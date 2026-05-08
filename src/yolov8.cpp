#include "yolov8.h"

YoloV8::YoloV8(const std::string &onnxModelPath, const YoloV8Config &config)
    : PROBABILITY_THRESHOLD(config.probabilityThreshold), NMS_THRESHOLD(config.nmsThreshold), TOP_K(config.topK),
      SEG_CHANNELS(config.segChannels), SEG_H(config.segH), SEG_W(config.segW), SEGMENTATION_THRESHOLD(config.segmentationThreshold),
      CLASS_NAMES(config.classNames), NUM_KPS(config.numKPS), KPS_THRESHOLD(config.kpsThreshold) {
    // Specify options for GPU inference
    Options options;
    options.optBatchSize = 1;
    options.maxBatchSize = 1;

    options.precision = config.precision;
    options.calibrationDataDirectoryPath = config.calibrationDataDirectory;

    if (options.precision == Precision::INT8) {
        if (options.calibrationDataDirectoryPath.empty()) {
            throw std::runtime_error("Error: Must supply calibration data path for INT8 calibration");
        }
    }

    // Create our TensorRT inference engine
    m_trtEngine = EngineFactory::createEngine(options);
    std::cout << "Building or loading TensorRT engine..." << std::endl;

    // Build the onnx model into a TensorRT engine file, cache the file to disk, and then load the TensorRT engine file into memory.
    // If the engine file already exists on disk, this function will not rebuild but only load into memory.
    // The engine file is rebuilt any time the above Options are changed.
    auto succ = m_trtEngine->buildLoadNetwork(onnxModelPath, SUB_VALS, DIV_VALS, NORMALIZE);
    if (!succ) {
        const std::string errMsg = "Error: Unable to build or load the TensorRT engine. "
                                   "Try increasing TensorRT log severity to kVERBOSE (in /libs/tensorrt-cpp-api/engine.cpp).";
        throw std::runtime_error(errMsg);
    }
    std::cout << "TensorRT engine built and loaded!" << std::endl;
}

std::vector<std::vector<cv::cuda::GpuMat>> YoloV8::preprocess(const cv::cuda::GpuMat &gpuImg) {
    // Populate the input vectors
    const auto &inputDims = m_trtEngine->getInputDims();

    // Convert the image from BGR to RGB
    cv::cuda::GpuMat rgbMat;
    cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGR2RGB);

    auto resized = rgbMat;

    // Resize to the model expected input size while maintaining the aspect ratio with the use of padding
    if (resized.rows != inputDims[0].d[1] || resized.cols != inputDims[0].d[2]) {
        // Only resize if not already the right size to avoid unecessary copy
        resized = EngineBase::resizeKeepAspectRatioPadRightBottom(rgbMat, inputDims[0].d[1], inputDims[0].d[2]);
    }

    // Convert to format expected by our inference engine
    // The reason for the strange format is because it supports models with multiple inputs as well as batching
    // In our case though, the model only has a single input and we are using a batch size of 1.
    std::vector<cv::cuda::GpuMat> input{std::move(resized)};
    std::vector<std::vector<cv::cuda::GpuMat>> inputs{std::move(input)};

    // These params will be used in the post-processing stage
    m_imgHeight = rgbMat.rows;
    m_imgWidth = rgbMat.cols;
    m_ratio = 1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols), inputDims[0].d[1] / static_cast<float>(rgbMat.rows));

    return inputs;
}

std::vector<Object> YoloV8::detectObjects(const cv::cuda::GpuMat &inputImageBGR) {
    // c33/c35 FP16 fast path: the engine writes outputs into its pooled host vectors; no
    // triple-nested vector and no per-frame allocation.
    if (auto *fp16Engine = dynamic_cast<EngineFP16 *>(m_trtEngine.get())) {
        float ratio = 1.0f;
        const bool succ = fp16Engine->runInferenceFromBGR(inputImageBGR, ratio, SUB_VALS, DIV_VALS, NORMALIZE);
        if (!succ) {
            throw std::runtime_error("Error: Unable to run inference (FP16 fast path).");
        }
        m_imgWidth = static_cast<float>(inputImageBGR.cols);
        m_imgHeight = static_cast<float>(inputImageBGR.rows);
        m_ratio = ratio;

        std::vector<Object> ret;
        if (m_trtEngine->getOutputDims().size() == 1) {
            ret = postprocessDetect(fp16Engine->hostOutputs()[0]);
        }
        return ret;
    }

    // FP32 / generic fallback: preprocess via OpenCV, legacy runInference, transformOutput.
    std::vector<std::vector<std::vector<float>>> featureVectors;
    const auto input = preprocess(inputImageBGR);
    const bool succ = m_trtEngine->runInference(input, featureVectors);
    if (!succ) {
        throw std::runtime_error("Error: Unable to run inference.");
    }

    std::vector<Object> ret;
    if (m_trtEngine->getOutputDims().size() == 1) {
        std::vector<float> featureVector;
        m_trtEngine->transformOutput(featureVectors, featureVector);
        ret = postprocessDetect(featureVector);
    }
    return ret;
}

std::vector<Object> YoloV8::detectObjects(const cv::Mat &inputImageBGR) {
    // Upload the image to GPU memory
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImageBGR);

    // Call detectObjects with the GPU image
    return detectObjects(gpuImg);
}

std::vector<Object> YoloV8::postprocessDetect(const std::vector<float> &featureVector) {
    const auto &outputDims = m_trtEngine->getOutputDims();
    auto numChannels = outputDims[0].d[1];
    auto numAnchors = outputDims[0].d[2];

    auto numClasses = CLASS_NAMES.size();

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    // featureVector is laid out as numChannels rows × numAnchors columns: channel c at anchor i
    // is at index `c * numAnchors + i`. Previously we constructed a Mat view + .t() (which
    // copies) and then iterated row-major over the transposed view. Indexing the original
    // layout directly drops one full Mat allocation per detection.
    const float *data = featureVector.data();
    bboxes.reserve(numAnchors / 16);
    scores.reserve(numAnchors / 16);
    labels.reserve(numAnchors / 16);

    for (int i = 0; i < numAnchors; ++i) {
        const float *xPtr = data + 0 * numAnchors + i;
        const float *yPtr = data + 1 * numAnchors + i;
        const float *wPtr = data + 2 * numAnchors + i;
        const float *hPtr = data + 3 * numAnchors + i;

        // Find the best class score for this anchor across all numClasses class channels.
        int bestLabel = 0;
        float bestScore = data[4 * numAnchors + i];
        for (int c = 1; c < static_cast<int>(numClasses); ++c) {
            const float s = data[(4 + c) * numAnchors + i];
            if (s > bestScore) {
                bestScore = s;
                bestLabel = c;
            }
        }

        if (bestScore <= PROBABILITY_THRESHOLD) {
            continue;
        }

        const float x = *xPtr;
        const float y = *yPtr;
        const float w = *wPtr;
        const float h = *hPtr;

        const float x0 = std::clamp((x - 0.5f * w) * m_ratio, 0.f, m_imgWidth);
        const float y0 = std::clamp((y - 0.5f * h) * m_ratio, 0.f, m_imgHeight);
        const float x1 = std::clamp((x + 0.5f * w) * m_ratio, 0.f, m_imgWidth);
        const float y1 = std::clamp((y + 0.5f * h) * m_ratio, 0.f, m_imgHeight);

        cv::Rect_<float> bbox;
        bbox.x = x0;
        bbox.y = y0;
        bbox.width = x1 - x0;
        bbox.height = y1 - y0;

        bboxes.push_back(bbox);
        labels.push_back(bestLabel);
        scores.push_back(bestScore);
    }

    // Run NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, PROBABILITY_THRESHOLD, NMS_THRESHOLD, indices);

    std::vector<Object> objects;

    // Choose the top k detections
    int cnt = 0;
    for (auto &chosenIdx : indices) {
        if (cnt >= TOP_K) {
            break;
        }

        Object obj{};
        obj.probability = scores[chosenIdx];
        obj.label = labels[chosenIdx];
        obj.rect = bboxes[chosenIdx];
        objects.push_back(obj);

        cnt += 1;
    }

    return objects;
}

void YoloV8::drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects, unsigned int scale, int squareHalfSize) {
    // Get the center of the screen
    int screenCenterX = image.cols / 2;
    int screenCenterY = image.rows / 2;

    // If segmentation information is present, start with that
    if (!objects.empty() && !objects[0].boxMask.empty()) {
        cv::Mat mask = image.clone();
        for (const auto &object : objects) {
            // Choose the color
            int colorIndex = object.label % CS2_COLORS.size();
            cv::Scalar color = cv::Scalar(CS2_COLORS[colorIndex][0], CS2_COLORS[colorIndex][1], CS2_COLORS[colorIndex][2]);

            // Add the mask for said object
            mask(object.rect).setTo(color * 255, object.boxMask);
        }
        // Add all the masks to our image
        cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
    }

    // Bounding boxes and annotations
    for (auto &object : objects) {
        // Choose the color
        int colorIndex = object.label % CS2_COLORS.size();
        cv::Scalar color = cv::Scalar(CS2_COLORS[colorIndex][0], CS2_COLORS[colorIndex][1], CS2_COLORS[colorIndex][2]);

        const auto &rect = object.rect;

        // Draw rectangles and text
        char text[256];
        sprintf(text, "%s %d", CLASS_NAMES[object.label].c_str(), object.label);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);

        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = object.rect.x;
        int y = object.rect.y + 1;

        cv::rectangle(image, rect, color * 255, scale + 1);

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)), txt_bk_color, -1);

        cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, color, scale);

        // Pose estimation
        if (!object.kps.empty()) {
            auto &kps = object.kps;
            for (int k = 0; k < NUM_KPS + 2; k++) {
                if (k < NUM_KPS) {
                    int kpsX = std::round(kps[k * 3]);
                    int kpsY = std::round(kps[k * 3 + 1]);
                    float kpsS = kps[k * 3 + 2];
                    if (kpsS > KPS_THRESHOLD) {
                        cv::Scalar kpsColor = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                        cv::circle(image, {kpsX, kpsY}, 5, kpsColor, -1);
                    }
                }
                auto &ske = SKELETON[k];
                int pos1X = std::round(kps[(ske[0] - 1) * 3]);
                int pos1Y = std::round(kps[(ske[0] - 1) * 3 + 1]);

                int pos2X = std::round(kps[(ske[1] - 1) * 3]);
                int pos2Y = std::round(kps[(ske[1] - 1) * 3 + 1]);

                float pos1S = kps[(ske[0] - 1) * 3 + 2];
                float pos2S = kps[(ske[1] - 1) * 3 + 2];

                if (pos1S > KPS_THRESHOLD && pos2S > KPS_THRESHOLD) {
                    cv::Scalar limbColor = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                    cv::line(image, {pos1X, pos1Y}, {pos2X, pos2Y}, limbColor, 2);
                }
            }
        }
    }
}
