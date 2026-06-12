#include "YoloDetector.h"
#include "YoloPostprocKernels.h"
#include "EngineFP16.h"

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

YoloDetector::YoloDetector(const std::string &onnxModelPath, const YoloConfig &config,
                           YoloVersion version)
    : PROBABILITY_THRESHOLD(config.probabilityThreshold),
      NMS_THRESHOLD(config.nmsThreshold),
      TOP_K(config.topK),
      SEG_CHANNELS(config.segChannels),
      SEG_H(config.segH),
      SEG_W(config.segW),
      SEGMENTATION_THRESHOLD(config.segmentationThreshold),
      CLASS_NAMES(config.classNames),
      NUM_KPS(config.numKPS),
      KPS_THRESHOLD(config.kpsThreshold),
      REG_MAX(config.regMax),
      MAX_DETECTIONS_V26(config.maxDetectionsV26),
      m_version(version) {

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

    m_trtEngine = EngineFactory::createEngine(options);
    std::cout << "Building or loading TensorRT engine..." << std::endl;

    auto succ = m_trtEngine->buildLoadNetwork(onnxModelPath, SUB_VALS, DIV_VALS, NORMALIZE);
    if (!succ) {
        const std::string errMsg = "Error: Unable to build or load the TensorRT engine. "
                                   "Try increasing TensorRT log severity to kVERBOSE.";
        throw std::runtime_error(errMsg);
    }
    std::cout << "TensorRT engine built and loaded!" << std::endl;
    std::cout.flush(); // force flush before anything that might crash

    // Cache fp16 engine pointer (avoids dynamic_cast per frame).
    m_fp16Engine = dynamic_cast<EngineFP16 *>(m_trtEngine.get());
    // If the engine's output tensors are FP32 (TensorRT upcast the FP16 ONNX),
    // keep the fast path for input (fused preproc works since input is still Half)
    // but skip the GPU postproc kernel — raw D2H + CPU postproc instead.
    if (m_fp16Engine && m_fp16Engine->isEngineFP32()) {
        std::cout << "[YoloDetector] Engine has FP32 output — keeping fast path, GPU postproc disabled" << std::endl;
        m_outputIsFP32 = true;
        // Pre-allocate host buffer for raw FP32 output D2H copy.
        m_rawOutputLen = static_cast<int>(m_fp16Engine->outputLength(0));
        m_hostRawOutput.resize(static_cast<size_t>(m_rawOutputLen));
    }

    // Cache output dimensions — never change after engine construction.
    if (m_trtEngine->getOutputDims().size() >= 1) {
        const auto &od = m_trtEngine->getOutputDims()[0];
        m_numAnchors = od.d[2];
        const int numChannels = od.d[1];

        if (m_version == YoloVersion::V26) {
            m_numClasses = static_cast<int>(CLASS_NAMES.size());
            std::cout << "[YoloDetector] V26 output dims: (" << od.d[0]
                      << ", " << od.d[1] << ", " << od.d[2] << ")"
                      << "  maxDetectionsV26=" << MAX_DETECTIONS_V26
                      << "  fp16Engine=" << (m_fp16Engine ? "yes" : "no")
                      << std::endl;
        } else if (m_version == YoloVersion::V11) {
            m_numClasses = numChannels - 4 * REG_MAX;
        } else {
            m_numClasses = numChannels - 4;
        }
    }

    if (config.precision == Precision::FP16) {
        std::cout << "[YoloDetector] Allocating postproc buffers..." << std::endl;
        allocatePostprocBuffers();
        std::cout << "[YoloDetector] Postproc buffers OK" << std::endl;
    }
    std::cout << "[YoloDetector] Constructor complete" << std::endl;
}

YoloDetector::~YoloDetector() {
    releaseGraph();
    if (m_captureBuffer) {
        cudaFree(m_captureBuffer);
        m_captureBuffer = nullptr;
    }
    freePostprocBuffers();
}

// ---------------------------------------------------------------------------
// GPU buffer management
// ---------------------------------------------------------------------------

void YoloDetector::allocatePostprocBuffers() {
    if (m_devSurvivorCount) { return; }

    uint32_t *tmpCount = nullptr;
    float *tmpSurvivors = nullptr;

    // Allocate both GPU buffers before updating members — if either fails,
    // we clean up and throw without leaving partial state.
    cudaError_t err1 = cudaMalloc(reinterpret_cast<void **>(&tmpCount), sizeof(uint32_t));
    if (err1 != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA malloc failed for survivor count: ") +
                                 cudaGetErrorString(err1));
    }

    cudaError_t err2 = cudaMalloc(reinterpret_cast<void **>(&tmpSurvivors),
                                  static_cast<size_t>(kMaxSurvivors) * kYoloSurvivorStride * sizeof(float));
    if (err2 != cudaSuccess) {
        cudaFree(tmpCount); // roll back first allocation
        throw std::runtime_error(std::string("CUDA malloc failed for survivor buffer: ") +
                                 cudaGetErrorString(err2));
    }

    m_devSurvivorCount = tmpCount;
    m_devSurvivors = tmpSurvivors;
    m_hostSurvivors.resize(static_cast<size_t>(kMaxSurvivors) * kYoloSurvivorStride);
}

void YoloDetector::freePostprocBuffers() {
    if (m_devSurvivorCount) { cudaFree(m_devSurvivorCount); m_devSurvivorCount = nullptr; }
    if (m_devSurvivors)     { cudaFree(m_devSurvivors);     m_devSurvivors = nullptr; }
}

// ---------------------------------------------------------------------------
// CUDA graph infrastructure
// ---------------------------------------------------------------------------

void YoloDetector::releaseGraph() {
    if (m_graphExec) { cudaGraphExecDestroy(m_graphExec); m_graphExec = nullptr; }
    if (m_graph)     { cudaGraphDestroy(m_graph);         m_graph = nullptr; }
    m_graphCaptured = false;
}

bool YoloDetector::runFp16InferenceOnStream(const cv::cuda::GpuMat &captureView,
                                            int numAnchors, int numClasses,
                                            cudaStream_t stream) {
    float ratio = 1.0f;
    if (!m_fp16Engine->runInferenceFromBGR(captureView, ratio, SUB_VALS, DIV_VALS, NORMALIZE)) {
        return false;
    }
    m_imgWidth = static_cast<float>(captureView.cols);
    m_imgHeight = static_cast<float>(captureView.rows);
    m_ratio = ratio;

    if (m_outputIsFP32) {
        // Engine output is FP32 — skip GPU postproc kernel, copy raw output.
        cudaMemcpyAsync(m_hostRawOutput.data(),
                        m_fp16Engine->outputDevicePtr(0),
                        static_cast<size_t>(m_rawOutputLen) * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    } else {
        cudaMemsetAsync(m_devSurvivorCount, 0, sizeof(uint32_t), stream);

        // Version-specific GPU postproc kernel (virtual dispatch)
        launchPostprocKernel(m_fp16Engine, numAnchors, numClasses, stream);

        cudaMemcpyAsync(&m_hostSurvivorCount, m_devSurvivorCount,
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(m_hostSurvivors.data(), m_devSurvivors,
                        static_cast<size_t>(kMaxSurvivors) * kYoloSurvivorStride * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Shared NMS postprocessing (used by V8 and V11; V26 overrides)
// ---------------------------------------------------------------------------

std::vector<Object> YoloDetector::postprocessFromSurvivors(uint32_t count) {
    const uint32_t n = std::min(count, static_cast<uint32_t>(kMaxSurvivors));

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    bboxes.reserve(n);
    scores.reserve(n);
    labels.reserve(n);

    for (uint32_t i = 0; i < n; ++i) {
        const float *s = &m_hostSurvivors[i * kYoloSurvivorStride];
        const float x0 = s[0], y0 = s[1], x1 = s[2], y1 = s[3];
        cv::Rect_<float> bbox;
        bbox.x = x0;
        bbox.y = y0;
        bbox.width = x1 - x0;
        bbox.height = y1 - y0;
        bboxes.push_back(bbox);
        scores.push_back(s[4]);
        labels.push_back(static_cast<int>(s[5]));
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, PROBABILITY_THRESHOLD,
                             NMS_THRESHOLD, indices);

    std::vector<Object> objects;
    int cnt = 0;
    for (auto &chosenIdx : indices) {
        if (cnt >= TOP_K) { break; }
        Object obj{};
        obj.probability = scores[chosenIdx];
        obj.label = labels[chosenIdx];
        obj.rect = bboxes[chosenIdx];
        objects.push_back(obj);
        cnt += 1;
    }
    return objects;
}

// ---------------------------------------------------------------------------
// Preprocessing (FP32 fallback)
// ---------------------------------------------------------------------------

std::vector<std::vector<cv::cuda::GpuMat>> YoloDetector::preprocess(const cv::cuda::GpuMat &gpuImg) {
    const auto &inputDims = m_trtEngine->getInputDims();

    cv::cuda::GpuMat rgbMat;
    if (gpuImg.channels() == 4)
        cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGRA2RGB);
    else
        cv::cuda::cvtColor(gpuImg, rgbMat, cv::COLOR_BGR2RGB);

    auto resized = rgbMat;
    if (resized.rows != inputDims[0].d[1] || resized.cols != inputDims[0].d[2]) {
        resized = EngineBase::resizeKeepAspectRatioPadRightBottom(rgbMat, inputDims[0].d[1], inputDims[0].d[2]);
    }

    std::vector<cv::cuda::GpuMat> input{std::move(resized)};
    std::vector<std::vector<cv::cuda::GpuMat>> inputs{std::move(input)};

    m_imgHeight = rgbMat.rows;
    m_imgWidth = rgbMat.cols;
    m_ratio = 1.f / std::min(inputDims[0].d[2] / static_cast<float>(rgbMat.cols),
                             inputDims[0].d[1] / static_cast<float>(rgbMat.rows));

    return inputs;
}

// ---------------------------------------------------------------------------
// Main detection entry point
// ---------------------------------------------------------------------------

std::vector<Object> YoloDetector::detectObjects(const cv::cuda::GpuMat &inputImageBGR) {
    // -- FP16 fast path (CUDA graph capture + GPU postproc) ------------------
    if (m_fp16Engine) {
        const cudaStream_t stream = m_fp16Engine->stream();

        // Lazy-init/capture source channel info on first frame.
        if (m_srcChannels == 0) {
            m_srcChannels = inputImageBGR.type() == CV_8UC3 ? 3 :
                           (inputImageBGR.type() == CV_8UC4 ? 4 : 0);
            if (m_srcChannels == 0) {
                throw std::runtime_error("Error: detectObjects requires CV_8UC3 (BGR) or CV_8UC4 (BGRA) GpuMat.");
            }
        }

        if (!m_captureBuffer) {
            m_captureWidth = inputImageBGR.cols;
            m_captureHeight = inputImageBGR.rows;
            m_captureChannels = m_srcChannels;
            m_captureBufferPitch = static_cast<size_t>(m_captureWidth) * m_captureChannels;
            const size_t bytes = m_captureBufferPitch * m_captureHeight;
            cudaMalloc(reinterpret_cast<void **>(&m_captureBuffer), bytes);
        } else if (inputImageBGR.cols != m_captureWidth ||
                   inputImageBGR.rows != m_captureHeight ||
                   m_srcChannels != m_captureChannels) {
            releaseGraph();
            cudaFree(m_captureBuffer);
            m_captureBuffer = nullptr;
            m_captureWidth = inputImageBGR.cols;
            m_captureHeight = inputImageBGR.rows;
            m_captureChannels = m_srcChannels;
            m_captureBufferPitch = static_cast<size_t>(m_captureWidth) * m_captureChannels;
            cudaMalloc(reinterpret_cast<void **>(&m_captureBuffer),
                       m_captureBufferPitch * m_captureHeight);
        }

        cudaMemcpy2DAsync(m_captureBuffer, m_captureBufferPitch,
                          inputImageBGR.ptr<uint8_t>(), inputImageBGR.step,
                          static_cast<size_t>(m_captureWidth) * m_captureChannels,
                          m_captureHeight, cudaMemcpyDeviceToDevice, stream);

        const int viewType = m_captureChannels == 3 ? CV_8UC3 : CV_8UC4;
        cv::cuda::GpuMat captureView(m_captureHeight, m_captureWidth, viewType,
                                     m_captureBuffer, m_captureBufferPitch);

        if (!m_graphCaptured) {
            // Warmup
            if (!runFp16InferenceOnStream(captureView, m_numAnchors, m_numClasses, stream)) {
                throw std::runtime_error("Error: FP16 warmup pass failed.");
            }
            cudaStreamSynchronize(stream);

            // Capture
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
            if (!runFp16InferenceOnStream(captureView, m_numAnchors, m_numClasses, stream)) {
                cudaStreamEndCapture(stream, &m_graph);
                cudaGraphDestroy(m_graph);
                m_graph = nullptr;
                throw std::runtime_error("Error: FP16 graph-capture pass failed.");
            }
            cudaStreamEndCapture(stream, &m_graph);

            const cudaError_t instErr = cudaGraphInstantiate(&m_graphExec, m_graph, nullptr, nullptr, 0);
            if (instErr != cudaSuccess) {
                cudaGraphDestroy(m_graph);
                m_graph = nullptr;
                throw std::runtime_error(std::string("cudaGraphInstantiate failed: ") +
                                         cudaGetErrorString(instErr));
            }
            m_graphCaptured = true;
            cudaStreamSynchronize(stream);
            if (m_outputIsFP32) {
                return postprocessDetect(m_hostRawOutput);
            }
            return postprocessFromSurvivors(m_hostSurvivorCount);
        }

        // Steady state: replay captured graph.
        cudaGraphLaunch(m_graphExec, stream);
        cudaStreamSynchronize(stream);
        if (m_outputIsFP32) {
            return postprocessDetect(m_hostRawOutput);
        }
        return postprocessFromSurvivors(m_hostSurvivorCount);
    }

    // -- FP32 / generic fallback ---------------------------------------------
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

std::vector<Object> YoloDetector::detectObjects(const cv::Mat &inputImageBGR) {
    cv::cuda::GpuMat gpuImg;
    gpuImg.upload(inputImageBGR);
    return detectObjects(gpuImg);
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

void YoloDetector::drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects,
                                    unsigned int scale, int /*squareHalfSize*/) {
    if (!objects.empty() && !objects[0].boxMask.empty()) {
        cv::Mat mask = image.clone();
        for (const auto &object : objects) {
            int colorIndex = object.label % CS2_COLORS.size();
            cv::Scalar color = cv::Scalar(CS2_COLORS[colorIndex][0],
                                          CS2_COLORS[colorIndex][1],
                                          CS2_COLORS[colorIndex][2]);
            mask(object.rect).setTo(color * 255, object.boxMask);
        }
        cv::addWeighted(image, 0.5, mask, 0.8, 1, image);
    }

    for (auto &object : objects) {
        int colorIndex = object.label % CS2_COLORS.size();
        cv::Scalar color = cv::Scalar(CS2_COLORS[colorIndex][0],
                                      CS2_COLORS[colorIndex][1],
                                      CS2_COLORS[colorIndex][2]);
        const auto &rect = object.rect;

        char text[256];
        snprintf(text, sizeof(text), "%s %d", CLASS_NAMES[object.label].c_str(), object.label);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                             0.35 * scale, scale, &baseLine);
        cv::Scalar txt_bk_color = color * 0.7 * 255;

        int x = object.rect.x;
        int y = object.rect.y + 1;

        cv::rectangle(image, rect, color * 255, scale + 1);
        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                     cv::Size(labelSize.width, labelSize.height + baseLine)),
                     txt_bk_color, -1);
        cv::putText(image, text, cv::Point(x, y + labelSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, color, scale);

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

// ---------------------------------------------------------------------------
// Auto-detection
// ---------------------------------------------------------------------------

YoloVersion YoloDetector::detectVersionFromOutput(const nvinfer1::Dims &outputDims,
                                                  int numClasses) {
    // outputDims layout: (batch, channels, anchors) or (batch, detections, boxinfo)
    const int channels = outputDims.d[1];
    const int anchors  = outputDims.d[2];

    // V26 end-to-end: output can be (1, 6, ~300) or (1, ~300, 6)
    // The 6 fields are: (x1, y1, x2, y2, confidence, class_id)
    if (channels == 6 && anchors >= 100 && anchors <= 500) {
        return YoloVersion::V26;
    }
    if (anchors == 6 && channels >= 100 && channels <= 500) {
        return YoloVersion::V26;
    }

    // V8 vs V11: output is (1, 4+C, 8400) or (1, 4*regMax+C, 8400)
    // V8:  channels = 4 + numClasses  (e.g. 4+80 = 84 for COCO)
    // V11: channels = 4*regMax + numClasses (e.g. 4*16+80 = 144 for COCO)
    //
    // Since regMax can vary (8, 16, 32), we detect V11 by checking if
    // (channels - numClasses) is divisible by 4 AND is much larger than 4.
    const int bboxChannels = channels - numClasses;

    if (bboxChannels == 4) {
        return YoloVersion::V8;
    }

    // If bboxChannels is a multiple of 4 and > 4, it's likely DFL (V11).
    // Common regMax values: 8 (32 bbox ch), 16 (64), 32 (128).
    if (bboxChannels > 4 && bboxChannels % 4 == 0 && bboxChannels <= 256) {
        // Sanity: regMax = bboxChannels/4 should be a reasonable value.
        const int regMax = bboxChannels / 4;
        if (regMax == 8 || regMax == 16 || regMax == 32) {
            return YoloVersion::V11;
        }
    }

    // Can't distinguish — default to V8.
    return YoloVersion::V8;
}

YoloVersion YoloDetector::detectVersionFromFilename(const std::string &modelPath) {
    // Extract the filename (last component of the path).
    std::string name = modelPath;
    const auto lastSlash = name.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        name = name.substr(lastSlash + 1);
    }
    // Lowercase for case-insensitive matching.
    for (auto &c : name) { c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }

    // Check for version strings in the filename.
    // Order matters: check more specific patterns first.
    if (name.find("yolo26") != std::string::npos ||
        name.find("yolov26") != std::string::npos) {
        return YoloVersion::V26;
    }
    if (name.find("yolo11") != std::string::npos ||
        name.find("yolov11") != std::string::npos) {
        return YoloVersion::V11;
    }
    if (name.find("yolo8") != std::string::npos ||
        name.find("yolov8") != std::string::npos) {
        return YoloVersion::V8;
    }
    return YoloVersion::AUTO; // no signal from filename
}