#include "yolov8.h"
#include "YoloPostprocKernels.h"

// ---------------------------------------------------------------------------
// V8 GPU postproc kernel — standard anchor-based decode (no DFL)
// ---------------------------------------------------------------------------

void YoloV8::launchPostprocKernel(EngineFP16 *fp16Engine,
                                  int numAnchors, int numClasses,
                                  cudaStream_t stream) {
    YoloFilterParams fp{};
    fp.output = static_cast<const __half *>(fp16Engine->outputDevicePtr(0));
    fp.numAnchors = numAnchors;
    fp.numClasses = numClasses;
    fp.ratio = m_ratio;
    fp.imgW = m_imgWidth;
    fp.imgH = m_imgHeight;
    fp.probThreshold = PROBABILITY_THRESHOLD;
    fp.outCount = m_devSurvivorCount;
    fp.outSurvivors = m_devSurvivors;
    fp.maxSurvivors = kMaxSurvivors;
    launchYoloFilterAndDecodeKernel(fp, stream);
}

// ---------------------------------------------------------------------------
// V8 CPU fallback postprocessing (FP32 path)
// ---------------------------------------------------------------------------

std::vector<Object> YoloV8::postprocessDetect(const std::vector<float> &featureVector) {
    const auto &outputDims = m_trtEngine->getOutputDims();
    auto numChannels = outputDims[0].d[1];
    auto numAnchors = outputDims[0].d[2];
    auto numClasses = CLASS_NAMES.size();

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    const float *data = featureVector.data();
    bboxes.reserve(numAnchors / 16);
    scores.reserve(numAnchors / 16);
    labels.reserve(numAnchors / 16);

    for (int i = 0; i < numAnchors; ++i) {
        const float x = data[0 * numAnchors + i];
        const float y = data[1 * numAnchors + i];
        const float w = data[2 * numAnchors + i];
        const float h = data[3 * numAnchors + i];

        int bestLabel = 0;
        float bestScore = data[4 * numAnchors + i];
        for (int c = 1; c < static_cast<int>(numClasses); ++c) {
            const float s = data[(4 + c) * numAnchors + i];
            if (s > bestScore) {
                bestScore = s;
                bestLabel = c;
            }
        }

        if (bestScore <= PROBABILITY_THRESHOLD) { continue; }

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
