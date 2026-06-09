#include "YoloV11.h"
#include "YoloPostprocKernels.h"

// ---------------------------------------------------------------------------
// V11 GPU postproc kernel — DFL decode + anchor-based detection
// ---------------------------------------------------------------------------

void YoloV11::launchPostprocKernel(EngineFP16 *fp16Engine,
                                   int numAnchors, int numClasses,
                                   cudaStream_t stream) {
    YoloV11FilterParams fp{};
    fp.output = static_cast<const __half *>(fp16Engine->outputDevicePtr(0));
    fp.numAnchors = numAnchors;
    fp.numClasses = numClasses;
    fp.regMax = REG_MAX;
    fp.ratio = m_ratio;
    fp.imgW = m_imgWidth;
    fp.imgH = m_imgHeight;
    fp.probThreshold = PROBABILITY_THRESHOLD;
    fp.outCount = m_devSurvivorCount;
    fp.outSurvivors = m_devSurvivors;
    fp.maxSurvivors = kMaxSurvivors;
    launchYoloV11FilterAndDecodeKernel(fp, stream);
}

// ---------------------------------------------------------------------------
// V11 CPU fallback postprocessing (FP32 path) — DFL decode on CPU
// ---------------------------------------------------------------------------

static inline float dflDecodeCoord(float *dist, int regMax) {
    // dist points to regMax logits for one coordinate at one anchor.
    // Modifies dist in-place (applies softmax). Returns the decoded coordinate value.
    float maxVal = dist[0];
    for (int b = 1; b < regMax; ++b) {
        if (dist[b] > maxVal) { maxVal = dist[b]; }
    }
    float sum = 0.0f;
    for (int b = 0; b < regMax; ++b) {
        dist[b] = std::exp(dist[b] - maxVal);
        sum += dist[b];
    }
    float coord = 0.0f;
    for (int b = 0; b < regMax; ++b) {
        coord += static_cast<float>(b) * (dist[b] / sum);
    }
    return coord;
}

std::vector<Object> YoloV11::postprocessDetect(const std::vector<float> &featureVector) {
    const auto &outputDims = m_trtEngine->getOutputDims();
    const int numChannels = outputDims[0].d[1];
    const int numAnchors  = outputDims[0].d[2];
    const int numClasses  = numChannels - 4 * REG_MAX;
    const int regMax = REG_MAX;

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    const float *data = featureVector.data();
    bboxes.reserve(numAnchors / 16);
    scores.reserve(numAnchors / 16);
    labels.reserve(numAnchors / 16);

    // Stack workspace for DFL softmax — regMax ≤ 16, no heap alloc needed.
    float tmpDist[32];

    for (int i = 0; i < numAnchors; ++i) {
        // DFL decode 4 coordinates
        float coords[4];
        for (int coord = 0; coord < 4; ++coord) {
            // Precompute base pointer for coordinate `coord` (invariant across bins).
            const float *coordBase = data + coord * regMax * numAnchors;
            for (int b = 0; b < regMax; ++b) {
                tmpDist[b] = coordBase[b * numAnchors + i];
            }
            coords[coord] = dflDecodeCoord(tmpDist, regMax);
        }

        const float x = coords[0];
        const float y = coords[1];
        const float w = coords[2];
        const float h = coords[3];

        // Find best class
        const int classOffset = 4 * regMax;
        int bestLabel = 0;
        float bestScore = data[(classOffset + 0) * numAnchors + i];
        for (int c = 1; c < numClasses; ++c) {
            const float s = data[(classOffset + c) * numAnchors + i];
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
