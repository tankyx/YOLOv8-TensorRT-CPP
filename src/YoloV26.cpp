#include "YoloV26.h"
#include "YoloPostprocKernels.h"

// ---------------------------------------------------------------------------
// V26 GPU postproc kernel — threshold only, no decode, no NMS
// ---------------------------------------------------------------------------

void YoloV26::launchPostprocKernel(EngineFP16 *fp16Engine,
                                   int /*numAnchors*/, int /*numClasses*/,
                                   cudaStream_t stream) {
    YoloV26FilterParams fp{};
    fp.output = static_cast<const __half *>(fp16Engine->outputDevicePtr(0));
    fp.maxDetections = MAX_DETECTIONS_V26;
    fp.probThreshold = PROBABILITY_THRESHOLD;
    fp.outCount = m_devSurvivorCount;
    fp.outSurvivors = m_devSurvivors;
    fp.maxSurvivors = kMaxSurvivors;
    launchYoloV26FilterKernel(fp, stream);
}

// ---------------------------------------------------------------------------
// V26 CPU postprocessing — NMS-free, confidence threshold only
// ---------------------------------------------------------------------------

std::vector<Object> YoloV26::postprocessFromSurvivors(uint32_t count) {
    // GPU kernel already applied the confidence threshold and wrote survivors
    // in (x1, y1, x2, y2, confidence, class_id) format. No NMS, no re-check.
    const uint32_t n = std::min(count, static_cast<uint32_t>(kMaxSurvivors));

    std::vector<Object> objects;
    objects.reserve(std::min(static_cast<int>(n), TOP_K));

    for (uint32_t i = 0; i < n; ++i) {
        if (static_cast<int>(objects.size()) >= TOP_K) { break; }

        const float *s = &m_hostSurvivors[i * kYoloSurvivorStride];
        Object obj{};
        obj.rect.x = s[0];
        obj.rect.y = s[1];
        obj.rect.width = s[2] - s[0];
        obj.rect.height = s[3] - s[1];
        obj.probability = s[4];
        obj.label = static_cast<int>(s[5]);
        objects.push_back(obj);
    }
    return objects;
}

// ---------------------------------------------------------------------------
// V26 CPU fallback postprocessing (FP32 path)
// ---------------------------------------------------------------------------

std::vector<Object> YoloV26::postprocessDetect(const std::vector<float> &featureVector) {
    const auto &outputDims = m_trtEngine->getOutputDims();
    // Output can be (1, detections, 6) or (1, 6, detections).
    // Compute the true detection count from total elements / 6 fields.
    const int totalElements = outputDims[0].d[1] * outputDims[0].d[2];
    const int maxDetections = totalElements / 6;

    // Output layout: (1, maxDetections, 6) — 6 fields contiguous per detection.
    const float *data = featureVector.data();

    std::vector<Object> objects;
    objects.reserve(std::min(maxDetections, TOP_K));

    for (int i = 0; i < maxDetections; ++i) {
        if (static_cast<int>(objects.size()) >= TOP_K) { break; }

        const float *det = &data[i * 6];
        const float x1   = det[0];
        const float y1   = det[1];
        const float x2   = det[2];
        const float y2   = det[3];
        const float conf = det[4];
        const float cls  = det[5];

        if (conf <= PROBABILITY_THRESHOLD) { continue; }

        Object obj{};
        obj.rect.x = x1;
        obj.rect.y = y1;
        obj.rect.width = x2 - x1;
        obj.rect.height = y2 - y1;
        obj.probability = conf;
        obj.label = static_cast<int>(cls);
        objects.push_back(obj);
    }
    return objects;
}
