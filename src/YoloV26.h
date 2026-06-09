#pragma once
#include "YoloDetector.h"

// ---------------------------------------------------------------------------
// YOLOv26 detector — end-to-end, NMS-free, threshold only
// ---------------------------------------------------------------------------

class YoloV26 : public YoloDetector {
public:
    YoloV26(const std::string &onnxModelPath, const YoloConfig &config)
        : YoloDetector(onnxModelPath, config, YoloVersion::V26) {}

protected:
    void launchPostprocKernel(EngineFP16 *fp16Engine,
                              int numAnchors, int numClasses,
                              cudaStream_t stream) override;

    std::vector<Object> postprocessFromSurvivors(uint32_t count) override;

    std::vector<Object> postprocessDetect(
        const std::vector<float> &featureVector) override;
};
