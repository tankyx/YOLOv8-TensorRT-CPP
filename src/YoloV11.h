#pragma once
#include "YoloDetector.h"

// ---------------------------------------------------------------------------
// YOLOv11 detector — DFL-aware bbox decode, NMS required
// ---------------------------------------------------------------------------

class YoloV11 : public YoloDetector {
public:
    YoloV11(const std::string &onnxModelPath, const YoloConfig &config)
        : YoloDetector(onnxModelPath, config, YoloVersion::V11) {}

protected:
    void launchPostprocKernel(EngineFP16 *fp16Engine,
                              int numAnchors, int numClasses,
                              cudaStream_t stream) override;

    std::vector<Object> postprocessDetect(
        const std::vector<float> &featureVector) override;
};
