#pragma once
#include "YoloDetector.h"
#include "yolov8.h"
#include "YoloV11.h"
#include "YoloV26.h"
#include <memory>

// ---------------------------------------------------------------------------
// DetectorFactory — creates the right YoloDetector subclass based on the
// model's ONNX output shape or an explicit version override.
// ---------------------------------------------------------------------------

class DetectorFactory {
public:
    /// Create a detector for the given ONNX model.
    ///
    /// If `version` is not AUTO, it is used directly (skips auto-detection).
    /// When AUTO (default), the version is inferred from filename first,
    /// then from the ONNX output shape after engine load.
    ///
    /// The engine is built and loaded during construction — the returned
    /// detector is ready for inference.
    static std::unique_ptr<YoloDetector> create(
        const std::string &onnxModelPath,
        const YoloConfig &config,
        YoloVersion version = YoloVersion::AUTO);

    /// Auto-detect from a built engine's output dims.
    /// Call after create() if you want to inspect what was detected.
    static YoloVersion detectFromEngine(const EngineBase &engine,
                                        int numClasses);
};
