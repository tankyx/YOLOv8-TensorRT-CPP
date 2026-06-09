#include "DetectorFactory.h"
#include "EngineFactory.h"

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

YoloVersion DetectorFactory::detectFromEngine(const EngineBase &engine,
                                              int numClasses) {
    const auto &outputDims = engine.getOutputDims();
    if (outputDims.empty()) {
        return YoloVersion::V8;
    }
    return YoloDetector::detectVersionFromOutput(outputDims[0], numClasses);
}

std::unique_ptr<YoloDetector> DetectorFactory::create(
    const std::string &onnxModelPath,
    const YoloConfig &config,
    YoloVersion version) {

    // If the caller specified an explicit version, use it directly.
    if (version != YoloVersion::AUTO) {
        switch (version) {
        case YoloVersion::V8:
            return std::make_unique<YoloV8>(onnxModelPath, config);
        case YoloVersion::V11:
            return std::make_unique<YoloV11>(onnxModelPath, config);
        case YoloVersion::V26:
            return std::make_unique<YoloV26>(onnxModelPath, config);
        default:
            break; // AUTO handled below
        }
    }

    // First pass: try filename-based detection. This is fast and works before
    // the engine is built.
    const auto fileVersion = YoloDetector::detectVersionFromFilename(onnxModelPath);
    if (fileVersion != YoloVersion::AUTO) {
        // Strong signal from filename. Use it directly — skip the temp-engine dance.
        std::cout << "[DetectorFactory] Detected "
                  << (fileVersion == YoloVersion::V11 ? "YOLOv11" :
                      fileVersion == YoloVersion::V26 ? "YOLOv26" : "YOLOv8")
                  << " from model filename." << std::endl;
        switch (fileVersion) {
        case YoloVersion::V8:
            return std::make_unique<YoloV8>(onnxModelPath, config);
        case YoloVersion::V11:
            return std::make_unique<YoloV11>(onnxModelPath, config);
        case YoloVersion::V26:
            return std::make_unique<YoloV26>(onnxModelPath, config);
        default:
            break;
        }
    }

    // Second pass: build the engine and inspect the output shape.
    // This handles renamed files and non-standard naming.
    auto temp = std::make_unique<YoloV8>(onnxModelPath, config);
    const auto shapeVersion = detectFromEngine(temp->getEngine(),
                                               static_cast<int>(config.classNames.size()));

    if (shapeVersion == YoloVersion::V8) {
        return temp; // correct already
    }

    // Shape says V11 or V26 — discard temp and re-create.
    // The engine .plan file is cached, so re-creation just deserializes.
    switch (shapeVersion) {
    case YoloVersion::V11:
        std::cout << "[DetectorFactory] Auto-detected YOLOv11 from output shape." << std::endl;
        return std::make_unique<YoloV11>(onnxModelPath, config);
    case YoloVersion::V26:
        std::cout << "[DetectorFactory] Auto-detected YOLOv26 from output shape." << std::endl;
        return std::make_unique<YoloV26>(onnxModelPath, config);
    default:
        return temp;
    }
}
