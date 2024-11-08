#pragma once

#include "EngineBase.h"
#include "EngineFP32.h"
#include "EngineFP16.h"
#include <memory>

class EngineFactory {
public:
    static std::unique_ptr<EngineBase> createEngine(const Options &options) {
        switch (options.precision) {
        case Precision::FP32:
            return std::make_unique<EngineFP32>(options);
        case Precision::FP16:
            return std::make_unique<EngineFP16>(options);
        case Precision::INT8:
            throw std::runtime_error("INT8 engine not implemented yet");
        default:
            throw std::runtime_error("Unknown precision type");
        }
    }
};