//EngineBase.h

#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <sstream>
#include <vector>

namespace Util {
inline bool doesFileExist(const std::string &filepath) {
    std::ifstream f(filepath.c_str());
    return f.good();
}

inline void checkCudaErrorCode(cudaError_t code) {
    if (code != 0) {
        std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + "(" + cudaGetErrorName(code) +
                             "), with message: " + cudaGetErrorString(code);
        std::cout << errMsg << std::endl;
        throw std::runtime_error(errMsg);
    }
}

// 64-bit FNV-1a over the bytes of `path`. Used to invalidate the cached TensorRT engine plan
// when the source ONNX changes — keying the cache filename on file path + options alone meant
// editing the .onnx in place silently loaded a stale plan.
inline std::string fnv1a64HexOfFile(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return std::string("nohash");
    }
    constexpr uint64_t kOffset = 1469598103934665603ULL;
    constexpr uint64_t kPrime = 1099511628211ULL;
    uint64_t h = kOffset;
    char buf[8192];
    while (f) {
        f.read(buf, sizeof(buf));
        std::streamsize n = f.gcount();
        for (std::streamsize i = 0; i < n; ++i) {
            h ^= static_cast<uint8_t>(buf[i]);
            h *= kPrime;
        }
    }
    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << h;
    return oss.str();
}
} // namespace Util

// Keeping existing Options and Precision enums
enum class Precision {
    FP32,
    FP16,
    INT8,
};

struct Options {
    Precision precision = Precision::FP32;
    std::string calibrationDataDirectoryPath;
    int32_t calibrationBatchSize = 128;
    int32_t optBatchSize = 1;
    int32_t maxBatchSize = 16;
    int deviceIndex = 0;
};

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept {
        // Only log VERBOSE and below — TensorRT severity is decreasing, so kINTERNAL_ERROR=0 < kVERBOSE=4
        // means this filter currently lets every level through. Keeping behavior unchanged in this PR.
        if (severity <= Severity::kVERBOSE) {
            std::cout << msg << std::endl;
        }
    }
};

class EngineBase {
public:
    virtual ~EngineBase() {
        if (m_stream) {
            cudaStreamDestroy(m_stream);
            m_stream = nullptr;
        }
    }

protected:
    explicit EngineBase(const Options &options) : m_options(options) {
        Util::checkCudaErrorCode(cudaStreamCreate(&m_stream));
    }

public:

    virtual bool buildLoadNetwork(std::string onnxModelPath, const std::array<float, 3> &subVals = {0.f, 0.f, 0.f},
                                  const std::array<float, 3> &divVals = {1.f, 1.f, 1.f}, bool normalize = true) = 0;

    virtual bool loadNetwork(std::string trtModelPath, const std::array<float, 3> &subVals = {0.f, 0.f, 0.f},
                             const std::array<float, 3> &divVals = {1.f, 1.f, 1.f}, bool normalize = true) = 0;

    virtual bool runInference(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs,
                              std::vector<std::vector<std::vector<float>>> &featureVectors) = 0;

    [[nodiscard]] virtual const std::vector<nvinfer1::Dims3> &getInputDims() const = 0;
    [[nodiscard]] virtual const std::vector<nvinfer1::Dims> &getOutputDims() const = 0;

    static cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(const cv::cuda::GpuMat &input, size_t height, size_t width,
                                                                const cv::Scalar &bgcolor = cv::Scalar(0, 0, 0)) {
        float r = std::min(width / (input.cols * 1.0), height / (input.rows * 1.0));
        int unpad_w = r * input.cols;
        int unpad_h = r * input.rows;
        cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
        cv::cuda::resize(input, re, re.size());
        cv::cuda::GpuMat out(height, width, CV_8UC3, bgcolor);
        re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
        return out;
    }

    template <typename T>
    static void transformOutput(std::vector<std::vector<std::vector<T>>> &input, std::vector<T> &output)
    {
        if (input.size() != 1 || input[0].size() != 1) {
            throw std::logic_error("The feature vector has incorrect dimensions!");
        }

        output = std::move(input[0][0]);
    }

protected:
    // -- Shared engine helpers (c19) ----------------------------------------------------------
    // Subclasses provide the precision suffix (e.g. "fp16" / "fp32") used in the cached engine
    // filename. The rest of serializeEngineOptions is identical between FP32 and FP16.
    virtual const char *precisionSuffix() const = 0;

    // Subclasses set precision-specific builder flags (kFP16, kTF32) on the IBuilderConfig.
    virtual void applyPrecisionFlags(nvinfer1::IBuilder &builder, nvinfer1::IBuilderConfig &config) = 0;

    // Build the cached engine filename. Identical across precisions modulo `precisionSuffix()`.
    std::string serializeEngineOptions(const Options &options, const std::string &onnxModelPath) {
        const auto filenamePos = onnxModelPath.find_last_of('/') + 1;
        std::string engineName = onnxModelPath.substr(filenamePos, onnxModelPath.find_last_of('.') - filenamePos) + ".engine";

        std::vector<std::string> deviceNames;
        getDeviceNames(deviceNames);

        if (static_cast<size_t>(options.deviceIndex) >= deviceNames.size()) {
            throw std::runtime_error("Error, provided device index is out of range!");
        }

        auto deviceName = deviceNames[options.deviceIndex];
        deviceName.erase(std::remove_if(deviceName.begin(), deviceName.end(), ::isspace), deviceName.end());
        engineName += "." + deviceName;
        engineName += ".";
        engineName += precisionSuffix();
        engineName += "." + std::to_string(options.maxBatchSize);
        engineName += "." + std::to_string(options.optBatchSize);
        if (!m_onnxHash.empty()) {
            engineName += "." + m_onnxHash;
        }
        return engineName;
    }

    static void getDeviceNames(std::vector<std::string> &deviceNames) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        for (int device = 0; device < numGPUs; device++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);
            deviceNames.push_back(std::string(prop.name));
        }
    }

    // FP32-flavored CHW float blob (uint8 BGR/RGB GpuMat -> float CHW). EngineFP16 builds on top
    // of this by feeding the result through its FP32->FP16 kernel.
    static cv::cuda::GpuMat blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput, const std::array<float, 3> &subVals,
                                            const std::array<float, 3> &divVals, bool normalize) {
        cv::cuda::GpuMat gpu_dst(1, batchInput[0].rows * batchInput[0].cols * batchInput.size(), CV_8UC3);

        size_t width = batchInput[0].cols * batchInput[0].rows;
        for (size_t img = 0; img < batchInput.size(); img++) {
            std::vector<cv::cuda::GpuMat> input_channels{
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width * 2 + width * 3 * img]))};
            cv::cuda::split(batchInput[img], input_channels);
        }

        cv::cuda::GpuMat mfloat;
        if (normalize) {
            gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
        } else {
            gpu_dst.convertTo(mfloat, CV_32FC3);
        }

        cv::cuda::subtract(mfloat, cv::Scalar(subVals[0], subVals[1], subVals[2]), mfloat, cv::noArray(), -1);
        cv::cuda::divide(mfloat, cv::Scalar(divVals[0], divVals[1], divVals[2]), mfloat, 1, -1);

        return mfloat;
    }

    // Parse the ONNX, configure an optimization profile, ask the builder to serialize, and write
    // the resulting plan next to the working directory using the cached engine filename. Identical
    // between FP32 and FP16 modulo applyPrecisionFlags() and the FP32-specific dynamic-batch dance,
    // which we collapse into the shared profile below (kMIN=1 in both cases works for inference).
    bool buildSerialized(const std::string &onnxModelPath) {
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
        if (!builder) {
            return false;
        }

        auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network) {
            return false;
        }

        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
        if (!parser) {
            return false;
        }

        std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            throw std::runtime_error("Unable to read onnx model");
        }

        if (!parser->parse(buffer.data(), buffer.size())) {
            return false;
        }

        const auto numInputs = network->getNbInputs();
        if (numInputs < 1) {
            throw std::runtime_error("Error, model needs at least 1 input!");
        }

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            return false;
        }

        applyPrecisionFlags(*builder, *config);

        nvinfer1::IOptimizationProfile *optProfile = builder->createOptimizationProfile();
        for (int32_t i = 0; i < numInputs; ++i) {
            const auto input = network->getInput(i);
            const auto inputName = input->getName();
            const auto inputDims = input->getDimensions();
            int32_t inputC = inputDims.d[1];
            int32_t inputH = inputDims.d[2];
            int32_t inputW = inputDims.d[3];

            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, inputC, inputH, inputW));
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT,
                                      nvinfer1::Dims4(m_options.optBatchSize, inputC, inputH, inputW));
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX,
                                      nvinfer1::Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
        }
        config->addOptimizationProfile(optProfile);

        cudaStream_t profileStream;
        Util::checkCudaErrorCode(cudaStreamCreate(&profileStream));
        config->setProfileStream(profileStream);

        std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        if (!plan) {
            Util::checkCudaErrorCode(cudaStreamDestroy(profileStream));
            return false;
        }

        const auto engineName = serializeEngineOptions(m_options, onnxModelPath);
        std::ofstream outfile(engineName, std::ofstream::binary);
        outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());

        Util::checkCudaErrorCode(cudaStreamDestroy(profileStream));
        return true;
    }

    // Members shared by both engines; subclasses populate them via loadNetwork / runInference.
    Options m_options;
    Logger m_logger;
    std::string m_onnxHash;
    // Per-engine CUDA stream reused across inferences (c21). Created in the ctor.
    cudaStream_t m_stream = nullptr;
};
