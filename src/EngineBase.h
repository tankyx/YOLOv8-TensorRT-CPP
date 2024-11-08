#pragma once

#include <NvInfer.h>
#include <array>
#include <cuda_runtime.h>
#include <memory>
#include <opencv2/core/cuda.hpp>
#include <vector>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>
#include <fstream>

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
        // Would advise using a proper logging utility such as
        // https://github.com/gabime/spdlog For the sake of this tutorial, will just
        // log to the console.

        // Only log Warnings or more important.
        if (severity <= Severity::kVERBOSE) {
            std::cout << msg << std::endl;
        }
    }
};

class EngineBase {
public:
    virtual ~EngineBase() = default;

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

    virtual void setCaptureDimensions(int height, int width) = 0;

protected:
    static cv::cuda::GpuMat blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput, const std::array<float, 3> &subVals,
                                            const std::array<float, 3> &divVals, bool normalize);
};