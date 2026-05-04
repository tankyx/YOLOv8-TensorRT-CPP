// EngineFP16.h

#pragma once

#include "EngineBase.h"
#include "EngineFP16Kernels.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

class EngineFP16 : public EngineBase {
public:
    explicit EngineFP16(const Options &options) : EngineBase(options) {}
    ~EngineFP16() override { clearGpuBuffers(); }

    bool buildLoadNetwork(std::string onnxModelPath, const std::array<float, 3> &subVals = {0.f, 0.f, 0.f},
                          const std::array<float, 3> &divVals = {1.f, 1.f, 1.f}, bool normalize = true) override {
        m_onnxHash = Util::fnv1a64HexOfFile(onnxModelPath);
        const auto engineName = serializeEngineOptions(m_options, onnxModelPath);
        std::cout << "Searching for engine file with name: " << engineName << std::endl;

        if (Util::doesFileExist(engineName)) {
            std::cout << "Engine found, not regenerating..." << std::endl;
        } else {
            if (!Util::doesFileExist(onnxModelPath)) {
                throw std::runtime_error("Could not find onnx model at path: " + onnxModelPath);
            }
            std::cout << "Engine not found, generating. This could take a while..." << std::endl;
            if (!buildSerialized(onnxModelPath)) {
                return false;
            }
        }

        return loadNetwork(engineName, subVals, divVals, normalize);
    }

    bool loadNetwork(std::string trtModelPath, const std::array<float, 3> &subVals = {0.f, 0.f, 0.f},
                     const std::array<float, 3> &divVals = {1.f, 1.f, 1.f}, bool normalize = true) override {
        m_subVals = subVals;
        m_divVals = divVals;
        m_normalize = normalize;

        if (!Util::doesFileExist(trtModelPath)) {
            std::cout << "Error, unable to read TensorRT model at path: " + trtModelPath << std::endl;
            return false;
        }

        std::ifstream file(trtModelPath, std::ios::binary | std::ios::ate);
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            throw std::runtime_error("Unable to read engine file");
        }

        m_runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(m_logger)};
        if (!m_runtime) {
            return false;
        }

        auto ret = cudaSetDevice(m_options.deviceIndex);
        if (ret != 0) {
            int numGPUs;
            cudaGetDeviceCount(&numGPUs);
            throw std::runtime_error("Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex));
        }

        m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
        if (!m_engine) {
            return false;
        }

        m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
        if (!m_context) {
            return false;
        }

        clearGpuBuffers();
        m_buffers.resize(m_engine->getNbIOTensors());
        m_outputLengths.clear();
        m_inputDims.clear();
        m_outputDims.clear();
        m_IOTensorNames.clear();

        cudaStream_t stream;
        Util::checkCudaErrorCode(cudaStreamCreate(&stream));

        for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
            const auto tensorName = m_engine->getIOTensorName(i);
            m_IOTensorNames.emplace_back(tensorName);
            const auto tensorType = m_engine->getTensorIOMode(tensorName);
            const auto tensorShape = m_engine->getTensorShape(tensorName);

            if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
                if (m_engine->getTensorDataType(tensorName) != nvinfer1::DataType::kHALF) {
                    throw std::runtime_error("Error, expected FP16 input in EngineFP16");
                }

                m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
                m_inputBatchSize = tensorShape.d[0];
            } else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {
                if (m_engine->getTensorDataType(tensorName) != nvinfer1::DataType::kHALF) {
                    throw std::runtime_error("Error, expected FP16 output in EngineFP16");
                }

                uint32_t outputLength = 1;
                m_outputDims.push_back(tensorShape);

                for (int j = 1; j < tensorShape.nbDims; ++j) {
                    outputLength *= tensorShape.d[j];
                }

                m_outputLengths.push_back(outputLength);

                // Make sure the allocation is aligned for FP16
                size_t allocSize = outputLength * m_options.maxBatchSize * sizeof(__half);
                allocSize = (allocSize + 31) & ~31; // 32-byte alignment

                Util::checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], allocSize, stream));
            }
        }

        Util::checkCudaErrorCode(cudaStreamSynchronize(stream));
        Util::checkCudaErrorCode(cudaStreamDestroy(stream));

        return true;
    }

    bool runInference(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs,
                      std::vector<std::vector<std::vector<float>>> &featureVectors) override {
        if (inputs.empty() || inputs[0].empty()) {
            std::cout << "Provided input vector is empty!" << std::endl;
            return false;
        }

        const auto numInputs = m_inputDims.size();
        if (inputs.size() != numInputs) {
            std::cout << "Incorrect number of inputs provided!" << std::endl;
            return false;
        }

        const auto batchSize = static_cast<int32_t>(inputs[0].size());
        cudaStream_t inferenceCudaStream;
        Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

        std::vector<cv::cuda::GpuMat> preprocessedInputs;
        for (size_t i = 0; i < numInputs; ++i) {
            const auto &batchInput = inputs[i];
            const auto &dims = m_inputDims[i];

            nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1], dims.d[2]};
            m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims);

            // Process in FP32 then convert to FP16 in one step (custom CUDA kernel).
            auto processed = blobFromGpuMatsFP16(batchInput, m_subVals, m_divVals, m_normalize);

            // Ensure 32-byte alignment for FP16 tensor cores.
            size_t pitch = (processed.step + 31) & ~31;
            if (processed.step != pitch) {
                cv::cuda::GpuMat aligned(processed.rows, processed.cols, CV_16FC3, pitch);
                processed.copyTo(aligned);
                preprocessedInputs.push_back(aligned);
                m_buffers[i] = aligned.ptr<void>();
            } else {
                preprocessedInputs.push_back(processed);
                m_buffers[i] = processed.ptr<void>();
            }
        }

        if (!m_context->allInputDimensionsSpecified()) {
            throw std::runtime_error("Not all required dimensions specified");
        }

        for (size_t i = 0; i < m_buffers.size(); ++i) {
            if (!m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i])) {
                return false;
            }
        }

        if (!m_context->enqueueV3(inferenceCudaStream)) {
            return false;
        }

        featureVectors.clear();
        for (int batch = 0; batch < batchSize; ++batch) {
            std::vector<std::vector<float>> batchOutputs;
            for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
                // Pull the FP16 output to host, then convert to FP32 for the consumer's vector<float> contract.
                std::vector<__half> fp16Output(m_outputLengths[outputBinding - numInputs]);
                Util::checkCudaErrorCode(cudaMemcpyAsync(
                    fp16Output.data(),
                    static_cast<char *>(m_buffers[outputBinding]) + (batch * sizeof(__half) * m_outputLengths[outputBinding - numInputs]),
                    m_outputLengths[outputBinding - numInputs] * sizeof(__half), cudaMemcpyDeviceToHost, inferenceCudaStream));

                std::vector<float> output(fp16Output.size());
                for (size_t j = 0; j < fp16Output.size(); ++j) {
                    output[j] = __half2float(fp16Output[j]);
                }

                batchOutputs.emplace_back(std::move(output));
            }
            featureVectors.emplace_back(std::move(batchOutputs));
        }

        Util::checkCudaErrorCode(cudaStreamSynchronize(inferenceCudaStream));
        Util::checkCudaErrorCode(cudaStreamDestroy(inferenceCudaStream));

        return true;
    }

    [[nodiscard]] const std::vector<nvinfer1::Dims3> &getInputDims() const override { return m_inputDims; }

    [[nodiscard]] const std::vector<nvinfer1::Dims> &getOutputDims() const override { return m_outputDims; }

protected:
    const char *precisionSuffix() const override { return "fp16"; }

    void applyPrecisionFlags(nvinfer1::IBuilder &builder, nvinfer1::IBuilderConfig &config) override {
        if (!builder.platformHasFastFp16()) {
            throw std::runtime_error("GPU does not support FP16");
        }
        config.setFlag(nvinfer1::BuilderFlag::kFP16);
        if (builder.platformHasTf32()) {
            config.setFlag(nvinfer1::BuilderFlag::kTF32);
        }
    }

private:
    void clearGpuBuffers() {
        if (m_buffers.empty() || !m_engine) {
            return;
        }
        cudaStream_t stream;
        Util::checkCudaErrorCode(cudaStreamCreate(&stream));
        const auto numInputs = m_inputDims.size();
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
            if (m_buffers[outputBinding]) {
                Util::checkCudaErrorCode(cudaFreeAsync(m_buffers[outputBinding], stream));
            }
        }
        Util::checkCudaErrorCode(cudaStreamSynchronize(stream));
        Util::checkCudaErrorCode(cudaStreamDestroy(stream));
        m_buffers.clear();
    }

    static void convertFP32ToFP16(const cv::cuda::GpuMat &input, cv::cuda::GpuMat &output) {
        output.create(input.rows, input.cols, CV_16FC3);
        const int numElements = input.rows * input.cols * 3;
        launchConvertFP32ToFP16Kernel(input.ptr<float>(), output.ptr<__half>(), numElements);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA error in FP16 conversion: " + std::string(cudaGetErrorString(error)));
        }
        cudaDeviceSynchronize();
    }

    // FP16-flavored input prep: build the FP32 blob via the shared base helper, then cast to FP16
    // via the custom kernel.
    static cv::cuda::GpuMat blobFromGpuMatsFP16(const std::vector<cv::cuda::GpuMat> &batchInput, const std::array<float, 3> &subVals,
                                                const std::array<float, 3> &divVals, bool normalize) {
        cv::cuda::GpuMat mfloat = blobFromGpuMats(batchInput, subVals, divVals, normalize);
        cv::cuda::GpuMat mhalf;
        convertFP32ToFP16(mfloat, mhalf);
        return mhalf;
    }

    std::array<float, 3> m_subVals{};
    std::array<float, 3> m_divVals{};
    bool m_normalize{true};

    std::vector<void *> m_buffers;
    std::vector<uint32_t> m_outputLengths{};
    std::vector<nvinfer1::Dims3> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;
    std::vector<std::string> m_IOTensorNames;
    int32_t m_inputBatchSize{-1};

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
};
