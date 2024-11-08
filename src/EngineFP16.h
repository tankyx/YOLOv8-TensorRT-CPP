#pragma once

#include "EngineBase.h"
#include "EngineFP16Kernels.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void convertFP32ToFP16Kernel(const float *input, __half *output, int size);

class EngineFP16 : public EngineBase {
public:
    explicit EngineFP16(const Options &options) : m_options(options) {}
    ~EngineFP16() { clearGpuBuffers(); }

    bool buildLoadNetwork(std::string onnxModelPath, const std::array<float, 3> &subVals = {0.f, 0.f, 0.f},
                          const std::array<float, 3> &divVals = {1.f, 1.f, 1.f}, bool normalize = true) override {
        const auto engineName = serializeEngineOptions(m_options, onnxModelPath);
        std::cout << "Searching for engine file with name: " << engineName << std::endl;

        if (Util::doesFileExist(engineName)) {
            std::cout << "Engine found, not regenerating..." << std::endl;
        } else {
            if (!Util::doesFileExist(onnxModelPath)) {
                throw std::runtime_error("Could not find onnx model at path: " + onnxModelPath);
            }

            std::cout << "Engine not found, generating. This could take a while..." << std::endl;
            auto ret = build(onnxModelPath, subVals, divVals, normalize);
            if (!ret) {
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

        m_outputLengths.clear();
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

            // Process and convert to FP16 in one step
            auto processed = blobFromGpuMats(batchInput, m_subVals, m_divVals, m_normalize);

            // Ensure alignment
            size_t pitch = (processed.step + 31) & ~31; // 32-byte alignment
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
                // Allocate space for FP16 data
                std::vector<__half> fp16Output(m_outputLengths[outputBinding - numInputs]);

                // Copy FP16 data from GPU
                Util::checkCudaErrorCode(cudaMemcpyAsync(
                    fp16Output.data(),
                    static_cast<char *>(m_buffers[outputBinding]) + (batch * sizeof(__half) * m_outputLengths[outputBinding - numInputs]),
                    m_outputLengths[outputBinding - numInputs] * sizeof(__half), cudaMemcpyDeviceToHost, inferenceCudaStream));

                // Convert FP16 to FP32
                std::vector<float> output(fp16Output.size());
                for (size_t i = 0; i < fp16Output.size(); ++i) {
                    output[i] = __half2float(fp16Output[i]);
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

    void setCaptureDimensions(int height, int width) override {
        captureHeight = height;
        captureWidth = width;
    }

private:
    bool build(std::string onnxModelPath, const std::array<float, 3> &subVals, const std::array<float, 3> &divVals, bool normalize) {
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
        if (!builder) {
            return false;
        }

        if (!builder->platformHasFastFp16()) {
            throw std::runtime_error("GPU does not support FP16");
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
            throw std::runtime_error("Unable to read engine file");
        }

        auto parsed = parser->parse(buffer.data(), buffer.size());
        if (!parsed) {
            return false;
        }

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            return false;
        }

        // Enable FP16 mode
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

        // Enable TF32 for better numerical stability with FP16
        if (builder->platformHasTf32()) {
            config->setFlag(nvinfer1::BuilderFlag::kTF32);
        }

        auto optProfile = builder->createOptimizationProfile();
        const auto numInputs = network->getNbInputs();

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
            return false;
        }

        const auto engineName = serializeEngineOptions(m_options, onnxModelPath);
        std::ofstream outfile(engineName, std::ofstream::binary);
        outfile.write(reinterpret_cast<const char *>(plan->data()), plan->size());

        Util::checkCudaErrorCode(cudaStreamDestroy(profileStream));
        return true;
    }

    void clearGpuBuffers() {
        if (!m_buffers.empty()) {
            const auto numInputs = m_inputDims.size();
            for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
                Util::checkCudaErrorCode(cudaFree(m_buffers[outputBinding]));
            }
            m_buffers.clear();
        }
    }

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
        engineName += ".fp16"; // Always FP16 for this engine
        engineName += "." + std::to_string(options.maxBatchSize);
        engineName += "." + std::to_string(options.optBatchSize);

        return engineName;
    }

        static void convertFP32ToFP16(const cv::cuda::GpuMat &input, cv::cuda::GpuMat &output) {
        // Create output mat for FP16
        output.create(input.rows, input.cols, CV_16FC3);

        const int numElements = input.rows * input.cols * 3; // 3 channels

        launchConvertFP32ToFP16Kernel(input.ptr<float>(), output.ptr<__half>(), numElements);

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA error in FP16 conversion: " + std::string(cudaGetErrorString(error)));
        }

        // Synchronize
        cudaDeviceSynchronize();
    }

    void getDeviceNames(std::vector<std::string> &deviceNames) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        for (int device = 0; device < numGPUs; device++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);
            deviceNames.push_back(std::string(prop.name));
        }
    }

    static cv::cuda::GpuMat blobFromGpuMats(const std::vector<cv::cuda::GpuMat> &batchInput, const std::array<float, 3> &subVals,
                                            const std::array<float, 3> &divVals, bool normalize) {
        // Process in FP32 first
        cv::cuda::GpuMat gpu_dst(1, batchInput[0].rows * batchInput[0].cols * batchInput.size(), CV_8UC3);

        size_t width = batchInput[0].cols * batchInput[0].rows;
        for (size_t img = 0; img < batchInput.size(); img++) {
            std::vector<cv::cuda::GpuMat> input_channels{
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[0 + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width + width * 3 * img])),
                cv::cuda::GpuMat(batchInput[0].rows, batchInput[0].cols, CV_8U, &(gpu_dst.ptr()[width * 2 + width * 3 * img]))};
            cv::cuda::split(batchInput[img], input_channels);
        }

        // Do preprocessing in FP32
        cv::cuda::GpuMat mfloat;
        if (normalize) {
            gpu_dst.convertTo(mfloat, CV_32FC3, 1.f / 255.f);
        } else {
            gpu_dst.convertTo(mfloat, CV_32FC3);
        }

        cv::cuda::subtract(mfloat, cv::Scalar(subVals[0], subVals[1], subVals[2]), mfloat, cv::noArray(), -1);
        cv::cuda::divide(mfloat, cv::Scalar(divVals[0], divVals[1], divVals[2]), mfloat, 1, -1);

        // Convert to FP16 using our custom function
        cv::cuda::GpuMat mhalf;
        try {
            convertFP32ToFP16(mfloat, mhalf);
        } catch (const std::runtime_error &e) {
            std::cerr << "Error in FP16 conversion: " << e.what() << std::endl;
            throw;
        }

        return mhalf;
    }

    // Member variables
    std::array<float, 3> m_subVals{};
    std::array<float, 3> m_divVals{};
    bool m_normalize;

    std::vector<void *> m_buffers;
    std::vector<uint32_t> m_outputLengths{};
    std::vector<nvinfer1::Dims3> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;
    std::vector<std::string> m_IOTensorNames;
    int32_t m_inputBatchSize;

    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    const Options m_options;
    Logger m_logger;

    int captureHeight;
    int captureWidth;
};