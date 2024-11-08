#pragma once

#include "EngineBase.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

class EngineFP32 : public EngineBase {
public:
    explicit EngineFP32(const Options &options) : m_options(options) {}
    ~EngineFP32() { clearGpuBuffers(); }

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

        std::cout << "Loading TensorRT engine file at path: " << trtModelPath << std::endl;

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
            auto errMsg = "Unable to set GPU device index to: " + std::to_string(m_options.deviceIndex) + ". Note, your device has " +
                          std::to_string(numGPUs) + " CUDA-capable GPU(s).";
            throw std::runtime_error(errMsg);
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
                if (m_engine->getTensorDataType(tensorName) != nvinfer1::DataType::kFLOAT) {
                    throw std::runtime_error("Error, only FP32 input is supported in EngineFP32");
                }

                m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
                m_inputBatchSize = tensorShape.d[0];
            } else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {
                if (m_engine->getTensorDataType(tensorName) != nvinfer1::DataType::kFLOAT) {
                    throw std::runtime_error("Error, only FP32 output is supported in EngineFP32");
                }

                uint32_t outputLength = 1;
                m_outputDims.push_back(tensorShape);

                for (int j = 1; j < tensorShape.nbDims; ++j) {
                    outputLength *= tensorShape.d[j];
                }

                m_outputLengths.push_back(outputLength);
                Util::checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], outputLength * m_options.maxBatchSize * sizeof(float), stream));
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

        if (inputs[0].size() > static_cast<size_t>(m_options.maxBatchSize)) {
            std::cout << "The batch size is larger than the model expects!" << std::endl;
            return false;
        }

        if (m_inputBatchSize != -1 && inputs[0].size() != static_cast<size_t>(m_inputBatchSize)) {
            std::cout << "The batch size is different from what the model expects!" << std::endl;
            return false;
        }

        const auto batchSize = static_cast<int32_t>(inputs[0].size());
        for (size_t i = 1; i < inputs.size(); ++i) {
            if (inputs[i].size() != static_cast<size_t>(batchSize)) {
                std::cout << "The batch size needs to be constant for all inputs!" << std::endl;
                return false;
            }
        }

        cudaStream_t inferenceCudaStream;
        Util::checkCudaErrorCode(cudaStreamCreate(&inferenceCudaStream));

        std::vector<cv::cuda::GpuMat> preprocessedInputs;
        for (size_t i = 0; i < numInputs; ++i) {
            const auto &batchInput = inputs[i];
            const auto &dims = m_inputDims[i];

            auto &input = batchInput[0];
            if (input.channels() != dims.d[0] || input.rows != dims.d[1] || input.cols != dims.d[2]) {
                std::cout << "Input dimensions mismatch!" << std::endl;
                return false;
            }

            nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1], dims.d[2]};
            m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims);

            auto processed = blobFromGpuMats(batchInput, m_subVals, m_divVals, m_normalize);
            preprocessedInputs.push_back(processed);
            m_buffers[i] = processed.ptr<void>();
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
                std::vector<float> output(m_outputLengths[outputBinding - numInputs]);
                Util::checkCudaErrorCode(cudaMemcpyAsync(
                    output.data(),
                    static_cast<char *>(m_buffers[outputBinding]) + (batch * sizeof(float) * m_outputLengths[outputBinding - numInputs]),
                    m_outputLengths[outputBinding - numInputs] * sizeof(float), cudaMemcpyDeviceToHost, inferenceCudaStream));
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

        const auto numInputs = network->getNbInputs();
        if (numInputs < 1) {
            throw std::runtime_error("Error, model needs at least 1 input!");
        }

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            return false;
        }

        nvinfer1::IOptimizationProfile *optProfile = builder->createOptimizationProfile();

        const auto input0Batch = network->getInput(0)->getDimensions().d[0];
        bool doesSupportDynamicBatch = (input0Batch == -1);

        for (int32_t i = 0; i < numInputs; ++i) {
            const auto input = network->getInput(i);
            const auto inputName = input->getName();
            const auto inputDims = input->getDimensions();
            int32_t inputC = inputDims.d[1];
            int32_t inputH = inputDims.d[2];
            int32_t inputW = inputDims.d[3];

            if (doesSupportDynamicBatch) {
                optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, inputC, inputH, inputW));
            } else {
                optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN,
                                          nvinfer1::Dims4(m_options.optBatchSize, inputC, inputH, inputW));
            }

            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT,
                                      nvinfer1::Dims4(m_options.optBatchSize, inputC, inputH, inputW));
            optProfile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX,
                                      nvinfer1::Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
        }
        config->addOptimizationProfile(optProfile);

        // FP32 specific - no precision flags needed as FP32 is default

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
        engineName += ".fp32";
        engineName += "." + std::to_string(options.maxBatchSize);
        engineName += "." + std::to_string(options.optBatchSize);

        return engineName;
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

    void clearGpuBuffers() {
        if (!m_buffers.empty()) {
            const auto numInputs = m_inputDims.size();
            for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
                Util::checkCudaErrorCode(cudaFree(m_buffers[outputBinding]));
            }
            m_buffers.clear();
        }
    }

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

private:
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