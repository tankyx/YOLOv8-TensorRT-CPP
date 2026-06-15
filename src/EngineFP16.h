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
        // When the engine is FP32 (TensorRT upcast), stash preproc params so
        // runInference can blend them via blobFromGpuMats (matching EngineFP32).
        m_subValsFP32 = subVals;
        m_divValsFP32 = divVals;
        m_normalizeFP32 = normalize;
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
        std::cout << "[EngineFP16] Execution context created, setting up I/O..." << std::endl;
        std::cout.flush();

        clearGpuBuffers();
        std::cout << "[EngineFP16] Buffers cleared, " << m_engine->getNbIOTensors() << " I/O tensors" << std::endl;
        std::cout.flush();
        m_buffers.resize(m_engine->getNbIOTensors());
        m_outputLengths.clear();
        m_inputDims.clear();
        m_outputDims.clear();
        m_IOTensorNames.clear();

        // Detect actual tensor precision — the ONNX may be exported as FP16 but
        // TensorRT may upcast to FP32 internally (common with certain opsets).
        bool engineIsFP16 = true;
        for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
            const auto name = m_engine->getIOTensorName(i);
            if (m_engine->getTensorDataType(name) != nvinfer1::DataType::kHALF) {
                engineIsFP16 = false;
                break;
            }
        }
        m_engineIsFP32 = !engineIsFP16;
        if (m_engineIsFP32) {
            std::cout << "[EngineFP16] TensorRT upcast FP16 ONNX to FP32 engine — using FP32 path internally" << std::endl;
        }

        for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
            const auto tensorName = m_engine->getIOTensorName(i);
            m_IOTensorNames.emplace_back(tensorName);
            const auto tensorType = m_engine->getTensorIOMode(tensorName);
            const auto tensorShape = m_engine->getTensorShape(tensorName);
            const bool isFloat = m_engine->getTensorDataType(tensorName) == nvinfer1::DataType::kFLOAT;

            if (tensorType == nvinfer1::TensorIOMode::kINPUT) {
                m_inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
                m_inputBatchSize = tensorShape.d[0];

                const int inputC = tensorShape.d[1];
                const int inputH = tensorShape.d[2];
                const int inputW = tensorShape.d[3];
                size_t inputBytes;
                if (isFloat) {
                    inputBytes = static_cast<size_t>(inputC) * inputH * inputW * m_options.maxBatchSize * sizeof(float);
                } else {
                    inputBytes = static_cast<size_t>(inputC) * inputH * inputW * m_options.maxBatchSize * sizeof(__half);
                }
                inputBytes = (inputBytes + 31) & ~31;

                Util::checkCudaErrorCode(cudaMallocAsync(&m_ownedInputBuffer, inputBytes, m_stream));
                m_buffers[i] = m_ownedInputBuffer;
                m_inputBufferIndex = i;
            } else if (tensorType == nvinfer1::TensorIOMode::kOUTPUT) {
                uint32_t outputLength = 1;
                m_outputDims.push_back(tensorShape);

                for (int j = 1; j < tensorShape.nbDims; ++j) {
                    outputLength *= tensorShape.d[j];
                }

                m_outputLengths.push_back(outputLength);

                size_t allocSize;
                if (isFloat) {
                    allocSize = outputLength * m_options.maxBatchSize * sizeof(float);
                } else {
                    allocSize = outputLength * m_options.maxBatchSize * sizeof(__half);
                }
                allocSize = (allocSize + 31) & ~31;

                Util::checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], allocSize, m_stream));
            }
        }

        Util::checkCudaErrorCode(cudaStreamSynchronize(m_stream));

        return true;
    }

    // c37: device-only fast path. The engine schedules preproc + enqueueV3 on m_stream and
    // returns immediately — no D2H, no synchronization, no host output. The caller is expected
    // to chain its own GPU postprocess kernel + D2H + sync on the same stream. Used by YoloV8
    // alongside launchYoloFilterAndDecodeKernel to keep all output processing on the GPU.
    //
    // Accessors below expose the engine's stream and the device output binding pointer / length
    // so the caller can hand them to a downstream kernel.
    [[nodiscard]] bool isEngineFP32() const { return m_engineIsFP32; }
    [[nodiscard]] cudaStream_t stream() const { return m_stream; }
    [[nodiscard]] void *outputDevicePtr(int outputIdx) const {
        return m_buffers[m_inputDims.size() + outputIdx];
    }
    [[nodiscard]] uint32_t outputLength(int outputIdx) const { return m_outputLengths[outputIdx]; }
    [[nodiscard]] int numOutputs() const { return static_cast<int>(m_outputLengths.size()); }

    // c33 fast path: take the raw uint8 BGR capture directly, run the fused preproc kernel into
    // the pre-allocated FP16 input buffer, and schedule enqueueV3 on m_stream. Returns immediately
    // after enqueue — output stays on device for downstream GPU postprocess (c37).
    //
    // outRatio is the postprocess scale factor: bbox_dst * outRatio = bbox_src. Matches what
    // YoloV8::preprocess used to set m_ratio to.
    bool runInferenceFromBGR(const cv::cuda::GpuMat &bgr, float &outRatio, const std::array<float, 3> &subVals,
                             const std::array<float, 3> &divVals, bool normalize) {
        if (bgr.empty() || (bgr.type() != CV_8UC3 && bgr.type() != CV_8UC4)) {
            std::cout << "runInferenceFromBGR expects a non-empty CV_8UC3 (BGR) or CV_8UC4 (BGRA) GpuMat" << std::endl;
            return false;
        }
        if (m_inputDims.size() != 1 || m_inputBufferIndex < 0) {
            std::cout << "runInferenceFromBGR requires a single FP16 input binding" << std::endl;
            return false;
        }
        const int srcChannels = bgr.type() == CV_8UC3 ? 3 : 4;

        const auto &dims = m_inputDims[0];
        const int dstC = dims.d[0];
        const int dstH = dims.d[1];
        const int dstW = dims.d[2];
        if (dstC != 3) {
            std::cout << "runInferenceFromBGR only supports 3-channel input" << std::endl;
            return false;
        }

        // Letterbox geometry (matches resizeKeepAspectRatioPadRightBottom).
        const float r = std::min(static_cast<float>(dstW) / static_cast<float>(bgr.cols),
                                 static_cast<float>(dstH) / static_cast<float>(bgr.rows));
        const int unpadW = static_cast<int>(r * bgr.cols);
        const int unpadH = static_cast<int>(r * bgr.rows);
        outRatio = 1.0f / r;

        // Restore our owned input pointer in case the legacy runInference path has overwritten it
        // with a transient OpenCV GpuMat pointer (the two paths are mutually exclusive in practice
        // — YoloV8 dispatches by engine type — but be defensive).
        m_buffers[m_inputBufferIndex] = m_ownedInputBuffer;

        FusedPreprocParams p{};
        p.src = bgr.ptr<uint8_t>();
        p.srcW = bgr.cols;
        p.srcH = bgr.rows;
        p.srcPitch = static_cast<int>(bgr.step);
        p.srcChannels = srcChannels;
        p.dst = static_cast<__half *>(m_buffers[m_inputBufferIndex]);
        p.dstW = dstW;
        p.dstH = dstH;
        p.invScale = 1.0f / r;
        p.unpadW = unpadW;
        p.unpadH = unpadH;
        p.scale255 = normalize ? (1.0f / 255.0f) : 1.0f;
        for (int c = 0; c < 3; ++c) {
            p.sub[c] = subVals[c];
            p.div[c] = divVals[c] != 0.0f ? divVals[c] : 1.0f;
        }
        launchFusedPreprocBGRtoFP16Kernel(p, m_stream);

        // Bind shapes & addresses. The output buffers were allocated in loadNetwork; the input
        // buffer is already at m_buffers[m_inputBufferIndex] and we just wrote it on m_stream.
        const nvinfer1::Dims4 inputDims4{1, dstC, dstH, dstW};
        m_context->setInputShape(m_IOTensorNames[m_inputBufferIndex].c_str(), inputDims4);
        if (!m_context->allInputDimensionsSpecified()) {
            throw std::runtime_error("Not all required dimensions specified");
        }

        for (size_t i = 0; i < m_buffers.size(); ++i) {
            if (!m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i])) {
                return false;
            }
        }

        if (!m_context->enqueueV3(m_stream)) {
            return false;
        }

        // c37: caller chains its own GPU postprocess + D2H + sync on m_stream. We deliberately do
        // not synchronize here so the next stage runs back-to-back with inference.
        return true;
    }

    // FP32 fallback runInference — used when TensorRT upcasts an FP16 ONNX to an FP32 engine.
    // Mirrors EngineFP32::runInference (blobFromGpuMats → enqueueV3 → D2H memcpy).
    bool runInference(const std::vector<std::vector<cv::cuda::GpuMat>> &inputs,
                      std::vector<std::vector<std::vector<float>>> &featureVectors) override {
        if (!m_engineIsFP32) {
            std::cout << "EngineFP16::runInference is only supported when the engine is FP32." << std::endl;
            return false;
        }
        if (inputs.empty() || inputs[0].empty()) {
            std::cout << "Provided input vector is empty!" << std::endl;
            return false;
        }

        const auto numInputs = m_inputDims.size();
        if (inputs.size() != numInputs || inputs[0].size() > static_cast<size_t>(m_options.maxBatchSize)) {
            std::cout << "Incorrect number of inputs or batch size too large!" << std::endl;
            return false;
        }

        const auto batchSize = static_cast<int32_t>(inputs[0].size());
        if (m_inputBatchSize != -1 && batchSize != m_inputBatchSize) {
            std::cout << "The batch size is different from what the model expects!" << std::endl;
            return false;
        }

        // Set input shapes and preprocess via blobFromGpuMats.
        std::vector<cv::cuda::GpuMat> preprocessedInputs;
        for (size_t i = 0; i < numInputs; ++i) {
            const auto &dims = m_inputDims[i];
            auto &input = inputs[i][0];
            if (input.channels() != dims.d[0] || input.rows != dims.d[1] || input.cols != dims.d[2]) {
                std::cout << "Input dimensions mismatch!" << std::endl;
                return false;
            }
            nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1], dims.d[2]};
            m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims);

            auto processed = blobFromGpuMats(inputs[i], m_subValsFP32, m_divValsFP32, m_normalizeFP32);
            preprocessedInputs.push_back(processed);
            m_buffers[i] = processed.ptr<void>();
        }

        if (!m_context->allInputDimensionsSpecified()) {
            throw std::runtime_error("Not all required dimensions specified");
        }

        // Bind tensor addresses.
        for (size_t i = 0; i < m_buffers.size(); ++i) {
            if (!m_context->setTensorAddress(m_IOTensorNames[i].c_str(), m_buffers[i])) {
                return false;
            }
        }

        // Enqueue and copy outputs back.
        if (!m_context->enqueueV3(m_stream)) {
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
                    m_outputLengths[outputBinding - numInputs] * sizeof(float), cudaMemcpyDeviceToHost, m_stream));
                batchOutputs.emplace_back(std::move(output));
            }
            featureVectors.emplace_back(std::move(batchOutputs));
        }

        Util::checkCudaErrorCode(cudaStreamSynchronize(m_stream));
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
        if (m_stream && m_ownedInputBuffer) {
            cudaFreeAsync(m_ownedInputBuffer, m_stream);
            m_ownedInputBuffer = nullptr;
            m_inputBufferIndex = -1;
        }
        if (m_buffers.empty() || !m_engine || !m_stream) {
            return;
        }
        const auto numInputs = m_inputDims.size();
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
            if (m_buffers[outputBinding]) {
                Util::checkCudaErrorCode(cudaFreeAsync(m_buffers[outputBinding], m_stream));
            }
        }
        Util::checkCudaErrorCode(cudaStreamSynchronize(m_stream));
        m_buffers.clear();
    }

    std::vector<void *> m_buffers;
    std::vector<uint32_t> m_outputLengths{};
    std::vector<nvinfer1::Dims3> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;
    std::vector<std::string> m_IOTensorNames;
    int32_t m_inputBatchSize{-1};

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    // c33: pre-allocated FP16 CHW input buffer used by the fused preproc fast path
    // (runInferenceFromBGR). Owned by the engine; its address is mirrored into m_buffers at the
    // input binding index. The legacy runInference path overwrites m_buffers[input] with a
    // transient OpenCV GpuMat pointer, so the fast path restores from m_ownedInputBuffer on entry.
    void *m_ownedInputBuffer = nullptr;
    int m_inputBufferIndex = -1;
    bool m_engineIsFP32 = false;  // set in loadNetwork when TRT upcasts FP16 ONNX → FP32 engine

    // Stashed preproc params for the FP32 fallback path (runInference).
    std::array<float, 3> m_subValsFP32{};
    std::array<float, 3> m_divValsFP32{};
    bool m_normalizeFP32{true};
};