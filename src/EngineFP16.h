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

                // Pre-allocate the FP16 CHW input buffer (c33). The fused preproc kernel writes
                // directly here, so we no longer need a per-call OpenCV GpuMat for the input.
                const int inputC = tensorShape.d[1];
                const int inputH = tensorShape.d[2];
                const int inputW = tensorShape.d[3];
                size_t inputBytes = static_cast<size_t>(inputC) * inputH * inputW * m_options.maxBatchSize * sizeof(__half);
                inputBytes = (inputBytes + 31) & ~31; // 32-byte alignment for FP16 tensor cores

                Util::checkCudaErrorCode(cudaMallocAsync(&m_ownedInputBuffer, inputBytes, m_stream));
                m_buffers[i] = m_ownedInputBuffer;
                m_inputBufferIndex = i;
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

                Util::checkCudaErrorCode(cudaMallocAsync(&m_buffers[i], allocSize, m_stream));
            }
        }

        // c35: pre-size the host output pool so the fast path doesn't allocate per frame. One
        // vector per output binding, sized to that binding's element count.
        m_hostOutputs.assign(m_outputLengths.size(), std::vector<float>{});
        for (size_t i = 0; i < m_outputLengths.size(); ++i) {
            m_hostOutputs[i].resize(m_outputLengths[i]);
        }

        Util::checkCudaErrorCode(cudaStreamSynchronize(m_stream));

        return true;
    }

    // Host-side outputs from the most recent runInferenceFromBGR call. One vector per output
    // binding. Backing storage is owned and reused across calls (c35).
    [[nodiscard]] const std::vector<std::vector<float>> &hostOutputs() const { return m_hostOutputs; }

    // c33 fast path: take the raw uint8 BGR capture directly, run the fused preproc kernel into
    // the pre-allocated FP16 input buffer, enqueue inference, and write outputs into the pooled
    // host vectors exposed by hostOutputs(). Skips the triple-nested vector allocation that the
    // legacy runInference path emits.
    //
    // outRatio is the postprocess scale factor: bbox_dst * outRatio = bbox_src. Matches what
    // YoloV8::preprocess used to set m_ratio to.
    bool runInferenceFromBGR(const cv::cuda::GpuMat &bgr, float &outRatio, const std::array<float, 3> &subVals,
                             const std::array<float, 3> &divVals, bool normalize) {
        if (bgr.empty() || bgr.type() != CV_8UC3) {
            std::cout << "runInferenceFromBGR expects a non-empty CV_8UC3 GpuMat" << std::endl;
            return false;
        }
        if (m_inputDims.size() != 1 || m_inputBufferIndex < 0) {
            std::cout << "runInferenceFromBGR requires a single FP16 input binding" << std::endl;
            return false;
        }

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

        // Output path (c34/c35): D2H raw FP16 + CPU-side cast into the pooled host vectors. No
        // per-call allocation — the host output vectors are sized once at loadNetwork time.
        const auto numInputs = m_inputDims.size();
        for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
            const size_t outIdx = outputBinding - numInputs;
            const uint32_t length = m_outputLengths[outIdx];

            if (m_hostFp16Staging.size() < length) {
                m_hostFp16Staging.resize(length);
            }
            const __half *deviceFp16 = reinterpret_cast<const __half *>(m_buffers[outputBinding]);
            Util::checkCudaErrorCode(cudaMemcpyAsync(m_hostFp16Staging.data(), deviceFp16, length * sizeof(__half),
                                                     cudaMemcpyDeviceToHost, m_stream));
            Util::checkCudaErrorCode(cudaStreamSynchronize(m_stream));

            auto &output = m_hostOutputs[outIdx];
            // Pool already sized in loadNetwork; assert via resize-no-op to keep us safe if a
            // future engine ever returns dynamic output shapes.
            if (output.size() != length) {
                output.resize(length);
            }
            const __half *src = m_hostFp16Staging.data();
            float *dst = output.data();
            for (uint32_t k = 0; k < length; ++k) {
                dst[k] = __half2float(src[k]);
            }
        }
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

        std::vector<cv::cuda::GpuMat> preprocessedInputs;
        for (size_t i = 0; i < numInputs; ++i) {
            const auto &batchInput = inputs[i];
            const auto &dims = m_inputDims[i];

            nvinfer1::Dims4 inputDims = {batchSize, dims.d[0], dims.d[1], dims.d[2]};
            m_context->setInputShape(m_IOTensorNames[i].c_str(), inputDims);

            // Process in FP32 then convert to FP16 in one step (custom CUDA kernel).
            auto processed = blobFromGpuMatsFP16(batchInput, m_subVals, m_divVals, m_normalize, m_stream);

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

        if (!m_context->enqueueV3(m_stream)) {
            return false;
        }

        featureVectors.clear();
        for (int batch = 0; batch < batchSize; ++batch) {
            std::vector<std::vector<float>> batchOutputs;
            for (int32_t outputBinding = numInputs; outputBinding < m_engine->getNbIOTensors(); ++outputBinding) {
                const uint32_t length = m_outputLengths[outputBinding - numInputs];

                // FP16 -> FP32 on the GPU, then a single host transfer. Replaces the previous
                // scalar host-side __half2float loop (c23).
                ensureFp32StagingCapacity(length);

                const __half *deviceFp16 = reinterpret_cast<const __half *>(static_cast<char *>(m_buffers[outputBinding]) +
                                                                            (batch * sizeof(__half) * length));
                launchConvertFP16ToFP32Kernel(deviceFp16, m_fp32Staging, static_cast<int>(length), m_stream);

                std::vector<float> output(length);
                Util::checkCudaErrorCode(
                    cudaMemcpyAsync(output.data(), m_fp32Staging, length * sizeof(float), cudaMemcpyDeviceToHost, m_stream));

                Util::checkCudaErrorCode(cudaStreamSynchronize(m_stream)); // own each batch's output before the next overwrites the staging buf
                batchOutputs.emplace_back(std::move(output));
            }
            featureVectors.emplace_back(std::move(batchOutputs));
        }

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
    // Reusable device-side FP32 staging buffer for the FP16->FP32 output kernel (c23).
    // Sized to the largest output binding × batch and reused across calls.
    void ensureFp32StagingCapacity(uint32_t lengthElems) {
        if (lengthElems <= m_fp32StagingCapacity) {
            return;
        }
        if (m_fp32Staging) {
            Util::checkCudaErrorCode(cudaFreeAsync(m_fp32Staging, m_stream));
            Util::checkCudaErrorCode(cudaStreamSynchronize(m_stream));
            m_fp32Staging = nullptr;
        }
        Util::checkCudaErrorCode(cudaMallocAsync(reinterpret_cast<void **>(&m_fp32Staging), lengthElems * sizeof(float), m_stream));
        Util::checkCudaErrorCode(cudaStreamSynchronize(m_stream));
        m_fp32StagingCapacity = lengthElems;
    }

    void clearGpuBuffers() {
        if (m_stream && m_fp32Staging) {
            cudaFreeAsync(m_fp32Staging, m_stream);
            m_fp32Staging = nullptr;
            m_fp32StagingCapacity = 0;
        }
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

    static void convertFP32ToFP16(const cv::cuda::GpuMat &input, cv::cuda::GpuMat &output, cudaStream_t stream) {
        output.create(input.rows, input.cols, CV_16FC3);
        const int numElements = input.rows * input.cols * 3;
        launchConvertFP32ToFP16Kernel(input.ptr<float>(), output.ptr<__half>(), numElements, stream);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error("CUDA error in FP16 conversion: " + std::string(cudaGetErrorString(error)));
        }
        Util::checkCudaErrorCode(cudaStreamSynchronize(stream));
    }

    // FP16-flavored input prep: build the FP32 blob via the shared base helper, then cast to FP16
    // via the custom kernel on the engine's stream.
    static cv::cuda::GpuMat blobFromGpuMatsFP16(const std::vector<cv::cuda::GpuMat> &batchInput, const std::array<float, 3> &subVals,
                                                const std::array<float, 3> &divVals, bool normalize, cudaStream_t stream) {
        cv::cuda::GpuMat mfloat = blobFromGpuMats(batchInput, subVals, divVals, normalize);
        cv::cuda::GpuMat mhalf;
        convertFP32ToFP16(mfloat, mhalf, stream);
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

    // Staging buffer for the FP16->FP32 output kernel (c23). The fast path no longer uses it
    // (c34: raw FP16 D2H + CPU-side conversion); kept for the legacy runInference path until the
    // helper itself is removed in the cleanup commit.
    float *m_fp32Staging = nullptr;
    uint32_t m_fp32StagingCapacity = 0;

    // c34: host-side FP16 staging for the fast path's raw-FP16 D2H copy. Resized lazily.
    std::vector<__half> m_hostFp16Staging;

    // c35: pooled host outputs. One vector per output binding, sized at loadNetwork. Reused
    // across runInferenceFromBGR calls so we don't allocate per frame.
    std::vector<std::vector<float>> m_hostOutputs;

    // c33: pre-allocated FP16 CHW input buffer used by the fused preproc fast path
    // (runInferenceFromBGR). Owned by the engine; its address is mirrored into m_buffers at the
    // input binding index. The legacy runInference path overwrites m_buffers[input] with a
    // transient OpenCV GpuMat pointer, so the fast path restores from m_ownedInputBuffer on entry.
    void *m_ownedInputBuffer = nullptr;
    int m_inputBufferIndex = -1;
};
