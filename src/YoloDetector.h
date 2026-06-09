#pragma once
#include "EngineFactory.h"
#include "YoloPostprocKernels.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <algorithm>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

struct Object {
    int label{};
    float probability{};
    cv::Rect_<float> rect;
    cv::Mat boxMask;
    std::vector<float> kps{};
};

// ---------------------------------------------------------------------------
// YOLO model version
// ---------------------------------------------------------------------------

enum class YoloVersion {
    AUTO = 0,  // auto-detect from model file
    V8,        // output (1, 4+C, 8400) – no DFL, NMS required
    V11,       // output (1, 4*regMax+C, 8400) – DFL decode, NMS required
    V26        // output (1, 300, 6) – end-to-end, no NMS
};

// ---------------------------------------------------------------------------
// Detector configuration (replaces YoloV8Config)
// ---------------------------------------------------------------------------

struct YoloConfig {
    Precision precision = Precision::FP16;
    std::string calibrationDataDirectory;
    float probabilityThreshold = 0.25f;
    float nmsThreshold = 0.65f;
    int topK = 100;
    // Segmentation
    int segChannels = 32;
    int segH = 160;
    int segW = 160;
    float segmentationThreshold = 0.5f;
    // Pose
    int numKPS = 17;
    float kpsThreshold = 0.5f;
    // Class names
    std::vector<std::string> classNames = {"c", "ch", "t", "th"};
    // V11 DFL
    int regMax = 16;            // distribution bins per bbox coordinate (default 16)
    // V26
    int maxDetectionsV26 = 300; // number of output detections from V26 e2e head
};

// Backward-compatible alias
using YoloV8Config = YoloConfig;

// ---------------------------------------------------------------------------
// Base detector — engine management, graph capture, GPU buffers
// Subclasses override postprocessing (GPU kernel + CPU NMS/threshold)
// ---------------------------------------------------------------------------

class YoloDetector {
public:
    YoloDetector(const std::string &onnxModelPath, const YoloConfig &config,
                 YoloVersion version);
    virtual ~YoloDetector();

    // -- Inference ----------------------------------------------------------
    std::vector<Object> detectObjects(const cv::Mat &inputImageBGR);
    std::vector<Object> detectObjects(const cv::cuda::GpuMat &inputImageBGR);

    // -- Drawing helpers ----------------------------------------------------
    void drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects,
                          unsigned int scale = 2, int squareHalfSize = 0);

    // -- Public data --------------------------------------------------------
    const std::vector<std::string> CLASS_NAMES;

    // -- Engine access (for factory auto-detection) -------------------------
    [[nodiscard]] const EngineBase &getEngine() const { return *m_trtEngine; }

    // -- Auto-detection -----------------------------------------------------
    // Infers version from the ONNX output shape after loading.
    // Returns V8 if indistinguishable from V8/V11 (caller can override).
    static YoloVersion detectVersionFromOutput(const nvinfer1::Dims &outputDims,
                                               int numClasses);

    // Infers version from the ONNX filename (e.g. "yolo11n.onnx" -> V11).
    // Used as a fallback when the output shape is ambiguous.
    static YoloVersion detectVersionFromFilename(const std::string &modelPath);

protected:
    // -- Version-specific GPU postproc kernel --------------------------------
    // Called inside runFp16InferenceOnStream() on the engine's stream after
    // enqueueV3. Subclasses schedule their own kernel + set up params.
    virtual void launchPostprocKernel(EngineFP16 *fp16Engine,
                                      int numAnchors, int numClasses,
                                      cudaStream_t stream) = 0;

    // -- CPU postprocessing from survivor buffer (GPU fast-path) -------------
    // Default implementation applies NMS (used by V8 and V11).
    // V26 overrides with its NMS-free version.
    virtual std::vector<Object> postprocessFromSurvivors(uint32_t count);

    // -- CPU fallback postprocessing (FP32 path) ----------------------------
    virtual std::vector<Object> postprocessDetect(
        const std::vector<float> &featureVector) = 0;

    // -- GPU buffer management ----------------------------------------------
    void allocatePostprocBuffers();
    void freePostprocBuffers();

    // -- CUDA graph infrastructure ------------------------------------------
    void releaseGraph();
    bool runFp16InferenceOnStream(const cv::cuda::GpuMat &captureView,
                                  int numAnchors, int numClasses,
                                  cudaStream_t stream);

    // -- FP32 fallback preprocess -------------------------------------------
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(
        const cv::cuda::GpuMat &gpuImg);

    // -- Engine -------------------------------------------------------------
    std::unique_ptr<EngineBase> m_trtEngine = nullptr;
    EngineFP16 *m_fp16Engine = nullptr;  // cached dynamic_cast result
    YoloVersion m_version;

    // -- Cached output dims (set once in constructor, never change) ----------
    int m_numAnchors = 0;
    int m_numClasses = 0;
    int m_srcChannels = 0; // 3 (CV_8UC3 BGR) or 4 (CV_8UC4 BGRA)

    // -- Graph capture state ------------------------------------------------
    uint8_t *m_captureBuffer = nullptr;
    size_t m_captureBufferPitch = 0;
    int m_captureWidth = 0;
    int m_captureHeight = 0;
    int m_captureChannels = 0; // 3 (CV_8UC3 BGR) or 4 (CV_8UC4 BGRA)

    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;
    bool m_graphCaptured = false;

    // -- Survivor buffer (GPU fast path) ------------------------------------
    static constexpr int kMaxSurvivors = 1024;
    uint32_t *m_devSurvivorCount = nullptr;
    float *m_devSurvivors = nullptr;
    uint32_t m_hostSurvivorCount = 0;
    std::vector<float> m_hostSurvivors;

    // -- Normalization constants (used by preproc + GPU kernel) --------------
    const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

    // -- Postprocess geometry -----------------------------------------------
    float m_ratio = 1;
    float m_imgWidth = 0;
    float m_imgHeight = 0;

    // -- Thresholds ---------------------------------------------------------
    const float PROBABILITY_THRESHOLD;
    const float NMS_THRESHOLD;
    const int TOP_K;

    // -- Segmentation -------------------------------------------------------
    const int SEG_CHANNELS;
    const int SEG_H;
    const int SEG_W;
    const float SEGMENTATION_THRESHOLD;

    // -- Pose ---------------------------------------------------------------
    const int NUM_KPS;
    const float KPS_THRESHOLD;

    // -- V11 DFL ------------------------------------------------------------
    const int REG_MAX; // distribution bins per bbox coordinate (e.g. 16)

    // -- V26 end-to-end ------------------------------------------------------
    const int MAX_DETECTIONS_V26; // number of output detections from V26 e2e head

    // -- Colours ------------------------------------------------------------
    const std::vector<std::vector<float>> CS2_COLORS = {
        {1.0, 0.0, 0.0},
        {1.0, 0.5, 0.5},
        {0.0, 0.0, 1.0},
        {0.5, 0.5, 1.0}
    };

    const std::vector<std::vector<unsigned int>> KPS_COLORS = {
        {0, 255, 0},   {0, 255, 0},   {0, 255, 0},   {0, 255, 0},
        {0, 255, 0},   {255, 128, 0}, {255, 128, 0}, {255, 128, 0},
        {255, 128, 0}, {255, 128, 0}, {51, 153, 255},{51, 153, 255},
        {51, 153, 255},{51, 153, 255},{51, 153, 255},{128, 0, 128},
        {255, 0, 255}
    };

    const std::vector<std::vector<unsigned int>> SKELETON = {
        {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12},
        {7, 13},  {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},
        {2, 3},   {1, 2},   {1, 3},   {2, 4},   {3, 5},   {4, 6},  {5, 7}
    };

    const std::vector<std::vector<unsigned int>> LIMB_COLORS = {
        {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255},
        {255, 51, 255}, {255, 51, 255}, {255, 51, 255}, {255, 128, 0},
        {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0}
    };
};
