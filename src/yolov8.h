#pragma once
#include "EngineFactory.h"
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <algorithm>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

// Utility method for checking if a file exists on disk
inline bool doesFileExist(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

struct Object {
    // The object class.
    int label{};
    // The detection's confidence probability.
    float probability{};
    // The object bounding box rectangle.
    cv::Rect_<float> rect;
    // Semantic segmentation mask
    cv::Mat boxMask;
    // Pose estimation key points
    std::vector<float> kps{};
};

// Config the behavior of the YoloV8 detector.
// Can pass these arguments as command line parameters.
struct YoloV8Config {
    // The precision to be used for inference
    Precision precision = Precision::FP16;
    // Calibration data directory. Must be specified when using INT8 precision.
    std::string calibrationDataDirectory;
    // Probability threshold used to filter detected objects
    float probabilityThreshold = 0.25f;
    // Non-maximum suppression threshold
    float nmsThreshold = 0.65f;
    // Max number of detected objects to return
    int topK = 100;
    // Segmentation config options
    int segChannels = 32;
    int segH = 160;
    int segW = 160;
    float segmentationThreshold = 0.5f;
    // Pose estimation options
    int numKPS = 17;
    float kpsThreshold = 0.5f;
    // Class thresholds (default are CS2 classes)
    std::vector<std::string> classNames = {"c", "ch", "t", "th"};
};

class YoloV8 {
public:
    YoloV8(const std::string &onnxModelPath, const YoloV8Config &config);
    ~YoloV8();

    std::vector<Object> detectObjects(const cv::Mat &inputImageBGR);
    std::vector<Object> detectObjects(const cv::cuda::GpuMat &inputImageBGR);
    void drawObjectLabels(cv::Mat &image, const std::vector<Object> &objects, unsigned int scale = 2, int squareHalfSize = 0);

    const std::vector<std::string> CLASS_NAMES;

private:
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat &gpuImg);
    std::vector<Object> postprocessDetect(const std::vector<float> &featureVector);

    // c37: GPU postprocess buffers used by the FP16 fast path. The filter+decode kernel writes
    // up to kMaxSurvivors compacted survivor records into m_devSurvivors and atomically counts
    // them in m_devSurvivorCount; both are D2H'd before the CPU NMS pass.
    void allocatePostprocBuffers();
    void freePostprocBuffers();
    std::vector<Object> postprocessFromSurvivors(uint32_t count);

    // c38: CUDA graph capture for the per-frame GPU sequence (preproc + enqueueV3 + memset +
    // filter+decode + D2H). Captured once on the first call after a warmup, replayed every
    // frame. Requires a stable device pointer for the preproc kernel's input — m_captureBuffer
    // is that stable buffer; per frame, the bgr capture is cudaMemcpy2D'd into it before the
    // graph is launched.
    bool runFp16InferenceOnStream(class EngineFP16 *fp16Engine, const cv::cuda::GpuMat &captureView,
                                  int numAnchors, int numClasses, cudaStream_t stream);
    void releaseGraph();

    uint8_t *m_captureBuffer = nullptr;
    size_t m_captureBufferPitch = 0;
    int m_captureWidth = 0;
    int m_captureHeight = 0;
    int m_captureChannels = 0; // 3 (CV_8UC3 BGR) or 4 (CV_8UC4 BGRA)

    cudaGraph_t m_graph = nullptr;
    cudaGraphExec_t m_graphExec = nullptr;
    bool m_graphCaptured = false;

    static constexpr int kMaxSurvivors = 1024;
    uint32_t *m_devSurvivorCount = nullptr;
    float *m_devSurvivors = nullptr;
    uint32_t m_hostSurvivorCount = 0;
    std::vector<float> m_hostSurvivors;

    std::unique_ptr<EngineBase> m_trtEngine = nullptr;

    const std::array<float, 3> SUB_VALS{0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS{1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

    float m_ratio = 1;
    float m_imgWidth = 0;
    float m_imgHeight = 0;

    const float PROBABILITY_THRESHOLD;
    const float NMS_THRESHOLD;
    const int TOP_K;

    const int SEG_CHANNELS;
    const int SEG_H;
    const int SEG_W;
    const float SEGMENTATION_THRESHOLD;

    const int NUM_KPS;
    const float KPS_THRESHOLD;

    const std::vector<std::vector<float>> CS2_COLORS = {
        {1.0, 0.0, 0.0}, // Red
        {1.0, 0.5, 0.5}, // Light Red
        {0.0, 0.0, 1.0}, // Blue
        {0.5, 0.5, 1.0}  // Light Blue
    };

    // Color list for drawing objects
    const std::vector<std::vector<float>> COLOR_LIST = {{1, 1, 1},
                                                        {0.098, 0.325, 0.850},
                                                        {0.125, 0.694, 0.929},
                                                        {0.556, 0.184, 0.494},
                                                        {0.188, 0.674, 0.466},
                                                        {0.933, 0.745, 0.301},
                                                        {0.184, 0.078, 0.635},
                                                        {0.300, 0.300, 0.300},
                                                        {0.600, 0.600, 0.600},
                                                        {0.000, 0.000, 1.000},
                                                        {0.000, 0.500, 1.000},
                                                        {0.000, 0.749, 0.749},
                                                        {0.000, 1.000, 0.000},
                                                        {1.000, 0.000, 0.000},
                                                        {1.000, 0.000, 0.667},
                                                        {0.000, 0.333, 0.333},
                                                        {0.000, 0.667, 0.333},
                                                        {0.000, 1.000, 0.333},
                                                        {0.000, 0.333, 0.667},
                                                        {0.000, 0.667, 0.667},
                                                        {0.000, 1.000, 0.667},
                                                        {0.000, 0.333, 1.000},
                                                        {0.000, 0.667, 1.000},
                                                        {0.000, 1.000, 1.000},
                                                        {0.500, 0.333, 0.000},
                                                        {0.500, 0.667, 0.000},
                                                        {0.500, 1.000, 0.000},
                                                        {0.500, 0.000, 0.333},
                                                        {0.500, 0.333, 0.333},
                                                        {0.500, 0.667, 0.333},
                                                        {0.500, 1.000, 0.333},
                                                        {0.500, 0.000, 0.667},
                                                        {0.500, 0.333, 0.667},
                                                        {0.500, 0.667, 0.667},
                                                        {0.500, 1.000, 0.667},
                                                        {0.500, 0.000, 1.000},
                                                        {0.500, 0.333, 1.000},
                                                        {0.500, 0.667, 1.000},
                                                        {0.500, 1.000, 1.000},
                                                        {1.000, 0.333, 0.000},
                                                        {1.000, 0.667, 0.000},
                                                        {1.000, 1.000, 0.000},
                                                        {1.000, 0.000, 0.333},
                                                        {1.000, 0.333, 0.333},
                                                        {1.000, 0.667, 0.333},
                                                        {1.000, 1.000, 0.333},
                                                        {1.000, 0.000, 0.667},
                                                        {1.000, 0.333, 0.667},
                                                        {1.000, 0.667, 0.667},
                                                        {1.000, 1.000, 0.667},
                                                        {1.000, 0.000, 1.000},
                                                        {1.000, 0.333, 1.000},
                                                        {1.000, 0.667, 1.000},
                                                        {0.000, 0.000, 0.333},
                                                        {0.000, 0.000, 0.500},
                                                        {0.000, 0.000, 0.667},
                                                        {0.000, 0.000, 0.833},
                                                        {0.000, 0.000, 1.000},
                                                        {0.000, 0.167, 0.000},
                                                        {0.000, 0.333, 0.000},
                                                        {0.000, 0.500, 0.000},
                                                        {0.000, 0.667, 0.000},
                                                        {0.000, 0.833, 0.000},
                                                        {0.000, 1.000, 0.000},
                                                        {0.167, 0.000, 0.000},
                                                        {0.333, 0.000, 0.000},
                                                        {0.500, 0.000, 0.000},
                                                        {0.667, 0.000, 0.000},
                                                        {0.833, 0.000, 0.000},
                                                        {1.000, 0.000, 0.000},
                                                        {0.000, 0.000, 0.000},
                                                        {0.143, 0.143, 0.143},
                                                        {0.286, 0.286, 0.286},
                                                        {0.429, 0.429, 0.429},
                                                        {0.571, 0.571, 0.571},
                                                        {0.714, 0.714, 0.714},
                                                        {0.857, 0.857, 0.857},
                                                        {0.741, 0.447, 0.000},
                                                        {0.741, 0.717, 0.314},
                                                        {0.000, 0.500, 0.500}};

    const std::vector<std::vector<unsigned int>> KPS_COLORS = {
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},   {255, 128, 0},
        {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0}, {51, 153, 255},
        {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}};

    const std::vector<std::vector<unsigned int>> SKELETON = {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
                                                             {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},  {2, 3},  {1, 2},
                                                             {1, 3},   {2, 4},   {3, 5},   {4, 6},   {5, 7}};

    const std::vector<std::vector<unsigned int>> LIMB_COLORS = {
        {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {255, 51, 255}, {255, 51, 255}, {255, 51, 255},
        {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {0, 255, 0},    {0, 255, 0},
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0}};
};