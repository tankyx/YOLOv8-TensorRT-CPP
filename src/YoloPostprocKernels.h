// YoloPostprocKernels.h
//
// GPU-side filter + decode for YOLO detection heads. One CUDA kernel per model
// version replaces the CPU anchor loop over N anchors with a single kernel that
// reads the engine's FP16 output and writes survivors to a compact buffer.
//
// All kernel outputs use the same survivor format (kYoloSurvivorStride = 6):
//   (x0, y0, x1, y1, score, label)

#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compact survivor record: 6 floats per surviving anchor.
// (x0, y0, x1, y1, score, label). label is float for storage simplicity;
// cast to int on host.
constexpr int kYoloSurvivorStride = 6;

// ---------------------------------------------------------------------------
// YOLOv8 filter+decode params
// ---------------------------------------------------------------------------

struct YoloFilterParams {
    const __half *output; // engine output, FP16, layout [1, 4 + numClasses, numAnchors]
    int numAnchors;
    int numClasses;

    float ratio;          // bbox_dst * ratio = bbox_src
    float imgW;
    float imgH;

    float probThreshold;

    uint32_t *outCount;      // single uint32_t, zeroed before launch
    float *outSurvivors;     // [maxSurvivors * kYoloSurvivorStride]
    int maxSurvivors;
};

void launchYoloFilterAndDecodeKernel(const YoloFilterParams &params, cudaStream_t stream);

// ---------------------------------------------------------------------------
// YOLOv11 filter+decode params (DFL-aware)
// ---------------------------------------------------------------------------

// YOLOv11 uses Distribution Focal Loss on its bbox regression. Each of the
// 4 bbox coordinates is encoded as a distribution over `regMax` bins. The
// engine output layout is [1, 4*regMax + numClasses, numAnchors].
//
// For each anchor the kernel:
//   1. Loads 4×regMax distribution values
//   2. Applies softmax over regMax bins per coordinate
//   3. Computes weighted sum: coord = Σ(bin_i * softmax(dist_i))
//   4. Decodes xywh → xyxy in source-image space
//   5. Finds best class score
//   6. Atomically appends to the survivor buffer

struct YoloV11FilterParams {
    const __half *output; // engine output, FP16, layout [1, 4*regMax + numClasses, numAnchors]
    int numAnchors;
    int numClasses;
    int regMax;            // distribution bins per coordinate (default 16)

    float ratio;
    float imgW;
    float imgH;

    float probThreshold;

    uint32_t *outCount;
    float *outSurvivors;
    int maxSurvivors;
};

void launchYoloV11FilterAndDecodeKernel(const YoloV11FilterParams &params,
                                        cudaStream_t stream);

// ---------------------------------------------------------------------------
// YOLOv26 filter params (end-to-end, NMS-free)
// ---------------------------------------------------------------------------

// YOLOv26 outputs final detections directly: (1, 300, 6) where each row is
// (x1, y1, x2, y2, confidence, class_id). No box decode, no class argmax,
// no NMS. The kernel only applies a confidence threshold and copies survivors.

struct YoloV26FilterParams {
    const __half *output; // engine output, FP16, layout [1, 6, maxDetections]
    int maxDetections;    // typically 300

    float probThreshold;

    uint32_t *outCount;
    float *outSurvivors;
    int maxSurvivors;
};

void launchYoloV26FilterKernel(const YoloV26FilterParams &params,
                               cudaStream_t stream);

#ifdef __cplusplus
}
#endif
