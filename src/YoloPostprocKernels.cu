// YoloPostprocKernels.cu

#include "YoloPostprocKernels.h"
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

// ---------------------------------------------------------------------------
// YOLOv8 — standard anchor decode (no DFL)
// ---------------------------------------------------------------------------

__global__ void yoloFilterAndDecodeKernel(YoloFilterParams p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.numAnchors) { return; }

    // Find best class score.
    int bestLabel = 0;
    float bestScore = __half2float(p.output[4 * p.numAnchors + i]);
    for (int c = 1; c < p.numClasses; ++c) {
        const float s = __half2float(p.output[(4 + c) * p.numAnchors + i]);
        if (s > bestScore) { bestScore = s; bestLabel = c; }
    }

    if (bestScore <= p.probThreshold) { return; }

    // Decode xywh → xyxy in source-image space.
    const float x = __half2float(p.output[0 * p.numAnchors + i]);
    const float y = __half2float(p.output[1 * p.numAnchors + i]);
    const float w = __half2float(p.output[2 * p.numAnchors + i]);
    const float h = __half2float(p.output[3 * p.numAnchors + i]);

    const float x0 = clampf((x - 0.5f * w) * p.ratio, 0.f, p.imgW);
    const float y0 = clampf((y - 0.5f * h) * p.ratio, 0.f, p.imgH);
    const float x1 = clampf((x + 0.5f * w) * p.ratio, 0.f, p.imgW);
    const float y1 = clampf((y + 0.5f * h) * p.ratio, 0.f, p.imgH);

    const uint32_t slot = atomicAdd(p.outCount, 1u);
    if (static_cast<int>(slot) >= p.maxSurvivors) { return; }
    float *dst = p.outSurvivors + slot * kYoloSurvivorStride;
    dst[0] = x0; dst[1] = y0;
    dst[2] = x1; dst[3] = y1;
    dst[4] = bestScore;
    dst[5] = static_cast<float>(bestLabel);
}

void launchYoloFilterAndDecodeKernel(const YoloFilterParams &params, cudaStream_t stream) {
    const int blockSize = 256;
    const int numBlocks = (params.numAnchors + blockSize - 1) / blockSize;
    yoloFilterAndDecodeKernel<<<numBlocks, blockSize, 0, stream>>>(params);
}

// ---------------------------------------------------------------------------
// YOLOv11 — DFL-aware anchor decode
// ---------------------------------------------------------------------------
//
// YOLOv11 bbox regression uses Distribution Focal Loss: each of the 4
// coordinates is distributed over `regMax` bins (default 16). The softmax
// over bins + weighted sum decodes a single float per coordinate.
//
// Output layout: [1, 4*regMax + numClasses, numAnchors]
//   channels [0 .. 4*regMax-1] = bbox distribution
//     coord 0: channels [0*regMax .. 1*regMax-1]
//     coord 1: channels [1*regMax .. 2*regMax-1]
//     coord 2: channels [2*regMax .. 3*regMax-1]
//     coord 3: channels [3*regMax .. 4*regMax-1]
//   channels [4*regMax .. 4*regMax+numClasses-1] = class scores
//
// Reference: ultralytics/utils/ops.py process_box_rm

__global__ void yoloV11FilterAndDecodeKernel(YoloV11FilterParams p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.numAnchors) { return; }

    const int regMax = p.regMax;

    // -- DFL decode: 4 coordinates, each distributed over regMax bins --------
    // Process coordinates sequentially to keep register pressure low.
    // A single dist[] of regMax floats is reused across the 4 coordinates.
    float coords[4];

    for (int coord = 0; coord < 4; ++coord) {
        // Load the regMax FP16 logits for coordinate `coord` at anchor `i`.
        // Stored as half in global memory; convert to float in registers.
        float dist[16]; // regMax is typically 16 (fits in registers w/o spill)
        float maxVal = -1e30f;
        for (int b = 0; b < regMax; ++b) {
            const float v = __half2float(p.output[(coord * regMax + b) * p.numAnchors + i]);
            dist[b] = v;
            if (v > maxVal) { maxVal = v; }
        }

        // Softmax (numerically stable with max subtraction).
        float sum = 0.0f;
#pragma unroll
        for (int b = 0; b < regMax; ++b) {
            const float e = expf(dist[b] - maxVal);
            dist[b] = e;
            sum += e;
        }
        const float invSum = (sum > 0.0f) ? (1.0f / sum) : 1.0f;

        // Weighted sum: coord = Σ(bin * softmax(bin)).
        float coordVal = 0.0f;
#pragma unroll
        for (int b = 0; b < regMax; ++b) {
            coordVal += static_cast<float>(b) * dist[b] * invSum;
        }
        coords[coord] = coordVal;
    }

    const float x = coords[0];
    const float y = coords[1];
    const float w = coords[2];
    const float h = coords[3];

    // -- Find best class ----------------------------------------------------
    const int classOffset = 4 * regMax;
    int bestLabel = 0;
    float bestScore = __half2float(p.output[(classOffset + 0) * p.numAnchors + i]);
    for (int c = 1; c < p.numClasses; ++c) {
        const float s = __half2float(p.output[(classOffset + c) * p.numAnchors + i]);
        if (s > bestScore) { bestScore = s; bestLabel = c; }
    }

    if (bestScore <= p.probThreshold) { return; }

    // -- Decode xywh → xyxy -------------------------------------------------
    const float x0 = clampf((x - 0.5f * w) * p.ratio, 0.f, p.imgW);
    const float y0 = clampf((y - 0.5f * h) * p.ratio, 0.f, p.imgH);
    const float x1 = clampf((x + 0.5f * w) * p.ratio, 0.f, p.imgW);
    const float y1 = clampf((y + 0.5f * h) * p.ratio, 0.f, p.imgH);

    const uint32_t slot = atomicAdd(p.outCount, 1u);
    if (static_cast<int>(slot) >= p.maxSurvivors) { return; }
    float *dst = p.outSurvivors + slot * kYoloSurvivorStride;
    dst[0] = x0; dst[1] = y0;
    dst[2] = x1; dst[3] = y1;
    dst[4] = bestScore;
    dst[5] = static_cast<float>(bestLabel);
}

void launchYoloV11FilterAndDecodeKernel(const YoloV11FilterParams &params, cudaStream_t stream) {
    const int blockSize = 256;
    const int numBlocks = (params.numAnchors + blockSize - 1) / blockSize;
    yoloV11FilterAndDecodeKernel<<<numBlocks, blockSize, 0, stream>>>(params);
}

// ---------------------------------------------------------------------------
// YOLOv26 — end-to-end filter (NMS-free)
// ---------------------------------------------------------------------------
//
// YOLOv26 output: (1, 6, maxDetections) where each row is
//   [x1, y1, x2, y2, confidence, class_id]
//
// The output is already decoded — no box math, no class argmax.
// Only confidence thresholding is needed.

__global__ void yoloV26FilterKernel(YoloV26FilterParams p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.maxDetections) { return; }

    // Row-major: detection i is at offset i*6 in the 2D output.
    // Output layout is [1, 6, maxDetections], so contiguous stride is maxDetections.
    const float x1     = __half2float(p.output[0 * p.maxDetections + i]);
    const float y1     = __half2float(p.output[1 * p.maxDetections + i]);
    const float x2     = __half2float(p.output[2 * p.maxDetections + i]);
    const float y2     = __half2float(p.output[3 * p.maxDetections + i]);
    const float conf   = __half2float(p.output[4 * p.maxDetections + i]);
    const float cls_id = __half2float(p.output[5 * p.maxDetections + i]);

    if (conf <= p.probThreshold) { return; }

    const uint32_t slot = atomicAdd(p.outCount, 1u);
    if (static_cast<int>(slot) >= p.maxSurvivors) { return; }
    float *dst = p.outSurvivors + slot * kYoloSurvivorStride;
    dst[0] = x1;   dst[1] = y1;
    dst[2] = x2;   dst[3] = y2;
    dst[4] = conf;
    dst[5] = cls_id;
}

void launchYoloV26FilterKernel(const YoloV26FilterParams &params, cudaStream_t stream) {
    const int blockSize = 256;
    const int numBlocks = (params.maxDetections + blockSize - 1) / blockSize;
    yoloV26FilterKernel<<<numBlocks, blockSize, 0, stream>>>(params);
}
