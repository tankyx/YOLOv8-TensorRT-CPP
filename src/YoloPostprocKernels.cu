// YoloPostprocKernels.cu

#include "YoloPostprocKernels.h"
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

// One thread per anchor. Find the best class score; if above threshold, decode the box and
// append to the survivor buffer with an atomic counter increment. Writes are guarded against
// maxSurvivors so a saturated buffer drops extras silently.
__global__ void yoloFilterAndDecodeKernel(YoloFilterParams p) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.numAnchors) {
        return;
    }

    // Find best class. Score channels are output[(4 + c) * numAnchors + i] for c in [0, numClasses).
    int bestLabel = 0;
    float bestScore = __half2float(p.output[4 * p.numAnchors + i]);
    for (int c = 1; c < p.numClasses; ++c) {
        const float s = __half2float(p.output[(4 + c) * p.numAnchors + i]);
        if (s > bestScore) {
            bestScore = s;
            bestLabel = c;
        }
    }

    if (bestScore <= p.probThreshold) {
        return;
    }

    // Decode xywh -> xyxy in source image space.
    const float x = __half2float(p.output[0 * p.numAnchors + i]);
    const float y = __half2float(p.output[1 * p.numAnchors + i]);
    const float w = __half2float(p.output[2 * p.numAnchors + i]);
    const float h = __half2float(p.output[3 * p.numAnchors + i]);

    const float x0 = clampf((x - 0.5f * w) * p.ratio, 0.f, p.imgW);
    const float y0 = clampf((y - 0.5f * h) * p.ratio, 0.f, p.imgH);
    const float x1 = clampf((x + 0.5f * w) * p.ratio, 0.f, p.imgW);
    const float y1 = clampf((y + 0.5f * h) * p.ratio, 0.f, p.imgH);

    const uint32_t slot = atomicAdd(p.outCount, 1u);
    if (static_cast<int>(slot) >= p.maxSurvivors) {
        return;
    }
    float *dst = p.outSurvivors + slot * kYoloSurvivorStride;
    dst[0] = x0;
    dst[1] = y0;
    dst[2] = x1;
    dst[3] = y1;
    dst[4] = bestScore;
    dst[5] = static_cast<float>(bestLabel);
}

void launchYoloFilterAndDecodeKernel(const YoloFilterParams &params, cudaStream_t stream) {
    const int blockSize = 256;
    const int numBlocks = (params.numAnchors + blockSize - 1) / blockSize;
    yoloFilterAndDecodeKernel<<<numBlocks, blockSize, 0, stream>>>(params);
}
