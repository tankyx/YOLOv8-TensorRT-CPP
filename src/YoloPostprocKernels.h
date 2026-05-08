// YoloPostprocKernels.h
//
// GPU-side filter + decode for the YOLOv8 detection head. Replaces the CPU loop over
// 8400 anchors in YoloV8::postprocessDetect with a single CUDA kernel that reads the
// engine's FP16 output (layout [1, 4 + numClasses, numAnchors] CHW) and writes only
// the anchors above the probability threshold to a compact survivor buffer.

#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compact survivor record written by the kernel. 6 floats per surviving anchor:
// (x0, y0, x1, y1, score, label). label is float for storage simplicity; cast to int on host.
constexpr int kYoloSurvivorStride = 6;

struct YoloFilterParams {
    const __half *output; // engine output, FP16, layout [1, 4 + numClasses, numAnchors]
    int numAnchors;
    int numClasses;

    // Postprocess scale: bbox_dst * ratio = bbox_src. Matches YoloV8::m_ratio.
    float ratio;
    // Clamp bounds in source (capture) image space — match the original CPU clamp.
    float imgW;
    float imgH;

    float probThreshold;

    // Survivor sink. outCount is a single uint32_t (zeroed before launch) atomically
    // incremented per survivor. outSurvivors is sized [maxSurvivors * kYoloSurvivorStride].
    // The kernel guards writes by `idx < maxSurvivors` so saturation is safe (any extras
    // are dropped on the floor and the host caps count at maxSurvivors).
    uint32_t *outCount;
    float *outSurvivors;
    int maxSurvivors;
};

void launchYoloFilterAndDecodeKernel(const YoloFilterParams &params, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
