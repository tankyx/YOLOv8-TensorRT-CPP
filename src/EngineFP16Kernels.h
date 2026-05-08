// EngineFP16Kernels.h

#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Input-side cast: FP32 -> FP16 (vectorized over half2 since c22).
void launchConvertFP32ToFP16Kernel(const float *input, __half *output, int size, cudaStream_t stream = nullptr);

// Output-side cast: FP16 -> FP32 (vectorized over half2). Used by EngineFP16::runInference to
// avoid the host-side scalar conversion loop.
void launchConvertFP16ToFP32Kernel(const __half *input, float *output, int size, cudaStream_t stream = nullptr);

// Fused uint8 BGR -> FP16 CHW preprocessing. Reads a packed uint8 BGR source image (with arbitrary
// row pitch in bytes) and writes a contiguous FP16 CHW tensor in RGB plane order, applying:
//   - bilinear letterbox resize (top-left aligned, zero-padded right/bottom — matches the previous
//     resizeKeepAspectRatioPadRightBottom helper)
//   - BGR -> RGB channel swap (so output plane 0 = R, 1 = G, 2 = B)
//   - per-channel normalize: out = (v * scale255 - sub[c]) / div[c], cast to half
//
// Replaces the previous OpenCV-based pipeline (cvtColor + resize + split + convertTo + sub + div +
// FP32->FP16 cast), which materialized a ~4.7 MB FP32 intermediate and required a stream sync
// mid-preprocess.
struct FusedPreprocParams {
    const uint8_t *src;
    int srcW;
    int srcH;
    int srcPitch; // bytes per source row

    __half *dst;
    int dstW;
    int dstH;

    // Bilinear sampling: src_xy = dst_xy * invScale (in pixels). Padding region is x>=unpadW or
    // y>=unpadH inside the dst canvas — those threads write zeros.
    float invScale;
    int unpadW;
    int unpadH;

    // Per-output-channel (R,G,B) normalization: out = (v * scale255 - sub[c]) / div[c].
    float scale255; // 1/255 if normalize else 1
    float sub[3];
    float div[3];
};

void launchFusedPreprocBGRtoFP16Kernel(const FusedPreprocParams &params, cudaStream_t stream);

#ifdef __cplusplus
}
#endif