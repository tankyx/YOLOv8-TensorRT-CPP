// EngineFP16Kernels.h

#pragma once

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

#ifdef __cplusplus
}
#endif