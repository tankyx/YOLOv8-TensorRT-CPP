// EngineFP16Kernels.h

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Declare the CUDA function that will be called from C++
void launchConvertFP32ToFP16Kernel(const float *input, __half *output, int size, cudaStream_t stream = nullptr);

#ifdef __cplusplus
}
#endif