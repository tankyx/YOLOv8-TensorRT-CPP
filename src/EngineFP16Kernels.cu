// EngineFP16Kernels.cu

#include "EngineFP16Kernels.h"
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

// Vectorized FP32 -> FP16 conversion. Each thread processes a half2 (two adjacent floats),
// halving global memory transactions vs. the scalar version. The trailing odd element (if
// `size` is odd) is handled by a second narrow launch on the last lane.
__global__ void convertFP32ToFP16Kernel_half2(const float2 *input, __half2 *output, int pairCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pairCount) {
        output[idx] = __float22half2_rn(input[idx]);
    }
}

__global__ void convertFP32ToFP16Kernel_tail(const float *input, __half *output, int offset, int size) {
    int idx = offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

void launchConvertFP32ToFP16Kernel(const float *input, __half *output, int size, cudaStream_t stream) {
    const int blockSize = 256;
    const int pairCount = size / 2;

    if (pairCount > 0) {
        const int numBlocks = (pairCount + blockSize - 1) / blockSize;
        convertFP32ToFP16Kernel_half2<<<numBlocks, blockSize, 0, stream>>>(reinterpret_cast<const float2 *>(input),
                                                                          reinterpret_cast<__half2 *>(output), pairCount);
    }

    const int tailStart = pairCount * 2;
    if (tailStart < size) {
        // Single trailing element; one block, one thread is fine.
        convertFP32ToFP16Kernel_tail<<<1, 1, 0, stream>>>(input, output, tailStart, size);
    }
}
