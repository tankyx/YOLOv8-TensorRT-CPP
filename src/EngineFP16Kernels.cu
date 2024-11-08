// EngineFP16Kernels.cu

#include "EngineFP16Kernels.h"
#include <device_launch_parameters.h>

__global__ void convertFP32ToFP16Kernel(const float *input, __half *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

void launchConvertFP32ToFP16Kernel(const float *input, __half *output, int size, cudaStream_t stream) {
    const int blockSize = 256;
    const int numBlocks = (size + blockSize - 1) / blockSize;

    convertFP32ToFP16Kernel<<<numBlocks, blockSize, 0, stream>>>(input, output, size);
}