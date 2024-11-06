#include "SoftwareFuser.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fuseFramesKernel(const uchar3 *captureCardFrame, const uchar3 *desktopFrame, uchar3 *outputFrame, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 ccPixel = captureCardFrame[idx];
        if (ccPixel.x != 0 || ccPixel.y != 0 || ccPixel.z != 0) {
            outputFrame[idx] = ccPixel;
        } else {
            outputFrame[idx] = desktopFrame[idx];
        }
    }
}

extern "C" void launchFuseFramesKernel(const uchar3 *captureCardFrame, const uchar3 *desktopFrame, uchar3 *outputFrame, int width,
                                       int height, cudaStream_t stream) {
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    fuseFramesKernel<<<gridSize, blockSize, 0, stream>>>(captureCardFrame, desktopFrame, outputFrame, width, height);
}