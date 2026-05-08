// EngineFP16Kernels.cu

#include "EngineFP16Kernels.h"
#include <cuda_fp16.h>
#include <device_launch_parameters.h>

// --- Fused uint8 BGR -> FP16 CHW preprocessing (c33) -------------------------------------------
//
// One pixel per thread. Bilinear letterbox + BGR->RGB swap + normalize + HWC->CHW + u8->fp16 in
// a single kernel. Replaces ~7 OpenCV-CUDA kernel launches plus a 4.7 MB FP32 intermediate (and
// the in-preprocess cudaStreamSynchronize that came with the FP32->FP16 cast helper).

__device__ __forceinline__ float bilerp(float p00, float p01, float p10, float p11, float ax, float ay) {
    const float top = p00 * (1.0f - ax) + p01 * ax;
    const float bot = p10 * (1.0f - ax) + p11 * ax;
    return top * (1.0f - ay) + bot * ay;
}

__global__ void fusedPreprocBGRtoFP16Kernel(FusedPreprocParams p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.dstW || y >= p.dstH) {
        return;
    }

    const int planeSize = p.dstW * p.dstH;
    const int idx = y * p.dstW + x;

    // Pad bottom/right: outside the resized region, write zeros to all 3 channels. Matches the
    // previous resizeKeepAspectRatioPadRightBottom (default bgcolor = (0,0,0)), and after the
    // (v - 0) / 1 normalize that's still zero, so we can short-circuit before sampling src.
    if (x >= p.unpadW || y >= p.unpadH) {
        const __half zero = __float2half(0.0f);
        p.dst[0 * planeSize + idx] = zero;
        p.dst[1 * planeSize + idx] = zero;
        p.dst[2 * planeSize + idx] = zero;
        return;
    }

    // Inverse-map dst -> src in float pixel coords, then bilinear-sample.
    const float fx = static_cast<float>(x) * p.invScale;
    const float fy = static_cast<float>(y) * p.invScale;
    int x0 = static_cast<int>(floorf(fx));
    int y0 = static_cast<int>(floorf(fy));
    const float ax = fx - static_cast<float>(x0);
    const float ay = fy - static_cast<float>(y0);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Clamp to source bounds (replicate edge — same as cv::cuda::resize default).
    x0 = max(0, min(x0, p.srcW - 1));
    x1 = max(0, min(x1, p.srcW - 1));
    y0 = max(0, min(y0, p.srcH - 1));
    y1 = max(0, min(y1, p.srcH - 1));

    const uint8_t *row0 = p.src + y0 * p.srcPitch;
    const uint8_t *row1 = p.src + y1 * p.srcPitch;

    // Source is packed BGR uint8: pixel = (B, G, R). Output planes are (R, G, B).
    const float b00 = static_cast<float>(row0[x0 * 3 + 0]);
    const float g00 = static_cast<float>(row0[x0 * 3 + 1]);
    const float r00 = static_cast<float>(row0[x0 * 3 + 2]);
    const float b01 = static_cast<float>(row0[x1 * 3 + 0]);
    const float g01 = static_cast<float>(row0[x1 * 3 + 1]);
    const float r01 = static_cast<float>(row0[x1 * 3 + 2]);
    const float b10 = static_cast<float>(row1[x0 * 3 + 0]);
    const float g10 = static_cast<float>(row1[x0 * 3 + 1]);
    const float r10 = static_cast<float>(row1[x0 * 3 + 2]);
    const float b11 = static_cast<float>(row1[x1 * 3 + 0]);
    const float g11 = static_cast<float>(row1[x1 * 3 + 1]);
    const float r11 = static_cast<float>(row1[x1 * 3 + 2]);

    float r = bilerp(r00, r01, r10, r11, ax, ay);
    float g = bilerp(g00, g01, g10, g11, ax, ay);
    float b = bilerp(b00, b01, b10, b11, ax, ay);

    r = (r * p.scale255 - p.sub[0]) / p.div[0];
    g = (g * p.scale255 - p.sub[1]) / p.div[1];
    b = (b * p.scale255 - p.sub[2]) / p.div[2];

    p.dst[0 * planeSize + idx] = __float2half(r);
    p.dst[1 * planeSize + idx] = __float2half(g);
    p.dst[2 * planeSize + idx] = __float2half(b);
}

void launchFusedPreprocBGRtoFP16Kernel(const FusedPreprocParams &params, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid((params.dstW + block.x - 1) / block.x, (params.dstH + block.y - 1) / block.y);
    fusedPreprocBGRtoFP16Kernel<<<grid, block, 0, stream>>>(params);
}
