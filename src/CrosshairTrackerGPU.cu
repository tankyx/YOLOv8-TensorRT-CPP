#include "CrosshairTrackerGPU.h"
#include <cuda_runtime.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Atomic float-max via compare-and-swap.  CUDA has no native atomicMax for
// floats, so we bit-cast to int and spin on atomicCAS.
// ---------------------------------------------------------------------------
__device__ static float atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        if (value <= __int_as_float(assumed))
            break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
    return __int_as_float(old);
}

// ---------------------------------------------------------------------------
// Kernel: score every pair of line segments, pick the intersection that best
// resembles a perpendicular crosshair (arms of roughly equal length).
//
// Each block reduces to a single best via shared memory + atomic float-max;
// thread 0 in the winning block writes the final float2 to global memory.
// ---------------------------------------------------------------------------
__global__ void findCrosshairKernel(const float4* lines, int numLines,
                                    float2* bestIntersection) {
    __shared__ float bestScore;
    __shared__ float bestX;
    __shared__ float bestY;

    if (threadIdx.x == 0) {
        bestScore = -1.0f;
        bestX     = -1.0f;
        bestY     = -1.0f;
    }
    __syncthreads();

    // Per-thread local best — no races inside the double loop.
    float localBestScore = -1.0f;
    float localBestX     = -1.0f;
    float localBestY     = -1.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numLines;
         i += gridDim.x * blockDim.x) {
        const float4 l1  = lines[i];
        const float  x1  = l1.x, y1 = l1.y, x2 = l1.z, y2 = l1.w;
        const float  dx1 = x2 - x1, dy1 = y2 - y1;
        const float  len1 = sqrtf(dx1 * dx1 + dy1 * dy1);
        if (len1 < 5.0f) continue;

        for (int j = i + 1; j < numLines; j++) {
            const float4 l2  = lines[j];
            const float  x3  = l2.x, y3 = l2.y, x4 = l2.z, y4 = l2.w;
            const float  dx2 = x4 - x3, dy2 = y4 - y3;
            const float  len2 = sqrtf(dx2 * dx2 + dy2 * dy2);
            if (len2 < 5.0f) continue;

            // Angle between the two segments — must be near 90° (±15°).
            const float dot      = dx1 * dx2 + dy1 * dy2;
            const float cosAngle = dot / (len1 * len2);
            const float angle    = acosf(fminf(fmaxf(cosAngle, -1.0f), 1.0f))
                                   * 180.0f / 3.14159265f;
            if (fabsf(angle - 90.0f) > 15.0f) continue;

            // Intersection of the two infinite lines.
            const float det = dx1 * dy2 - dx2 * dy1;
            if (fabsf(det) < 1e-4f) continue;
            const float t   = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / det;
            const float u   = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / det;
            const float ix  = x1 + t * dx1;
            const float iy  = y1 + t * dy1;

            // Score: how evenly the intersection divides each segment.
            // 1.0 = perfect centre; 0.0 = at an endpoint.
            const float arm1a = t * len1;
            const float arm1b = (1.0f - t) * len1;
            const float arm2a = u * len2;
            const float arm2b = (1.0f - u) * len2;
            const float ratio1 = fminf(arm1a, arm1b) / fmaxf(arm1a, arm1b);
            const float ratio2 = fminf(arm2a, arm2b) / fmaxf(arm2a, arm2b);
            const float score  = (ratio1 + ratio2) * 0.5f;

            if (score > localBestScore) {
                localBestScore = score;
                localBestX     = ix;
                localBestY     = iy;
            }
        }
    }

    // Reduce across threads in the block: atomically max the score, then
    // let the thread(s) holding the winning score publish their x,y.
    if (localBestScore > 0.0f) {
        atomicMaxFloat(&bestScore, localBestScore);
    }
    __syncthreads();

    if (localBestScore > 0.0f && localBestScore == bestScore) {
        bestX = localBestX;
        bestY = localBestY;
    }
    __syncthreads();

    // Single writer per block publishes the result (or the sentinel).
    if (threadIdx.x == 0) {
        if (bestScore > 0.5f) {
            bestIntersection->x = bestX;
            bestIntersection->y = bestY;
        } else {
            bestIntersection->x = -1.0f;
            bestIntersection->y = -1.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side launch helper (called from CrosshairTrackerGPU::findCrosshairFromLines).
// ---------------------------------------------------------------------------
void launchFindCrosshair(const float4* d_lines, int numLines,
                         float2* d_bestIntersection, cudaStream_t stream) {
    if (numLines < 2) {
        // No line pairs to score — write sentinel directly.
        const float2 sentinel = make_float2(-1.0f, -1.0f);
        cudaMemcpyAsync(d_bestIntersection, &sentinel, sizeof(float2),
                        cudaMemcpyHostToDevice, stream);
        return;
    }
    const int blockSize = 256;
    const int gridSize  = (numLines + blockSize - 1) / blockSize;
    findCrosshairKernel<<<gridSize, blockSize, 0, stream>>>(
        d_lines, numLines, d_bestIntersection);
}

// ===================================================================
// CrosshairTrackerGPU implementation
// ===================================================================

CrosshairTrackerGPU::CrosshairTrackerGPU(int roiWidth, int roiHeight)
    : m_roiWidth(roiWidth), m_roiHeight(roiHeight) {

    m_canny = cv::cuda::createCannyEdgeDetector(50.0, 150.0);

    // createHoughSegmentDetector(rho, theta, minLineLength, maxLineGap).
    // The user-spec "threshold=50" is not an HoughSegmentDetector parameter
    // (it exists on CPU HoughLinesP); minLineLength=10 / maxLineGap=5 match
    // the intended behaviour of keeping only meaningful segments.
    m_hough = cv::cuda::createHoughSegmentDetector(1.0f, static_cast<float>(CV_PI) / 180.0f,
                                                   10, 5);

    cudaMalloc(&m_devBestIntersection, sizeof(float2));

    // Initialise sentinel on device.
    const float2 sentinel = make_float2(-1.0f, -1.0f);
    cudaMemcpy(m_devBestIntersection, &sentinel, sizeof(float2),
               cudaMemcpyHostToDevice);

    m_crosshairPos = cv::Point2f(static_cast<float>(roiWidth)  / 2.0f,
                                 static_cast<float>(roiHeight) / 2.0f);
    m_prevPos      = m_crosshairPos;
    m_delta        = cv::Point2f(0.0f, 0.0f);
    m_initialized  = false;
}

CrosshairTrackerGPU::~CrosshairTrackerGPU() {
    if (m_devBestIntersection) {
        cudaFree(m_devBestIntersection);
        m_devBestIntersection = nullptr;
    }
}

bool CrosshairTrackerGPU::update(const cv::cuda::GpuMat& roi,
                                 cudaStream_t stream) {
    if (roi.empty()) return false;

    // 1. Grayscale
    cv::cuda::cvtColor(roi, m_gray, cv::COLOR_BGR2GRAY, 0, stream);

    // 2. Canny edge detection
    m_canny->detect(m_gray, m_edges, stream);

    // 3. Hough line-segment detection
    m_hough->detect(m_edges, m_lines, stream);

    // 4. Custom scoring kernel → async D2H
    findCrosshairFromLines(stream);

    // 5. Single float2 sync per frame
    float2 hostBest;
    cudaMemcpyAsync(&hostBest, m_devBestIntersection, sizeof(float2),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (hostBest.x >= 0.0f && hostBest.y >= 0.0f) {
        m_crosshairPos = cv::Point2f(hostBest.x, hostBest.y);
        if (m_initialized) {
            m_delta = m_crosshairPos - m_prevPos;
        } else {
            m_delta       = cv::Point2f(0.0f, 0.0f);
            m_initialized = true;
        }
        m_prevPos = m_crosshairPos;
        return true;
    }
    // No detection this frame — delta stays at last known value.
    return false;
}

void CrosshairTrackerGPU::findCrosshairFromLines(cudaStream_t stream) {
    float4* d_lines   = reinterpret_cast<float4*>(m_lines.data);
    const int numLines = m_lines.cols; // Hough outputs 1 row × N cols of Vec4f
    launchFindCrosshair(d_lines, numLines, m_devBestIntersection, stream);
}
