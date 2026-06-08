#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>

// Pure-GPU crosshair shape tracker. Detects the crosshair by its geometric
// structure — a pair of roughly-perpendicular lines intersecting near their
// midpoints — rather than template-matching against a stored image.
//
// Pipeline (all on GPU, single float2 D2H copy per frame):
//   1. Grayscale conversion
//   2. Canny edge detection
//   3. Hough line-segment detection
//   4. Custom CUDA kernel: score every line-pair intersection, pick best
//   5. Async copy single float2 back to host
class CrosshairTrackerGPU {
public:
    // roiWidth / roiHeight: pixel dimensions of the region the tracker will
    // receive in update(). Used to initialise the centre fallback position.
    CrosshairTrackerGPU(int roiWidth, int roiHeight);
    ~CrosshairTrackerGPU();

    // Process the cropped ROI (GPU mat, BGRA or BGR) and update the internal
    // crosshair centre. stream must outlive the call — the tracker enqueues
    // work onto it but does NOT synchronise (the caller syncs after the
    // async D2H copy). Returns true if a crosshair was detected this frame.
    bool update(const cv::cuda::GpuMat& roi, cudaStream_t stream);

    // Latest crosshair coordinates (pixels, relative to ROI top-left).
    cv::Point2f getPosition() const { return m_crosshairPos; }

    // Frame-to-frame movement delta (current – previous). Zero on the first
    // frame where a detection occurs. Use for direct recoil compensation.
    cv::Point2f getDelta() const { return m_delta; }

private:
    cv::Ptr<cv::cuda::CannyEdgeDetector>   m_canny;
    cv::Ptr<cv::cuda::HoughSegmentDetector> m_hough;
    cv::cuda::GpuMat m_gray, m_edges, m_lines;

    cv::Point2f m_crosshairPos;
    cv::Point2f m_prevPos;
    cv::Point2f m_delta;
    int m_roiWidth, m_roiHeight;
    bool m_initialized;

    float2* m_devBestIntersection; // device buffer for kernel output

    // Extract device pointer + line count from m_lines, launch the scoring
    // kernel, and enqueue an async D2H copy into m_devBestIntersection.
    void findCrosshairFromLines(cudaStream_t stream);
};
