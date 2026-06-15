#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>

// GPU crosshair tracker for dot-type crosshairs.
// Finds the brightest spot in a small search window around the last known
// position.  No template matching — just argmax + weighted centroid.
//
// Pipeline:
//   1. Convert ROI to grayscale (GPU)
//   2. Crop 31×31 search window around tracking anchor (GPU, zero-copy)
//   3. Download 961 bytes → find brightest pixel + weighted centroid (CPU)
//   4. Brightness gate + EMA temporal smoothing → m_crosshairPos
class CrosshairTrackerGPU {
public:
    CrosshairTrackerGPU(int roiWidth, int roiHeight,
                        const std::string& templatePath = "crosshair.png");
    ~CrosshairTrackerGPU();

    // Always ready — no template to load.
    bool isReady() const { return true; }

    // Process the cropped ROI (GPU mat, BGRA or BGR) and update the internal
    // crosshair centre.  stream must outlive the call.
    bool update(const cv::cuda::GpuMat& roi, cudaStream_t stream);

    // Latest crosshair coordinates (pixels, relative to ROI top-left).
    cv::Point2f getPosition() const { return m_crosshairPos; }

    // Frame-to-frame movement delta (current – previous).
    cv::Point2f getDelta() const { return m_delta; }

    // Dot size for overlay bounding box (fixed at 9×9).
    int getTemplateW() const { return 9; }
    int getTemplateH() const { return 9; }

private:
    cv::cuda::GpuMat m_gray;       // full-ROI grayscale (reused)

    cv::Point2f m_crosshairPos;
    cv::Point2f m_prevPos;
    cv::Point2f m_delta;
    int m_roiWidth, m_roiHeight;
    bool m_initialized;

    // Tracking anchor for search window + EMA.
    float2 m_trackAnchor;
};
