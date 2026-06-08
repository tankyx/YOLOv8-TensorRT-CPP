#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>

// Pure-GPU crosshair tracker using template matching against a stored
// crosshair image.  Much more robust than geometric (Hough) detection for
// complex game backgrounds — the template captures the exact pixel pattern.
//
// Pipeline (all on GPU):
//   1. Convert ROI to grayscale
//   2. Template matching (NCC-normalised) against the crosshair template
//   3. Find best-match position via minMaxLoc
//   4. Search-window gate + EMA temporal smoothing → m_crosshairPos
class CrosshairTrackerGPU {
public:
    // roiWidth / roiHeight: pixel dimensions of the region the tracker will
    // receive in update().  templatePath: path to crosshair PNG (grayscale or
    // colour — will be converted to single-channel on load).
    CrosshairTrackerGPU(int roiWidth, int roiHeight,
                        const std::string& templatePath = "crosshair.png");
    ~CrosshairTrackerGPU();

    // Returns true if the template was loaded successfully.
    bool isReady() const { return m_ready; }

    // Process the cropped ROI (GPU mat, BGRA or BGR) and update the internal
    // crosshair centre.  stream must outlive the call.
    bool update(const cv::cuda::GpuMat& roi, cudaStream_t stream);

    // Latest crosshair coordinates (pixels, relative to ROI top-left).
    cv::Point2f getPosition() const { return m_crosshairPos; }

    // Frame-to-frame movement delta (current – previous).
    cv::Point2f getDelta() const { return m_delta; }

private:
    cv::Ptr<cv::cuda::TemplateMatching> m_matcher;
    cv::cuda::GpuMat m_templateGpu;
    cv::cuda::GpuMat m_gray, m_matchResult;

    cv::Point2f m_crosshairPos;
    cv::Point2f m_prevPos;
    cv::Point2f m_delta;
    int m_roiWidth, m_roiHeight;
    int m_templateW, m_templateH;
    bool m_ready;
    bool m_initialized;

    // Tracking anchor for the search-window gate + EMA.
    float2 m_trackAnchor;

    // Debug
    int m_debugFrame;
};
