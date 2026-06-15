#include "CrosshairTrackerGPU.h"
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/core/cuda_stream_accessor.hpp>

// ===================================================================
// CrosshairTrackerGPU — bright-spot tracker for dot crosshairs
// ===================================================================

CrosshairTrackerGPU::CrosshairTrackerGPU(int roiWidth, int roiHeight,
                                         const std::string& /*templatePath*/)
    : m_roiWidth(roiWidth), m_roiHeight(roiHeight),
      m_initialized(false) {

    m_trackAnchor = make_float2(static_cast<float>(roiWidth)  / 2.0f,
                                static_cast<float>(roiHeight) / 2.0f);

    m_crosshairPos = cv::Point2f(m_trackAnchor.x, m_trackAnchor.y);
    m_prevPos      = m_crosshairPos;
    m_delta        = cv::Point2f(0.0f, 0.0f);

    std::cout << "[Crosshair] bright-spot tracker ready (dot mode, "
              << roiWidth << "x" << roiHeight << ")" << std::endl;
}

CrosshairTrackerGPU::~CrosshairTrackerGPU() {}

bool CrosshairTrackerGPU::update(const cv::cuda::GpuMat& roi,
                                 cudaStream_t stream) {
    if (roi.empty()) return false;

    cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);

    // 1. Convert ROI to grayscale on GPU.
    const int cvtCode = (roi.channels() == 4) ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY;
    cv::cuda::cvtColor(roi, m_gray, cvtCode, 0, cvStream);

    // 2. Crop 31×31 search window around the tracking anchor.
    constexpr int halfWin = 15;
    int sx = static_cast<int>(m_trackAnchor.x) - halfWin;
    int sy = static_cast<int>(m_trackAnchor.y) - halfWin;
    int sw = halfWin * 2 + 1;
    int sh = sw;
    // Clamp to ROI bounds.
    if (sx < 0) { sw += sx; sx = 0; }
    if (sy < 0) { sh += sy; sy = 0; }
    if (sx + sw > m_roiWidth)  sw = m_roiWidth  - sx;
    if (sy + sh > m_roiHeight) sh = m_roiHeight - sy;
    if (sw < 3 || sh < 3) return false;

    const cv::Rect searchRect(sx, sy, sw, sh);
    cv::cuda::GpuMat window = m_gray(searchRect); // zero-copy GPU ROI view

    // 3. Download the tiny search window (≤ 961 bytes).
    cv::Mat hostWindow;
    window.download(hostWindow, cvStream);
    cvStream.waitForCompletion();

    // 4. Find the brightest pixel and compute weighted centroid.
    float maxVal = 0.0f;
    int mx = 0, my = 0;
    for (int y = 0; y < hostWindow.rows; ++y) {
        const uint8_t* row = hostWindow.ptr<uint8_t>(y);
        for (int x = 0; x < hostWindow.cols; ++x) {
            const float v = static_cast<float>(row[x]);
            if (v > maxVal) { maxVal = v; mx = x; my = y; }
        }
    }

    // 5. Brightness gate: dot must be significantly brighter than background.
    //    A typical green/cyan dot crosshair is 200–255 in grayscale.
    if (maxVal < 170.0f) return false;

    // 6. Weighted centroid of pixels near the max (sub-pixel precision).
    const float threshold = maxVal * 0.75f;
    float sumX = 0.0f, sumY = 0.0f, sumW = 0.0f;
    for (int y = 0; y < hostWindow.rows; ++y) {
        const uint8_t* row = hostWindow.ptr<uint8_t>(y);
        for (int x = 0; x < hostWindow.cols; ++x) {
            const float v = static_cast<float>(row[x]);
            if (v > threshold) {
                const float w = v - threshold;
                sumX += static_cast<float>(x) * w;
                sumY += static_cast<float>(y) * w;
                sumW += w;
            }
        }
    }

    const float bestX = (sumW > 0.0f) ? (sumX / sumW + static_cast<float>(sx)) : (static_cast<float>(mx + sx) + 0.5f);
    const float bestY = (sumW > 0.0f) ? (sumY / sumW + static_cast<float>(sy)) : (static_cast<float>(my + sy) + 0.5f);

    // 7. Search-window gate: reject if the detected position is too far from anchor.
    constexpr float maxWindow = 60.0f;
    if (m_initialized) {
        const float dx = bestX - m_trackAnchor.x;
        const float dy = bestY - m_trackAnchor.y;
        if (dx * dx + dy * dy > maxWindow * maxWindow) {
            return false;
        }
    }

    // 8. EMA smoothing.
    constexpr float alpha = 0.85f;
    m_trackAnchor.x = alpha * bestX + (1.0f - alpha) * m_trackAnchor.x;
    m_trackAnchor.y = alpha * bestY + (1.0f - alpha) * m_trackAnchor.y;

    m_crosshairPos = cv::Point2f(m_trackAnchor.x, m_trackAnchor.y);

    if (m_initialized) {
        m_delta = m_crosshairPos - m_prevPos;
    } else {
        m_delta = cv::Point2f(0.0f, 0.0f);
        m_initialized = true;
    }
    m_prevPos = m_crosshairPos;

    return true;
}
