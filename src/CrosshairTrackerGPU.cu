#include "CrosshairTrackerGPU.h"
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h> // CV_TM_CCOEFF_NORMED

// ===================================================================
// CrosshairTrackerGPU — template-matching implementation
// ===================================================================

CrosshairTrackerGPU::CrosshairTrackerGPU(int roiWidth, int roiHeight,
                                         const std::string& templatePath)
    : m_roiWidth(roiWidth), m_roiHeight(roiHeight),
      m_ready(false), m_initialized(false) {

    // Load the crosshair template from disk.  Try several paths.
    cv::Mat templ;
    std::string paths[] = {
        templatePath,
        "images/" + templatePath,
        "dep/" + templatePath
    };
    for (const auto& p : paths) {
        templ = cv::imread(p, cv::IMREAD_GRAYSCALE);
        if (!templ.empty()) {
            std::cout << "[Crosshair] loaded template: " << p
                      << " (" << templ.cols << "x" << templ.rows << ")" << std::endl;
            break;
        }
    }
    if (templ.empty()) {
        std::cerr << "[Crosshair] ERROR: could not load template '"
                  << templatePath << "' from any search path" << std::endl;
        return;
    }

    m_templateW = templ.cols;
    m_templateH = templ.rows;

    // Upload to GPU.
    m_templateGpu.upload(templ);

    // Create template matcher — NCC-normalised, CV_8U input, grayscale.
    // CV_TM_CCOEFF_NORMED is invariant to brightness, robust against
    // varying game backgrounds.
    m_matcher = cv::cuda::createTemplateMatching(CV_8UC1, CV_TM_CCOEFF_NORMED);

    // Tracking anchor starts at ROI centre.
    m_trackAnchor = make_float2(static_cast<float>(roiWidth)  / 2.0f,
                                static_cast<float>(roiHeight) / 2.0f);

    m_crosshairPos = cv::Point2f(m_trackAnchor.x, m_trackAnchor.y);
    m_prevPos      = m_crosshairPos;
    m_delta        = cv::Point2f(0.0f, 0.0f);
    m_debugFrame   = 0;
    m_ready        = true;
}

CrosshairTrackerGPU::~CrosshairTrackerGPU() {}

bool CrosshairTrackerGPU::update(const cv::cuda::GpuMat& roi,
                                 cudaStream_t stream) {
    if (!m_ready || roi.empty()) return false;

    cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);

    // 1. Convert ROI to grayscale.
    int cvtCode = (roi.channels() == 4) ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY;
    cv::cuda::cvtColor(roi, m_gray, cvtCode, 0, cvStream);

    // 2. Template matching — search full ROI.
    m_matcher->match(m_gray, m_templateGpu, m_matchResult, cvStream);

    // 3. Download match result and find best on CPU
    //    (cv::cuda::minMaxLoc not available in this OpenCV build).
    cv::Mat hostResult;
    m_matchResult.download(hostResult, cvStream);

    // Wait for GPU operations to finish.
    cvStream.waitForCompletion();

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(hostResult, &minVal, &maxVal, &minLoc, &maxLoc);

    // CV_TM_CCOEFF_NORMED: higher = better.  maxLoc is top-left of best match.
    const float bestX = static_cast<float>(maxLoc.x) + static_cast<float>(m_templateW) * 0.5f;
    const float bestY = static_cast<float>(maxLoc.y) + static_cast<float>(m_templateH) * 0.5f;
    const float score = static_cast<float>(maxVal);

    // 4. Search-window gate: if we have a reasonable prior, reject matches
    //    too far from the tracking anchor.
    bool accepted = true;
    const float maxWindow = 50.0f;
    if (m_initialized) {
        const float dx = bestX - m_trackAnchor.x;
        const float dy = bestY - m_trackAnchor.y;
        const float dist = sqrtf(dx * dx + dy * dy);
        if (dist > maxWindow) {
            accepted = false;
        }
    }

    // --- Debug print (~once per second) ---
    if (++m_debugFrame >= 240) {
        m_debugFrame = 0;
        std::cout << "[Crosshair] score: " << score
                  << "  best: (" << bestX << ", " << bestY << ")"
                  << "  accepted: " << (accepted ? "YES" : "no (outside window)")
                  << "  anchor: (" << m_trackAnchor.x << ", " << m_trackAnchor.y << ")"
                  << std::endl;
    }

    if (accepted) {
        // EMA smoothing: alpha=0.4 — responsive to recoil, suppresses jitter.
        const float alpha = 0.4f;
        m_trackAnchor.x = alpha * bestX + (1.0f - alpha) * m_trackAnchor.x;
        m_trackAnchor.y = alpha * bestY + (1.0f - alpha) * m_trackAnchor.y;

        m_crosshairPos = cv::Point2f(m_trackAnchor.x, m_trackAnchor.y);

        if (m_initialized) {
            m_delta = m_crosshairPos - m_prevPos;
        } else {
            m_delta       = cv::Point2f(0.0f, 0.0f);
            m_initialized = true;
        }
        m_prevPos = m_crosshairPos;
        return true;
    }
    return false;
}
