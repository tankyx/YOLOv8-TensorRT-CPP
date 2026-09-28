// LinuxCapture.h
//
// Linux equivalent of DXGICaptureCUDA for Wayland sessions.
// Capture path: xdg-desktop-portal ScreenCast (D-Bus) -> PipeWire stream -> pinned host buffer
// (cudaHostAlloc) -> cudaMemcpy2DAsync -> linear BGRA GpuMat. Output is BGRA (CV_8UC4); the
// fused preproc kernel ignores the alpha channel, same as the Windows path.
//
// The first CaptureScreen call after construction triggers no work: the portal handshake
// (including the one-time KDE consent dialog) happens in the constructor. CaptureScreen
// returns false while the PipeWire stream has not produced a frame yet.
//
// Self-contained: all PipeWire/D-Bus details live in LinuxCapture.cpp. Requires
// <opencv2/core/cuda.hpp> and <cuda_runtime.h> only.

#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <opencv2/core/cuda.hpp>

class LinuxCapture {
public:
    // Runs the full portal handshake (D-Bus) and connects the PipeWire stream.
    // Throws std::runtime_error on hard failure (no session bus, portal refused,
    // PipeWire connect failed). May block on the KDE permission dialog the first time.
    LinuxCapture();
    ~LinuxCapture();

    LinuxCapture(const LinuxCapture &) = delete;
    LinuxCapture &operator=(const LinuxCapture &) = delete;

    // Capture one frame into `frame` (CV_8UC4 BGRA). Sized to the screen resolution; the
    // caller crops to ROI afterwards. Returns false when no new frame is available yet
    // (caller retries next loop iteration), true when a fresh frame was copied onto `stream`.
    bool CaptureScreen(cv::cuda::GpuMat &frame, cudaStream_t stream);

    int screenWidth() const;
    int screenHeight() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
