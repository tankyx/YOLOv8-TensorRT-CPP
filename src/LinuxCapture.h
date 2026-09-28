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
    // `targetFps` caps the rate at which frames are copied to the GPU (0 = unlimited).
    // The value is offered to the portal as the stream framerate and enforced again in
    // the PipeWire callback, so the host copies never run at the compositor's delivery
    // rate (which scales with the game's own frame rate).
    //
    // `roiWidth`/`roiHeight` describe the centred window the caller actually uses
    // (CaptureWidth/CaptureHeight in the ini). Only that window is copied host-side and
    // uploaded; the GpuMat keeps its full screen size, so the caller's crop maths and
    // overlay offsets stay unchanged. 0x0 copies the whole frame.
    explicit LinuxCapture(int targetFps = 0, int roiWidth = 0, int roiHeight = 0);
    ~LinuxCapture();

    LinuxCapture(const LinuxCapture &) = delete;
    LinuxCapture &operator=(const LinuxCapture &) = delete;

    // Capture one frame into `frame` (CV_8UC4 BGRA). Sized to the screen resolution; only the
    // configured ROI window is refreshed (nothing reads the rest, so it keeps its previous
    // contents) — host traffic is (ROI area / frame area) of the old full-frame copy. Blocks
    // up to ~8 ms waiting for a new frame, then returns false if none arrived (so callers need
    // not poll); returns true when a fresh frame was copied onto `stream`.
    bool CaptureScreen(cv::cuda::GpuMat &frame, cudaStream_t stream);

    int screenWidth() const;
    int screenHeight() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
