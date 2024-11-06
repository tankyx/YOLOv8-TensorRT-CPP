#pragma once

#include "DXGICapture.h" //My DXGI/DDA capture
#include "threadsafe_queue.h"
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Forward declaration of CUDA kernel
extern "C" void
launchFuseFramesKernel(const uchar3 *captureCardFrame, const uchar3 *desktopFrame, uchar3 *outputFrame, int width, int height,
                       cudaStream_t stream);

class SoftwareFuser {
public:
    SoftwareFuser(int captureWidth, int captureHeight, int outputWidth, int outputHeight, FrameQueue &desktopQueue)
        : m_captureWidth(captureWidth), m_captureHeight(captureHeight), m_outputWidth(outputWidth), m_outputHeight(outputHeight),
          m_desktopQueue(desktopQueue) {
        initializeCapture();
    }

    void startFusion() {
        m_running = true;
        m_fusionThread = std::thread(&SoftwareFuser::fusionLoop, this);
    }

    void stopFusion() {
        m_running = false;
        if (m_fusionThread.joinable())
            m_fusionThread.join();
    }

    FrameQueue &getFusedQueue() { return m_fusedQueue; }

private:
    void initializeCapture() {
        m_captureCard.open(1, cv::CAP_DSHOW); // Adjust index as needed for your capture card

        if (!m_captureCard.isOpened()) {
            std::cout << "Failed to open capture card" << std::endl;
            throw std::runtime_error("Failed to open capture card");
        }

        m_captureCard.set(cv::CAP_PROP_FRAME_WIDTH, m_captureWidth);
        m_captureCard.set(cv::CAP_PROP_FRAME_HEIGHT, m_captureHeight);
    }

    void SoftwareFuser::fusionLoop() {
        cv::Mat cpuCaptureFrame, cpuDesktopFrame, cpuOutputFrame;
        cv::cuda::GpuMat gpuCaptureFrame, upscaledCaptureFrame, gpuDesktopFrame, outputFrame;

        cudaStreamCreate(&m_cudaStream);

        while (m_running) {
            m_captureCard >> cpuCaptureFrame;
            if (cpuCaptureFrame.empty()) {
                std::cout << "Capture card frame is empty" << std::endl;
                continue;
            }

            gpuCaptureFrame.upload(cpuCaptureFrame);
            cv::cuda::resize(gpuCaptureFrame, upscaledCaptureFrame, cv::Size(m_outputWidth, m_outputHeight));

            // Get desktop frame and upload to GPU
            cpuDesktopFrame = m_desktopQueue.pop();
            gpuDesktopFrame.upload(cpuDesktopFrame);

            outputFrame.create(m_outputHeight, m_outputWidth, CV_8UC3);

            launchFuseFramesKernel(upscaledCaptureFrame.ptr<uchar3>(), gpuDesktopFrame.ptr<uchar3>(), outputFrame.ptr<uchar3>(),
                                   m_outputWidth, m_outputHeight, m_cudaStream);

            cudaStreamSynchronize(m_cudaStream);

            // Download the result back to CPU
            outputFrame.download(cpuOutputFrame);

            m_fusedQueue.push(cpuOutputFrame);
        }

        cudaStreamDestroy(m_cudaStream);
    }

    cv::VideoCapture m_captureCard;
    FrameQueue &m_desktopQueue;
    FrameQueue m_fusedQueue;
    std::thread m_fusionThread;
    std::atomic<bool> m_running{false};
    int m_captureWidth, m_captureHeight, m_outputWidth, m_outputHeight;
    cudaStream_t m_cudaStream;
};