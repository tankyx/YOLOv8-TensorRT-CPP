#define NOMINMAX

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include "DXGICapture.h"
#include "GDIOverlay.h"
#include "MouseController.h"
#include "threadsafe_queue.h"
#include "yolov8.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <variant>
#include <windows.h>

using namespace std::chrono;
using YoloV8Variant = std::variant<YoloV8<float>, YoloV8<__half>>;

// Function prototypes
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
cv::Point detectCrosshairCPU(const cv::Mat &frame, const cv::Mat &templ);
void moveToTop();
void clearScreen();
void logAverageLatencies(LatencyQueue &captureLatency, LatencyQueue &detectionLatency, LatencyQueue &overlayLatency,
                         SafeQueue<std::string> &logQueue);

class Runners {
public:
    void captureThread(FrameQueue &captureQueue, LatencyQueue &latencyQueue, DXGICapture &capture, HWND targetWnd, int captureWidth,
                       int captureHeight, std::atomic<bool> &running);
    void detectionThread(FrameQueue &captureQueue, DetectionQueue &detectionQueue, LatencyQueue &latencyQueue, YoloV8Variant &yoloV8,
                         MouseController &mouseController, cv::Mat &templateImg, int captureWidth, int captureHeight,
                         std::atomic<bool> &running);
    void overlayThread(DetectionQueue &detectionQueue, LatencyQueue &latencyQueue, SafeQueue<std::string> &logQueue, GDIOverlay &overlay,
                       std::atomic<bool> &running);
};

// Main function
int main(int argc, char *argv[]) {
    try {
        // Parse command line arguments
        if (argc != 7) {
            std::cerr << "Usage: " << argv[0]
                      << " <path to TensorRT engine file> <width> <height> <float/half> [Head Label ID 1] [Head Label ID 2]" << std::endl;
            return -1;
        }

        std::string onnxModelPath = argv[1];
        int captureHeight = std::stoi(argv[2]);
        int captureWidth = std::stoi(argv[3]);
        std::string precision = argv[4];
        int HL1 = std::stoi(argv[5]);
        int HL2 = std::stoi(argv[6]);

        // Setup YoloV8 model
        YoloV8Config config;
        std::unique_ptr<YoloV8Variant> yoloV8;

        if (precision == "float") {
            config.precision = Precision::FP32;
            yoloV8 = std::make_unique<YoloV8Variant>(YoloV8<float>(onnxModelPath, config));
        } else if (precision == "half") {
            config.precision = Precision::FP16;
            yoloV8 = std::make_unique<YoloV8Variant>(YoloV8<__half>(onnxModelPath, config));
        } else {
            throw std::runtime_error("Invalid precision: " + precision + ". Use 'float' or 'half'.");
        }

        // Setup screen capture and mouse control
        int screenWidth = GetSystemMetrics(SM_CXSCREEN);
        int screenHeight = GetSystemMetrics(SM_CYSCREEN);
        MouseController mouseController(screenWidth, screenHeight, captureWidth, captureHeight, 0.80f, 55, 0.25f, 0.65f, 15, HL1, HL2,
                                        3000);

        // Load crosshair template
        cv::Mat templateImg = cv::imread("crosshair.png", cv::IMREAD_COLOR);
        if (templateImg.empty()) {
            throw std::runtime_error("Failed to load template image");
        }

        // Setup queues and flags
        SafeQueue<std::string> logQueue;
        LatencyQueue captureLatency, detectionLatency, overlayLatency;
        std::atomic<bool> running(true);
        FrameQueue captureQueue;
        DetectionQueue detectionQueue;

        // Find target window
        HWND targetWnd = FindWindow(NULL, "Counter-Strike 2");
        if (!targetWnd) {
            throw std::runtime_error("Failed to find target window.");
        }

        // Initialize capture and overlay
        DXGICapture capture(targetWnd);
        // GDIOverlay overlay(targetWnd, screenWidth, screenHeight);

        // Start threads
        Runners runners;
        std::thread captureThreadObj(
            [&]() { runners.captureThread(captureQueue, captureLatency, capture, targetWnd, captureWidth, captureHeight, running); });

        std::thread detectionThreadObj([&]() {
            runners.detectionThread(captureQueue, detectionQueue, detectionLatency, *yoloV8, mouseController, templateImg, captureWidth,
                                    captureHeight, running);
        });

        // std::thread overlayThreadObj([&]() {
        //     runners.overlayThread(detectionQueue, overlayLatency, logQueue, overlay, running);
        // });

        // Main loop
        clearScreen();
        while (running) {
            MSG msg;
            if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) {
                    running = false;
                }
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }

            if (GetAsyncKeyState(VK_INSERT) & 0x8000) {
                running = false;
            }

            logAverageLatencies(captureLatency, detectionLatency, overlayLatency, logQueue);
        }

        // Cleanup
        logQueue.push("exit");
        cv::destroyAllWindows();
        if (captureThreadObj.joinable())
            captureThreadObj.join();
        if (detectionThreadObj.joinable())
            detectionThreadObj.join();
        // if (overlayThreadObj.joinable()) overlayThreadObj.join();

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

// Window procedure implementation
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    if (uMsg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// Crosshair detection function
cv::Point detectCrosshairCPU(const cv::Mat &frame, const cv::Mat &templ) {
    cv::Mat result;
    cv::matchTemplate(frame, templ, result, cv::TM_CCOEFF_NORMED);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    cv::Point crosshairCenter(maxLoc.x + templ.cols / 2, maxLoc.y + templ.rows / 2);
    cv::rectangle(frame, maxLoc, cv::Point(maxLoc.x + templ.cols, maxLoc.y + templ.rows), cv::Scalar(0, 255, 0), 2);

    return crosshairCenter;
}

// Console utility functions
void moveToTop() { std::cout << "\033[H"; }

void clearScreen() { std::cout << "\033[2J\033[H"; }

void logAverageLatencies(LatencyQueue &captureLatency, LatencyQueue &detectionLatency, LatencyQueue &overlayLatency,
                         SafeQueue<std::string> &logQueue) {
    moveToTop();
    double avgCaptureLatency = captureLatency.getAverageLatency();
    double avgDetectionLatency = detectionLatency.getAverageLatency();
    double avgOverlayLatency = overlayLatency.getAverageLatency();

    std::cout << "Average Capture Latency (last 500ms): " << avgCaptureLatency << "ms\n"
              << "Average Detection Latency (last 500ms): " << avgDetectionLatency << "ms\n"
              << "Average Overlay Latency (last 500ms): " << avgOverlayLatency << "ms\n";
}

// Runners class method implementations
void Runners::captureThread(FrameQueue &captureQueue, LatencyQueue &latencyQueue, DXGICapture &capture, HWND targetWnd, int captureWidth,
                            int captureHeight, std::atomic<bool> &running) {
    const int targetFPS = 1000;
    const std::chrono::nanoseconds frameDuration(1000000000 / targetFPS);
    cv::Mat frame1(captureHeight, captureWidth, CV_8UC3);
    cv::Mat frame2(captureHeight, captureWidth, CV_8UC3);
    cv::Mat *currentFrame = &frame1;
    cv::Mat *nextFrame = &frame2;

    while (running) {
        auto start = std::chrono::high_resolution_clock::now();

        capture.CaptureScreen(*currentFrame);

        if (!currentFrame->empty()) {
            captureQueue.push(std::move(*currentFrame));
        }

        std::swap(currentFrame, nextFrame);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = end - start;

        latencyQueue.push(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
    }
}

void Runners::detectionThread(FrameQueue &captureQueue, DetectionQueue &detectionQueue, LatencyQueue &latencyQueue, YoloV8Variant &yoloV8,
                              MouseController &mouseController, cv::Mat &templateImg, int captureWidth, int captureHeight,
                              std::atomic<bool> &running) {
    while (running) {
        cv::Mat frame = captureQueue.pop();

        auto detectionStartTime = std::chrono::high_resolution_clock::now();

        int centerX = frame.cols / 2;
        int centerY = frame.rows / 2;

        int roiWidth = captureWidth;
        int roiHeight = captureHeight;
        int x = std::max(centerX - roiWidth / 2, 0);
        int y = std::max(centerY - roiHeight / 2, 0);
        x = std::min(x, frame.cols - roiWidth);
        y = std::min(y, frame.rows - roiHeight);
        cv::Rect roi(x, y, roiWidth, roiHeight);

        int smallRoiSize = 160;
        int smallX = std::max(centerX - smallRoiSize / 2, 0);
        int smallY = std::max(centerY - smallRoiSize / 2, 0);
        smallX = std::min(smallX, frame.cols - smallRoiSize);
        smallY = std::min(smallY, frame.rows - smallRoiSize);
        cv::Rect smallRoi(smallX, smallY, smallRoiSize, smallRoiSize);

        cv::Mat croppedFrame = frame(roi);
        cv::Mat smallCroppedFrame = frame(smallRoi);

        cv::Point crosshairPos = detectCrosshairCPU(smallCroppedFrame, templateImg);

        crosshairPos.x += smallX - x;
        crosshairPos.y += smallY - y;

        std::vector<Object> detections;
        std::visit([&](auto &yolo) { detections = yolo.detectObjects(croppedFrame); }, yoloV8);

        mouseController.setCrosshairPosition(crosshairPos.x, crosshairPos.y);
        mouseController.aim(detections);
        mouseController.triggerLeftClickIfCenterWithinDetection(detections);

        detectionQueue.push(detections);

        auto detectionEndTime = std::chrono::high_resolution_clock::now();
        auto detectionDuration = detectionEndTime - detectionStartTime;

        latencyQueue.push(std::chrono::duration_cast<std::chrono::milliseconds>(detectionDuration).count());
    }
}

void Runners::overlayThread(DetectionQueue &detectionQueue, LatencyQueue &latencyQueue, SafeQueue<std::string> &logQueue,
                            GDIOverlay &overlay, std::atomic<bool> &running) {
    const int targetFPS = 500;
    const std::chrono::nanoseconds frameDuration(1000000000 / targetFPS);

    MSG msg = {};
    while (msg.message != WM_QUIT && running) {
        auto overlayStartTime = std::chrono::high_resolution_clock::now();

        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        } else {
            overlay.cleanScreen();
            overlay.drawDetectionsPen(detectionQueue.pop());
            overlay.render();
            auto overlayEndTime = std::chrono::high_resolution_clock::now();
            auto overlayDuration = overlayEndTime - overlayStartTime;

            if (overlayDuration < frameDuration) {
                std::this_thread::sleep_for(frameDuration - overlayDuration);
            }

            overlayEndTime = std::chrono::high_resolution_clock::now();
            overlayDuration = overlayEndTime - overlayStartTime;
            latencyQueue.push(std::chrono::duration_cast<std::chrono::milliseconds>(overlayDuration).count());
        }
    }
}