#define NOMINMAX

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include "DXGICapture.h"
#include "GDIOverlay.h"
//#include "Overlay.h"
#include "MouseController.h"
#include "cmd_line_util.h"
#include "threadsafe_queue.h"
#include "yolov8.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <stdlib.h>
#include <thread>
#include <variant>
#include <windows.h>

using namespace std::chrono;
using YoloV8Variant = std::variant<YoloV8<float>, YoloV8<__half>>;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    if (uMsg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// Function to detect crosshair using OpenCV template matching and draw around it
cv::Point detectCrosshairCPU(const cv::Mat &frame, const cv::Mat &templ) {
    // Perform template matching
    cv::Mat result;
    cv::matchTemplate(frame, templ, result, cv::TM_CCOEFF_NORMED);

    // Localize the best match with minMaxLoc
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    cv::Point crosshairCenter = cv::Point(maxLoc.x + templ.cols / 2, maxLoc.y + templ.rows / 2);

    // Draw a rectangle around the detected crosshair in the original colored frame
    cv::rectangle(frame, maxLoc, cv::Point(maxLoc.x + templ.cols, maxLoc.y + templ.rows), cv::Scalar(0, 255, 0), 2);

    return crosshairCenter;
}


// Function to move the cursor to the top of the log section
void moveToTop() {
    std::cout << "\033[H"; // Move the cursor to the top-left corner
}

void clearScreen() {
    std::cout << "\033[2J\033[H"; // Clear the screen and move the cursor to the top-left corner
}

void logAverageLatencies(LatencyQueue &captureLatency, LatencyQueue &detectionLatency, LatencyQueue &overlayLatency,
                         SafeQueue<std::string> &logQueue) {
    // Clear the screen and move the cursor to the top
    moveToTop();

    // Get the average latencies
    double avgCaptureLatency = captureLatency.getAverageLatency();
    double avgDetectionLatency = detectionLatency.getAverageLatency();
    double avgOverlayLatency = overlayLatency.getAverageLatency();

    std::cout << "Average Capture Latency (last 500ms): " << std::to_string(avgCaptureLatency) + "ms\n"
              << "Average Detection Latency (last 500ms): " << std::to_string(avgDetectionLatency) + "ms\n"
              << "Average Overlay Latency (last 500ms): " << std::to_string(avgOverlayLatency) + "ms\n";
}

class Runners {
public:
    void captureThread(FrameQueue &captureQueue, LatencyQueue &latencyQueue, DXGICapture &capture, HWND targetWnd, int captureWidth,
                       int captureHeight, std::atomic<bool> &running) {
        const int targetFPS = 1000;
        const std::chrono::nanoseconds frameDuration(1000000000 / targetFPS); // 1s / FPS
        cv::Mat frame1(captureHeight, captureWidth, CV_8UC3);
        cv::Mat frame2(captureHeight, captureWidth, CV_8UC3);
        cv::Mat *currentFrame = &frame1;
        cv::Mat *nextFrame = &frame2;

        while (running) {
            auto start = std::chrono::high_resolution_clock::now();

            capture.CaptureScreen(*currentFrame);

            if (!currentFrame->empty()) {
                captureQueue.push(std::move(*currentFrame)); // Use move semantics to avoid copying
            }

            std::swap(currentFrame, nextFrame);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = end - start;

            latencyQueue.push(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());

        }
    }

void detectionThread(FrameQueue &captureQueue, DetectionQueue &detectionQueue, LatencyQueue &latencyQueue, YoloV8Variant &yoloV8,
                         MouseController &mouseController, cv::Mat &templateImg, int captureWidth, int captureHeight,
                         std::atomic<bool> &running) {
        const int targetFPS = 1000;
        const std::chrono::nanoseconds frameDuration(1000000000 / targetFPS);

        while (running) {
            cv::Mat frame = captureQueue.pop();

            auto detectionStartTime = std::chrono::high_resolution_clock::now();

            // Calculate the region of interest (center capture area)
            int centerX = frame.cols / 2;
            int centerY = frame.rows / 2;

            // Define the capture ROI within the frame
            int roiWidth = captureWidth;
            int roiHeight = captureHeight;
            int x = std::max(centerX - roiWidth / 2, 0);
            int y = std::max(centerY - roiHeight / 2, 0);
            if (x + roiWidth > frame.cols)
                x = frame.cols - roiWidth;
            if (y + roiHeight > frame.rows)
                y = frame.rows - roiHeight;
            cv::Rect roi(x, y, roiWidth, roiHeight);

            // Define the smaller 160x160 ROI within the capture frame
            int smallRoiSize = 160;
            int smallX = std::max(centerX - smallRoiSize / 2, 0);
            int smallY = std::max(centerY - smallRoiSize / 2, 0);
            if (smallX + smallRoiSize > frame.cols)
                smallX = frame.cols - smallRoiSize;
            if (smallY + smallRoiSize > frame.rows)
                smallY = frame.rows - smallRoiSize;
            cv::Rect smallRoi(smallX, smallY, smallRoiSize, smallRoiSize);

            // Use references to the ROIs instead of cloning
            cv::Mat croppedFrame = frame(roi);
            cv::Mat smallCroppedFrame = frame(smallRoi);

            cv::Point crosshairPos = detectCrosshairCPU(smallCroppedFrame, templateImg);

            // Adjust the coordinates to the full frame
            crosshairPos.x += smallX - x;
            crosshairPos.y += smallY - y;

            // Perform object detection on the captured frame
            std::vector<Object> detections;
            std::visit([&](auto &yolo) { 
                detections = yolo.detectObjects(croppedFrame);
                //yolo.drawObjectLabels(croppedFrame, detections);
            }, yoloV8);

            /* cv::imshow("croppedFrame", croppedFrame);
            cv::imshow("smallCroppedFrame", smallCroppedFrame);
            if (cv::waitKey(1) == 27) {
                running = false;
                break;
            }*/

            // Use the crosshairPos for further processing, like mouse control, etc.
            mouseController.setCrosshairPosition(crosshairPos.x, crosshairPos.y);
            mouseController.aim(detections);
            mouseController.triggerLeftClickIfCenterWithinDetection(detections);

            detectionQueue.push(detections);

            auto detectionEndTime = std::chrono::high_resolution_clock::now();
            auto detectionDuration = detectionEndTime - detectionStartTime;

            latencyQueue.push(std::chrono::duration_cast<std::chrono::milliseconds>(detectionDuration).count());
        }
    }


    void overlayThread(DetectionQueue &detectionQueue, LatencyQueue &latencyQueue, SafeQueue<std::string> &logQueue, GDIOverlay &overlay,
                       std::atomic<bool> &running) {
        const int targetFPS = 500;
        const std::chrono::nanoseconds frameDuration(1000000000 / targetFPS);

        MSG msg = {};
        while (msg.message != WM_QUIT || running) {
            auto overlayStartTime = std::chrono::high_resolution_clock::now();

            if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            } else {
                overlay.cleanScreen();
                //overlay.drawLog("Yup");
                overlay.drawDetectionsPen(detectionQueue.pop());
                overlay.render();
                auto overlayEndTime = std::chrono::high_resolution_clock::now();
                auto overlayDuration = overlayEndTime - overlayStartTime;

                if (overlayDuration < frameDuration) {
                    std::this_thread::sleep_for(frameDuration - overlayDuration);
                    // Busy wait for the remaining time
                    while (std::chrono::high_resolution_clock::now() - overlayEndTime < frameDuration - overlayDuration) {
                        // Busy wait
                    }
                }

                overlayEndTime = std::chrono::high_resolution_clock::now();
                overlayDuration = overlayEndTime - overlayStartTime;
                latencyQueue.push(std::chrono::duration_cast<std::chrono::milliseconds>(overlayDuration).count());
            }
        }
    }
};

int main(int argc, char *argv[]) {
    try {
        YoloV8Config config;
        std::string onnxModelPath, precision;
        int captureWidth, captureHeight, HL1, HL2;

        if (argc != 7) {
            std::cerr << "Usage: " << argv[0] << " <path to TensorRT engine file> <width> <height> <float/half> [Head Label ID 1][Head Label ID 2]" << std::endl;
            return -1;
        }

        onnxModelPath = argv[1];
        captureHeight = strtol(argv[2], NULL, 10);
        captureWidth = strtol(argv[3], NULL, 10);
        precision = argv[4];
        HL1 = strtol(argv[5], NULL, 10);
        HL2 = strtol(argv[6], NULL, 10);

        std::unique_ptr<YoloV8Variant> yoloV8;

        if (precision == "float") {
            yoloV8 = std::make_unique<YoloV8Variant>(YoloV8<float>(onnxModelPath, config));
        } else if (precision == "half") {
            yoloV8 = std::make_unique<YoloV8Variant>(YoloV8<__half>(onnxModelPath, config));
        } else {
            std::cerr << "Invalid precision: " << precision << ". Use 'float' or 'half'." << std::endl;
            return -1;
        }

        // Get the screen dimensions
        int screenWidth = GetSystemMetrics(SM_CXSCREEN);
        int screenHeight = GetSystemMetrics(SM_CYSCREEN);

        // Create MouseController instance
        MouseController mouseController(screenWidth, screenHeight, captureWidth, captureHeight, 0.80f, 55, 0.25f, 0.65f, 15, HL1, HL2, 3000);

        cv::Mat templateImg = cv::imread("crosshair.png", cv::IMREAD_COLOR);
        if (templateImg.empty()) {
            std::cerr << "Failed to load template image" << std::endl;
            return -1;
        }

        SafeQueue<std::string> logQueue;
        LatencyQueue captureLatency;
        LatencyQueue detectionLatency;
        LatencyQueue overlayLatency;
        std::atomic<bool> running(true);
        FrameQueue captureQueue;
        DetectionQueue detectionQueue;

        HWND targetWnd = FindWindow(NULL, "Counter-Strike 2");
        if (!targetWnd) {
            std::cerr << "Failed to find target window." << std::endl;
            return -1;
        }

        DXGICapture capture(targetWnd);
        //GDIOverlay overlay(targetWnd, screenWidth, screenHeight);

        Runners runners;

        std::thread captureThreadObj(
            [&runners, &captureQueue, &captureLatency, &capture, targetWnd, captureWidth, captureHeight, &running]() {
                runners.captureThread(captureQueue, captureLatency, capture, targetWnd, captureWidth, captureHeight, running);
            });

        std::thread detectionThreadObj([&runners, &captureQueue, &detectionQueue, &detectionLatency, &yoloV8, &mouseController,
                                        &templateImg, captureWidth, captureHeight, &running]() {
            runners.detectionThread(captureQueue, detectionQueue, detectionLatency, *yoloV8, mouseController, templateImg, captureWidth,
                                    captureHeight, running);
        });

        //std::thread overlayThreadObj([&runners, &detectionQueue, &overlayLatency, &logQueue, &overlay, &running]() {
		//	runners.overlayThread(detectionQueue, overlayLatency, logQueue, overlay, running);
		//});

        clearScreen();
        while (true) {
            std::string logMessage;
            MSG msg;

            if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) {
                    running = false;
                }
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }

            // Check for INS key press
            if (GetAsyncKeyState(VK_INSERT) & 0x8000) {
                running = false;
                break;
            }

            logAverageLatencies(captureLatency, detectionLatency, overlayLatency, logQueue);
        }

        running = false;
        logQueue.push("exit");
        cv::destroyAllWindows();
        if (captureThreadObj.joinable())
            captureThreadObj.join();
        if (detectionThreadObj.joinable())
            detectionThreadObj.join();
        //if (overlayThreadObj.joinable())
        //    overlayThreadObj.join();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
