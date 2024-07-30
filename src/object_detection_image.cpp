#define NOMINMAX

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include "crosshair_detection.cuh"
#include "threadsafe_queue.h"
#include "DXGICapture.h"
#include "MouseController.h"
#include "cmd_line_util.h"
#include "yolov8.h"
#include "GDIOverlay.h"

#include <chrono>
#include <stdlib.h>
#include <thread>
#include <windows.h>
#include <variant>
#include <algorithm>

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

void loggingThreadFunc(SafeQueue<std::string> &logQueue, std::atomic<bool> &running) {
    clearScreen();
    while (running) {
        std::string logMessage = logQueue.pop();
        if (logMessage == "exit")
            break;
        moveToTop();
        std::cout << logMessage << std::endl;
    }
}

int main(int argc, char *argv[]) {
    try {
        YoloV8Config config;
        std::string onnxModelPath, precision;
        int captureWidth, captureHeight, crossHeight, crossWidth;

        if (argc != 5) {
            std::cerr << "Usage: " << argv[0] << " <path to TensorRT engine file> <width> <height> <float/half>" << std::endl;
            return -1;
        }

        onnxModelPath = argv[1];
        captureHeight = strtol(argv[2], NULL, 10);
        captureWidth = strtol(argv[3], NULL, 10);
        crossHeight = captureHeight / 4;
        crossWidth = captureWidth / 4;
        precision = argv[4];

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

        // Calculate the capture region (center of the screen)
        int x = (screenWidth - captureWidth) / 2;
        int y = (screenHeight - captureHeight) / 2;
        std::cout << "Capture region: " << x << "," << y << " " << captureWidth << "x" << captureHeight << std::endl;

        // Create MouseController instance
        MouseController mouseController(screenWidth, screenHeight, captureWidth, captureHeight, 1.00, 30, 0.25f, 0.95f, 15);

        cv::Mat templateImg = cv::imread("crosshair.png", cv::IMREAD_COLOR);
        if (templateImg.empty()) {
            std::cerr << "Failed to load template image" << std::endl;
            return -1;
        }

        SafeQueue<std::string> logQueue;
        std::atomic<bool> running(true);
        FrameQueue detectionsQueue;
        std::thread logThread(loggingThreadFunc, std::ref(logQueue), std::ref(running));

        HWND targetWnd = FindWindow(NULL, "Counter-Strike 2");
        if (!targetWnd) {
            std::cerr << "Failed to find target window." << std::endl;
            return -1;
        }

        DXGICapture capture(targetWnd);
        //WindowCapture capture(targetWnd);
        GDIOverlay overlay(targetWnd, screenWidth, screenHeight);

        const int targetFPS = 60;
        const int frameDuration = 1000 / targetFPS; // Frame duration in milliseconds

        MSG msg = {};
        while (msg.message != WM_QUIT) {
            static std::string logMessage;
            auto loopStartTime = std::chrono::high_resolution_clock::now();

            // Capture the screen
            cv::Mat frame(captureHeight, captureWidth, CV_8UC3);
            auto captureStartTime = std::chrono::high_resolution_clock::now();
            RECT rect;
            GetWindowRect(targetWnd, &rect);
            capture.CaptureScreen(frame);
            auto captureEndTime = std::chrono::high_resolution_clock::now();

            if (frame.empty()) {
                std::cerr << "Error: Captured empty frame." << std::endl;
                continue;
            }

            auto detectionStartTime = std::chrono::high_resolution_clock::now();

            // Calculate the region of interest (center 640x640 area)
            int centerX = frame.cols / 2;
            int centerY = frame.rows / 2;

            // Define the 640x640 ROI within the frame
            int captureWidth = 640;
            int captureHeight = 640;
            int x = std::max(centerX - captureWidth / 2, 0);
            int y = std::max(centerY - captureHeight / 2, 0);
            if (x + captureWidth > frame.cols)
                x = frame.cols - captureWidth;
            if (y + captureHeight > frame.rows)
                y = frame.rows - captureHeight;
            cv::Rect roi(x, y, captureWidth, captureHeight);

            // Define the smaller 160x160 ROI within the 640x640 frame
            int smallRoiSize = 160;
            int smallX = std::max(centerX - smallRoiSize / 2, 0);
            int smallY = std::max(centerY - smallRoiSize / 2, 0);
            if (smallX + smallRoiSize > frame.cols)
                smallX = frame.cols - smallRoiSize;
            if (smallY + smallRoiSize > frame.rows)
                smallY = frame.rows - smallRoiSize;
            cv::Rect smallRoi(smallX, smallY, smallRoiSize, smallRoiSize);

            // Crop the center area
            cv::Mat croppedFrame = frame(roi).clone();

            // Crop the center area for crosshair detection
            cv::Mat smallCroppedFrame = frame(smallRoi).clone();


            cv::Point crosshairPos = detectCrosshairCPU(smallCroppedFrame, templateImg);

            // Adjust the coordinates to the full 640x640 frame
            crosshairPos.x += smallX - x;
            crosshairPos.y += smallY - y;

            // Perform object detection on the full 640x640 frame
            std::vector<Object> detections;
            std::visit(
                [&](auto &yolo) {
                    detections = yolo.detectObjects(croppedFrame);
                },
                *yoloV8);

            auto detectionEndTime = std::chrono::high_resolution_clock::now();
            // Use the crosshairPos for further processing, like mouse control, etc.
            auto aimStartTime = std::chrono::high_resolution_clock::now();
            mouseController.setCrosshairPosition(crosshairPos.x, crosshairPos.y);
            mouseController.aim(detections);
            mouseController.triggerLeftClickIfCenterWithinDetection(detections);
            auto aimEndTime = std::chrono::high_resolution_clock::now();

            long long updateDuration = 0;

            auto showStartTime = std::chrono::high_resolution_clock::now();
            if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            } else {
                overlay.cleanScreen();
                overlay.drawDetectionsPen(detections);
                overlay.drawLog(logMessage);
                overlay.render();
            }
            auto showEndTime = std::chrono::high_resolution_clock::now();

            auto waitStartTime = std::chrono::high_resolution_clock::now();
            // Check for ESC key press
            if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
                break;
            }
            auto waitEndTime = std::chrono::high_resolution_clock::now();
            auto loopEndTime = std::chrono::high_resolution_clock::now();

            auto sleepTimer = std::chrono::milliseconds(frameDuration) - (loopEndTime - loopStartTime);
            if (sleepTimer > std::chrono::milliseconds(0)) {
                std::this_thread::sleep_for(sleepTimer);
            }
            // Log the time taken for the loop iteration, screen capture, and detection
            auto detectionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(detectionEndTime - detectionStartTime).count();
            auto loopDuration = std::chrono::duration_cast<std::chrono::milliseconds>((loopEndTime + sleepTimer) - loopStartTime).count();
            auto aimDuration = std::chrono::duration_cast<std::chrono::milliseconds>(aimEndTime - aimStartTime).count();
            auto showDuration = std::chrono::duration_cast<std::chrono::milliseconds>(showEndTime - showStartTime).count();
            auto waitDuration = std::chrono::duration_cast<std::chrono::milliseconds>(waitEndTime - waitStartTime).count();
            auto captureDuration = std::chrono::duration_cast<std::chrono::milliseconds>(captureEndTime - captureStartTime).count();
            auto sleepDuration = std::chrono::duration_cast<std::chrono::milliseconds>(sleepTimer).count();

            logMessage = "FPS: " + std::to_string(1000 / (loopDuration)) + "fps\n" + "Capture time: " + std::to_string(captureDuration) + "ms\n" +
                "Detection time: " + std::to_string(detectionDuration) + "ms\n" + "Aim time: " + std::to_string(aimDuration) + "ms\n" +
                "Show time: " + std::to_string(showDuration) + "ms\n" + "Wait time: " + std::to_string(waitDuration) + "ms\n" +
                "Sleep time:" + std::to_string(sleepDuration) + "ms\n" + "Loop time: " + std::to_string(loopDuration) + "ms\n";
            logQueue.push(std::string(logMessage));
        }

        running = false;
        logQueue.push("exit");
        cv::destroyAllWindows();
        if (logThread.joinable()) logThread.join();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
