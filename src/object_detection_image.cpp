#include "MouseController.h"
#include "cmd_line_util.h"
#include "yolov8.h"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <stdlib.h> // for strtol
#include <thread>
#include <windows.h>

using namespace std::chrono;

// Function to initialize screen capture
void initializeScreenCapture(HDC &hScreenDC, HDC &hMemoryDC, HBITMAP &hBitmap, int width, int height) {
    hScreenDC = GetDC(nullptr);
    hMemoryDC = CreateCompatibleDC(hScreenDC);
    hBitmap = CreateCompatibleBitmap(hScreenDC, width, height);
    SelectObject(hMemoryDC, hBitmap);
}

void captureScreen(HDC hScreenDC, HDC hMemoryDC, HBITMAP hBitmap, int x, int y, int width, int height, cv::Mat &mat) {
    BitBlt(hMemoryDC, 0, 0, width, height, hScreenDC, x, y, SRCCOPY);

    BITMAPINFOHEADER bi;
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height; // negative height to flip the image vertically
    bi.biPlanes = 1;
    bi.biBitCount = 24; // 3 bytes per pixel (BGR)
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0;
    bi.biXPelsPerMeter = 0;
    bi.biYPelsPerMeter = 0;
    bi.biClrUsed = 0;
    bi.biClrImportant = 0;

    // Create a temporary buffer to store the captured data
    std::vector<BYTE> buffer(width * height * 3);

    // Copy the captured data into the temporary buffer
    GetDIBits(hMemoryDC, hBitmap, 0, height, buffer.data(), (BITMAPINFO *)&bi, DIB_RGB_COLORS);

    // Copy the buffer data into the cv::Mat object
    memcpy(mat.data, buffer.data(), buffer.size());
}

// Function to release resources
void releaseScreenCapture(HDC hScreenDC, HDC hMemoryDC, HBITMAP hBitmap) {
    DeleteObject(hBitmap);
    DeleteDC(hMemoryDC);
    ReleaseDC(nullptr, hScreenDC);
}

int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;
    int captureWidth, captureHeight;

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <path to TensorRT engine file> <width> <height>" << std::endl;
        return -1;
    }

    onnxModelPath = argv[1];
    captureHeight = strtol(argv[2], NULL, 10);
    captureWidth = strtol(argv[3], NULL, 10);

    // Create the YoloV8 engine
    YoloV8 yoloV8(onnxModelPath, config);

    // Get the screen dimensions
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    // Initialize screen capture
    HDC hScreenDC, hMemoryDC;
    HBITMAP hBitmap;
    std::cout << "Initializing screen capture..." << std::endl;
    initializeScreenCapture(hScreenDC, hMemoryDC, hBitmap, captureWidth, captureHeight);

    // Calculate the capture region (center of the screen)
    int x = (screenWidth - captureWidth) / 2;
    int y = (screenHeight - captureHeight) / 2;
    std::cout << "Capture region: " << x << "," << y << " " << captureWidth << "x" << captureHeight << std::endl;

    // Create MouseController instance
    MouseController mouseController(screenWidth, screenHeight, captureWidth, captureHeight, 1.25, 30);

    // Prepare a cv::Mat to store the captured frame
    cv::Mat frame(captureHeight, captureWidth, CV_8UC3);

    cv::namedWindow("Detections", cv::WINDOW_NORMAL);
    cv::resizeWindow("Detections", captureWidth, captureHeight);

    // Timer to manage frame capture at 144Hz
    auto frameInterval = duration_cast<nanoseconds>(duration<double>(1.0 / 144.0));
    auto nextFrameTime = high_resolution_clock::now();

    while (true) {
        auto now = high_resolution_clock::now();
        if (now < nextFrameTime) {
            std::this_thread::sleep_for(nextFrameTime - now);
        }
        nextFrameTime += frameInterval;

        // Capture the screen
        captureScreen(hScreenDC, hMemoryDC, hBitmap, x, y, captureWidth, captureHeight, frame);

        if (frame.empty()) {
            std::cerr << "Error: Captured empty frame." << std::endl;
            continue;
        }

        std::vector<int> classIds;
        std::vector<float> confidences;

        const auto detections = yoloV8.detectObjects(frame);

        // Aim at the closest CH/TH detection
        mouseController.aim(detections);
        mouseController.triggerLeftClickIfCenterWithinDetection(detections);

        yoloV8.drawObjectLabels(frame, detections, 1, 30);

        cv::imshow("Detections", frame);

        // Break the loop on ESC key press
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // Release resources
    releaseScreenCapture(hScreenDC, hMemoryDC, hBitmap);

    cv::destroyAllWindows();

    return 0;
}
