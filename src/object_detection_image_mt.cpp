#define NOMINMAX

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include "DXGICapture.h"
#include "INIParser.h" // Make sure to include the INI parser we created earlier
#include "MouseController.h"
#include "threadsafe_queue.h"
#include "yolov8.h"
#include "SoftwareFuser.h"
#include "D3D11Overlay.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <variant>
#include <windows.h>

using namespace std::chrono;

class ObjectDetectionSystem {
public:
    ObjectDetectionSystem(const std::string &iniFile);
    void run();

private:
    void loadConfigFromINI(const std::string &iniFile);
    void initializeSystem();
    void startThreads();
    void mainLoop();
    void cleanup();

    static cv::Point detectCrosshairCPU(const cv::Mat &frame, const cv::Mat &templ);
    static void moveToTop();
    static void clearScreen();
    static void logAverageLatencies(LatencyQueue &captureLatency, LatencyQueue &detectionLatency, LatencyQueue &overlayLatency,
                                    SafeQueue<std::string> &logQueue);

    void captureThread();
    void detectionThread();
    void overlayThread();

    INIParser config;
    std::unique_ptr<YoloV8> yoloV8;
    std::unique_ptr<DXGICapture> capture;
    std::unique_ptr<MouseController> mouseController;

    cv::Mat templateImg;
    int captureWidth;
    int captureHeight;
    int overlayX;
    int overlayY;
    int captureFPS;
    int displayFPS;
    int screenWidth;
    int screenHeight;
    int displayThreadCore;
    int captureThreadCore;
    HWND targetWnd;

    SafeQueue<std::string> logQueue;
    LatencyQueue captureLatency, detectionLatency, overlayLatency;
    std::atomic<bool> running;
    FrameQueue captureQueue;
    DetectionQueue detectionQueue;

    std::thread captureThreadObj;
    std::thread detectionThreadObj;
    std::thread overlayThreadObj;

    bool useOverlay;
    bool useFusion;
    bool pinThreads;
    bool trackCrosshair;
};

ObjectDetectionSystem::ObjectDetectionSystem(const std::string &iniFile) : running(true) {
    loadConfigFromINI(iniFile);
    initializeSystem();
}

void ObjectDetectionSystem::loadConfigFromINI(const std::string &iniFile) {
    if (!config.loadFile(iniFile)) {
        throw std::runtime_error("Failed to load INI file: " + iniFile);
    }

    captureWidth = config.getInt("CaptureWidth", 640);
    captureHeight = config.getInt("CaptureHeight", 640);
    trackCrosshair = config.getBool("TrackCrosshair", false);
    useOverlay = config.getBool("UseOverlay", false);
    useFusion = config.getBool("UseFusion", false);
    captureFPS = config.getInt("CaptureFPS", 240);
    displayFPS = config.getInt("DisplayFPS", 144);

    // Thread settings
    pinThreads = config.getBool("PinThreads", false);
    captureThreadCore = config.getInt("CaptureThreadCore", 0);
    displayThreadCore = config.getInt("DisplayThreadCore", 1);

    // Display settings
    overlayX = config.getInt("OverlayX", 0);
    overlayY = config.getInt("OverlayY", 0);
}

void ObjectDetectionSystem::initializeSystem() {
    YoloV8Config yoloConfig;
    yoloConfig.classNames = config.getStringArray("Labels");

    // Modify this section
    std::string precision = config.getString("Precision", "float");
    if (precision == "float") {
        yoloConfig.precision = Precision::FP32;
    } else if (precision == "half") {
        yoloConfig.precision = Precision::FP16;
    } else {
        throw std::runtime_error("Invalid precision in INI file. Use 'float' or 'half'.");
    }

    // Create YoloV8 directly - it will use the factory internally
    yoloV8 = std::make_unique<YoloV8>(config.getString("ModelPath"), yoloConfig);

    screenWidth = GetSystemMetrics(SM_CXSCREEN);
    screenHeight = GetSystemMetrics(SM_CYSCREEN);
    mouseController = std::make_unique<MouseController>(
        screenWidth, screenHeight, captureWidth, captureHeight, config.getFloat("MouseSensitivity", 0.80f), config.getInt("AimFOV", 55),
        config.getFloat("MinGain", 0.25f), config.getFloat("MaxGain", 0.65f), config.getInt("MaxSpeed", 15),
        config.getInt("HeadLabelID1", 0), config.getInt("HeadLabelID2", 1), config.getInt("CPI", 3000), config.getStringArray("Labels").size());

    if (trackCrosshair) {
        templateImg = cv::imread(config.getString("CrosshairTemplate", "crosshair.png"), cv::IMREAD_COLOR);
        if (templateImg.empty()) {
            throw std::runtime_error("Failed to load template image");
        }
    }

    capture = std::make_unique<DXGICapture>();
}

void ObjectDetectionSystem::startThreads() {
    captureThreadObj = std::thread(&ObjectDetectionSystem::captureThread, this);
    detectionThreadObj = std::thread(&ObjectDetectionSystem::detectionThread, this);
    if (useOverlay) {
        overlayThreadObj = std::thread(&ObjectDetectionSystem::overlayThread, this);
    }
}

void ObjectDetectionSystem::mainLoop() {
    clearScreen();
    std::cout << "OpenCV CUDA support: " << cv::cuda::getCudaEnabledDeviceCount() << " devices" << std::endl;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    while (running) {
        MSG msg = {};

        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        if (GetAsyncKeyState(VK_INSERT) & 0x8000) {
            running = false;
        }

        logAverageLatencies(captureLatency, detectionLatency, overlayLatency, logQueue);
        Sleep(100);
    }
}

void ObjectDetectionSystem::cleanup() {
    logQueue.push("exit");
    cv::destroyAllWindows();
    if (captureThreadObj.joinable())
        captureThreadObj.join();
    if (detectionThreadObj.joinable())
        detectionThreadObj.join();
    if (overlayThreadObj.joinable())
        overlayThreadObj.join();
}

void ObjectDetectionSystem::run() {
    startThreads();
    mainLoop();
    cleanup();
}

cv::Point ObjectDetectionSystem::detectCrosshairCPU(const cv::Mat &frame, const cv::Mat &templ) {
    cv::Mat result;
    cv::matchTemplate(frame, templ, result, cv::TM_CCOEFF_NORMED);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    cv::Point crosshairCenter(maxLoc.x + templ.cols / 2, maxLoc.y + templ.rows / 2);
    cv::rectangle(frame, maxLoc, cv::Point(maxLoc.x + templ.cols, maxLoc.y + templ.rows), cv::Scalar(0, 255, 0), 2);

    return crosshairCenter;
}

void ObjectDetectionSystem::moveToTop() { std::cout << "\033[H"; }

void ObjectDetectionSystem::clearScreen() { std::cout << "\033[2J\033[H"; }

void ObjectDetectionSystem::logAverageLatencies(LatencyQueue &captureLatency, LatencyQueue &detectionLatency, LatencyQueue &overlayLatency,
                                                SafeQueue<std::string> &logQueue) {
    double avgCaptureLatency = captureLatency.getAverageLatency();
    double avgDetectionLatency = detectionLatency.getAverageLatency();
    double avgOverlayLatency = overlayLatency.getAverageLatency();

    std::cout << "\rAverage Capture Latency: " << avgCaptureLatency << "ms | "
              << "Detection Latency: " << avgDetectionLatency << "ms | "
              << "Overlay Latency: " << avgOverlayLatency << "ms" << std::flush;
}

void ObjectDetectionSystem::captureThread() {
    const int targetFPS = config.getInt("CaptureFPS", 1000);
    const std::chrono::nanoseconds frameDuration(1000000000 / targetFPS);
    cv::Mat frame1(captureHeight, captureWidth, CV_8UC3);
    cv::Mat frame2(captureHeight, captureWidth, CV_8UC3);
    cv::Mat *currentFrame = &frame1;
    cv::Mat *nextFrame = &frame2;

    while (running) {
        auto start = std::chrono::high_resolution_clock::now();

        capture->CaptureScreen(*currentFrame);

        if (!currentFrame->empty()) {
            captureQueue.push(std::move(*currentFrame));
        }

        std::swap(currentFrame, nextFrame);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = end - start;

        captureLatency.push(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
    }
}

void ObjectDetectionSystem::detectionThread() {
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

        cv::Mat croppedFrame = frame(roi);

        cv::Point crosshairPos;
        if (trackCrosshair) {
            int smallRoiSize = config.getInt("SmallRoiSize", 160);
            int smallX = std::max(centerX - smallRoiSize / 2, 0);
            int smallY = std::max(centerY - smallRoiSize / 2, 0);
            smallX = std::min(smallX, frame.cols - smallRoiSize);
            smallY = std::min(smallY, frame.rows - smallRoiSize);
            cv::Rect smallRoi(smallX, smallY, smallRoiSize, smallRoiSize);

            cv::Mat smallCroppedFrame = frame(smallRoi);
            crosshairPos = detectCrosshairCPU(smallCroppedFrame, templateImg);
            crosshairPos.x += smallX - x;
            crosshairPos.y += smallY - y;
        } else {
            crosshairPos = cv::Point(roiWidth / 2, roiHeight / 2);
        }

        std::vector<Object> detections;
        detections = yoloV8->detectObjects(croppedFrame);
        yoloV8->drawObjectLabels(croppedFrame, detections);
        cv::imshow("Detection", croppedFrame);
        cv::waitKey(1);

        mouseController->setCrosshairPosition(crosshairPos.x, crosshairPos.y);
        mouseController->aim(detections);
        mouseController->triggerLeftClickIfCenterWithinDetection(detections);

        detectionQueue.push(detections);

        auto detectionEndTime = std::chrono::high_resolution_clock::now();
        auto detectionDuration = detectionEndTime - detectionStartTime;

        detectionLatency.push(std::chrono::duration_cast<std::chrono::milliseconds>(detectionDuration).count());
    }
}

void ObjectDetectionSystem::overlayThread() {
    if (!useOverlay)
        return;

    // Initialize overlay with screen dimensions and position
    D3D11Overlay overlay(screenWidth, screenHeight);
    overlay.setPosition(overlayX, overlayY);
    overlay.setClickthrough(true);

    // Set thread priority and affinity if configured
    if (pinThreads) {
        SetThreadAffinityMask(GetCurrentThread(), 1ULL << displayThreadCore);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    }

    // Calculate frame timing for desired FPS
    const auto frameTime = std::chrono::nanoseconds(1000000000 / displayFPS);
    auto lastFrameTime = std::chrono::steady_clock::now();

    while (running) {
        auto frameStart = std::chrono::steady_clock::now();

        // Get latest detections
        const std::vector<Object> &detections = detectionQueue.pop();

        // Draw detections
        overlay.drawDetections(detections);

        // Prepare and draw log
        std::stringstream logMessage;
        logMessage << "Detections: " << detections.size() << "\n";
        for (size_t i = 0; i < std::min(detections.size(), size_t(5)); ++i) {
            logMessage << "D" << i << ": L" << detections[i].label << " C" << std::fixed << std::setprecision(2)
                       << detections[i].probability << "\n";
        }

        // Add performance metrics to log
        logMessage << "Capture: " << captureLatency.getAverageLatency() << "ms\n"
                   << "Detection: " << detectionLatency.getAverageLatency() << "ms";

        overlay.drawLog(logMessage.str());
        overlay.render();

        // Frame timing and latency tracking
        auto frameEnd = std::chrono::steady_clock::now();
        auto workDuration = frameEnd - frameStart;
        auto elapsed = frameEnd - lastFrameTime;

        overlayLatency.push(std::chrono::duration_cast<std::chrono::milliseconds>(workDuration).count());

        // Precise timing for target FPS
        if (elapsed < frameTime) {
            auto sleepTime = frameTime - elapsed;
            if (sleepTime > std::chrono::microseconds(500)) {
                std::this_thread::sleep_for(sleepTime - std::chrono::microseconds(500));
            }
            while (std::chrono::steady_clock::now() - lastFrameTime < frameTime) {
                _mm_pause();
            }
        }

        lastFrameTime = std::chrono::steady_clock::now();

        // Check for exit condition
        if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
            running = false;
        }
    }
}

int main(int argc, char *argv[]) {
    try {
        if (argc != 2) {
            throw std::runtime_error("Usage: " + std::string(argv[0]) + " <path to INI file>");
        }
        ObjectDetectionSystem system(argv[1]);
        system.run();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}