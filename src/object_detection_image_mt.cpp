#define NOMINMAX

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include "DXGICapture.h"
#include "DiscordOverlay.h"
#include "INIParser.h" // Make sure to include the INI parser we created earlier
#include "MouseController.h"
#include "threadsafe_queue.h"
#include "yolov8.h"
#include "SoftwareFuser.h"

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

    static void moveToTop();
    static void clearScreen();
    static void logAverageLatencies(LatencyQueue &captureLatency, LatencyQueue &detectionLatency, SafeQueue<std::string> &logQueue);

    void captureThread();
    void detectionThread();

    INIParser config;
    std::unique_ptr<YoloV8> yoloV8;
    std::unique_ptr<DXGICapture> capture;
    std::unique_ptr<MouseController> mouseController;
    std::unique_ptr<DiscordOverlay> debugOverlay;
    std::vector<std::string> labelNames;

    cv::Mat templateImg;
    // GPU template-matching state for crosshair tracking (c27). Lives on the detection thread;
    // single-consumer so no synchronization needed.
    cv::Ptr<cv::cuda::TemplateMatching> crosshairMatcher;
    cv::cuda::GpuMat gpuCrosshairTemplate;
    cv::cuda::GpuMat gpuCrosshairRoi;
    cv::cuda::GpuMat gpuCrosshairResult;
    int captureWidth;
    int captureHeight;
    int captureFPS;
    int screenWidth;
    int screenHeight;
    int captureThreadCore;
    HWND targetWnd;

    SafeQueue<std::string> logQueue;
    LatencyQueue captureLatency, detectionLatency;
    std::atomic<bool> running;
    FrameQueue captureQueue;
    DetectionQueue detectionQueue;

    std::thread captureThreadObj;
    std::thread detectionThreadObj;

    bool useFusion;
    bool pinThreads;
    bool trackCrosshair;
    bool debugView;
    std::string debugOverlayTargetProcess;
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
    useFusion = config.getBool("UseFusion", false);
    captureFPS = config.getInt("CaptureFPS", 240);

    // Thread settings
    pinThreads = config.getBool("PinThreads", false);
    captureThreadCore = config.getInt("CaptureThreadCore", 0);

    // Debug viewer (c29 -> c30) — INI-gated Discord-overlay debug renderer for verifying
    // detection visually. Off by default. Requires the legacy Discord in-game overlay
    // enabled and Discord running. Cost when on: ~50–150 KB/frame memcpy + a render thread
    // at ~120 Hz; the detection thread itself only does cheap atomics + a mutex'd vector copy.
    debugView                 = config.getBool("DebugView", false);
    debugOverlayTargetProcess = config.getString("DebugOverlayTargetProcess", "cs2.exe");
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
    // HID identity (c15) — defaults preserve the originally-hardcoded device.
    const uint16_t hidVid = static_cast<uint16_t>(config.getInt("HidVendorId", 0x0812));
    const uint16_t hidPid = static_cast<uint16_t>(config.getInt("HidProductId", 0x2205));
    const std::string hidSerialNarrow = config.getString("HidSerial", "DF625857C74E132B");
    std::wstring hidSerialWide(hidSerialNarrow.begin(), hidSerialNarrow.end()); // ASCII-only serials

    mouseController = std::make_unique<MouseController>(
        screenWidth, screenHeight, captureWidth, captureHeight, config.getFloat("MouseSensitivity", 0.80f), config.getInt("AimFOV", 55),
        config.getFloat("MinGain", 0.25f), config.getFloat("MaxGain", 0.65f), config.getInt("MaxSpeed", 15),
        config.getInt("HeadLabelID1", 0), config.getInt("HeadLabelID2", 1), config.getInt("CPI", 3000),
        static_cast<int>(config.getStringArray("Labels").size()),
        yoloConfig.probabilityThreshold,
        hidVid, hidPid, std::move(hidSerialWide));

    if (trackCrosshair) {
        templateImg = cv::imread(config.getString("CrosshairTemplate", "crosshair.png"), cv::IMREAD_COLOR);
        if (templateImg.empty()) {
            throw std::runtime_error("Failed to load template image");
        }
        gpuCrosshairTemplate.upload(templateImg);
        crosshairMatcher = cv::cuda::createTemplateMatching(templateImg.type(), cv::TM_CCOEFF_NORMED);
    }

    capture = std::make_unique<DXGICapture>();

    detectionQueue.setMoveThresholdPx(config.getInt("DetectionMoveThresholdPx", 5));

    labelNames = config.getStringArray("Labels");

    if (debugView) {
        const DWORD pid = DiscordOverlay::findProcessIdByName(debugOverlayTargetProcess);
        if (pid == 0) {
            std::cerr << "DebugView: target process '" << debugOverlayTargetProcess
                      << "' not running; overlay disabled. Launch the game first or set"
                         " DebugOverlayTargetProcess in the INI."
                      << std::endl;
        } else {
            debugOverlay = std::make_unique<DiscordOverlay>(pid);
            if (!debugOverlay->start()) {
                std::cerr << "DebugView: Discord overlay failed to start; continuing without overlay."
                          << std::endl;
                debugOverlay.reset();
            } else {
                debugOverlay->setLabelNames(labelNames);
            }
        }
    }
}

void ObjectDetectionSystem::startThreads() {
    captureThreadObj = std::thread(&ObjectDetectionSystem::captureThread, this);
    detectionThreadObj = std::thread(&ObjectDetectionSystem::detectionThread, this);
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

        logAverageLatencies(captureLatency, detectionLatency, logQueue);
        Sleep(100);
    }
}

void ObjectDetectionSystem::cleanup() {
    logQueue.push("exit");
    if (debugOverlay) {
        debugOverlay->stop();
        debugOverlay.reset();
    }
    if (captureThreadObj.joinable())
        captureThreadObj.join();
    if (detectionThreadObj.joinable())
        detectionThreadObj.join();
}

void ObjectDetectionSystem::run() {
    startThreads();
    mainLoop();
    cleanup();
}

void ObjectDetectionSystem::moveToTop() { std::cout << "\033[H"; }

void ObjectDetectionSystem::clearScreen() { std::cout << "\033[2J\033[H"; }

void ObjectDetectionSystem::logAverageLatencies(LatencyQueue &captureLatency, LatencyQueue &detectionLatency,
                                                SafeQueue<std::string> &logQueue) {
    double avgCaptureLatency = captureLatency.getAverageLatency();
    double avgDetectionLatency = detectionLatency.getAverageLatency();

    std::cout << "\rAverage Capture Latency: " << avgCaptureLatency << "ms | "
              << "Detection Latency: " << avgDetectionLatency << "ms" << std::flush;
}

void ObjectDetectionSystem::captureThread() {
    try {
        const int targetFPS = config.getInt("CaptureFPS", 1000);
        const std::chrono::nanoseconds frameDuration(1000000000 / targetFPS);
        cv::Mat frame1(captureHeight, captureWidth, CV_8UC3);
        cv::Mat frame2(captureHeight, captureWidth, CV_8UC3);
        cv::Mat *currentFrame = &frame1;
        cv::Mat *nextFrame = &frame2;

        while (running) {
            auto start = std::chrono::high_resolution_clock::now();

            bool got = capture->CaptureScreen(*currentFrame);

            if (got && !currentFrame->empty()) {
                captureQueue.push(std::move(*currentFrame));
                std::swap(currentFrame, nextFrame);
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = end - start;

            captureLatency.push(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
        }
    } catch (const std::exception &e) {
        std::cerr << "captureThread: fatal exception: " << e.what() << ". Stopping." << std::endl;
        running = false;
    } catch (...) {
        std::cerr << "captureThread: fatal unknown exception. Stopping." << std::endl;
        running = false;
    }
}

void ObjectDetectionSystem::detectionThread() {
    try {
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
                gpuCrosshairRoi.upload(smallCroppedFrame);
                crosshairMatcher->match(gpuCrosshairRoi, gpuCrosshairTemplate, gpuCrosshairResult);
                double maxVal = 0.0;
                cv::Point maxLoc;
                cv::cuda::minMaxLoc(gpuCrosshairResult, nullptr, &maxVal, nullptr, &maxLoc);

                crosshairPos.x = maxLoc.x + templateImg.cols / 2 + smallX - x;
                crosshairPos.y = maxLoc.y + templateImg.rows / 2 + smallY - y;
            } else {
                crosshairPos = cv::Point(roiWidth / 2, roiHeight / 2);
            }

            std::vector<Object> detections;
            detections = yoloV8->detectObjects(croppedFrame);

            mouseController->setCrosshairPosition(crosshairPos.x, crosshairPos.y);
            mouseController->aim(detections);
            mouseController->triggerLeftClickIfCenterWithinDetection(detections);

            detectionQueue.push(detections);

            auto detectionEndTime = std::chrono::high_resolution_clock::now();
            auto detectionDuration = detectionEndTime - detectionStartTime;

            detectionLatency.push(std::chrono::duration_cast<std::chrono::milliseconds>(detectionDuration).count());

            if (debugOverlay && debugOverlay->isRunning()) {
                // Translate detections + crosshair from cropped-ROI space to game-framebuffer
                // space. For fullscreen play at native resolution, framebuffer == screen.
                std::vector<DiscordOverlay::DetectionBox> boxes;
                boxes.reserve(detections.size());
                for (const auto &d : detections) {
                    DiscordOverlay::DetectionBox b{};
                    b.x          = d.rect.x + static_cast<float>(x);
                    b.y          = d.rect.y + static_cast<float>(y);
                    b.w          = d.rect.width;
                    b.h          = d.rect.height;
                    b.label      = d.label;
                    b.confidence = d.probability;
                    boxes.push_back(b);
                }
                debugOverlay->setDetections(std::move(boxes));
                debugOverlay->setCrosshair(static_cast<float>(crosshairPos.x + x),
                                            static_cast<float>(crosshairPos.y + y));
                debugOverlay->setStats(detectionLatency.getAverageLatency(), 0);
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "detectionThread: fatal exception: " << e.what() << ". Stopping." << std::endl;
        running = false;
    } catch (...) {
        std::cerr << "detectionThread: fatal unknown exception. Stopping." << std::endl;
        running = false;
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