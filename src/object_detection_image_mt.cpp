#define NOMINMAX

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include "DXGICapture.h"
#include "DXGICaptureCUDA.h"
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
    static void logAverageLatencies(LatencyQueue &captureLatency, LatencyQueue &detectionLatency,
                                    LatencyQueue &renderLatency, SafeQueue<std::string> &logQueue);

    void captureThread();
    void detectionThread();
    void captureThreadGpu();
    void detectionThreadGpu();

    INIParser config;
    std::unique_ptr<YoloV8> yoloV8;
    std::unique_ptr<DXGICapture> capture;
    std::unique_ptr<DXGICaptureCUDA> captureCUDA;
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
    LatencyQueue captureLatency, detectionLatency, renderLatency;
    std::atomic<bool> running;
    FrameQueue captureQueue;
    GpuFrameQueue gpuCaptureQueue;
    DetectionQueue detectionQueue;

    std::thread captureThreadObj;
    std::thread detectionThreadObj;

    bool useFusion;
    bool pinThreads;
    bool trackCrosshair;
    bool debugView;
    bool useDirectGpuCapture;
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

    // c39c: opt into the DXGI/CUDA-interop capture path. When true the capture stays on the
    // GPU end-to-end (no CPU map, no cvtColor, no upload). Default false until validated.
    useDirectGpuCapture = config.getBool("UseDirectGpuCapture", false);
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
    // HID identity — defaults point at the ESP32-P4 bridge cloned as the OP1
    // 8K V2 (VID 0x3367 PID 0x1978, no iSerialNumber). To target the old
    // RP2040 instead, override HidVendorId/HidProductId/HidSerial in the INI.
    const uint16_t hidVid = static_cast<uint16_t>(config.getInt("HidVendorId", 0x3367));
    const uint16_t hidPid = static_cast<uint16_t>(config.getInt("HidProductId", 0x1978));
    const std::string hidSerialNarrow = config.getString("HidSerial", "");
    std::wstring hidSerialWide(hidSerialNarrow.begin(), hidSerialNarrow.end()); // ASCII-only serials

    mouseController = std::make_unique<MouseController>(
        screenWidth, screenHeight, captureWidth, captureHeight, config.getFloat("MouseSensitivity", 0.80f), config.getInt("AimFOV", 55),
        config.getFloat("MinGain", 0.25f), config.getFloat("MaxGain", 0.65f), config.getInt("MaxSpeed", 15),
        config.getInt("HeadLabelID1", 0), config.getInt("HeadLabelID2", 1), config.getInt("CPI", 3000),
        static_cast<int>(config.getStringArray("Labels").size()),
        yoloConfig.probabilityThreshold,
        hidVid, hidPid, std::move(hidSerialWide),
        config.getFloat("Smoothing", 5.0f),
        config.getFloat("DebugSnapGain", 3.0f),
        config.getBool("DebugAimEnabled", true));

    if (trackCrosshair) {
        templateImg = cv::imread(config.getString("CrosshairTemplate", "crosshair.png"), cv::IMREAD_COLOR);
        if (templateImg.empty()) {
            throw std::runtime_error("Failed to load template image");
        }
        gpuCrosshairTemplate.upload(templateImg);
        crosshairMatcher = cv::cuda::createTemplateMatching(templateImg.type(), cv::TM_CCOEFF_NORMED);
    }

    if (useDirectGpuCapture) {
        // TrackCrosshair runs a CPU-side cv::Mat template match on the small ROI; the direct-GPU
        // capture path keeps the frame as a BGRA GpuMat throughout, with no CPU snapshot, so the
        // combination is genuinely broken. The Discord debug overlay, OTOH, only consumes
        // detection boxes / crosshair coords / latency stats — no frame data — so it composes
        // cleanly with the GPU path. (c40)
        if (trackCrosshair) {
            throw std::runtime_error(
                "UseDirectGpuCapture is incompatible with TrackCrosshair (CPU template path). Disable one.");
        }
        captureCUDA = std::make_unique<DXGICaptureCUDA>();
    } else {
        capture = std::make_unique<DXGICapture>();
    }

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
    if (useDirectGpuCapture) {
        captureThreadObj = std::thread(&ObjectDetectionSystem::captureThreadGpu, this);
        detectionThreadObj = std::thread(&ObjectDetectionSystem::detectionThreadGpu, this);
    } else {
        captureThreadObj = std::thread(&ObjectDetectionSystem::captureThread, this);
        detectionThreadObj = std::thread(&ObjectDetectionSystem::detectionThread, this);
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

        logAverageLatencies(captureLatency, detectionLatency, renderLatency, logQueue);
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
                                                LatencyQueue &renderLatency, SafeQueue<std::string> &logQueue) {
    double avgCaptureLatency = captureLatency.getAverageLatency();
    double avgDetectionLatency = detectionLatency.getAverageLatency();
    double avgRenderLatency = renderLatency.getAverageLatency();

    std::cout << "\rCapture: " << avgCaptureLatency << "ms | "
              << "Detection: " << avgDetectionLatency << "ms | "
              << "Render: " << avgRenderLatency << "ms       " << std::flush;
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
                auto renderStart = std::chrono::high_resolution_clock::now();
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
                auto renderEnd = std::chrono::high_resolution_clock::now();
                renderLatency.push(std::chrono::duration_cast<std::chrono::milliseconds>(renderEnd - renderStart).count());
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

// c39c: GPU capture thread. Owns a private CUDA stream so the cudaMemcpy2DFromArrayAsync
// inside DXGICaptureCUDA::CaptureScreen runs concurrently with the detection thread's engine
// stream. Synchronizes the local stream before pushing so the consumer reads stable data.
void ObjectDetectionSystem::captureThreadGpu() {
    try {
        cudaStream_t captureStream = nullptr;
        cudaStreamCreate(&captureStream);

        cv::cuda::GpuMat frame1;
        cv::cuda::GpuMat frame2;
        cv::cuda::GpuMat *currentFrame = &frame1;
        cv::cuda::GpuMat *nextFrame = &frame2;

        while (running) {
            auto start = std::chrono::high_resolution_clock::now();

            const bool got = captureCUDA->CaptureScreen(*currentFrame, captureStream);
            if (got) {
                cudaStreamSynchronize(captureStream);
                gpuCaptureQueue.push(std::move(*currentFrame));
                std::swap(currentFrame, nextFrame);
            }

            auto end = std::chrono::high_resolution_clock::now();
            captureLatency.push(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        }

        cudaStreamDestroy(captureStream);
    } catch (const std::exception &e) {
        std::cerr << "captureThreadGpu: fatal exception: " << e.what() << ". Stopping." << std::endl;
        running = false;
    } catch (...) {
        std::cerr << "captureThreadGpu: fatal unknown exception. Stopping." << std::endl;
        running = false;
    }
}

void ObjectDetectionSystem::detectionThreadGpu() {
    try {
        while (running) {
            cv::cuda::GpuMat frame = gpuCaptureQueue.pop();

            auto detectionStartTime = std::chrono::high_resolution_clock::now();

            const int centerX = frame.cols / 2;
            const int centerY = frame.rows / 2;

            int x = std::max(centerX - captureWidth / 2, 0);
            int y = std::max(centerY - captureHeight / 2, 0);
            x = std::min(x, frame.cols - captureWidth);
            y = std::min(y, frame.rows - captureHeight);
            const cv::Rect roi(x, y, captureWidth, captureHeight);

            cv::cuda::GpuMat croppedFrame = frame(roi); // ROI view, no copy

            // Crosshair tracking is refused at init when direct GPU capture is on (CPU template
            // match needs a cv::Mat). The overlay only consumes detection metadata, so it works.
            const cv::Point crosshairPos(captureWidth / 2, captureHeight / 2);

            std::vector<Object> detections = yoloV8->detectObjects(croppedFrame);

            mouseController->setCrosshairPosition(crosshairPos.x, crosshairPos.y);
            mouseController->aim(detections);
            mouseController->triggerLeftClickIfCenterWithinDetection(detections);

            detectionQueue.push(detections);

            auto detectionEndTime = std::chrono::high_resolution_clock::now();
            detectionLatency.push(
                std::chrono::duration_cast<std::chrono::milliseconds>(detectionEndTime - detectionStartTime).count());

            // c40: same overlay-publish block as the legacy detectionThread. Translates from
            // cropped-ROI coords into game-framebuffer coords (assumes fullscreen-native).
            if (debugOverlay && debugOverlay->isRunning()) {
                auto renderStart = std::chrono::high_resolution_clock::now();
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
                auto renderEnd = std::chrono::high_resolution_clock::now();
                renderLatency.push(
                    std::chrono::duration_cast<std::chrono::milliseconds>(renderEnd - renderStart).count());
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "detectionThreadGpu: fatal exception: " << e.what() << ". Stopping." << std::endl;
        running = false;
    } catch (...) {
        std::cerr << "detectionThreadGpu: fatal unknown exception. Stopping." << std::endl;
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