#define NOMINMAX

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include "DXGICaptureCUDA.h"
#include "DiscordOverlay.h"
#include "DetectorFactory.h"
#include "INIParser.h"
#include "MetricsWriter.h"
#include "MouseController.h"
#include "threadsafe_queue.h"
#include "yolov8.h"
#include "YoloV11.h"
#include "YoloV26.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <windows.h>

using namespace std::chrono;

class ObjectDetectionSystem {
public:
    ObjectDetectionSystem(const std::string &iniFile);
    void run();

private:
    void loadConfigFromINI(const std::string &iniFile);
    void initializeSystem();
    void mainLoop();
    void cleanup();

    static void logAverageLatencies(LatencyQueue &captureLatency, LatencyQueue &detectionLatency,
                                    LatencyQueue &renderLatency);

    void captureThread();
    void detectionThread();

    INIParser config;
    std::unique_ptr<YoloDetector> yoloDetector;
    std::unique_ptr<DXGICaptureCUDA> capture;
    std::unique_ptr<MouseController> mouseController;
    std::unique_ptr<DiscordOverlay> debugOverlay;
    std::vector<std::string> labelNames;

    int captureWidth;
    int captureHeight;
    int captureFPS;
    int screenWidth;
    int screenHeight;
    int captureThreadCore;
    HWND targetWnd;

    LatencyQueue captureLatency, detectionLatency, renderLatency;
    std::atomic<bool> running;
    GpuFrameQueue gpuCaptureQueue;
    DetectionQueue detectionQueue;

    std::thread captureThreadObj;
    std::thread detectionThreadObj;

    bool pinThreads;
    bool debugView;
    std::string debugOverlayTargetProcess;

    // Monitoring
    MetricsWriter m_metrics;
    std::string m_metricsPath;
    int m_detectionCountAccum = 0;
    int m_frameCountAccum = 0;
};

ObjectDetectionSystem::ObjectDetectionSystem(const std::string &iniFile) : running(true) {
    loadConfigFromINI(iniFile);
    initializeSystem();
    std::cout << "[CTOR] initializeSystem complete, entering run()" << std::endl;
}

void ObjectDetectionSystem::loadConfigFromINI(const std::string &iniFile) {
    if (!config.loadFile(iniFile)) {
        throw std::runtime_error("Failed to load INI file: " + iniFile);
    }

    captureWidth  = config.getInt("CaptureWidth", 640);
    captureHeight = config.getInt("CaptureHeight", 640);
    captureFPS    = config.getInt("CaptureFPS", 240);

    pinThreads         = config.getBool("PinThreads", false);
    captureThreadCore  = config.getInt("CaptureThreadCore", 0);

    debugView                 = config.getBool("DebugView", false);
    debugOverlayTargetProcess = config.getString("DebugOverlayTargetProcess", "cs2.exe");

    m_metricsPath = config.getString("MetricsStatus", "");
    if (!m_metricsPath.empty()) {
        m_metrics.open(m_metricsPath);
    }
}

void ObjectDetectionSystem::initializeSystem() {
    YoloConfig yoloConfig;
    yoloConfig.classNames = config.getStringArray("Labels");

    std::string precision = config.getString("Precision", "half");
    if (precision == "float") {
        yoloConfig.precision = Precision::FP32;
    } else if (precision == "half") {
        yoloConfig.precision = Precision::FP16;
    } else {
        throw std::runtime_error("Invalid precision in INI file. Use 'float' or 'half'.");
    }

    YoloVersion modelVersion = YoloVersion::AUTO;
    const std::string versionStr = config.getString("ModelVersion", "auto");
    if (versionStr == "v8" || versionStr == "V8")       modelVersion = YoloVersion::V8;
    else if (versionStr == "v11" || versionStr == "V11") modelVersion = YoloVersion::V11;
    else if (versionStr == "v26" || versionStr == "V26") modelVersion = YoloVersion::V26;

    std::cout << "[INIT] Creating YOLO detector..." << std::endl;
    yoloDetector = DetectorFactory::create(config.getString("ModelPath"), yoloConfig, modelVersion);
    std::cout << "[INIT] YOLO detector OK" << std::endl;

    screenWidth  = GetSystemMetrics(SM_CXSCREEN);
    screenHeight = GetSystemMetrics(SM_CYSCREEN);

    const uint16_t hidVid = static_cast<uint16_t>(config.getInt("HidVendorId", 0x3367));
    const uint16_t hidPid = static_cast<uint16_t>(config.getInt("HidProductId", 0x1978));
    const std::string hidSerialNarrow = config.getString("HidSerial", "");
    std::wstring hidSerialWide(hidSerialNarrow.begin(), hidSerialNarrow.end());

    mouseController = std::make_unique<MouseController>(
        screenWidth, screenHeight, captureWidth, captureHeight,
        config.getFloat("MouseSensitivity", 0.80f), config.getInt("AimFOV", 55),
        config.getFloat("MinGain", 0.25f), config.getFloat("MaxGain", 0.65f),
        config.getInt("MaxSpeed", 15),
        config.getInt("HeadLabelID1", 0), config.getInt("HeadLabelID2", 1),
        config.getInt("CPI", 3000),
        static_cast<int>(config.getStringArray("Labels").size()),
        yoloConfig.probabilityThreshold,
        hidVid, hidPid, std::move(hidSerialWide),
        config.getFloat("Smoothing", 5.0f),
        config.getFloat("DebugSnapGain", 3.0f),
        config.getBool("DebugAimEnabled", true));

    // Pixel-perfect FOV-based calibration.
    {
        const float gameFov = config.getFloat("GameFOV", 90.0f);
        const std::string modelPath = config.getString("ModelPath");
        std::string game;
        if (modelPath.find("cs2") != std::string::npos || modelPath.find("CS2") != std::string::npos) {
            game = "CS2";
        } else if (modelPath.find("val") != std::string::npos || modelPath.find("Val") != std::string::npos
                   || modelPath.find("VAL") != std::string::npos) {
            game = "VALORANT";
        }
        if (!game.empty()) {
            mouseController->setGameCalibration(config.getFloat("MouseSensitivity", 0.80f), gameFov, game);
        }
    }

    capture = std::make_unique<DXGICaptureCUDA>();

    detectionQueue.setMoveThresholdPx(config.getInt("DetectionMoveThresholdPx", 5));

    labelNames = config.getStringArray("Labels");

    if (debugView) {
        const DWORD pid = DiscordOverlay::findProcessIdByName(debugOverlayTargetProcess);
        if (pid == 0) {
            std::cerr << "DebugView: target process '" << debugOverlayTargetProcess
                      << "' not running; overlay disabled." << std::endl;
        } else {
            debugOverlay = std::make_unique<DiscordOverlay>(pid);
            if (!debugOverlay->start()) {
                std::cerr << "DebugView: Discord overlay failed to start." << std::endl;
                debugOverlay.reset();
            } else {
                debugOverlay->setLabelNames(labelNames);
            }
        }
    }
}

void ObjectDetectionSystem::mainLoop() {
    std::cout << "OpenCV CUDA support: " << cv::cuda::getCudaEnabledDeviceCount() << " devices" << std::endl;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::cout.flush();

    std::cout << "Waiting for first captured frame..." << std::endl;
    for (int i = 0; i < 50 && running; i++) {
        if (captureLatency.getAverageLatency() > 0.0 || detectionLatency.getAverageLatency() > 0.0) {
            std::cout << "Capture: OK (frames flowing)" << std::endl;
            break;
        }
        Sleep(100);
    }
    if (captureLatency.getAverageLatency() <= 0.0 && detectionLatency.getAverageLatency() <= 0.0) {
        std::cout << "Capture: BLOCKED (no frames after 5s)" << std::endl;
    }

    int metricsTick = 0;

    while (running) {
        MSG msg = {};
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        if (GetAsyncKeyState(VK_INSERT) & 0x8000) {
            running = false;
        }

        logAverageLatencies(captureLatency, detectionLatency, renderLatency);

        if (!m_metricsPath.empty() && ++metricsTick >= 10) {
            metricsTick = 0;

            const char *modelStr = "v8";
            bool graphOk = false;
            if (yoloDetector) {
                const auto &engine = yoloDetector->getEngine();
                graphOk = (engine.getOutputDims().size() == 1);
                const auto &od = engine.getOutputDims();
                if (!od.empty()) {
                    const int ch = od[0].d[1];
                    const int na = od[0].d[2];
                    if (ch == 6 && na >= 100 && na <= 500) modelStr = "v26";
                    else if (ch > 4 + static_cast<int>(labelNames.size())) modelStr = "v11";
                }
            }

            const double avgDets = (m_frameCountAccum > 0)
                ? static_cast<double>(m_detectionCountAccum) / m_frameCountAccum : 0.0;

            m_metrics.update(
                captureLatency.getAverageLatency(),
                captureLatency.getMinLatency(), captureLatency.getMaxLatency(),
                detectionLatency.getAverageLatency(),
                detectionLatency.getMinLatency(), detectionLatency.getMaxLatency(),
                renderLatency.getAverageLatency(),
                renderLatency.getMinLatency(), renderLatency.getMaxLatency(),
                avgDets, modelStr, graphOk, "fp16");

            m_detectionCountAccum = 0;
            m_frameCountAccum = 0;
        }

        Sleep(100);
    }
}

void ObjectDetectionSystem::cleanup() {
    if (debugOverlay) {
        debugOverlay->stop();
        debugOverlay.reset();
    }
    if (captureThreadObj.joinable())  captureThreadObj.join();
    if (detectionThreadObj.joinable()) detectionThreadObj.join();
}

void ObjectDetectionSystem::run() {
    captureThreadObj  = std::thread(&ObjectDetectionSystem::captureThread, this);
    detectionThreadObj = std::thread(&ObjectDetectionSystem::detectionThread, this);
    mainLoop();
    cleanup();
}

void ObjectDetectionSystem::logAverageLatencies(LatencyQueue &captureLatency, LatencyQueue &detectionLatency,
                                                LatencyQueue &renderLatency) {
    std::cout << "\rCapture: " << captureLatency.getAverageLatency() << "ms | "
              << "Detection: " << detectionLatency.getAverageLatency() << "ms | "
              << "Render: " << renderLatency.getAverageLatency() << "ms       " << std::flush;
}

// ── GPU capture thread ────────────────────────────────────────────────

void ObjectDetectionSystem::captureThread() {
    try {
        cudaStream_t captureStream = nullptr;
        cudaStreamCreate(&captureStream);

        cv::cuda::GpuMat frame1, frame2;
        cv::cuda::GpuMat *currentFrame = &frame1;
        cv::cuda::GpuMat *nextFrame    = &frame2;

        while (running) {
            auto start = std::chrono::high_resolution_clock::now();

            const bool got = capture->CaptureScreen(*currentFrame, captureStream);
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
        std::cerr << "captureThread: " << e.what() << std::endl;
        running = false;
    } catch (...) {
        std::cerr << "captureThread: unknown error" << std::endl;
        running = false;
    }
}

// ── GPU detection thread ──────────────────────────────────────────────

void ObjectDetectionSystem::detectionThread() {
    try {
        while (running) {
            cv::cuda::GpuMat frame = gpuCaptureQueue.pop();

            auto t0 = std::chrono::high_resolution_clock::now();

            const int centerX = frame.cols / 2;
            const int centerY = frame.rows / 2;
            int x = std::max(centerX - captureWidth  / 2, 0);
            int y = std::max(centerY - captureHeight / 2, 0);
            x = std::min(x, frame.cols - captureWidth);
            y = std::min(y, frame.rows - captureHeight);
            const cv::Rect roi(x, y, captureWidth, captureHeight);
            cv::cuda::GpuMat croppedFrame = frame(roi);

            // Crosshair defaults to screen centre.
            cv::Point crosshairPos(screenWidth / 2, screenHeight / 2);

            std::vector<Object> detections = yoloDetector->detectObjects(croppedFrame);

            m_detectionCountAccum += static_cast<int>(detections.size());
            m_frameCountAccum += 1;

            mouseController->setCrosshairPosition(crosshairPos.x, crosshairPos.y);
            mouseController->aim(detections);
            mouseController->triggerLeftClickIfCenterWithinDetection(detections);

            detectionQueue.push(detections);

            auto t1 = std::chrono::high_resolution_clock::now();
            detectionLatency.push(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

            // Overlay
            if (debugOverlay && debugOverlay->isRunning()) {
                auto tR = std::chrono::high_resolution_clock::now();
                std::vector<DiscordOverlay::DetectionBox> boxes;
                boxes.reserve(detections.size());
                for (const auto &d : detections) {
                    DiscordOverlay::DetectionBox b{};
                    b.x = d.rect.x + static_cast<float>(x);
                    b.y = d.rect.y + static_cast<float>(y);
                    b.w = d.rect.width;
                    b.h = d.rect.height;
                    b.label = d.label;
                    b.confidence = d.probability;
                    boxes.push_back(b);
                }
                debugOverlay->setDetections(std::move(boxes));
                debugOverlay->setStats(detectionLatency.getAverageLatency(), 0);
                auto tR2 = std::chrono::high_resolution_clock::now();
                renderLatency.push(std::chrono::duration_cast<std::chrono::milliseconds>(tR2 - tR).count());
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "detectionThread: " << e.what() << std::endl;
        running = false;
    } catch (...) {
        std::cerr << "detectionThread: unknown error" << std::endl;
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
