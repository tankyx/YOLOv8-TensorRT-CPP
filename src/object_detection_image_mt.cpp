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
#include "YoloV11.h"
#include "YoloV26.h"
#include "DetectorFactory.h"
#include "SoftwareFuser.h"
#include "CrosshairTrackerGPU.h"
#include "GdiDebugWindow.h"
#include "MetricsWriter.h"

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
                                    LatencyQueue &renderLatency, LatencyQueue &crosshairLatency,
                                    SafeQueue<std::string> &logQueue);

    void captureThread();
    void detectionThread();
    void captureThreadGpu();
    void detectionThreadGpu();

    INIParser config;
    std::unique_ptr<YoloDetector> yoloDetector;
    std::unique_ptr<DXGICapture> capture;
    std::unique_ptr<DXGICaptureCUDA> captureCUDA;
    std::unique_ptr<MouseController> mouseController;
    std::unique_ptr<DiscordOverlay> debugOverlay;
    std::vector<std::string> labelNames;

    // Pure-GPU shape-based crosshair tracker (replaces template matching).
    // Runs Canny + Hough + custom intersection-scoring kernel entirely on
    // device; single float2 D2H copy per frame.
    std::unique_ptr<CrosshairTrackerGPU> crosshairTracker;
    int captureWidth;
    int captureHeight;
    int captureFPS;
    int screenWidth;
    int screenHeight;
    int captureThreadCore;
    HWND targetWnd;

    SafeQueue<std::string> logQueue;
    LatencyQueue captureLatency, detectionLatency, renderLatency, crosshairLatency;
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
    bool debugViewOpenCV;
    bool useDirectGpuCapture;
    std::string debugOverlayTargetProcess;
    std::string crosshairTemplatePath;
    GdiDebugWindow m_debugWin;

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
    crosshairTemplatePath     = config.getString("CrosshairTemplate", "crosshair.png");
    debugViewOpenCV           = config.getBool("DebugViewOpenCV", false);
    std::cout << "[CONFIG] DebugViewOpenCV = " << (debugViewOpenCV ? "true" : "false") << std::endl;

    // Write a test file at startup to verify I/O works at this path
    {
        std::ofstream test("C:/Users/tanguy/Documents/GitHub/YOLOv8-TensorRT-CPP/STARTUP_TEST.txt");
        test << "startup OK" << std::endl;
    }

    // c39c: opt into the DXGI/CUDA-interop capture path. When true the capture stays on the
    // GPU end-to-end (no CPU map, no cvtColor, no upload). Default false until validated.
    useDirectGpuCapture = config.getBool("UseDirectGpuCapture", false);

    // Optional metrics status file for external monitoring (e.g. codewhale).
    // If set, a compact JSON status file is overwritten once per second.
    // If omitted or empty, no file is written.
    m_metricsPath = config.getString("MetricsStatus", "");
    if (!m_metricsPath.empty()) {
        m_metrics.open(m_metricsPath);
    }
}

void ObjectDetectionSystem::initializeSystem() {
    YoloConfig yoloConfig;
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

    // Determine YOLO version: explicit INI override, or auto-detect.
    // ModelVersion can be "v8", "v11", "v26", or "auto" (default).
    YoloVersion modelVersion = YoloVersion::AUTO;
    const std::string versionStr = config.getString("ModelVersion", "auto");
    if (versionStr == "v8" || versionStr == "V8") {
        modelVersion = YoloVersion::V8;
    } else if (versionStr == "v11" || versionStr == "V11") {
        modelVersion = YoloVersion::V11;
    } else if (versionStr == "v26" || versionStr == "V26") {
        modelVersion = YoloVersion::V26;
    }
    // "auto" or anything else: modelVersion stays AUTO

    std::cout << "[INIT] Creating YOLO detector..." << std::endl;
    yoloDetector = DetectorFactory::create(config.getString("ModelPath"), yoloConfig, modelVersion);
    std::cout << "[INIT] YOLO detector OK" << std::endl;

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
        // Shape-based GPU tracker replaces the old CPU template-match path.
        // Works with both direct GPU capture and the legacy CPU upload path
        // because the tracker only ever touches GpuMats.
        crosshairTracker = std::make_unique<CrosshairTrackerGPU>(captureWidth, captureHeight,
                                                                    crosshairTemplatePath);
    }

    if (useDirectGpuCapture) {
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
    // Startup marker — proves mainLoop was reached even if stdout is swallowed.
    { std::ofstream m("C:/Users/tanguy/Documents/GitHub/YOLOv8-TensorRT-CPP/MAINLOOP_OK.txt"); m << "reached" << std::endl; }
    clearScreen();
    std::cout << "OpenCV CUDA support: " << cv::cuda::getCudaEnabledDeviceCount() << " devices" << std::endl;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::cout.flush();

    std::cout << "Waiting for first captured frame..." << std::endl;
    // If no frames arrive within 5 seconds, capture is blocked.
    for (int i = 0; i < 50 && running; i++) {
        if (captureLatency.getAverageLatency() > 0.0 || detectionLatency.getAverageLatency() > 0.0) {
            std::cout << "Capture: OK (frames flowing)" << std::endl;
            break;
        }
        Sleep(100);
    }
    if (captureLatency.getAverageLatency() <= 0.0 && detectionLatency.getAverageLatency() <= 0.0) {
        std::cout << "Capture: BLOCKED (no frames after 5s) — Valorant likely blocks DXGI DD" << std::endl;
    }

    // GDI debug window — only when DebugViewOpenCV is enabled
    if (debugViewOpenCV) {
        m_debugWin.create(L"YOLO Detections", captureWidth, captureHeight);
    }

    // Metrics counter — write status file once per second (every 10th loop iteration
    // since Sleep(100) gives ~10 Hz loop rate).
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

        logAverageLatencies(captureLatency, detectionLatency, renderLatency, crosshairLatency, logQueue);

        // Write metrics status file once per second.
        if (!m_metricsPath.empty() && ++metricsTick >= 10) {
            metricsTick = 0;

            // Determine model version string and graph state.
            const char *modelStr = "v8";
            bool graphOk = false;
            if (yoloDetector) {
                const auto &engine = yoloDetector->getEngine();
                graphOk = (engine.getOutputDims().size() == 1);
                // Infer version from output shape.
                const auto &od = engine.getOutputDims();
                if (!od.empty()) {
                    const int ch = od[0].d[1];
                    const int na = od[0].d[2];
                    if (ch == 6 && na >= 100 && na <= 500) modelStr = "v26";
                    else if (ch > 4 + static_cast<int>(labelNames.size())) modelStr = "v11";
                }
            }

            const double fps = (m_frameCountAccum > 0)
                ? static_cast<double>(m_frameCountAccum)
                : 0.0;
            const double avgDets = (m_frameCountAccum > 0)
                ? static_cast<double>(m_detectionCountAccum) / m_frameCountAccum
                : 0.0;

            m_metrics.update(
                captureLatency.getAverageLatency(),
                captureLatency.getMinLatency(),
                captureLatency.getMaxLatency(),
                detectionLatency.getAverageLatency(),
                detectionLatency.getMinLatency(),
                detectionLatency.getMaxLatency(),
                renderLatency.getAverageLatency(),
                renderLatency.getMinLatency(),
                renderLatency.getMaxLatency(),
                avgDets,
                modelStr,
                graphOk,
                "fp16");

            m_detectionCountAccum = 0;
            m_frameCountAccum = 0;
        }

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
    { std::ofstream m("C:/Users/tanguy/Documents/GitHub/YOLOv8-TensorRT-CPP/RUN_OK.txt"); m << "entered" << std::endl; }
    startThreads();
    { std::ofstream m("C:/Users/tanguy/Documents/GitHub/YOLOv8-TensorRT-CPP/THREADS_OK.txt"); m << "started" << std::endl; }
    mainLoop();
    cleanup();
}

void ObjectDetectionSystem::moveToTop() { std::cout << "\033[H"; }

void ObjectDetectionSystem::clearScreen() { std::cout << "\033[2J\033[H"; }

void ObjectDetectionSystem::logAverageLatencies(LatencyQueue &captureLatency, LatencyQueue &detectionLatency,
                                                LatencyQueue &renderLatency, LatencyQueue &crosshairLatency,
                                                SafeQueue<std::string> &logQueue) {
    double avgCaptureLatency = captureLatency.getAverageLatency();
    double avgDetectionLatency = detectionLatency.getAverageLatency();
    double avgRenderLatency = renderLatency.getAverageLatency();
    double avgCrosshairLatency = crosshairLatency.getAverageLatency();

    std::cout << "\rCapture: " << avgCaptureLatency << "ms | "
              << "Detection: " << avgDetectionLatency << "ms | "
              << "Render: " << avgRenderLatency << "ms | "
              << "Crosshair: " << std::fixed << std::setprecision(3)
              << (avgCrosshairLatency / 1000.0) << "ms       " << std::flush;
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
    // Hardcoded startup marker to confirm this function is entered at all
    {
        std::ofstream m("C:/Users/tanguy/Documents/GitHub/YOLOv8-TensorRT-CPP/DET_THREAD_ENTERED.txt");
        m << "detectionThread entered OK" << std::endl;
    }
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

            // Shape-based GPU crosshair tracker — same pipeline as detectionThreadGpu.
            // The CPU path uploads the cropped frame to a temporary GpuMat for the
            // tracker, then downloads the position. One extra upload per frame, but
            // the cost is negligible compared to the already-present YOLO inference.
            cv::Point crosshairPos(roiWidth / 2, roiHeight / 2);
            if (crosshairTracker) {
                auto crosshairStart = std::chrono::high_resolution_clock::now();
                cv::cuda::GpuMat gpuCropped;
                gpuCropped.upload(croppedFrame);
                cudaStream_t trackerStream = nullptr;
                bool detected = crosshairTracker->update(gpuCropped, trackerStream);
                auto crosshairEnd = std::chrono::high_resolution_clock::now();
                crosshairLatency.push(
                    std::chrono::duration_cast<std::chrono::microseconds>(crosshairEnd - crosshairStart).count());
                if (detected) {
                    cv::Point2f pos = crosshairTracker->getPosition();
                    crosshairPos = cv::Point(static_cast<int>(pos.x), static_cast<int>(pos.y));
                    cv::Point2f delta = crosshairTracker->getDelta();
                    if (delta.x != 0.0f || delta.y != 0.0f) {
                        mouseController->applyRecoilCompensation(-delta.x, -delta.y);
                    }
                }
            }

            std::vector<Object> detections;
            detections = yoloDetector->detectObjects(croppedFrame);

            // Accumulate for per-second metrics averaging.
            m_detectionCountAccum += static_cast<int>(detections.size());
            m_frameCountAccum += 1;

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
                if (crosshairTracker) {
                    debugOverlay->setCrosshair(static_cast<float>(crosshairPos.x + x),
                                                static_cast<float>(crosshairPos.y + y));
                }
                debugOverlay->setStats(detectionLatency.getAverageLatency(), 0);
                auto renderEnd = std::chrono::high_resolution_clock::now();
                renderLatency.push(std::chrono::duration_cast<std::chrono::milliseconds>(renderEnd - renderStart).count());
            }

            // GDI debug window — draw detection boxes and push to native window
            if (debugViewOpenCV) {
                cv::Mat vis = croppedFrame.clone();
                for (const auto& d : detections) {
                    cv::rectangle(vis, d.rect, cv::Scalar(0, 255, 0), 2);
                    const char* lbl = (d.label >= 0 && static_cast<size_t>(d.label) < labelNames.size())
                                          ? labelNames[d.label].c_str() : "?";
                    cv::putText(vis, lbl, cv::Point(d.rect.x, d.rect.y - 4), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
                }
                m_debugWin.updateFrame(vis);
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

            // Shape-based GPU crosshair tracker (pure device pipeline: Canny → Hough
            // → custom intersection-scoring kernel → single float2 D2H copy).
            // Falls back to ROI centre when tracking is disabled or no detection.
            cv::Point crosshairPos(captureWidth / 2, captureHeight / 2);
            if (crosshairTracker) {
                auto crosshairStart = std::chrono::high_resolution_clock::now();
                // Use the default stream (0) for simplicity; the tracker synchronises
                // after its single async D2H copy.
                cudaStream_t trackerStream = nullptr;
                bool detected = crosshairTracker->update(croppedFrame, trackerStream);
                auto crosshairEnd = std::chrono::high_resolution_clock::now();
                crosshairLatency.push(
                    std::chrono::duration_cast<std::chrono::microseconds>(crosshairEnd - crosshairStart).count());
                if (detected) {
                    cv::Point2f pos = crosshairTracker->getPosition();
                    crosshairPos = cv::Point(static_cast<int>(pos.x), static_cast<int>(pos.y));
                    // Apply recoil compensation from frame-to-frame delta.
                    cv::Point2f delta = crosshairTracker->getDelta();
                    if (delta.x != 0.0f || delta.y != 0.0f) {
                        mouseController->applyRecoilCompensation(-delta.x, -delta.y);
                    }
                }
            }

            std::vector<Object> detections = yoloDetector->detectObjects(croppedFrame);

            // Accumulate for per-second metrics averaging.
            m_detectionCountAccum += static_cast<int>(detections.size());
            m_frameCountAccum += 1;

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
                if (crosshairTracker) {
                    debugOverlay->setCrosshair(static_cast<float>(crosshairPos.x + x),
                                                static_cast<float>(crosshairPos.y + y));
                }
                debugOverlay->setStats(detectionLatency.getAverageLatency(), 0);
                auto renderEnd = std::chrono::high_resolution_clock::now();
                renderLatency.push(
                    std::chrono::duration_cast<std::chrono::milliseconds>(renderEnd - renderStart).count());
            }

            // GDI debug window — download GpuMat, draw boxes, push to native window
            if (debugViewOpenCV) {
                cv::Mat vis;
                croppedFrame.download(vis);
                for (const auto& d : detections) {
                    cv::rectangle(vis, d.rect, cv::Scalar(0, 255, 0), 2);
                    const char* lbl = (d.label >= 0 && static_cast<size_t>(d.label) < labelNames.size())
                                          ? labelNames[d.label].c_str() : "?";
                    cv::putText(vis, lbl, cv::Point(d.rect.x, d.rect.y - 4), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
                }
                m_debugWin.updateFrame(vis);
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