#include "yolov8.h"
#include "threadsafe_queue.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <windows.h>
#include <setupapi.h>
#include <hidsdi.h>
#include <hidclass.h>

// Cubic-bezier path generator for screen-space mouse smoothing. Ported from
// CS2Miam (bezier_curve.hpp) using cv::Point2f instead of Vector3 since we
// already pull in OpenCV. Time-based progression makes the curve duration
// independent of frame rate.
class BezierCurve2D {
public:
    static constexpr float CONTROL_RANDOMNESS = 0.3f;
    static constexpr float PROGRESS_SPEED_MULTIPLIER = 0.00005f;
    static constexpr float TARGET_CHANGE_THRESHOLD = 5.0f;

    BezierCurve2D()
        : _rng(static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count())),
          _t(0.0f), _active(false) {}

    void initialize(const cv::Point2f &start, const cv::Point2f &end) {
        _start = start;
        _end = end;
        _t = 0.0f;
        std::uniform_real_distribution<float> dist(-CONTROL_RANDOMNESS, CONTROL_RANDOMNESS);
        const cv::Point2f delta = end - start;
        _control1 = cv::Point2f(start.x + delta.x * 0.33f + delta.x * dist(_rng),
                                start.y + delta.y * 0.33f + delta.y * dist(_rng));
        _control2 = cv::Point2f(start.x + delta.x * 0.67f + delta.x * dist(_rng),
                                start.y + delta.y * 0.67f + delta.y * dist(_rng));
        _active = true;
        _lastUpdate = std::chrono::steady_clock::now();
    }

    cv::Point2f calculate(float t) const {
        const float u = 1.0f - t, tt = t * t, uu = u * u, uuu = uu * u, ttt = tt * t;
        return cv::Point2f(uuu * _start.x + 3 * uu * t * _control1.x + 3 * u * tt * _control2.x + ttt * _end.x,
                           uuu * _start.y + 3 * uu * t * _control1.y + 3 * u * tt * _control2.y + ttt * _end.y);
    }

    void update(float smoothingFactor) {
        if (!_active) return;
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - _lastUpdate).count();
        _lastUpdate = now;
        _t += (1.0f - smoothingFactor) * PROGRESS_SPEED_MULTIPLIER * static_cast<float>(elapsed);
        if (_t > 1.0f) _t = 1.0f;
    }

    cv::Point2f getCurrentPosition() const { return calculate(_t); }
    void updateStartPosition(const cv::Point2f &movement) { _start = _start - movement; }
    void deactivate() { _active = false; }
    bool isActive() const { return _active; }

    bool shouldReinitialize(const cv::Point2f &newTarget) const {
        return !_active || std::abs(newTarget.x - _end.x) > TARGET_CHANGE_THRESHOLD
            || std::abs(newTarget.y - _end.y) > TARGET_CHANGE_THRESHOLD;
    }

private:
    std::mt19937 _rng;
    float _t;
    cv::Point2f _start, _end, _control1, _control2;
    bool _active;
    std::chrono::steady_clock::time_point _lastUpdate;
};

// HID protocol-level constants — these describe the report format the firmware exposes,
// not the device identity. Hardcoded on purpose; the per-device VID/PID/serial are now
// constructor parameters (c15).
#define TARGET_USAGE_PAGE 0xFF00 // Vendor-defined usage page
#define TARGET_USAGE 0x01        // Vendor-defined usage

class MouseController {
public:
    MouseController(int screenWidth, int screenHeight, int detectionZoneWidth, int detectionZoneHeight, float sensitivity,
                    int centralSquareSize, float minGain, float maxGain, float maxSpeed, int HL1, int HL2, int cpi,
                    int nLab, float probabilityThreshold,
                    uint16_t hidVendorId, uint16_t hidProductId, std::wstring hidSerial,
                    float smoothing);

    void setSmoothing(float userVal);
    float getSmoothing() const { return _userSmoothing; }
    ~MouseController();
    void aim(const std::vector<Object> &detections);
    void triggerLeftClickIfCenterWithinDetection(const std::vector<Object> &detections);
    void setCrosshairPosition(int x, int y);
    int getCrosshairX() const { return crosshairX; }
    int getCrosshairY() const { return crosshairY; }
    int getMovementX() const { return _dx; }
    int getMovementY() const { return _dy; }

private:
    int screenWidth;
    int screenHeight;
    int detectionZoneWidth;
    int detectionZoneHeight;
    int detectionZoneX;
    int detectionZoneY;
    int crosshairX;
    int crosshairY;
    int _dx;
    int _dy;
    float sensitivity;
    int centralSquareSize;
    int maxSpeed;
    float minGain;
    float maxGain;
    HANDLE hidDevice;

    int headLabel1;
    int headLabel2;
    int nLabels;
    float probabilityThreshold;

    // HID device identity (c15)
    uint16_t hidVendorId;
    uint16_t hidProductId;
    std::wstring hidSerial;

    int cpi;

    bool isLeftClicking;
    bool hidWarningLogged = false;

    // Smoothing pipeline (ported from CS2Miam aimbot.hpp). _userSmoothing is the
    // user-facing 1-10 knob; _smoothVal is the internal multiplier applied to
    // per-frame deltas. Sub-pixel residue is carried frame-to-frame so slow
    // movements don't get truncated to zero. Bezier curve smooths the initial
    // flick on a fresh target and is reinitialised when the target jumps.
    float _userSmoothing = 5.0f;
    float _smoothVal = 0.05f;
    float _residX = 0.0f;
    float _residY = 0.0f;
    BezierCurve2D _bezier;

    static float remapSmoothing(float userVal) {
        // Smoothing knob 1-10 maps to internal "smoothing percent" 50-100, which
        // smoothingToInternal() then inverts to a per-frame multiplier in [0.005, 0.50].
        // Wider than CS2Miam's original 90-100 band — that one capped at 10% per frame
        // which is too conservative for a ~220 Hz pipeline. Smoothing=1 = snap, =10 = smooth.
        const float clamped = std::clamp(userVal, 1.0f, 10.0f);
        return 50.0f + (clamped - 1.0f) * (50.0f / 9.0f);
    }
    static float smoothingToInternal(float userVal) {
        return (std::max)(0.005f, 1.0f - (remapSmoothing(userVal) / 100.0f));
    }

    // Non-blocking trigger-click state machine (c12). Cooldown between releases
    // is randomised to break up the obvious "click every N ms" pattern.
    bool triggerPressed = false;
    std::chrono::steady_clock::time_point triggerReleaseAt{};
    std::chrono::steady_clock::time_point triggerNextAllowedAt{};
    std::mt19937 _triggerRng{std::random_device{}()};

    bool isLeftMouseButtonPressed();
    bool isRightMouseButtonPressed();
    bool isMouseButton4Pressed();
    bool isMouseButton5Pressed();
    void leftClick();
    void releaseLeftClick();
    Object findClosestDetection(const std::vector<Object> &detections);
    void sendHIDReport(int16_t dx, int16_t dy, uint8_t button);
    bool ConnectToDevice();
    bool processHIDReport(std::vector<uint8_t> &report);
    float calculateSpeedScaling(const cv::Rect &rect);
};

