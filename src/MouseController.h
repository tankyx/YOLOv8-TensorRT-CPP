#include "yolov8.h"
#include "threadsafe_queue.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <windows.h>
#include <setupapi.h>
#include <hidsdi.h>
#include <hidclass.h>

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
                    uint16_t hidVendorId, uint16_t hidProductId, std::wstring hidSerial);
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

    // Non-blocking trigger-click state machine (c12)
    bool triggerPressed = false;
    std::chrono::steady_clock::time_point triggerReleaseAt{};
    std::chrono::steady_clock::time_point triggerNextAllowedAt{};

    bool isLeftMouseButtonPressed();
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

