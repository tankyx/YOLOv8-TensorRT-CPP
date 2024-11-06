#include "yolov8.h"
#include "threadsafe_queue.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <windows.h>
#include <setupapi.h>
#include <hidsdi.h>
#include <hidclass.h>

#define VENDOR_ID 0x0812
#define PRODUCT_ID 0x2205
#define TARGET_SERIAL L"DF625857C74E132B" // Replace with your target serial number
#define TARGET_USAGE_PAGE 0xFF00      // Vendor-defined usage page
#define TARGET_USAGE 0x01             // Vendor-defined usage

class MouseController {
public:
    MouseController(int screenWidth, int screenHeight, int detectionZoneWidth, int detectionZoneHeight, float sensitivity,
                    int centralSquareSize, float minGain, float maxGain, float maxSpeed, int HL1, int HL2, int cpi);
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
    int currentSpeedX;
    int currentSpeedY;
    HANDLE hidDevice;
    float alpha; // Smoothing factor (between 0 and 1)
    float smoothedTargetX;
    float smoothedTargetY;

    // PID Controller state
    float integralX;
    float integralY;
    float prevErrorX;
    float prevErrorY;

    DWORD lastTime;

    int headLabel1;
    int headLabel2;

    int cpi;

    bool isLeftClicking;

    SafeQueue<std::vector<uint8_t>> reportQueue;
    std::thread reportThread;
    std::atomic<bool> running;

    bool isLeftMouseButtonPressed();
    bool isMouseButton4Pressed();
    bool isMouseButton5Pressed();
    void moveMouseRelative(int dx, int dy);
    void moveMouseAbsolute(int x, int y);
    void leftClick();
    void releaseLeftClick();
    Object findClosestDetection(const std::vector<Object> &detections);
    void sendHIDReport(int16_t dx, int16_t dy, uint8_t button);
    bool ConnectToDevice();
    bool processHIDReport(std::vector<uint8_t> &report);
    void processHIDReports();
    void resetSpeed();
    float calculateSpeedScaling(const cv::Rect &rect);
};

