#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <windows.h>
#include "yolov8.h"

class MouseController {
public:
    MouseController(int screenWidth, int screenHeight, int detectionZoneWidth, int detectionZoneHeight, float sensitivity, int centralSquareSize);
    void aim(const std::vector<Object> &detections);
    void triggerLeftClickIfCenterWithinDetection(const std::vector<Object> &detections);

private:
    int screenWidth;
    int screenHeight;
    int detectionZoneWidth;
    int detectionZoneHeight;
    int detectionZoneX;
    int detectionZoneY;
    float sensitivity;
    int centralSquareSize;

    bool isLeftMouseButtonPressed();
    bool isMouseButton5Pressed();
    void moveMouseRelative(int dx, int dy);
    void moveMouseAbsolute(int x, int y);
    void leftClick();
    Object findClosestDetection(const std::vector<Object> &detections);
};

MouseController::MouseController(int screenWidth, int screenHeight, int detectionZoneWidth, int detectionZoneHeight, float sensitivity, int centralSquareSize)
    : screenWidth(screenWidth), screenHeight(screenHeight), detectionZoneWidth(detectionZoneWidth),
      detectionZoneHeight(detectionZoneHeight), sensitivity(sensitivity), centralSquareSize(centralSquareSize) {
    // Calculate the top-left corner of the detection zone
    detectionZoneX = (screenWidth - detectionZoneWidth) / 2;
    detectionZoneY = (screenHeight - detectionZoneHeight) / 2;
}

void MouseController::aim(const std::vector<Object> &detections) {
    if (isLeftMouseButtonPressed()) {
        Object closestDetection = findClosestDetection(detections);
        if (closestDetection.probability > 0.0f) {
            int targetX = closestDetection.rect.x + closestDetection.rect.width / 2;
            int targetY = closestDetection.rect.y + closestDetection.rect.height / 2;

            // Convert target coordinates from detection zone to screen coordinates
            int screenTargetX = detectionZoneX + targetX;
            int screenTargetY = detectionZoneY + targetY;

            // Calculate the relative movement
            int centerX = screenWidth / 2;
            int centerY = screenHeight / 2;
            int dx = screenTargetX - centerX;
            int dy = screenTargetY - centerY;

            // Ignore detections outside the central square
            if (dx < -centralSquareSize || dx > centralSquareSize || dy < -centralSquareSize || dy > centralSquareSize) {
                return;
            }

            // Apply sensitivity and direct movement
            int scaledDx = static_cast<int>(dx * sensitivity);
            int scaledDy = static_cast<int>(dy * sensitivity);

            moveMouseRelative(scaledDx, scaledDy);
        }
    }
}

bool MouseController::isLeftMouseButtonPressed() { return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0; }
bool MouseController::isMouseButton5Pressed() {
    return (GetAsyncKeyState(VK_XBUTTON1) & 0x8000) != 0; // Mouse Button 5
}

void MouseController::moveMouseRelative(int dx, int dy) {
    INPUT input = {0};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_MOVE;
    input.mi.dx = dx;
    input.mi.dy = dy;
    SendInput(1, &input, sizeof(INPUT));
}

void MouseController::moveMouseAbsolute(int x, int y) {
    INPUT input = {0};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE;
    input.mi.dx = static_cast<LONG>((x * 65535) / screenWidth);
    input.mi.dy = static_cast<LONG>((y * 65535) / screenHeight);
    SendInput(1, &input, sizeof(INPUT));
}

Object MouseController::findClosestDetection(const std::vector<Object> &detections) {
    Object closestDetection;
    closestDetection.probability = 0.0f; // Initialize with no detection

    int centerX = detectionZoneWidth / 2;
    int centerY = detectionZoneHeight / 2;
    float closestDistance = FLT_MAX;

    for (const auto &detection : detections) {
        if (detection.label == 1 || detection.label == 3) { // Replace with actual label values
            int detectionCenterX = detection.rect.x + detection.rect.width / 2;
            int detectionCenterY = detection.rect.y + detection.rect.height / 2;
            float distance = std::sqrt(std::pow(detectionCenterX - centerX, 2) + std::pow(detectionCenterY - centerY, 2));
            if (distance < closestDistance) {
                closestDistance = distance;
                closestDetection = detection;
            }
        }
    }

    return closestDetection;
}

void MouseController::triggerLeftClickIfCenterWithinDetection(const std::vector<Object> &detections) {
    if (isMouseButton5Pressed()) {
        int centerX = detectionZoneWidth / 2;
        int centerY = detectionZoneHeight / 2;

        for (const auto &detection : detections) {
            if (detection.label >= 0 && detection.label <= 3) { // Replace with actual label values
                if (centerX >= detection.rect.x && centerX <= detection.rect.x + detection.rect.width && centerY >= detection.rect.y &&
                    centerY <= detection.rect.y + detection.rect.height) {
                    Sleep(40);
                    leftClick();
                    Sleep(20);
                    break;
                }
            }
        }
    }
}

void MouseController::leftClick() {
    INPUT input = {0};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    SendInput(1, &input, sizeof(INPUT));

    ZeroMemory(&input, sizeof(INPUT));
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
    SendInput(1, &input, sizeof(INPUT));
}
