#include "MouseController.h"


MouseController::MouseController(int screenWidth, int screenHeight, int detectionZoneWidth, int detectionZoneHeight, float sensitivity,
                                 int centralSquareSize, float minGain, float maxGain, float maxSpeed, int HL1, int HL2, int cpi)
    : screenWidth(screenWidth), screenHeight(screenHeight), detectionZoneWidth(detectionZoneWidth),
      detectionZoneHeight(detectionZoneHeight), sensitivity(sensitivity), centralSquareSize(centralSquareSize),
      minGain(minGain), maxSpeed(maxSpeed), maxGain(maxGain), hidDevice(nullptr), headLabel1(HL1), headLabel2(HL2), cpi(cpi) {
    // Calculate the top-left corner of the detection zone
    detectionZoneX = (screenWidth - detectionZoneWidth) / 2;
    detectionZoneY = (screenHeight - detectionZoneHeight) / 2;

    crosshairX = screenWidth / 2;
    crosshairY = screenHeight / 2;

    integralX = 0.0f;
    integralY = 0.0f;
    alpha = 0.35f;
    prevErrorX= 0.0f;
    prevErrorY = 0.0f;
    smoothedTargetX = 0.0f;
    smoothedTargetY = 0.0f;

    ConnectToDevice();
}

MouseController::~MouseController() {
    if (hidDevice) {
        CloseHandle(hidDevice);
    }
}

bool MouseController::ConnectToDevice() {
    // Initialize HID library
    GUID hidGuid;
    HidD_GetHidGuid(&hidGuid);

    // Get a handle to the device information set
    HDEVINFO deviceInfoSet = SetupDiGetClassDevs(&hidGuid, NULL, NULL, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
    if (deviceInfoSet == INVALID_HANDLE_VALUE) {
        std::cerr << "Failed to get device information set" << std::endl;
        return false;
    }

    // Enumerate devices
    SP_DEVICE_INTERFACE_DATA deviceInterfaceData;
    deviceInterfaceData.cbSize = sizeof(SP_DEVICE_INTERFACE_DATA);
    DWORD deviceIndex = 0;

    while (SetupDiEnumDeviceInterfaces(deviceInfoSet, NULL, &hidGuid, deviceIndex++, &deviceInterfaceData)) {
        // Get the size of the device interface detail data
        DWORD requiredSize = 0;
        SetupDiGetDeviceInterfaceDetail(deviceInfoSet, &deviceInterfaceData, NULL, 0, &requiredSize, NULL);
        std::vector<BYTE> buffer(requiredSize);
        SP_DEVICE_INTERFACE_DETAIL_DATA *deviceInterfaceDetailData = reinterpret_cast<SP_DEVICE_INTERFACE_DETAIL_DATA *>(buffer.data());
        deviceInterfaceDetailData->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);

        // Get the device interface detail data
        if (SetupDiGetDeviceInterfaceDetail(deviceInfoSet, &deviceInterfaceData, deviceInterfaceDetailData, requiredSize, NULL, NULL)) {
            // Open a handle to the device
            HANDLE deviceHandle = CreateFile(deviceInterfaceDetailData->DevicePath, GENERIC_WRITE | GENERIC_READ,
                                             FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
            if (deviceHandle != INVALID_HANDLE_VALUE) {
                std::cout << "Device opened successfully: " << deviceInterfaceDetailData->DevicePath << std::endl;

                // Get the device attributes
                HIDD_ATTRIBUTES attributes;
                attributes.Size = sizeof(HIDD_ATTRIBUTES);
                if (HidD_GetAttributes(deviceHandle, &attributes)) {
                    if (attributes.VendorID == VENDOR_ID && attributes.ProductID == PRODUCT_ID) {
                        std::cout << "Found HIDVendor device" << std::endl;

                        // Get the serial number
                        wchar_t serialNumber[256];
                        if (HidD_GetSerialNumberString(deviceHandle, serialNumber, sizeof(serialNumber))) {
                            if (wcscmp(serialNumber, TARGET_SERIAL) == 0) {
                                // Check interface descriptor for vendor-defined usage page and usage
                                PHIDP_PREPARSED_DATA preparsedData;
                                HIDP_CAPS caps;
                                if (HidD_GetPreparsedData(deviceHandle, &preparsedData)) {
                                    if (HidP_GetCaps(preparsedData, &caps) == HIDP_STATUS_SUCCESS) {
                                        if (caps.UsagePage == TARGET_USAGE_PAGE && caps.Usage == TARGET_USAGE) {
                                            hidDevice = deviceHandle;
                                            HidD_FreePreparsedData(preparsedData);
                                            std::wcout << L"Found HIDVendor device with serial number: " << serialNumber << std::endl;
                                            std::cout << "Vendor ID: " << std::hex << attributes.VendorID << std::endl;
                                            std::cout << "Product ID: " << std::hex << attributes.ProductID << std::dec << std::endl;
                                            SetupDiDestroyDeviceInfoList(deviceInfoSet);
                                            return true; // Successfully connected to the device
                                        }
                                    }
                                    HidD_FreePreparsedData(preparsedData);
                                } else {
                                    std::cerr << "Failed to get preparsed data" << std::endl;
                                }
                            } else {
                                std::cerr << "Serial number does not match" << std::endl;
                            }
                        } else {
                            std::cerr << "Failed to get serial number" << std::endl;
                        }
                    }
                } else {
                    std::cerr << "Failed to get device attributes" << std::endl;
                }
                CloseHandle(deviceHandle);
            }
        }
    }

    SetupDiDestroyDeviceInfoList(deviceInfoSet);

    if (!hidDevice) {
        std::cerr << "No suitable HIDVendor device found." << std::endl;
        return false;
    }

    std::cout << "Device opened successfully" << std::endl;
    return true;
}

void MouseController::setCrosshairPosition(int x, int y) {
    crosshairX = x;
    crosshairY = y;
}

// Implementation of resetSpeed method
void MouseController::resetSpeed() {
    currentSpeedX = 0.0f;
    currentSpeedY = 0.0f;
    integralX = 0.0f;
    integralY = 0.0f;
    prevErrorX = 0.0f;
    prevErrorY = 0.0f;
    _dx = 0;
    _dy = 0;
}

void MouseController::processHIDReports() {
    while (running) {
        auto report = reportQueue.pop();
        processHIDReport(report);
    }
}

bool MouseController::processHIDReport(std::vector<uint8_t> &report) {
    if (hidDevice != nullptr) {
        BOOL res = HidD_SetOutputReport(hidDevice, report.data(), report.size());
        if (!res) {
            int err = GetLastError();
            std::cerr << "Failed to send HID report using HidD_SetOutputReport: " << err << std::endl;
            if (err == 995 || err == 1167) {
                std::cerr << "Device disconnected" << std::endl;
                CloseHandle(hidDevice);
                hidDevice = nullptr;
                std::this_thread::sleep_for(std::chrono::seconds(2)); // Wait for device to reconnect
                ConnectToDevice();
                std::cout << "\033[2J\033[H";
            } else {
                exit(1);
            }
        }
    } else {
        ConnectToDevice();
    }
    return true;
}

void MouseController::sendHIDReport(int16_t dx, int16_t dy) {
    std::vector<uint8_t> report(5, 0);

    report[0] = 0x06; // Report ID
    report[1] = dx & 0xFF;
    report[2] = (dx >> 8) & 0xFF;
    report[3] = dy & 0xFF;
    report[4] = (dy >> 8) & 0xFF;

   processHIDReport(report);
}

float MouseController::calculateSpeedScaling(const cv::Rect &rect) {
    // Define the thresholds for small, medium, and large detection boxes
    const float smallBoxThreshold = 7.0f;
    const float largeBoxThreshold = 40.0f;
    const float minScaling = 0.5f; // Minimum scaling factor
    const float maxScaling = 1.0f; // Maximum scaling factor

    float boxSize = (rect.width < rect.height) ? rect.width : rect.height; // Use the smaller dimension as the box size

    if (boxSize < smallBoxThreshold) {
        return minScaling; // Scale down to minScaling
    } else if (boxSize > largeBoxThreshold) {
        return maxScaling; // Use 100% of max speed
    } else {
        // Linearly interpolate between smallBoxThreshold and largeBoxThreshold
        return minScaling + (maxScaling - minScaling) * ((boxSize - smallBoxThreshold) / (largeBoxThreshold - smallBoxThreshold));
    }
}

void MouseController::aim(const std::vector<Object> &detections) {
    const float deadZoneThreshold = 5.0f;

    if (isLeftMouseButtonPressed() || isMouseButton5Pressed()) {
        Object closestDetection = findClosestDetection(detections);
        if (closestDetection.probability > 0.0f) {
            int targetX = closestDetection.rect.x + closestDetection.rect.width / 2;
            int targetY = closestDetection.rect.y + closestDetection.rect.height / 2;

            // Convert target coordinates from detection zone to screen coordinates
            int screenTargetX = detectionZoneX + targetX;
            int screenTargetY = detectionZoneY + targetY;

            // Convert crosshair coordinates from detection zone to screen coordinates
            int screenCrosshairX = detectionZoneX + crosshairX;
            int screenCrosshairY = detectionZoneY + crosshairY;

            // Calculate the relative movement
            int dx = static_cast<int>(screenTargetX - screenCrosshairX);
            int dy = static_cast<int>(screenTargetY - screenCrosshairY);

            // Calculate the distance to the target
            float distance = sqrt(dx * dx + dy * dy);

            // Ignore detections outside the central square or within the dead zone
            if (distance < deadZoneThreshold || dx < -centralSquareSize || dx > centralSquareSize || dy < -centralSquareSize ||
                dy > centralSquareSize) {
                resetSpeed();
                return;
            }

            // Calculate the maximum distance within the detection zone (diagonal of the detection zone)
            float detectionZoneDiagonal = sqrt(centralSquareSize * centralSquareSize + centralSquareSize * centralSquareSize);

            // Apply exponential scaling to the proportional gain
            float normalizedDistance = distance / detectionZoneDiagonal;
            float proportionalGain = minGain + (maxGain - minGain) * std::exp(-normalizedDistance);

            // Apply dynamic proportional control
            currentSpeedX = proportionalGain * dx;
            currentSpeedY = proportionalGain * dy;

            // Calculate the speed scaling factor based on the detection box size
            float speedScaling = calculateSpeedScaling(closestDetection.rect);

            // Scale the speed
            currentSpeedX *= speedScaling;
            currentSpeedY *= speedScaling;

            // Cap the speed to maxSpeed
            currentSpeedX = std::clamp(currentSpeedX, -maxSpeed, maxSpeed);
            currentSpeedY = std::clamp(currentSpeedY, -maxSpeed, maxSpeed);

            // Apply sensitivity and movement
            _dx = static_cast<int>(currentSpeedX * sensitivity);
            _dy = static_cast<int>(currentSpeedY * sensitivity);
        }
    } else {
        resetSpeed();
    }

    if (_dx != 0 || _dy != 0) {
        sendHIDReport(_dx, _dy);
    }
}




bool MouseController::isLeftMouseButtonPressed() { return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0; }
bool MouseController::isMouseButton4Pressed() { return (GetAsyncKeyState(VK_XBUTTON1) & 0x8000) != 0; } // Mouse Button 4
bool MouseController::isMouseButton5Pressed() { return (GetAsyncKeyState(VK_XBUTTON2) & 0x8000) != 0; } // Mouse Button 5

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

// Modify findClosestDetection to use crosshair position
Object MouseController::findClosestDetection(const std::vector<Object> &detections) {
    Object closestDetection;
    closestDetection.probability = 0.0f; // Initialize with no detection

    float closestDistance = FLT_MAX;

    for (const auto &detection : detections) {
        if (detection.label == headLabel1 || detection.label == headLabel2) { // Replace with actual label values
            int detectionCenterX = detection.rect.x + detection.rect.width / 2;
            int detectionCenterY = detection.rect.y + detection.rect.height / 2;
            float distance = std::sqrt(std::pow(detectionCenterX - crosshairX, 2) + std::pow(detectionCenterY - crosshairY, 2));
            if (distance < closestDistance) {
                closestDistance = distance;
                closestDetection = detection;
            }
        }
    }

    return closestDetection;
}

void MouseController::triggerLeftClickIfCenterWithinDetection(const std::vector<Object> &detections) {
    if (isMouseButton4Pressed()) {
        int centerX = crosshairX;
        int centerY = crosshairY;

        for (const auto &detection : detections) {
            if (detection.label == 1 || detection.label == 3) { // Replace with actual label values
                if (centerX >= detection.rect.x && centerX <= detection.rect.x + detection.rect.width && centerY >= detection.rect.y &&
                    centerY <= detection.rect.y + detection.rect.height) {
                    leftClick();
                    Sleep(250);
                    break;
                }
            }
        }
    }
}