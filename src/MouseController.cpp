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
    running = true;
    std::thread(&MouseController::processHIDReports, this).detach();
}

MouseController::~MouseController() {
    if (hidDevice) {
        CloseHandle(hidDevice);
    }
    running = false;
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

    while (SetupDiEnumDeviceInterfaces(deviceInfoSet, NULL, &hidGuid, deviceIndex, &deviceInterfaceData)) {
        std::cout << "Enumerating device at index: " << deviceIndex++ << std::endl;
        // Get the size of the device interface detail data
        DWORD requiredSize = 0;
        SetupDiGetDeviceInterfaceDetail(deviceInfoSet, &deviceInterfaceData, NULL, 0, &requiredSize, NULL);
        std::vector<BYTE> buffer(requiredSize);
        SP_DEVICE_INTERFACE_DETAIL_DATA *deviceInterfaceDetailData = reinterpret_cast<SP_DEVICE_INTERFACE_DETAIL_DATA *>(buffer.data());
        deviceInterfaceDetailData->cbSize = sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA);

        // Get the device interface detail data
        if (SetupDiGetDeviceInterfaceDetail(deviceInfoSet, &deviceInterfaceData, deviceInterfaceDetailData, requiredSize, NULL, NULL)) {
            // Open a handle to the device
            std::wcout << L"Device path: " << deviceInterfaceDetailData->DevicePath << std::endl;
            HANDLE deviceHandle = CreateFile(deviceInterfaceDetailData->DevicePath, GENERIC_WRITE | GENERIC_READ,
                                             FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
            if (deviceHandle != INVALID_HANDLE_VALUE) {
                std::cout << "Device opened successfully: " << deviceInterfaceDetailData->DevicePath << std::endl;

                // Get the device attributes
                HIDD_ATTRIBUTES attributes;
                attributes.Size = sizeof(HIDD_ATTRIBUTES);
                if (HidD_GetAttributes(deviceHandle, &attributes)) {
                    if (attributes.VendorID == VENDOR_ID && attributes.ProductID == PRODUCT_ID) {
                        std::cout << "Found matching device: Vendor ID: " << std::hex << attributes.VendorID
                                  << ", Product ID: " << attributes.ProductID << std::dec << std::endl;

                        wchar_t serialNumber[256];
                        if (HidD_GetSerialNumberString(deviceHandle, serialNumber, sizeof(serialNumber))) {
                            std::wcout << L"Device serial number: " << serialNumber << std::endl;
                            if (wcscmp(serialNumber, TARGET_SERIAL) == 0) {
                                std::wcout << L"Serial number matches: " << serialNumber << std::endl;

                                // Check the usage page and usage (optional, based on your needs)
                                PHIDP_PREPARSED_DATA preparsedData;
                                HIDP_CAPS caps;
                                if (HidD_GetPreparsedData(deviceHandle, &preparsedData)) {
                                    if (HidP_GetCaps(preparsedData, &caps) == HIDP_STATUS_SUCCESS) {
                                        if (caps.UsagePage == TARGET_USAGE_PAGE && caps.Usage == TARGET_USAGE) {
                                            hidDevice = deviceHandle;
                                            std::cout << "Device has the correct UsagePage and Usage." << std::endl;
                                            HidD_FreePreparsedData(preparsedData);
                                            break;
                                        }
                                    }
                                    HidD_FreePreparsedData(preparsedData);
                                }
                            } else {
                                std::wcout << L"Serial number does not match. Expected: " << TARGET_SERIAL << L", Received: "
                                           << serialNumber << std::endl;
                            }
                        } else {
                            std::cerr << "Failed to get serial number" << std::endl;
                        }
                    }
                } else {
                    std::cerr << "Failed to get device attributes" << std::endl;
                }
                CloseHandle(deviceHandle);
            } else {
                std::cerr << "Failed to open device: " << GetLastError() << std::endl;
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
        Sleep(2);
    }
}

bool MouseController::processHIDReport(std::vector<uint8_t> &report) {
    if (hidDevice != nullptr) {
        DWORD bytesWritten = 0;
        BOOL res = WriteFile(hidDevice, report.data(), report.size(), &bytesWritten, NULL);

        if (!res || bytesWritten != report.size()) {
            int err = GetLastError();
            std::cerr << "Failed to send HID report using WriteFile: " << err << std::endl;

            if (err == 995 || err == 1167) { // Device disconnected errors
                std::cerr << "Device disconnected" << std::endl;
                CloseHandle(hidDevice);
                hidDevice = nullptr;
                std::this_thread::sleep_for(std::chrono::seconds(2)); // Wait for device to reconnect
                ConnectToDevice();
                std::cout << "\033[2J\033[H"; // Clear terminal (optional)
            } else {
                exit(1); // Handle the error appropriately
            }
        }
    } else {
        ConnectToDevice(); // Reconnect if the device is not available
    }
    return true;
}

void MouseController::sendHIDReport(int16_t dx, int16_t dy) {
    // Create a 64-byte report
    std::vector<uint8_t> report(65, 0);

    // The first byte could be the Report ID, as per your descriptor
    report[0] = 0x02; // Report ID, this is arbitrary but should match your device's expectations

    // Assuming dx and dy are data you want to send as part of the report
    report[1] = dx & 0xFF;        // Low byte of dx
    report[2] = (dx >> 8) & 0xFF; // High byte of dx
    report[3] = dy & 0xFF;        // Low byte of dy
    report[4] = (dy >> 8) & 0xFF; // High byte of dy

    // Send the report for processing
    reportQueue.push(report);
    //processHIDReport(report);
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
            if (detection.label == headLabel1 || detection.label == headLabel2) { // Replace with actual label values
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