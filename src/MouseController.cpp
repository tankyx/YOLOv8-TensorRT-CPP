#include "MouseController.h"


MouseController::MouseController(int screenWidth, int screenHeight, int detectionZoneWidth, int detectionZoneHeight, float sensitivity,
                                 int centralSquareSize, float minGain, float maxGain, float maxSpeed, int HL1, int HL2, int cpi, int nLab,
                                 float probabilityThreshold, uint16_t hidVendorId, uint16_t hidProductId, std::wstring hidSerial,
                                 float smoothing, float debugSnapGain, bool debugAimEnabled)
    : screenWidth(screenWidth), screenHeight(screenHeight), detectionZoneWidth(detectionZoneWidth),
      detectionZoneHeight(detectionZoneHeight), sensitivity(sensitivity), centralSquareSize(centralSquareSize),
      minGain(minGain), maxSpeed(maxSpeed), maxGain(maxGain), hidDevice(nullptr), headLabel1(HL1), headLabel2(HL2), cpi(cpi), nLabels(nLab),
      probabilityThreshold(probabilityThreshold), hidVendorId(hidVendorId), hidProductId(hidProductId), hidSerial(std::move(hidSerial)) {
    setDebugSnapGain(debugSnapGain);
    setDebugAimEnabled(debugAimEnabled);
    // Calculate the top-left corner of the detection zone
    detectionZoneX = (screenWidth - detectionZoneWidth) / 2;
    detectionZoneY = (screenHeight - detectionZoneHeight) / 2;

    crosshairX = screenWidth / 2;
    crosshairY = screenHeight / 2;

    isLeftClicking = false;

    setSmoothing(smoothing);

    ConnectToDevice();
}

void MouseController::setSmoothing(float userVal) {
    _userSmoothing = std::clamp(userVal, 1.0f, 10.0f);
    _smoothVal = smoothingToInternal(_userSmoothing);
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
                    if (attributes.VendorID == hidVendorId && attributes.ProductID == hidProductId) {
                        std::cout << "Found matching device: Vendor ID: " << std::hex << attributes.VendorID
                                  << ", Product ID: " << attributes.ProductID << std::dec << std::endl;

                        wchar_t serialNumber[256];
                        if (HidD_GetSerialNumberString(deviceHandle, serialNumber, sizeof(serialNumber))) {
                            std::wcout << L"Device serial number: " << serialNumber << std::endl;
                            if (wcscmp(serialNumber, hidSerial.c_str()) == 0) {
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
                                std::wcout << L"Serial number does not match. Expected: " << hidSerial << L", Received: "
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

bool MouseController::processHIDReport(std::vector<uint8_t> &report) {
    if (hidDevice == nullptr) {
        if (!hidWarningLogged) {
            std::cerr << "MouseController: HID device unavailable; aim/click suppressed until reconnect." << std::endl;
            hidWarningLogged = true;
        }
        ConnectToDevice();
        return false;
    }

    DWORD bytesWritten = 0;
    BOOL res = WriteFile(hidDevice, report.data(), report.size(), &bytesWritten, NULL);
    if (res && bytesWritten == report.size()) {
        hidWarningLogged = false; // back to healthy — re-arm the one-shot
        return true;
    }

    int err = GetLastError();
    std::cerr << "MouseController: WriteFile failed (err=" << err << ")." << std::endl;

    if (err == 995 || err == 1167) { // device removed / pending I/O cancelled
        std::cerr << "MouseController: device disconnected; will retry on next call." << std::endl;
        CloseHandle(hidDevice);
        hidDevice = nullptr;
        return false;
    }

    // Any other write failure: don't exit(1); let the caller keep running. The device may still
    // be present but transiently unavailable (e.g. USB suspend). Drop the report this tick.
    return false;
}

void MouseController::sendHIDReport(int16_t dx, int16_t dy, uint8_t button) {
    // Create a 64-byte report
    std::vector<uint8_t> report(65, 0);

    // The first byte could be the Report ID, as per your descriptor
    report[0] = 0x02; // Report ID, this is arbitrary but should match your device's expectations

    // Assuming dx and dy are data you want to send as part of the report
    report[1] = dx & 0xFF;        // Low byte of dx
    report[2] = (dx >> 8) & 0xFF; // High byte of dx
    report[3] = dy & 0xFF;        // Low byte of dy
    report[4] = (dy >> 8) & 0xFF; // High byte of dy
    report[5] = button;           // Button state

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
    // Hold-to-aim gate: any of LMB, MB5, or RMB activates the aim. RMB is a
    // debug-only mode — aim runs but the click bit stays 0 so the firmware
    // never sends a press to the game. clickThrough tracks "the user actually
    // wants to shoot" (LMB or MB5) and gates both the HID button bit and the
    // isLeftClicking release-on-deactivation bookkeeping.
    const bool clickThrough = isLeftMouseButtonPressed() || isMouseButton5Pressed();
    const bool debugAimHeld = _debugAimEnabled && isRightMouseButtonPressed();
    const bool aimingActive = clickThrough || debugAimHeld;

    if (!aimingActive) {
        // Don't release LMB if the triggerbot is mid-hold — that would cut its
        // shot short. The trigger's own release path will handle it.
        if (isLeftClicking && !triggerPressed) {
            releaseLeftClick();
            isLeftClicking = false;
        }
        _bezier.deactivate();
        _residX = 0.0f;
        _residY = 0.0f;
        return;
    }

    // Released LMB/MB5 but still aiming via RMB — drop the click bit cleanly.
    // Same trigger-coexistence rule applies.
    if (isLeftClicking && !clickThrough && !triggerPressed) {
        releaseLeftClick();
        isLeftClicking = false;
    }
    if (clickThrough) {
        isLeftClicking = true;
    }

    const Object closest = findClosestDetection(detections);
    if (closest.probability <= probabilityThreshold) {
        _bezier.deactivate();
        return;
    }

    const float targetX = static_cast<float>(closest.rect.x + closest.rect.width / 2);
    const float targetY = static_cast<float>(closest.rect.y + closest.rect.height / 2);
    const float deltaX = targetX - static_cast<float>(crosshairX);
    const float deltaY = targetY - static_cast<float>(crosshairY);

    // Tracking dead zone — if the crosshair is essentially on target, don't
    // move (stops jitter from sub-pixel oscillation).
    if (deltaX * deltaX + deltaY * deltaY < 1.0f) {
        _bezier.deactivate();
        return;
    }

    float movementX;
    float movementY;
    if (clickThrough) {
        // Real aim path: Bezier-smoothed flick, re-anchored when the target jumps.
        const cv::Point2f currentTarget(deltaX, deltaY);
        if (_bezier.shouldReinitialize(currentTarget)) {
            _bezier.initialize(cv::Point2f(0.0f, 0.0f), currentTarget);
        }
        _bezier.update(_smoothVal);
        cv::Point2f bezierPos = _bezier.getCurrentPosition();
        movementX = bezierPos.x;
        movementY = bezierPos.y;
        _bezier.updateStartPosition(cv::Point2f(movementX, movementY));
    } else {
        // Debug aim (RMB-only): no smoothing — snap straight at the target.
        // _debugSnapGain converts ROI pixels into mouse counts for the user's
        // in-game sensitivity (1 count ≠ 1 ROI px, so the raw delta under-rotates).
        _bezier.deactivate();
        movementX = deltaX * _debugSnapGain;
        movementY = deltaY * _debugSnapGain;
    }

    // Carry sub-pixel residue across frames so slow tracking doesn't truncate
    // to zero counts every report.
    movementX += _residX;
    movementY += _residY;
    const int16_t dX = static_cast<int16_t>(movementX);
    const int16_t dY = static_cast<int16_t>(movementY);
    _residX = movementX - static_cast<float>(dX);
    _residY = movementY - static_cast<float>(dY);

    _dx = dX;
    _dy = dY;
    // Coexist with the triggerbot: if the trigger is mid-hold this tick, keep
    // the LMB bit set in our movement report so we don't yank LMB up before
    // the hold completes.
    const bool buttonHeld = clickThrough || triggerPressed;
    if (dX != 0 || dY != 0) {
        sendHIDReport(dX, dY, buttonHeld ? 0x01 : 0x00);
    }
}

bool MouseController::isLeftMouseButtonPressed() { return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0; }
bool MouseController::isRightMouseButtonPressed() { return (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0; }
bool MouseController::isTriggerKeyPressed() { return (GetAsyncKeyState(VK_LSHIFT) & 0x8000) != 0; } // Triggerbot hold key
bool MouseController::isMouseButton5Pressed() { return (GetAsyncKeyState(VK_XBUTTON2) & 0x8000) != 0; } // Mouse Button 5

void MouseController::leftClick() { sendHIDReport(0, 0, 0x01); }

void MouseController::releaseLeftClick() { sendHIDReport(0, 0, 0x00); }

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
    using namespace std::chrono;
    constexpr auto holdDuration = milliseconds(100);
    constexpr int cooldownMeanMs = 200;
    constexpr int cooldownJitterMs = 35;

    const auto now = steady_clock::now();

    // If a press is in flight, see whether it's time to release. This runs every detection
    // tick instead of blocking the thread with Sleep(). Cooldown is sampled fresh per shot
    // in [mean-jitter, mean+jitter] ms so the trigger cadence isn't a perfect metronome.
    //
    // We unconditionally release here. We can't reliably check "is the user physically
    // holding LMB?" because GetAsyncKeyState(VK_LBUTTON) reflects the aggregate OS state
    // across all mouse devices — including our own HID, which set LMB high at press time.
    // If the user really is still holding LMB, aim()'s next iteration will re-assert it.
    if (triggerPressed && now >= triggerReleaseAt) {
        releaseLeftClick();
        triggerPressed = false;
        std::uniform_int_distribution<int> jitter(-cooldownJitterMs, cooldownJitterMs);
        triggerNextAllowedAt = now + milliseconds(cooldownMeanMs + jitter(_triggerRng));
    }

    if (!isTriggerKeyPressed() || triggerPressed || now < triggerNextAllowedAt) {
        return;
    }

    const int centerX = crosshairX;
    const int centerY = crosshairY;

    for (const auto &detection : detections) {
        if (detection.label >= nLabels) {
            continue;
        }
        if (centerX >= detection.rect.x && centerX <= detection.rect.x + detection.rect.width &&
            centerY >= detection.rect.y && centerY <= detection.rect.y + detection.rect.height) {
            leftClick();
            triggerPressed = true;
            triggerReleaseAt = now + holdDuration;
            break;
        }
    }
}