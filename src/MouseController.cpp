#include "MouseController.h"

#ifndef _WIN32
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <glob.h>
#include <linux/input.h>
#include <linux/uinput.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif


MouseController::MouseController(int screenWidth, int screenHeight, int detectionZoneWidth, int detectionZoneHeight, float sensitivity,
                                 int centralSquareSize, float minGain, float maxGain, float maxSpeed, int HL1, int HL2, int cpi, int nLab,
                                 float probabilityThreshold, uint16_t hidVendorId, uint16_t hidProductId, std::wstring hidSerial,
                                 float smoothing, float debugSnapGain, bool debugAimEnabled)
    : screenWidth(screenWidth), screenHeight(screenHeight), detectionZoneWidth(detectionZoneWidth),
      detectionZoneHeight(detectionZoneHeight), sensitivity(sensitivity), centralSquareSize(centralSquareSize),
      minGain(minGain), maxSpeed(maxSpeed), maxGain(maxGain),
#ifdef _WIN32
      hidDevice(nullptr),
#endif
      headLabel1(HL1), headLabel2(HL2), cpi(cpi), nLabels(nLab),
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

void MouseController::setGameCalibration(float sens, float fov, const std::string& game) {
    sensitivity = sens;
    gameFOV = fov;
    if (game == "VALORANT" || game == "Valorant" || game == "valorant") {
        m_yaw = 0.07f;
    } else if (game == "CS2" || game == "cs2") {
        m_yaw = 0.022f;
    } else {
        m_yaw = 0.022f;
    }
    useFovMethod = true;

    // Precompute focal length and angle-per-count for the hot path.
    const float fovRad = gameFOV * 3.14159265f / 180.0f;
    m_focalLength = (static_cast<float>(screenWidth) / 2.0f) / tanf(fovRad / 2.0f);
    m_anglePerCountRad = sensitivity * m_yaw * (3.14159265f / 180.0f);
}

std::pair<float, float> MouseController::pixelDeltaToCounts(float deltaX, float deltaY) const {
    if (deltaX == 0.0f && deltaY == 0.0f) return {0.0f, 0.0f};
    const float angleX = atan2f(deltaX, m_focalLength);
    const float angleY = atan2f(deltaY, m_focalLength);
    return {_countsScale * angleX / m_anglePerCountRad, _countsScale * angleY / m_anglePerCountRad};
}

MouseController::~MouseController() {
#ifdef _WIN32
    if (hidDevice) {
        CloseHandle(hidDevice);
    }
#else
    if (m_uinputFd >= 0) {
        ioctl(m_uinputFd, UI_DEV_DESTROY);
        close(m_uinputFd);
        m_uinputFd = -1;
    }
    if (m_evdevMouseFd >= 0) {
        close(m_evdevMouseFd);
        m_evdevMouseFd = -1;
    }
    if (m_evdevKbdFd >= 0) {
        close(m_evdevKbdFd);
        m_evdevKbdFd = -1;
    }
#endif
}

#ifdef _WIN32
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
#else
namespace {
// Test whether bit `code` is set in an EVIOCGBIT capability bitmask.
bool evdevHasBit(const unsigned long *bits, size_t wordCount, unsigned int code) {
    constexpr size_t wordBits = sizeof(unsigned long) * 8;
    if (code / wordBits >= wordCount) return false;
    return (bits[code / wordBits] >> (code % wordBits)) & 1UL;
}
} // namespace

bool MouseController::ConnectToDevice() {
    // Output side: create a virtual mouse on /dev/uinput. Movement and click
    // reports are emitted as evdev events (see sendHIDReport).
    m_uinputFd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if (m_uinputFd < 0) {
        std::cerr << "MouseController: cannot open /dev/uinput: " << std::strerror(errno) << ". "
                  << "Add your user to the 'input' group and install a udev rule such as "
                  << "KERNEL==\"uinput\", MODE=\"0660\", GROUP=\"input\", OPTIONS+=\"static_node=uinput\" "
                  << "(then re-login). Aim/click output is disabled." << std::endl;
        return false;
    }

    if (ioctl(m_uinputFd, UI_SET_EVBIT, EV_REL) < 0 ||
        ioctl(m_uinputFd, UI_SET_RELBIT, REL_X) < 0 ||
        ioctl(m_uinputFd, UI_SET_RELBIT, REL_Y) < 0 ||
        ioctl(m_uinputFd, UI_SET_EVBIT, EV_KEY) < 0 ||
        ioctl(m_uinputFd, UI_SET_KEYBIT, BTN_LEFT) < 0 ||
        ioctl(m_uinputFd, UI_SET_KEYBIT, BTN_RIGHT) < 0 ||
        ioctl(m_uinputFd, UI_SET_KEYBIT, BTN_MIDDLE) < 0) {
        std::cerr << "MouseController: uinput capability setup failed: " << std::strerror(errno) << std::endl;
        close(m_uinputFd);
        m_uinputFd = -1;
        return false;
    }

    struct uinput_setup setup {};
    std::snprintf(setup.name, UINPUT_MAX_NAME_SIZE, "yolo-virtual-mouse");
    setup.id.bustype = BUS_USB;
    setup.id.vendor = hidVendorId;
    setup.id.product = hidProductId;
    setup.id.version = 1;

    if (ioctl(m_uinputFd, UI_DEV_SETUP, &setup) < 0 || ioctl(m_uinputFd, UI_DEV_CREATE) < 0) {
        std::cerr << "MouseController: uinput device creation failed: " << std::strerror(errno) << std::endl;
        close(m_uinputFd);
        m_uinputFd = -1;
        return false;
    }
    std::cout << "Virtual mouse created on /dev/uinput" << std::endl;

    // Input side: locate the physical mouse (and a keyboard for the trigger
    // hold key). Without these, button queries gracefully read as released.
    if (!openInputDevices()) {
        std::cerr << "MouseController: no readable physical mouse/keyboard under /dev/input/. "
                  << "Button/trigger state will read as not pressed. Add your user to the "
                  << "'input' group or install a udev rule for /dev/input/event* (then re-login)." << std::endl;
    }
    return true;
}

bool MouseController::openInputDevices() {
    glob_t g {};
    if (glob("/dev/input/event[0-9]*", 0, nullptr, &g) != 0) {
        globfree(&g);
        return false;
    }

    constexpr size_t keyWords = (KEY_MAX + 8 * sizeof(unsigned long)) / (8 * sizeof(unsigned long));
    constexpr size_t relWords = (REL_MAX + 8 * sizeof(unsigned long)) / (8 * sizeof(unsigned long));

    for (size_t i = 0; i < g.gl_pathc && (m_evdevMouseFd < 0 || m_evdevKbdFd < 0); ++i) {
        const int fd = open(g.gl_pathv[i], O_RDONLY | O_NONBLOCK);
        if (fd < 0) continue; // permission denied or node vanished — try the next one

        char name[256] = {0};
        if (ioctl(fd, EVIOCGNAME(sizeof(name) - 1), name) < 0) name[0] = '\0';
        // Never read back our own synthetic device (or any other uinput clone).
        if (std::strcmp(name, "yolo-virtual-mouse") == 0 || std::strstr(name, "uinput") != nullptr) {
            close(fd);
            continue;
        }

        unsigned long keyBits[keyWords] = {0};
        unsigned long relBits[relWords] = {0};
        ioctl(fd, EVIOCGBIT(EV_KEY, sizeof(keyBits)), keyBits);
        ioctl(fd, EVIOCGBIT(EV_REL, sizeof(relBits)), relBits);
        const bool hasRelX = evdevHasBit(relBits, relWords, REL_X);

        // Physical mouse: relative X movement plus a left button.
        if (m_evdevMouseFd < 0 && hasRelX && evdevHasBit(keyBits, keyWords, BTN_LEFT)) {
            m_evdevMouseFd = fd;
            std::cout << "Physical mouse: " << g.gl_pathv[i] << " (" << name << ")" << std::endl;
            continue;
        }
        // Keyboard: has the trigger key plus real alpha keys, and no relative axes
        // (avoids matching mice that expose a few key codes).
        if (m_evdevKbdFd < 0 && !hasRelX && evdevHasBit(keyBits, keyWords, KEY_LEFTSHIFT) &&
            evdevHasBit(keyBits, keyWords, KEY_A)) {
            m_evdevKbdFd = fd;
            std::cout << "Trigger-key keyboard: " << g.gl_pathv[i] << " (" << name << ")" << std::endl;
            continue;
        }
        close(fd);
    }
    globfree(&g);
    return m_evdevMouseFd >= 0 && m_evdevKbdFd >= 0;
}

void MouseController::drainInputDevices() {
    struct input_event ev {};
    if (m_evdevMouseFd >= 0) {
        while (read(m_evdevMouseFd, &ev, sizeof(ev)) == static_cast<ssize_t>(sizeof(ev))) {
            if (ev.type == EV_KEY) {
                if (ev.code == BTN_LEFT) {
                    m_leftPressed = (ev.value != 0);
                } else if (ev.code == BTN_RIGHT) {
                    m_rightPressed = (ev.value != 0);
                }
            }
        }
    }
    if (m_evdevKbdFd >= 0) {
        while (read(m_evdevKbdFd, &ev, sizeof(ev)) == static_cast<ssize_t>(sizeof(ev))) {
            if (ev.type == EV_KEY && ev.code == KEY_LEFTSHIFT) {
                m_triggerKeyHeld = (ev.value != 0);
            }
        }
    }
}
#endif

void MouseController::setCrosshairPosition(int x, int y) {
    crosshairX = x;
    crosshairY = y;
}

void MouseController::applyRecoilCompensation(float dx, float dy) {
    const int16_t idx = static_cast<int16_t>(dx);
    const int16_t idy = static_cast<int16_t>(dy);
    if (idx != 0 || idy != 0) {
        sendHIDReport(idx, idy, isLeftClicking ? 0x01 : 0x00);
    }
}

#ifdef _WIN32
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
    BOOL res = WriteFile(hidDevice, report.data(), (DWORD)report.size(), &bytesWritten, NULL);
    if (res && bytesWritten == report.size()) {
        hidWarningLogged = false;
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
#endif

#ifdef _WIN32
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
#else
void MouseController::sendHIDReport(int16_t dx, int16_t dy, uint8_t button) {
    if (m_uinputFd < 0) {
        if (!hidWarningLogged) {
            std::cerr << "MouseController: uinput device unavailable; aim/click suppressed until reconnect." << std::endl;
            hidWarningLogged = true;
        }
        ConnectToDevice();
        if (m_uinputFd < 0) return;
    }

    // Mirror the Windows report semantics: dx/dy little-endian deltas plus a
    // button byte whose bit0 is LMB (see the report layout above). BTN_LEFT
    // is re-emitted on every report just like the firmware re-applies the
    // button byte; the kernel dedupes unchanged key states.
    auto emit = [this](uint16_t type, uint16_t code, int32_t value) {
        struct input_event ev {};
        ev.type = type;
        ev.code = code;
        ev.value = value;
        return write(m_uinputFd, &ev, sizeof(ev)) == static_cast<ssize_t>(sizeof(ev));
    };

    bool ok = true;
    if (dx != 0) ok = emit(EV_REL, REL_X, dx) && ok;
    if (dy != 0) ok = emit(EV_REL, REL_Y, dy) && ok;
    ok = emit(EV_KEY, BTN_LEFT, (button & 0x01) ? 1 : 0) && ok;
    ok = emit(EV_SYN, SYN_REPORT, 0) && ok;

    if (!ok) {
        std::cerr << "MouseController: uinput write failed: " << std::strerror(errno)
                  << ". Will recreate the device on next call." << std::endl;
        close(m_uinputFd);
        m_uinputFd = -1;
        return;
    }
    hidWarningLogged = false;
}
#endif

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
    // Hold-to-aim gate: LMB or RMB activates the aim. RMB is a debug-only mode —
    // aim runs but the click bit stays 0 so the firmware never sends a press to
    // the game. clickThrough tracks "the user actually wants to shoot" (LMB) and
    // gates both the HID button bit and the isLeftClicking release-on-
    // deactivation bookkeeping. (MB5 was previously a second trigger; unbound c41.)
    const bool clickThrough = isLeftMouseButtonPressed();
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

    // Released LMB but still aiming via RMB — drop the click bit cleanly. Same
    // trigger-coexistence rule applies.
    if (isLeftClicking && !clickThrough && !triggerPressed) {
        releaseLeftClick();
        isLeftClicking = false;
    }
    if (clickThrough) {
        isLeftClicking = true;
    }

    // Sort detections by screen-space distance to crosshair (closest first).
    // findClosestDetection would already pick the closest, but sorting gives
    // the caller a distance-ordered list for any multi-target logic.
    std::vector<Object> sortedDets = detections;
    std::sort(sortedDets.begin(), sortedDets.end(),
        [this](const Object &a, const Object &b) {
            const float ax = static_cast<float>(detectionZoneX + a.rect.x + a.rect.width  / 2) - static_cast<float>(crosshairX);
            const float ay = static_cast<float>(detectionZoneY + a.rect.y + a.rect.height / 2) - static_cast<float>(crosshairY);
            const float bx = static_cast<float>(detectionZoneX + b.rect.x + b.rect.width  / 2) - static_cast<float>(crosshairX);
            const float by = static_cast<float>(detectionZoneY + b.rect.y + b.rect.height / 2) - static_cast<float>(crosshairY);
            return (ax * ax + ay * ay) < (bx * bx + by * by);
        });

    const Object closest = findClosestDetection(sortedDets);
    if (closest.probability <= probabilityThreshold) {
        _bezier.deactivate();
        return;
    }

    // Target bounding box in screen space.
    const float boxL = static_cast<float>(detectionZoneX + closest.rect.x);
    const float boxT = static_cast<float>(detectionZoneY + closest.rect.y);
    const float boxR = boxL + static_cast<float>(closest.rect.width);
    const float boxB = boxT + static_cast<float>(closest.rect.height);

    // Crosshair position in screen space (from GPU tracker when enabled).
    const float chX = static_cast<float>(crosshairX);
    const float chY = static_cast<float>(crosshairY);

    // Recoil-containment delta: move crosshair to the nearest box *edge*,
    // not the centre.  When the crosshair is inside the box, no movement
    // is needed — the aimbot only activates when recoil pushes it out.
    float movePxX = 0.0f, movePxY = 0.0f;
    if      (chX < boxL)  movePxX = boxL - chX;
    else if (chX > boxR)  movePxX = boxR - chX;

    if      (chY < boxT)  movePxY = boxT - chY;
    else if (chY > boxB)  movePxY = boxB - chY;

    const bool inside = (movePxX == 0.0f && movePxY == 0.0f);

    if (inside) {
        _bezier.deactivate();
        return;
    }

    // Aim-FOV gate: skip if the containment distance exceeds the configured
    // aim field-of-view radius.  The RMB debug path ignores this gate.
    const float distToBox = std::sqrt(movePxX * movePxX + movePxY * movePxY);
    if (clickThrough && distToBox > static_cast<float>(centralSquareSize)) {
        _bezier.deactivate();
        return;
    }

    float movementX;
    float movementY;
    if (clickThrough) {
        // LMB path: fixed-fraction tracking identical to CS2Miam's spraying
        // formula (aimbot.hpp) — per-frame gain = _smoothVal * AIM_SPEED, flat
        // with distance. At Smoothing=4 that's 0.25 per tick, replacing the old
        // distance-boosted gain (up to 0.40 near the target) that snapped.
        _bezier.deactivate();
        auto [fullCountX, fullCountY] = pixelDeltaToCounts(movePxX, movePxY);
        movementX = fullCountX * _smoothVal * AIM_SPEED;
        movementY = fullCountY * _smoothVal * AIM_SPEED;
    } else {
#ifndef _WIN32
        // Linux input-path auto-calibration. The delivery chain (compositor,
        // session scaling) can amplify raw counts by an unknown factor k. On
        // the first RMB press with a target, send a fixed count burst, then
        // measure how far the SAME target actually moved on screen (we capture
        // it): k = observed_px / expected_px, and _countsScale = 1/k. Skipped
        // when the user set CountsScale explicitly in the INI.
        if (_countsScale == 1.0f && _calState != CAL_DONE) {
            const float boxCX = (boxL + boxR) * 0.5f;
            const float boxCY = (boxT + boxB) * 0.5f;
            if (_calState == CAL_IDLE) {
                _calTargetX = boxCX;
                _calTargetY = boxCY;
                _calLabel = closest.label;
                sendHIDReport(CAL_COUNTS, 0, 0x00);
                _calWaitFrames = 3; // let the view settle (~15 ms)
                _calState = CAL_SENT;
                return;
            }
            if (_calState == CAL_SENT) {
                if (--_calWaitFrames > 0) {
                    return;
                }
                _calState = CAL_MEASURE;
                return;
            }
            if (_calState == CAL_MEASURE) {
                // The view rotated right, so the scene shifted LEFT by the
                // pixel distance matching the rotation.
                const float expectedPx =
                    m_focalLength * tanf(static_cast<float>(CAL_COUNTS) * m_anglePerCountRad);
                const float observedPx = _calTargetX - boxCX;
                if (closest.label == _calLabel && std::abs(boxCY - _calTargetY) < 80.0f &&
                    observedPx > expectedPx * 0.2f) {
                    const float k = observedPx / expectedPx;
                    _countsScale = std::clamp(1.0f / k, 0.05f, 4.0f);
                    std::cout << "[MouseController] Auto-calibration: sent " << CAL_COUNTS
                              << " counts, target moved " << observedPx << "px (expected "
                              << expectedPx << "px) -> amplification x" << k
                              << ", CountsScale=" << _countsScale << std::endl;
                    _calState = CAL_DONE;
                    return;
                }
                // Target lost or ambiguous (left the ROI) — retry once from scratch.
                if (_calRetries++ < 3) {
                    _calState = CAL_IDLE;
                } else {
                    std::cerr << "[MouseController] Auto-calibration failed (target lost); "
                                 "keeping CountsScale=1. Set it manually in the INI." << std::endl;
                    _calState = CAL_DONE;
                }
                return;
            }
        }
#endif
        // RMB debug path: snap to box centre (pixel-perfect at gain=1.0).
        _bezier.deactivate();
        const float boxCX = (boxL + boxR) * 0.5f;
        const float boxCY = (boxT + boxB) * 0.5f;
        const float dX = boxCX - chX;
        const float dY = boxCY - chY;
        auto [snapCountX, snapCountY] = pixelDeltaToCounts(dX, dY);
        movementX = snapCountX * _debugSnapGain;
        movementY = snapCountY * _debugSnapGain;
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

#ifdef _WIN32
bool MouseController::isLeftMouseButtonPressed() { return (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0; }
bool MouseController::isRightMouseButtonPressed() { return (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0; }
bool MouseController::isTriggerKeyPressed() { return (GetAsyncKeyState(VK_LSHIFT) & 0x8000) != 0; } // Triggerbot hold key
#else
bool MouseController::isLeftMouseButtonPressed() {
    if (m_evdevMouseFd < 0) {
        if (!m_evdevWarnLogged) {
            std::cerr << "MouseController: no readable physical mouse evdev node; "
                      << "button queries return false. Check 'input' group / udev permissions." << std::endl;
            m_evdevWarnLogged = true;
        }
        return false;
    }
    drainInputDevices();
    return m_leftPressed;
}

bool MouseController::isRightMouseButtonPressed() {
    if (m_evdevMouseFd < 0) {
        if (!m_evdevWarnLogged) {
            std::cerr << "MouseController: no readable physical mouse evdev node; "
                      << "button queries return false. Check 'input' group / udev permissions." << std::endl;
            m_evdevWarnLogged = true;
        }
        return false;
    }
    drainInputDevices();
    return m_rightPressed;
}

// Triggerbot hold key: VK_LSHIFT on Windows -> KEY_LEFTSHIFT on Linux, read
// from a keyboard evdev node (mouse nodes don't carry keyboard keys).
bool MouseController::isTriggerKeyPressed() {
    if (m_evdevKbdFd < 0) {
        if (!m_evdevWarnLogged) {
            std::cerr << "MouseController: no readable keyboard evdev node; "
                      << "trigger key queries return false. Check 'input' group / udev permissions." << std::endl;
            m_evdevWarnLogged = true;
        }
        return false;
    }
    drainInputDevices();
    return m_triggerKeyHeld;
}
#endif

void MouseController::leftClick() { sendHIDReport(0, 0, 0x01); }

void MouseController::releaseLeftClick() { sendHIDReport(0, 0, 0x00); }

// Find the detection closest to the crosshair. Detection boxes are in capture
// space; the crosshair is in screen space. We translate detection centers to
// screen space for a correct distance comparison.
Object MouseController::findClosestDetection(const std::vector<Object> &detections) {
    Object closestDetection;
    closestDetection.probability = 0.0f;

    float closestDistance = FLT_MAX;

    for (const auto &detection : detections) {
        if (detection.label == headLabel1 || detection.label == headLabel2) {
            const float screenCX = static_cast<float>(detectionZoneX + detection.rect.x + detection.rect.width / 2);
            const float screenCY = static_cast<float>(detectionZoneY + detection.rect.y + detection.rect.height / 2);
            const float dx = screenCX - static_cast<float>(crosshairX);
            const float dy = screenCY - static_cast<float>(crosshairY);
            const float distance = dx * dx + dy * dy;
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
    constexpr auto holdDuration = milliseconds(60);
    constexpr auto armDelay = milliseconds(30);
    constexpr int cooldownMeanMs = 110;
    constexpr int cooldownJitterMs = 18;

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

    // Releasing the trigger key cancels a pending armed shot.
    if (!isTriggerKeyPressed()) {
        triggerArmed = false;
        return;
    }

    if (triggerPressed) {
        return;
    }

    // Armed shot reached its fire time — commit unconditionally. Detection is fast enough
    // that a re-check at fire time would just add latency without improving accuracy.
    if (triggerArmed && now >= triggerFireAt) {
        leftClick();
        triggerPressed = true;
        triggerReleaseAt = now + holdDuration;
        triggerArmed = false;
        return;
    }

    if (triggerArmed || now < triggerNextAllowedAt) {
        return;
    }

    // Crosshair is screen-space; detection rects are capture-space.
    // Convert crosshair to capture-space for the containment check.
    const int capCHX = crosshairX - detectionZoneX;
    const int capCHY = crosshairY - detectionZoneY;

    for (const auto &detection : detections) {
        if (detection.label >= nLabels) {
            continue;
        }
        if (capCHX >= detection.rect.x && capCHX <= detection.rect.x + detection.rect.width &&
            capCHY >= detection.rect.y && capCHY <= detection.rect.y + detection.rect.height) {
            triggerArmed = true;
            triggerFireAt = now + armDelay;
            break;
        }
    }
}