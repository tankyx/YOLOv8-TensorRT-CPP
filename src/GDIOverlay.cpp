// GDIOverlay.cpp
#include "GDIOverlay.h"

GDIOverlay::GDIOverlay(HWND targetWnd, int screenWidth, int screenHeight) : targetWnd(targetWnd), _width(screenWidth), _height(screenHeight) {
    createOverlayWindow();
}

void GDIOverlay::createOverlayWindow() {
    overlayWnd = CreateWindowEx(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST, "STATIC", NULL, WS_POPUP | WS_VISIBLE, 0, 0, _width,
                                _height, NULL, NULL, NULL, NULL);
    SetLayeredWindowAttributes(overlayWnd, RGB(0, 0, 0), 255, LWA_ALPHA);
    MARGINS margin = {-1};
    DwmExtendFrameIntoClientArea(overlayWnd, &margin);

    GetWindowRect(targetWnd, &rect);

    hdc = GetDC(overlayWnd);
    memDC = CreateCompatibleDC(hdc);
    hbmMem = CreateCompatibleBitmap(hdc, _width, _height);
    SelectObject(memDC, hbmMem);
    hbrBkgnd = CreateSolidBrush(RGB(0, 0, 0));

    hFont = CreateFont(20, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                       DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, "Arial");

    // Create a pen for drawing
    hPen = CreatePen(PS_SOLID, 2, RGB(0, 0, 0));

    ShowWindow(overlayWnd, SW_SHOW);
}

void GDIOverlay::cleanScreen() {
    // Clear the memory DC
    FillRect(memDC, &rect, hbrBkgnd);
}

void GDIOverlay::drawDetectionsPen(const std::vector<Object> &detections) {
    // Calculate the offset to center the 640x640 capture zone
    int offsetX = (_width - 640) / 2;
    int offsetY = (_height - 640) / 2;

    // Draw the detections
    for (const auto &detection : detections) {
        int x = static_cast<int>(detection.rect.x) + offsetX;
        int y = static_cast<int>(detection.rect.y) + offsetY;
        int width = static_cast<int>(detection.rect.width);
        int height = static_cast<int>(detection.rect.height);

        // Change the pen color based on detection label
        SelectObject(memDC, PEN_LIST[detection.label % PEN_LIST.size()]);
        SelectObject(memDC, GetStockObject(NULL_BRUSH));

        // Draw the rectangle
        POINT points[5] = {
            {x, y}, {x + width, y}, {x + width, y + height}, {x, y + height}, {x, y} // Closing the rectangle
        };

        Polygon(memDC, points, 5);
    }
}

void GDIOverlay::drawDetections(const std::vector<Object> &detections) {
    // Calculate the offset to center the 640x640 capture zone
    int offsetX = (_width - 640) / 2;
    int offsetY = (_height - 640) / 2;

    // Draw the detections
    for (const auto &detection : detections) {
        auto brush = BRUSH_LIST[detection.label % BRUSH_LIST.size()];
        int x = static_cast<int>(detection.rect.x) + offsetX;
        int y = static_cast<int>(detection.rect.y) + offsetY;
        int width = static_cast<int>(detection.rect.width);
        int height = static_cast<int>(detection.rect.height);

        if (!brush) {
            std::cerr << "Failed to create brush." << std::endl;
            continue;
        }

        RECT detectionRect = {x, y, x + width, y + height};
        if (!FrameRect(memDC, &detectionRect, brush)) {
            std::cerr << "Failed to frame rect." << std::endl;
        }
    }
}

void GDIOverlay::drawText(const std::string &text, int x, int y, COLORREF color, int fontSize) {
    // Set the text color and background mode
    SetTextColor(memDC, color);
    SetBkMode(memDC, TRANSPARENT);

    HFONT hOldFont = (HFONT)SelectObject(memDC, hFont);

    // Draw the text line by line
    int lineHeight = fontSize + 2; // Adjust line height as needed
    int yPos = y;
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        TextOut(memDC, x, yPos, line.c_str(), line.length());
        yPos += lineHeight;
    }
}

// Usage in your drawing or rendering function
void GDIOverlay::drawLog(const std::string &logMessage) {
    // Draw the log message
    drawText(logMessage, 10, 10, RGB(255, 0, 255), 20); // Adjust position and color as needed
}

void GDIOverlay::render() {
    // Copy the memory DC to the window DC
    BitBlt(hdc, 0, 0, _width, _height, memDC, 0, 0, SRCCOPY);
}
