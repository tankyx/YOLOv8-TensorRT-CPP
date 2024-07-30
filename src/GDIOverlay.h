// GDIOverlay.h
#pragma once
#include "yolov8.h"
#include <vector>
#include <windows.h>
#include <dwmapi.h>

class GDIOverlay {
public:
    GDIOverlay(HWND targetWnd, int screenWidth, int screenHeight);
    void drawDetections(const std::vector<Object> &detections);
    void drawDetectionsPen(const std::vector<Object> &detections);
    void drawLog(const std::string &logMessage);
    void render();
    void cleanScreen();

private:
    HWND targetWnd;
    HWND overlayWnd;
    HDC hdc;
    HDC memDC;
    HBITMAP hbmMem;
    HBRUSH hbrBkgnd;
    HFONT hFont; 
    HPEN hPen;
    RECT rect;
    int _width;
    int _height;
    void createOverlayWindow();
    void drawText(const std::string &text, int x, int y, COLORREF color, int fontSize);

    const std::vector<std::tuple<int, int, int>> COLOR_LIST = {
        {255, 0, 0},   // Red
        {255, 120, 120}, // Light Red
        {0, 0, 255},   // Blue
        {120, 120, 255}  // Light Blue
    };

    const std::vector<HBRUSH> BRUSH_LIST = {
		CreateSolidBrush(RGB(255, 0, 0)),   // Red
		CreateSolidBrush(RGB(255, 150, 150)), // Light Red
		CreateSolidBrush(RGB(0, 0, 255)),   // Blue
		CreateSolidBrush(RGB(150, 150, 255))  // Light Blue
	};

    const std::vector<HPEN> PEN_LIST = {
        CreatePen(PS_SOLID, 2, RGB(255, 0, 0)),     // Red
        CreatePen(PS_DOT, 2, RGB(255, 150, 150)), // Light Red
        CreatePen(PS_SOLID, 2, RGB(0, 0, 255)),     // Blue
        CreatePen(PS_DOT, 2, RGB(150, 150, 255))  // Light Blue
    };
};
