// Simple GDI debug window — renders a shared frame buffer.
// No OpenCV highgui dependency.  The detection thread updates the buffer
// and calls InvalidateRect; the main-loop message pump handles WM_PAINT.
class GdiDebugWindow {
public:
    GdiDebugWindow() : m_hwnd(nullptr), m_wndClass{} {}
    ~GdiDebugWindow() { destroy(); }

    bool create(const wchar_t* title, int width, int height) {
        if (m_hwnd) return true;

        m_wndClass.cbSize        = sizeof(WNDCLASSEXW);
        m_wndClass.style         = CS_HREDRAW | CS_VREDRAW;
        m_wndClass.lpfnWndProc   = s_wndProc;
        m_wndClass.hInstance     = GetModuleHandleW(nullptr);
        m_wndClass.hCursor       = LoadCursorW(nullptr, IDC_ARROW);
        m_wndClass.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        m_wndClass.lpszClassName = L"YoloDebugWindow";
        RegisterClassExW(&m_wndClass);

        // Centre the window
        int x = (GetSystemMetrics(SM_CXSCREEN) - width) / 2;
        int y = (GetSystemMetrics(SM_CYSCREEN) - height) / 2;

        RECT rc = {0, 0, width, height};
        AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

        m_hwnd = CreateWindowExW(0, m_wndClass.lpszClassName, title,
            WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME, x, y,
            rc.right - rc.left, rc.bottom - rc.top,
            nullptr, nullptr, m_wndClass.hInstance, this);
        if (!m_hwnd) return false;

        ShowWindow(m_hwnd, SW_SHOW);
        UpdateWindow(m_hwnd);
        return true;
    }

    void updateFrame(const cv::Mat& frame) {
        if (!m_hwnd || frame.empty()) return;
        cv::Mat bgr;
        if (frame.channels() == 4) cv::cvtColor(frame, bgr, cv::COLOR_BGRA2BGR);
        else if (frame.channels() == 1) cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
        else bgr = frame;

        BITMAPINFO bi = {};
        bi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
        bi.bmiHeader.biWidth       = bgr.cols;
        bi.bmiHeader.biHeight      = -bgr.rows;  // top-down
        bi.bmiHeader.biPlanes      = 1;
        bi.bmiHeader.biBitCount    = 24;
        bi.bmiHeader.biCompression = BI_RGB;

        HDC hdc = GetDC(m_hwnd);
        RECT client;
        GetClientRect(m_hwnd, &client);
        SetStretchBltMode(hdc, COLORONCOLOR);
        StretchDIBits(hdc, 0, 0, client.right, client.bottom,
                      0, 0, bgr.cols, bgr.rows, bgr.data, &bi,
                      DIB_RGB_COLORS, SRCCOPY);
        ReleaseDC(m_hwnd, hdc);
    }

    void destroy() {
        if (m_hwnd) { DestroyWindow(m_hwnd); m_hwnd = nullptr; }
        if (m_wndClass.lpszClassName[0]) UnregisterClassW(m_wndClass.lpszClassName, m_wndClass.hInstance);
    }

private:
    static LRESULT CALLBACK s_wndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
        GdiDebugWindow* self = nullptr;
        if (msg == WM_CREATE) {
            auto* cs = reinterpret_cast<CREATESTRUCT*>(lp);
            self = static_cast<GdiDebugWindow*>(cs->lpCreateParams);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
        } else {
            self = reinterpret_cast<GdiDebugWindow*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        }
        if (!self) return DefWindowProcW(hwnd, msg, wp, lp);

        switch (msg) {
        case WM_CLOSE:  ShowWindow(hwnd, SW_HIDE); return 0;
        case WM_DESTROY: self->m_hwnd = nullptr; PostQuitMessage(0); return 0;
        }
        return DefWindowProcW(hwnd, msg, wp, lp);
    }

    HWND m_hwnd;
    WNDCLASSEXW m_wndClass;
};
