#pragma once

#include <windows.h>
#include <d2d1.h>
#include <d2d1helper.h>
#include <dwrite.h>
#include <wincodec.h>
#include <tlhelp32.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")

// Discord-overlay-backed debug renderer. Adapted from CS2Miam/overlay_discord.hpp
// (contract reverse-engineered by Samuel Tulach's OverlayCord).
//
// We render YOLO detection boxes + the resolved crosshair position into a CPU-side
// WIC bitmap with Direct2D, memcpy the dirty rows into Discord's whitelisted shared-
// memory framebuffer, and bump FrameCount. Discord's already-injected DLL composites
// our pixels into the game's swap chain at Present time — so the overlay rides the
// game's frame instead of a separate top-most window.
//
// Requires the *legacy* Discord in-game overlay (User Settings -> Game Overlay ->
// Enable in-game overlay). The new (post-2024) overlay does not expose this mapping.
class DiscordOverlay {
public:
    // Detection box in *game framebuffer* coordinates (i.e. screen-space if the game
    // is fullscreen at native resolution). Caller is responsible for translating from
    // YOLO's cropped-ROI space to screen space.
    struct DetectionBox {
        float x, y, w, h;
        int label;
        float confidence;
    };

private:
#pragma pack(push, 1)
    struct OverlayHeader {
        UINT Magic;
        UINT FrameCount;
        UINT NoClue;
        UINT Width;
        UINT Height;
        BYTE Buffer[1];
    };
#pragma pack(pop)

    // Half-open box used to track which pixels we actually touched. Discord composites
    // the full framebuffer regardless, but only memcpying the changed rect keeps memory
    // bandwidth down and lets us skip the FrameCount bump entirely when nothing changed.
    struct DirtyRect {
        int left{INT_MAX}, top{INT_MAX}, right{INT_MIN}, bottom{INT_MIN};
        bool empty() const { return left >= right || top >= bottom; }
        void reset() {
            left = top = INT_MAX;
            right = bottom = INT_MIN;
        }
        void extendBox(float x0, float y0, float x1, float y1, float pad) {
            if (x0 > x1) std::swap(x0, x1);
            if (y0 > y1) std::swap(y0, y1);
            left   = (std::min)(left,   static_cast<int>(std::floor(x0 - pad)));
            top    = (std::min)(top,    static_cast<int>(std::floor(y0 - pad)));
            right  = (std::max)(right,  static_cast<int>(std::ceil (x1 + pad)));
            bottom = (std::max)(bottom, static_cast<int>(std::ceil (y1 + pad)));
        }
        void unionWith(const DirtyRect &o) {
            if (o.empty()) return;
            if (empty()) { *this = o; return; }
            left   = (std::min)(left,   o.left);
            top    = (std::min)(top,    o.top);
            right  = (std::max)(right,  o.right);
            bottom = (std::max)(bottom, o.bottom);
        }
        void clip(int w, int h) {
            if (left   < 0) left   = 0;
            if (top    < 0) top    = 0;
            if (right  > w) right  = w;
            if (bottom > h) bottom = h;
            if (left > right || top > bottom) reset();
        }
    };

    DWORD _targetPid;

    // Discord shared mapping
    HANDLE         _mapHandle{nullptr};
    OverlayHeader *_mappedHeader{nullptr};
    UINT           _bufferWidth{0};
    UINT           _bufferHeight{0};

    // D2D / DWrite / WIC — touched ONLY from the render thread
    ID2D1Factory       *_d2dFactory{nullptr};
    IWICImagingFactory *_wicFactory{nullptr};
    IWICBitmap         *_wicBitmap{nullptr};
    ID2D1RenderTarget  *_renderTarget{nullptr};

    ID2D1SolidColorBrush *_redBrush{nullptr};
    ID2D1SolidColorBrush *_greenBrush{nullptr};
    ID2D1SolidColorBrush *_yellowBrush{nullptr};
    ID2D1SolidColorBrush *_whiteBrush{nullptr};

    IDWriteFactory    *_dwriteFactory{nullptr};
    IDWriteTextFormat *_labelFormat{nullptr};
    IDWriteTextFormat *_statsFormat{nullptr};

    // Threading
    std::thread       _renderThread;
    std::atomic<bool> _running{false};
    std::atomic<bool> _ready{false};

    // Shared state — written by detection thread, read by render thread
    std::mutex                 _stateMutex;
    std::vector<DetectionBox>  _detections;
    std::vector<std::string>   _labelNames;
    float                      _crosshairX{0.f};
    float                      _crosshairY{0.f};
    bool                       _hasCrosshair{false};

    std::atomic<double> _detectionLatencyMs{0.0};
    std::atomic<int>    _detectionFps{0};

    // Render-thread-local
    DirtyRect _currentDirty;
    DirtyRect _lastDirty;
    bool      _firstPublish{true};

    bool openDiscordMapping() {
        const std::string name = "DiscordOverlay_Framebuffer_Memory_" + std::to_string(_targetPid);
        _mapHandle = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
        if (!_mapHandle || _mapHandle == INVALID_HANDLE_VALUE) {
            std::fprintf(stderr,
                "[discord-overlay] OpenFileMappingA('%s') failed (err=%lu).\n"
                "[discord-overlay]   - Discord must be running with the LEGACY in-game overlay enabled.\n"
                "[discord-overlay]   - User Settings > Game Overlay > Enable in-game overlay.\n"
                "[discord-overlay]   - The new (post-2024) Discord overlay does NOT expose this mapping.\n",
                name.c_str(), GetLastError());
            return false;
        }

        _mappedHeader = static_cast<OverlayHeader *>(MapViewOfFile(_mapHandle, FILE_MAP_ALL_ACCESS, 0, 0, 0));
        if (!_mappedHeader) {
            std::fprintf(stderr, "[discord-overlay] MapViewOfFile failed (err=%lu)\n", GetLastError());
            CloseHandle(_mapHandle);
            _mapHandle = nullptr;
            return false;
        }

        _bufferWidth  = _mappedHeader->Width;
        _bufferHeight = _mappedHeader->Height;

        if (_bufferWidth == 0 || _bufferHeight == 0 || _bufferWidth > 7680 || _bufferHeight > 4320) {
            std::fprintf(stderr,
                "[discord-overlay] header dimensions look bogus (%ux%u) — "
                "Discord may not have initialised the buffer yet.\n",
                _bufferWidth, _bufferHeight);
            UnmapViewOfFile(_mappedHeader);
            CloseHandle(_mapHandle);
            _mappedHeader = nullptr;
            _mapHandle = nullptr;
            return false;
        }

        std::printf("[discord-overlay] connected to pid=%lu, framebuffer %ux%u\n", _targetPid, _bufferWidth, _bufferHeight);
        return true;
    }

    void closeDiscordMapping() {
        if (_mappedHeader) {
            UnmapViewOfFile(_mappedHeader);
            _mappedHeader = nullptr;
        }
        if (_mapHandle) {
            CloseHandle(_mapHandle);
            _mapHandle = nullptr;
        }
    }

    bool initGfx() {
        HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
            std::fprintf(stderr, "[discord-overlay] CoInitializeEx failed (0x%08lx)\n", hr);
            return false;
        }

        if (FAILED(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &_d2dFactory))) return false;
        if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&_wicFactory)))) return false;

        if (FAILED(_wicFactory->CreateBitmap(_bufferWidth, _bufferHeight, GUID_WICPixelFormat32bppPBGRA,
                                              WICBitmapCacheOnLoad, &_wicBitmap)))
            return false;

        D2D1_RENDER_TARGET_PROPERTIES rtProps =
            D2D1::RenderTargetProperties(D2D1_RENDER_TARGET_TYPE_DEFAULT,
                                          D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED),
                                          96.0f, 96.0f);
        if (FAILED(_d2dFactory->CreateWicBitmapRenderTarget(_wicBitmap, rtProps, &_renderTarget))) return false;

        _renderTarget->CreateSolidColorBrush(D2D1::ColorF(1.0f, 0.1f, 0.1f, 1.0f), &_redBrush);
        _renderTarget->CreateSolidColorBrush(D2D1::ColorF(0.2f, 1.0f, 0.2f, 1.0f), &_greenBrush);
        _renderTarget->CreateSolidColorBrush(D2D1::ColorF(1.0f, 1.0f, 0.0f, 1.0f), &_yellowBrush);
        _renderTarget->CreateSolidColorBrush(D2D1::ColorF(1.0f, 1.0f, 1.0f, 1.0f), &_whiteBrush);

        _renderTarget->SetAntialiasMode(D2D1_ANTIALIAS_MODE_PER_PRIMITIVE);

        if (SUCCEEDED(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory),
                                          reinterpret_cast<IUnknown **>(&_dwriteFactory)))) {
            _dwriteFactory->CreateTextFormat(L"Consolas", nullptr, DWRITE_FONT_WEIGHT_BOLD, DWRITE_FONT_STYLE_NORMAL,
                                              DWRITE_FONT_STRETCH_NORMAL, 12.0f, L"en-us", &_labelFormat);
            _dwriteFactory->CreateTextFormat(L"Consolas", nullptr, DWRITE_FONT_WEIGHT_BOLD, DWRITE_FONT_STYLE_NORMAL,
                                              DWRITE_FONT_STRETCH_NORMAL, 14.0f, L"en-us", &_statsFormat);
        }

        return true;
    }

    void cleanupGfx() {
        if (_redBrush)      { _redBrush->Release();      _redBrush      = nullptr; }
        if (_greenBrush)    { _greenBrush->Release();    _greenBrush    = nullptr; }
        if (_yellowBrush)   { _yellowBrush->Release();   _yellowBrush   = nullptr; }
        if (_whiteBrush)    { _whiteBrush->Release();    _whiteBrush    = nullptr; }
        if (_labelFormat)   { _labelFormat->Release();   _labelFormat   = nullptr; }
        if (_statsFormat)   { _statsFormat->Release();   _statsFormat   = nullptr; }
        if (_dwriteFactory) { _dwriteFactory->Release(); _dwriteFactory = nullptr; }
        if (_renderTarget)  { _renderTarget->Release();  _renderTarget  = nullptr; }
        if (_wicBitmap)     { _wicBitmap->Release();     _wicBitmap     = nullptr; }
        if (_wicFactory)    { _wicFactory->Release();    _wicFactory    = nullptr; }
        if (_d2dFactory)    { _d2dFactory->Release();    _d2dFactory    = nullptr; }
        CoUninitialize();
    }

    void publishToDiscord() {
        if (!_wicBitmap || !_mappedHeader) return;

        DirtyRect total;
        total.unionWith(_currentDirty);
        total.unionWith(_lastDirty);
        _lastDirty = _currentDirty;

        const int bw = static_cast<int>(_bufferWidth);
        const int bh = static_cast<int>(_bufferHeight);

        if (_firstPublish) {
            total.left = 0; total.top = 0;
            total.right = bw; total.bottom = bh;
            _firstPublish = false;
        } else {
            total.clip(bw, bh);
            if (total.empty()) return; // skip the FrameCount bump entirely
        }

        const int rectW = total.right - total.left;
        const int rectH = total.bottom - total.top;

        WICRect lockRect = {0, 0, static_cast<INT>(_bufferWidth), static_cast<INT>(_bufferHeight)};
        IWICBitmapLock *lock = nullptr;
        if (FAILED(_wicBitmap->Lock(&lockRect, WICBitmapLockRead, &lock)) || !lock) return;

        UINT  srcStride = 0;
        UINT  srcSize   = 0;
        BYTE *srcData   = nullptr;
        lock->GetStride(&srcStride);
        lock->GetDataPointer(&srcSize, &srcData);

        const UINT  dstStride = _bufferWidth * 4;
        const UINT  rowBytes  = static_cast<UINT>(rectW) * 4;
        BYTE       *dst       = _mappedHeader->Buffer + static_cast<size_t>(total.top) * dstStride
                                                       + static_cast<size_t>(total.left) * 4;
        const BYTE *src       = srcData + static_cast<size_t>(total.top) * srcStride
                                         + static_cast<size_t>(total.left) * 4;

        for (int y = 0; y < rectH; ++y) {
            std::memcpy(dst + y * dstStride, src + y * srcStride, rowBytes);
        }

        lock->Release();
        _mappedHeader->FrameCount++;
    }

    static std::wstring widen(const std::string &narrow) {
        std::wstring w(narrow.begin(), narrow.end());
        return w;
    }

    void render() {
        if (!_renderTarget) return;

        std::vector<DetectionBox> dets;
        std::vector<std::string>  labelNames;
        float chx = 0.f, chy = 0.f;
        bool  hasCh = false;
        {
            std::lock_guard<std::mutex> lock(_stateMutex);
            dets       = _detections;
            labelNames = _labelNames;
            chx        = _crosshairX;
            chy        = _crosshairY;
            hasCh      = _hasCrosshair;
        }

        const double latencyMs = _detectionLatencyMs.load(std::memory_order_relaxed);
        const int    fps       = _detectionFps.load(std::memory_order_relaxed);

        _currentDirty.reset();

        _renderTarget->BeginDraw();
        _renderTarget->Clear(D2D1::ColorF(0.0f, 0.0f, 0.0f, 0.0f));

        // Detection boxes + labels
        for (const auto &d : dets) {
            D2D1_RECT_F rect = D2D1::RectF(d.x, d.y, d.x + d.w, d.y + d.h);
            ID2D1SolidColorBrush *brush = _redBrush; // simplest scheme: any class -> red
            _renderTarget->DrawRectangle(rect, brush, 1.5f);
            _currentDirty.extendBox(d.x, d.y, d.x + d.w, d.y + d.h, 2.0f);

            if (_labelFormat) {
                wchar_t labelBuf[64];
                const char *className = (d.label >= 0 && static_cast<size_t>(d.label) < labelNames.size())
                                            ? labelNames[d.label].c_str()
                                            : "?";
                std::wstring wide = widen(className);
                swprintf_s(labelBuf, L"%ls %.0f%%", wide.c_str(), d.confidence * 100.0f);
                D2D1_RECT_F textRect = D2D1::RectF(d.x, (std::max)(0.f, d.y - 14.f), d.x + 200.f, d.y);
                _renderTarget->DrawText(labelBuf, static_cast<UINT32>(wcslen(labelBuf)), _labelFormat, textRect, brush);
                _currentDirty.extendBox(textRect.left, textRect.top, textRect.right, textRect.bottom, 2.0f);
            }
        }

        // Crosshair debug box — hollow yellow rectangle at the detected crosshair position
        if (hasCh) {
            const float halfSize = 8.0f; // 16×16 pixel box
            D2D1_RECT_F rect = D2D1::RectF(chx - halfSize, chy - halfSize, chx + halfSize, chy + halfSize);
            _renderTarget->DrawRectangle(rect, _yellowBrush, 1.5f);
            _currentDirty.extendBox(rect.left, rect.top, rect.right, rect.bottom, 2.0f);
        }

        // Stats line top-left
        if (_statsFormat) {
            wchar_t statsBuf[96];
            swprintf_s(statsBuf, L"YOLO  det=%.2fms  fps=%d  n=%zu", latencyMs, fps, dets.size());
            D2D1_RECT_F textRect = D2D1::RectF(10.0f, 10.0f, 400.0f, 30.0f);
            _renderTarget->DrawText(statsBuf, static_cast<UINT32>(wcslen(statsBuf)), _statsFormat, textRect, _whiteBrush);
            _currentDirty.extendBox(textRect.left, textRect.top, textRect.right, textRect.bottom, 2.0f);
        }

        _renderTarget->EndDraw();
        publishToDiscord();
    }

    void renderThreadMain() {
        if (!openDiscordMapping()) {
            _running.store(false, std::memory_order_release);
            _ready.store(true, std::memory_order_release);
            return;
        }
        if (!initGfx()) {
            closeDiscordMapping();
            _running.store(false, std::memory_order_release);
            _ready.store(true, std::memory_order_release);
            return;
        }
        _ready.store(true, std::memory_order_release);

        // ~120 Hz overlay refresh — generous for a debug view, well under the game's
        // present rate. Discord composites only on frames where FrameCount changed.
        const auto frameInterval = std::chrono::microseconds(8333);
        auto       nextFrame     = std::chrono::steady_clock::now() + frameInterval;

        while (_running.load(std::memory_order_acquire)) {
            auto now = std::chrono::steady_clock::now();
            if (now >= nextFrame) {
                render();
                nextFrame += frameInterval;
                if (nextFrame < now) nextFrame = now + frameInterval;
            } else {
                std::this_thread::yield();
            }
        }

        // Best-effort: blank the buffer so we don't leave stale boxes on screen.
        if (_renderTarget) {
            _renderTarget->BeginDraw();
            _renderTarget->Clear(D2D1::ColorF(0.0f, 0.0f, 0.0f, 0.0f));
            _renderTarget->EndDraw();
            _firstPublish = true;
            publishToDiscord();
        }

        cleanupGfx();
        closeDiscordMapping();
    }

public:
    explicit DiscordOverlay(DWORD targetPid) : _targetPid(targetPid) {}
    ~DiscordOverlay() { stop(); }

    DiscordOverlay(const DiscordOverlay &) = delete;
    DiscordOverlay &operator=(const DiscordOverlay &) = delete;

    bool start() {
        if (_running.exchange(true)) return _mappedHeader != nullptr;
        _ready.store(false, std::memory_order_release);
        _renderThread = std::thread(&DiscordOverlay::renderThreadMain, this);
        while (!_ready.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return _running.load(std::memory_order_acquire) && _mappedHeader != nullptr;
    }

    void stop() {
        _running.store(false, std::memory_order_release);
        if (_renderThread.joinable()) _renderThread.join();
    }

    bool isRunning() const { return _mappedHeader != nullptr && _running.load(std::memory_order_relaxed); }

    UINT getBufferWidth() const { return _bufferWidth; }
    UINT getBufferHeight() const { return _bufferHeight; }

    void setLabelNames(std::vector<std::string> names) {
        std::lock_guard<std::mutex> lock(_stateMutex);
        _labelNames = std::move(names);
    }

    void setDetections(std::vector<DetectionBox> boxes) {
        std::lock_guard<std::mutex> lock(_stateMutex);
        _detections = std::move(boxes);
    }

    void setCrosshair(float x, float y) {
        std::lock_guard<std::mutex> lock(_stateMutex);
        _crosshairX  = x;
        _crosshairY  = y;
        _hasCrosshair = true;
    }

    void clearCrosshair() {
        std::lock_guard<std::mutex> lock(_stateMutex);
        _hasCrosshair = false;
    }

    void setStats(double latencyMs, int fps) {
        _detectionLatencyMs.store(latencyMs, std::memory_order_relaxed);
        _detectionFps.store(fps, std::memory_order_relaxed);
    }

    // Find the first running process whose exe name matches `exeName` (case-insensitive,
    // narrow ASCII). Returns 0 if not found.
    static DWORD findProcessIdByName(const std::string &exeName) {
        HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (snapshot == INVALID_HANDLE_VALUE) return 0;

        PROCESSENTRY32 pe{};
        pe.dwSize = sizeof(pe);
        DWORD pid = 0;

        auto iequals = [](const char *a, const char *b) {
            for (; *a && *b; ++a, ++b) {
                if (std::tolower(static_cast<unsigned char>(*a)) != std::tolower(static_cast<unsigned char>(*b))) {
                    return false;
                }
            }
            return *a == 0 && *b == 0;
        };

        if (Process32First(snapshot, &pe)) {
            do {
                if (iequals(pe.szExeFile, exeName.c_str())) {
                    pid = pe.th32ProcessID;
                    break;
                }
            } while (Process32Next(snapshot, &pe));
        }

        CloseHandle(snapshot);
        return pid;
    }
};