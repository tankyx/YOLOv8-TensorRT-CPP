#include "Overlay.h"
#include <TlHelp32.h>
#include <iostream>

Overlay::Overlay()
    : hwnd(NULL), SwapChain(nullptr), d3d11Device(nullptr), d3d11DevCon(nullptr), renderTargetView(nullptr), depthStencilView(nullptr),
      depthStencilBuffer(nullptr), VS(nullptr), PS(nullptr), VS_Buffer(nullptr), PS_Buffer(nullptr), vertLayout(nullptr),
      cbPerObjectBuffer(nullptr), rot(0.01f), memDC(nullptr), hFont(nullptr) {}

Overlay::~Overlay() { CleanUp(); }

bool Overlay::Initialize() {
    hwnd = hijackNvidiaOverlay();
    if (!hwnd)
        return false;
    if (!InitializeDirect3d11App())
        return false;

    // Create memory device context
    HDC hdc = GetDC(hwnd);
    memDC = CreateCompatibleDC(hdc);
    ReleaseDC(hwnd, hdc);

    // Create font
    hFont = CreateFont(24, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_OUTLINE_PRECIS, CLIP_DEFAULT_PRECIS,
                       CLEARTYPE_QUALITY, VARIABLE_PITCH, TEXT("Arial"));

    // Create a pen for drawing
    hPen = CreatePen(PS_SOLID, 2, RGB(0, 0, 0));

    return true;
}

void Overlay::Run() {
    MSG msg;
    ZeroMemory(&msg, sizeof(MSG));
    while (true) {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT)
                break;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        } else {
            UpdateScene();
            DrawScene();
        }
    }
}

void Overlay::CleanUp() {
    if (SwapChain)
        SwapChain->Release();
    if (d3d11Device)
        d3d11Device->Release();
    if (d3d11DevCon)
        d3d11DevCon->Release();
    if (renderTargetView)
        renderTargetView->Release();
    if (VS)
        VS->Release();
    if (PS)
        PS->Release();
    if (VS_Buffer)
        VS_Buffer->Release();
    if (PS_Buffer)
        PS_Buffer->Release();
    if (vertLayout)
        vertLayout->Release();
    if (depthStencilView)
        depthStencilView->Release();
    if (depthStencilBuffer)
        depthStencilBuffer->Release();
    if (cbPerObjectBuffer)
        cbPerObjectBuffer->Release();
    if (memDC)
        DeleteDC(memDC);
    if (hFont)
        DeleteObject(hFont);
}

HWND Overlay::hijackNvidiaOverlay() {
    HWND hwndTarget = NULL;
    // Use FindWindowEx or EnumWindows to find the target NVIDIA Overlay window
    // For simplicity, let's use FindWindowEx with known class name from your screenshot
    hwndTarget = FindWindowEx(NULL, NULL, L"CEF-OSC-WIDGET", L"NVIDIA GeForce Overlay");
    if (hwndTarget) {
        SetWindowLongPtr(hwndTarget, GWL_EXSTYLE, WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT);
        SetLayeredWindowAttributes(hwndTarget, RGB(0, 0, 0), 0, LWA_COLORKEY);
    }
    return hwndTarget;
}

bool Overlay::InitializeDirect3d11App() {
    // Describe our SwapChain Buffer
    DXGI_MODE_DESC bufferDesc;
    ZeroMemory(&bufferDesc, sizeof(DXGI_MODE_DESC));
    bufferDesc.Width = GetSystemMetrics(SM_CXSCREEN);
    bufferDesc.Height = GetSystemMetrics(SM_CYSCREEN);
    bufferDesc.RefreshRate.Numerator = 60;
    bufferDesc.RefreshRate.Denominator = 1;
    bufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    bufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
    bufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;

    // Describe our SwapChain
    DXGI_SWAP_CHAIN_DESC swapChainDesc;
    ZeroMemory(&swapChainDesc, sizeof(DXGI_SWAP_CHAIN_DESC));
    swapChainDesc.BufferDesc = bufferDesc;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.SampleDesc.Quality = 0;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.BufferCount = 1;
    swapChainDesc.OutputWindow = hwnd;
    swapChainDesc.Windowed = TRUE;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    // Create our SwapChain
    HRESULT hr = D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, NULL, NULL, NULL, D3D11_SDK_VERSION, &swapChainDesc,
                                               &SwapChain, &d3d11Device, NULL, &d3d11DevCon);
    if (FAILED(hr))
        return false;

    // Create our BackBuffer
    ID3D11Texture2D *BackBuffer;
    hr = SwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void **)&BackBuffer);
    if (FAILED(hr))
        return false;

    // Create our Render Target
    hr = d3d11Device->CreateRenderTargetView(BackBuffer, NULL, &renderTargetView);
    BackBuffer->Release();
    if (FAILED(hr))
        return false;

    // Describe our Depth/Stencil Buffer
    D3D11_TEXTURE2D_DESC depthStencilDesc;
    depthStencilDesc.Width = GetSystemMetrics(SM_CXSCREEN);
    depthStencilDesc.Height = GetSystemMetrics(SM_CYSCREEN);
    depthStencilDesc.MipLevels = 1;
    depthStencilDesc.ArraySize = 1;
    depthStencilDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    depthStencilDesc.SampleDesc.Count = 1;
    depthStencilDesc.SampleDesc.Quality = 0;
    depthStencilDesc.Usage = D3D11_USAGE_DEFAULT;
    depthStencilDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    depthStencilDesc.CPUAccessFlags = 0;
    depthStencilDesc.MiscFlags = 0;

    // Create the Depth/Stencil View
    hr = d3d11Device->CreateTexture2D(&depthStencilDesc, NULL, &depthStencilBuffer);
    if (FAILED(hr))
        return false;

    hr = d3d11Device->CreateDepthStencilView(depthStencilBuffer, NULL, &depthStencilView);
    if (FAILED(hr))
        return false;

    // Set our Render Target
    d3d11DevCon->OMSetRenderTargets(1, &renderTargetView, depthStencilView);

    // Create the buffer to send to the cbuffer in effect file
    D3D11_BUFFER_DESC cbbd;
    ZeroMemory(&cbbd, sizeof(D3D11_BUFFER_DESC));
    cbbd.Usage = D3D11_USAGE_DEFAULT;
    cbbd.ByteWidth = sizeof(XMMATRIX);
    cbbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbbd.CPUAccessFlags = 0;
    cbbd.MiscFlags = 0;
    hr = d3d11Device->CreateBuffer(&cbbd, NULL, &cbPerObjectBuffer);
    if (FAILED(hr))
        return false;

    // Camera information
    camPosition = XMVectorSet(0.0f, 3.0f, -8.0f, 0.0f);
    camTarget = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
    camUp = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

    // Set the View matrix
    camView = XMMatrixLookAtLH(camPosition, camTarget, camUp);

    // Set the Projection matrix
    camProjection =
        XMMatrixPerspectiveFovLH(0.4f * 3.14f, (float)GetSystemMetrics(SM_CXSCREEN) / GetSystemMetrics(SM_CYSCREEN), 1.0f, 1000.0f);

    return true;
}

void Overlay::DrawScene() {
    // Clear our backbuffer
    float bgColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    d3d11DevCon->ClearRenderTargetView(renderTargetView, bgColor);

    // Refresh the Depth/Stencil view
    d3d11DevCon->ClearDepthStencilView(depthStencilView, D3D11_CLEAR_DEPTH | D3D11_CLEAR_STENCIL, 1.0f, 0);

    // Present the backbuffer to the screen
    SwapChain->Present(0, 0);
}

void Overlay::DrawDetectionsPen(const std::vector<Object> &detections) {
    // Calculate the offset to center the 640x640 capture zone
    int offsetX = (GetSystemMetrics(SM_CXSCREEN) - 640) / 2;
    int offsetY = (GetSystemMetrics(SM_CYSCREEN) - 640) / 2;

    // Draw the detections
    for (const auto &detection : detections) {
        int x = detection.rect.left + offsetX;
        int y = detection.rect.top + offsetY;
        int width = detection.rect.right - detection.rect.left;
        int height = detection.rect.bottom - detection.rect.top;

        // Change the pen color based on detection label
        // Implement a method to select pen based on detection label
        SelectObject(memDC, PEN_LIST[detection.label % PEN_LIST.size()]);
        SelectObject(memDC, GetStockObject(NULL_BRUSH));

        // Draw the rectangle
        POINT points[5] = {
            {x, y}, {x + width, y}, {x + width, y + height}, {x, y + height}, {x, y} // Closing the rectangle
        };

        Polygon(memDC, points, 5);
    }
}

void Overlay::DrawText(const std::string &text, int x, int y, COLORREF color, int fontSize) {
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

    SelectObject(memDC, hOldFont);
}

LRESULT CALLBACK Overlay::WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) {
            DestroyWindow(hwnd);
        }
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}
