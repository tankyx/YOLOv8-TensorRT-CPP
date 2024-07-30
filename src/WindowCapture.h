#include <d3d11.h>
#include <d3d11sdklayers.h>
#include <dxgi1_2.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

class WindowCapture {
public:
    WindowCapture(HWND hwnd) : hwnd(hwnd) { Initialize(); }

    ~WindowCapture() { Cleanup(); }

    void Capture(cv::Mat &frame);

private:
    void Initialize();
    void Cleanup();

    HWND hwnd;
    ComPtr<ID3D11Device> d3dDevice;
    ComPtr<ID3D11DeviceContext> d3dDeviceContext;
    ComPtr<IDXGISwapChain1> swapChain;
    ComPtr<ID3D11Texture2D> backBuffer;
};

void WindowCapture::Initialize() {
    UINT createDeviceFlags = 0;

    // Create a D3D11 device and context
    D3D_FEATURE_LEVEL featureLevels[] = {D3D_FEATURE_LEVEL_11_0};
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, featureLevels, ARRAYSIZE(featureLevels),
                                   D3D11_SDK_VERSION, &d3dDevice, &featureLevels[0], &d3dDeviceContext);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create D3D11 device");
    }

    std::cout << "Created D3D11 device" << std::endl;

    // Get DXGI device
    ComPtr<IDXGIDevice> dxgiDevice;
    hr = d3dDevice.As(&dxgiDevice);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to get DXGI device");
    }

    std::cout << "Got DXGI device" << std::endl;

    // Get DXGI adapter
    ComPtr<IDXGIAdapter> dxgiAdapter;
    hr = dxgiDevice->GetAdapter(&dxgiAdapter);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to get DXGI adapter");
    }

    std::cout << "Got DXGI adapter" << std::endl;

    // Get DXGI factory
    ComPtr<IDXGIFactory2> dxgiFactory;
    hr = dxgiAdapter->GetParent(__uuidof(IDXGIFactory2), reinterpret_cast<void **>(dxgiFactory.GetAddressOf()));
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to get DXGI factory");
    }

    std::cout << "Got DXGI factory" << std::endl;

    // Get the client rect of the window
    RECT rect;
    if (!GetClientRect(hwnd, &rect)) {
        throw std::runtime_error("Failed to get client rect");
    }
    UINT width = rect.right - rect.left;
    UINT height = rect.bottom - rect.top;

    // Describe the swap chain
    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.BufferCount = 2;
    swapChainDesc.Width = width;
    swapChainDesc.Height = height;
    swapChainDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.SampleDesc.Quality = 0;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

    // Create the swap chain
    hr = dxgiFactory->CreateSwapChainForHwnd(d3dDevice.Get(), hwnd, &swapChainDesc, nullptr, nullptr, &swapChain);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create swap chain");
    }

    std::cout << "Created swap chain" << std::endl;

    // Get the back buffer
    hr = swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void **>(backBuffer.GetAddressOf()));
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to get back buffer");
    }

    std::cout << "Got back buffer" << std::endl;
}

void WindowCapture::Cleanup() {
    if (swapChain) {
        swapChain->SetFullscreenState(FALSE, nullptr);
    }
}

void WindowCapture::Capture(cv::Mat &frame) {
    // Map the back buffer
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    HRESULT hr = d3dDeviceContext->Map(backBuffer.Get(), 0, D3D11_MAP_READ, 0, &mappedResource);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to map back buffer");
    }

    // Create an OpenCV matrix from the mapped back buffer
    D3D11_TEXTURE2D_DESC desc;
    backBuffer->GetDesc(&desc);
    frame = cv::Mat(desc.Height, desc.Width, CV_8UC4, mappedResource.pData, mappedResource.RowPitch);

    // Unmap the back buffer
    d3dDeviceContext->Unmap(backBuffer.Get(), 0);
}
