#include <d3d11.h>
#include <dxgi1_2.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>
#include <windows.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

class DXGICapture {
public:
    DXGICapture() { Initialize(); }

    ~DXGICapture() { Cleanup(); }

    void CaptureScreen(cv::Mat &frame) {
        ComPtr<IDXGIResource> desktopResource;
        DXGI_OUTDUPL_FRAME_INFO frameInfo;
        HRESULT hr = deskDupl->AcquireNextFrame(0, &frameInfo, &desktopResource);
        if (FAILED(hr)) {
            if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
                return; // No new frame available
            }
            throw std::runtime_error("Failed to acquire next frame.");
        }

        ComPtr<ID3D11Texture2D> acquiredDesktopImage;
        hr = desktopResource.As(&acquiredDesktopImage);
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to query for IDXGISurface interface.");
        }

        // Copy the entire screen to the CPU accessible texture
        d3dDeviceContext->CopyResource(cpuImage.Get(), acquiredDesktopImage.Get());

        D3D11_MAPPED_SUBRESOURCE mapped;
        hr = d3dDeviceContext->Map(cpuImage.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to map the copied frame.");
        }

        cv::Mat bgraFrame(screenHeight, screenWidth, CV_8UC4, mapped.pData, mapped.RowPitch);
        cv::cvtColor(bgraFrame, frame, cv::COLOR_BGRA2BGR);
        d3dDeviceContext->Unmap(cpuImage.Get(), 0);

        hr = deskDupl->ReleaseFrame();
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to release the frame.");
        }
    }

private:
    void Initialize() {
        HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
                                       D3D11_SDK_VERSION, &d3dDevice, nullptr, &d3dDeviceContext);
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to create D3D11 device.");
        }

        ComPtr<IDXGIDevice> dxgiDevice;
        hr = d3dDevice.As(&dxgiDevice);
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to get DXGI device.");
        }

        ComPtr<IDXGIAdapter> dxgiAdapter;
        hr = dxgiDevice->GetAdapter(&dxgiAdapter);
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to get DXGI adapter.");
        }

        ComPtr<IDXGIOutput> dxgiOutput;
        hr = dxgiAdapter->EnumOutputs(0, &dxgiOutput);
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to get DXGI output.");
        }

        ComPtr<IDXGIOutput1> dxgiOutput1;
        hr = dxgiOutput.As(&dxgiOutput1);
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to get DXGI output1.");
        }

        hr = dxgiOutput1->DuplicateOutput(d3dDevice.Get(), &deskDupl);
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to create desktop duplication.");
        }

        // Get the screen dimensions
        screenWidth = GetSystemMetrics(SM_CXSCREEN);
        screenHeight = GetSystemMetrics(SM_CYSCREEN);

        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = screenWidth;
        desc.Height = screenHeight;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        desc.BindFlags = 0;
        hr = d3dDevice->CreateTexture2D(&desc, nullptr, &cpuImage);
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to create CPU-accessible texture.");
        }
    }

    void Cleanup() {
        if (deskDupl) {
            deskDupl->ReleaseFrame();
        }
    }

    ComPtr<ID3D11Device> d3dDevice;
    ComPtr<ID3D11DeviceContext> d3dDeviceContext;
    ComPtr<IDXGIOutputDuplication> deskDupl;
    ComPtr<ID3D11Texture2D> cpuImage;
    int screenWidth;
    int screenHeight;
};