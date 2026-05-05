#pragma once

#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>
#include <windows.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

class DXGICapture {
public:
    DXGICapture() {
        if (!CreateDevice()) {
            throw std::runtime_error("Failed to create D3D11 device for DXGI capture.");
        }
        if (!CreateDuplication()) {
            throw std::runtime_error("Failed to create DXGI desktop duplication.");
        }
    }

    ~DXGICapture() {
        if (deskDupl) {
            deskDupl->ReleaseFrame();
        }
    }

    // Returns true if `frame` was filled with a new frame.
    // Returns false on transient errors (no new frame, recoverable access loss); the caller should keep looping.
    bool CaptureScreen(cv::Mat &frame) {
        if (!deskDupl) {
            // Try to recover from a previous access loss
            if (!CreateDuplication()) {
                return false;
            }
        }

        ComPtr<IDXGIResource> desktopResource;
        DXGI_OUTDUPL_FRAME_INFO frameInfo;
        HRESULT hr = deskDupl->AcquireNextFrame(0, &frameInfo, &desktopResource);
        if (FAILED(hr)) {
            if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
                return false; // No new frame this tick
            }
            if (hr == DXGI_ERROR_ACCESS_LOST) {
                std::cerr << "DXGI: access lost, will recreate duplication on next call." << std::endl;
                deskDupl.Reset();
                return false;
            }
            std::cerr << "DXGI: AcquireNextFrame failed (hr=0x" << std::hex << hr << std::dec << "), continuing." << std::endl;
            return false;
        }

        ComPtr<ID3D11Texture2D> acquiredDesktopImage;
        hr = desktopResource.As(&acquiredDesktopImage);
        if (FAILED(hr)) {
            std::cerr << "DXGI: failed to query ID3D11Texture2D from desktop resource." << std::endl;
            deskDupl->ReleaseFrame();
            return false;
        }

        d3dDeviceContext->CopyResource(cpuImage.Get(), acquiredDesktopImage.Get());

        D3D11_MAPPED_SUBRESOURCE mapped;
        hr = d3dDeviceContext->Map(cpuImage.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) {
            std::cerr << "DXGI: Map failed (hr=0x" << std::hex << hr << std::dec << ")." << std::endl;
            deskDupl->ReleaseFrame();
            return false;
        }

        cv::Mat bgraFrame(screenHeight, screenWidth, CV_8UC4, mapped.pData, mapped.RowPitch);
        cv::cvtColor(bgraFrame, frame, cv::COLOR_BGRA2BGR);
        d3dDeviceContext->Unmap(cpuImage.Get(), 0);

        hr = deskDupl->ReleaseFrame();
        if (FAILED(hr)) {
            // Releasing a frame can fail in transient access-loss scenarios; flag for recreate next tick.
            if (hr == DXGI_ERROR_ACCESS_LOST) {
                deskDupl.Reset();
            }
        }
        return true;
    }

private:
    bool CreateDevice() {
        HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
                                       D3D11_SDK_VERSION, &d3dDevice, nullptr, &d3dDeviceContext);
        return SUCCEEDED(hr);
    }

    bool CreateDuplication() {
        ComPtr<IDXGIDevice> dxgiDevice;
        if (FAILED(d3dDevice.As(&dxgiDevice))) {
            return false;
        }

        ComPtr<IDXGIAdapter> dxgiAdapter;
        if (FAILED(dxgiDevice->GetAdapter(&dxgiAdapter))) {
            return false;
        }

        ComPtr<IDXGIOutput> dxgiOutput;
        if (FAILED(dxgiAdapter->EnumOutputs(0, &dxgiOutput))) {
            return false;
        }

        ComPtr<IDXGIOutput1> dxgiOutput1;
        if (FAILED(dxgiOutput.As(&dxgiOutput1))) {
            return false;
        }

        if (FAILED(dxgiOutput1->DuplicateOutput(d3dDevice.Get(), &deskDupl))) {
            return false;
        }

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
        cpuImage.Reset();
        if (FAILED(d3dDevice->CreateTexture2D(&desc, nullptr, &cpuImage))) {
            return false;
        }
        return true;
    }

    ComPtr<ID3D11Device> d3dDevice;
    ComPtr<ID3D11DeviceContext> d3dDeviceContext;
    ComPtr<IDXGIOutputDuplication> deskDupl;
    ComPtr<ID3D11Texture2D> cpuImage;
    int screenWidth = 0;
    int screenHeight = 0;
};
