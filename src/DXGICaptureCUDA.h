// DXGICaptureCUDA.h
//
// Direct GPU capture path: DXGI Desktop Duplication -> GPU-only staging texture (D3D11) ->
// cudaGraphicsD3D11RegisterResource -> cudaArray -> cudaMemcpy2DFromArrayAsync -> linear BGRA
// GpuMat. Replaces the GPU->CPU staging map + CPU cvtColor + GpuMat::upload that the legacy
// DXGICapture path goes through. Output is BGRA (CV_8UC4); the fused preproc kernel ignores the
// alpha channel.

#pragma once

#include <cuda_d3d11_interop.h>
#include <cuda_runtime.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <stdexcept>
#include <windows.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

class DXGICaptureCUDA {
public:
    DXGICaptureCUDA() {
        if (!CreateDevice()) {
            throw std::runtime_error("DXGICaptureCUDA: failed to create D3D11 device.");
        }
        if (!CreateDuplication()) {
            throw std::runtime_error("DXGICaptureCUDA: failed to create DXGI duplication.");
        }
    }

    ~DXGICaptureCUDA() {
        if (m_cudaResource) {
            cudaGraphicsUnregisterResource(m_cudaResource);
            m_cudaResource = nullptr;
        }
        if (m_deskDupl) {
            m_deskDupl->ReleaseFrame();
        }
    }

    // Capture one frame into `frame` (CV_8UC4 BGRA). Sized to the screen resolution; the caller
    // crops to ROI afterwards. Returns false on transient errors (no new frame, access lost).
    bool CaptureScreen(cv::cuda::GpuMat &frame, cudaStream_t stream) {
        if (!m_deskDupl) {
            if (!CreateDuplication()) {
                return false;
            }
        }

        ComPtr<IDXGIResource> desktopResource;
        DXGI_OUTDUPL_FRAME_INFO frameInfo;
        HRESULT hr = m_deskDupl->AcquireNextFrame(0, &frameInfo, &desktopResource);
        if (FAILED(hr)) {
            if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
                return false;
            }
            if (hr == DXGI_ERROR_ACCESS_LOST) {
                std::cerr << "DXGICaptureCUDA: access lost, will recreate duplication on next call." << std::endl;
                m_deskDupl.Reset();
                return false;
            }
            std::cerr << "DXGICaptureCUDA: AcquireNextFrame failed (hr=0x" << std::hex << hr << std::dec << ")." << std::endl;
            return false;
        }

        ComPtr<ID3D11Texture2D> desktopTex;
        hr = desktopResource.As(&desktopTex);
        if (FAILED(hr)) {
            m_deskDupl->ReleaseFrame();
            return false;
        }

        // Copy desktop -> our owned staging tex (GPU->GPU, no CPU touch).
        m_d3dContext->CopyResource(m_stagingTex.Get(), desktopTex.Get());

        // Map the staging texture for CUDA reads. The mapped array is read-only — we never
        // write back from CUDA, so this is cheap and safe.
        cudaError_t cerr = cudaGraphicsMapResources(1, &m_cudaResource, stream);
        if (cerr != cudaSuccess) {
            std::cerr << "DXGICaptureCUDA: cudaGraphicsMapResources failed: " << cudaGetErrorString(cerr) << std::endl;
            m_deskDupl->ReleaseFrame();
            return false;
        }

        cudaArray_t mappedArray = nullptr;
        cerr = cudaGraphicsSubResourceGetMappedArray(&mappedArray, m_cudaResource, 0, 0);
        if (cerr != cudaSuccess) {
            cudaGraphicsUnmapResources(1, &m_cudaResource, stream);
            m_deskDupl->ReleaseFrame();
            return false;
        }

        if (frame.rows != m_screenHeight || frame.cols != m_screenWidth || frame.type() != CV_8UC4) {
            frame.create(m_screenHeight, m_screenWidth, CV_8UC4);
        }

        const size_t widthBytes = static_cast<size_t>(m_screenWidth) * 4;
        cerr = cudaMemcpy2DFromArrayAsync(frame.ptr<uint8_t>(), frame.step, mappedArray, 0, 0, widthBytes,
                                          static_cast<size_t>(m_screenHeight), cudaMemcpyDeviceToDevice, stream);

        cudaGraphicsUnmapResources(1, &m_cudaResource, stream);
        m_deskDupl->ReleaseFrame();

        if (cerr != cudaSuccess) {
            std::cerr << "DXGICaptureCUDA: cudaMemcpy2DFromArrayAsync failed: " << cudaGetErrorString(cerr) << std::endl;
            return false;
        }
        return true;
    }

    int screenWidth() const { return m_screenWidth; }
    int screenHeight() const { return m_screenHeight; }

private:
    bool CreateDevice() {
        HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
                                       D3D11_SDK_VERSION, &m_d3dDevice, nullptr, &m_d3dContext);
        return SUCCEEDED(hr);
    }

    bool CreateDuplication() {
        ComPtr<IDXGIDevice> dxgiDevice;
        if (FAILED(m_d3dDevice.As(&dxgiDevice))) {
            return false;
        }

        ComPtr<IDXGIAdapter> dxgiAdapter;
        if (FAILED(dxgiDevice->GetAdapter(&dxgiAdapter))) {
            return false;
        }

        // CUDA-D3D11 interop requires both devices on the same adapter. cudaD3D11GetDevice
        // returns the CUDA device index for a given DXGI adapter; bind it before any CUDA
        // resource registration on this thread.
        int cudaDevice = -1;
        cudaError_t cerr = cudaD3D11GetDevice(&cudaDevice, dxgiAdapter.Get());
        if (cerr != cudaSuccess) {
            std::cerr << "DXGICaptureCUDA: cudaD3D11GetDevice failed (no CUDA device matches the D3D11 adapter): "
                      << cudaGetErrorString(cerr) << std::endl;
            return false;
        }
        cudaSetDevice(cudaDevice);

        ComPtr<IDXGIOutput> dxgiOutput;
        if (FAILED(dxgiAdapter->EnumOutputs(0, &dxgiOutput))) {
            return false;
        }

        ComPtr<IDXGIOutput1> dxgiOutput1;
        if (FAILED(dxgiOutput.As(&dxgiOutput1))) {
            return false;
        }

        if (FAILED(dxgiOutput1->DuplicateOutput(m_d3dDevice.Get(), &m_deskDupl))) {
            return false;
        }

        m_screenWidth = GetSystemMetrics(SM_CXSCREEN);
        m_screenHeight = GetSystemMetrics(SM_CYSCREEN);

        // GPU-only staging texture. Different from DXGICapture's CPU-readable staging — this one
        // never touches host memory; CUDA reads from it via the registered cudaArray.
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = m_screenWidth;
        desc.Height = m_screenHeight;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.CPUAccessFlags = 0;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE; // required for CUDA registration
        desc.MiscFlags = 0;

        m_stagingTex.Reset();
        if (m_cudaResource) {
            cudaGraphicsUnregisterResource(m_cudaResource);
            m_cudaResource = nullptr;
        }
        if (FAILED(m_d3dDevice->CreateTexture2D(&desc, nullptr, &m_stagingTex))) {
            return false;
        }

        cerr = cudaGraphicsD3D11RegisterResource(&m_cudaResource, m_stagingTex.Get(), cudaGraphicsRegisterFlagsNone);
        if (cerr != cudaSuccess) {
            std::cerr << "DXGICaptureCUDA: cudaGraphicsD3D11RegisterResource failed: " << cudaGetErrorString(cerr) << std::endl;
            return false;
        }
        return true;
    }

    ComPtr<ID3D11Device> m_d3dDevice;
    ComPtr<ID3D11DeviceContext> m_d3dContext;
    ComPtr<IDXGIOutputDuplication> m_deskDupl;
    ComPtr<ID3D11Texture2D> m_stagingTex;
    cudaGraphicsResource_t m_cudaResource = nullptr;
    int m_screenWidth = 0;
    int m_screenHeight = 0;
};
