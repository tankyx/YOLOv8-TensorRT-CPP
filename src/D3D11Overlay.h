#pragma once
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dwmapi.h>
#include <dxgi1_2.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

class D3D11Overlay {
public:
    D3D11Overlay(int width, int height) : m_width(width), m_height(height) {
        initD3D();
        createWindow();
        createResources();
    }

    ~D3D11Overlay() {
        if (m_hwnd) {
            DestroyWindow(m_hwnd);
            UnregisterClass("D3D11Overlay", GetModuleHandle(nullptr));
        }
    }

    void drawDetections(const std::vector<Object> &detections) {
        // Clear with transparent black
        m_canvas = cv::Scalar(0, 0, 0, 0);

        for (const auto &det : detections) {
            // Draw rectangle
            cv::rectangle(m_canvas, det.rect, cv::Scalar(0, 255, 0, 255), 2);

            // Create label with confidence
            std::string label = std::to_string(det.label) + ": " + std::to_string(int(det.probability * 100)) + "%";

            // Draw label background
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(m_canvas, cv::Point(det.rect.x, det.rect.y - textSize.height - 5),
                          cv::Point(det.rect.x + textSize.width, det.rect.y), cv::Scalar(0, 255, 0, 180), cv::FILLED);

            // Draw label text
            cv::putText(m_canvas, label, cv::Point(det.rect.x, det.rect.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0, 255), 1,
                        cv::LINE_AA);
        }
    }

    void drawLog(const std::string &text) {
        // Split text into lines and draw
        std::istringstream stream(text);
        std::string line;
        int y = 20;

        while (std::getline(stream, line)) {
            // Draw text background
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(m_canvas, cv::Point(10, y - textSize.height), cv::Point(15 + textSize.width, y + baseline),
                          cv::Scalar(0, 128, 255, 180), // Orange background
                          cv::FILLED);

            // Draw text
            cv::putText(m_canvas, line, cv::Point(10, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 255), 1, cv::LINE_AA);

            y += textSize.height + 5;
        }
    }

    void render() {
        // Update texture
        D3D11_MAPPED_SUBRESOURCE mapped;
        m_context->Map(m_texture.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);

        for (int i = 0; i < m_height; i++) {
            memcpy((BYTE *)mapped.pData + i * mapped.RowPitch, m_canvas.ptr(i), m_width * 4);
        }

        m_context->Unmap(m_texture.Get(), 0);

        // Clear and draw
        float clearColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        m_context->ClearRenderTargetView(m_renderTargetView.Get(), clearColor);
        m_context->Draw(4, 0);
        m_swapChain->Present(0, 0);
    }

    void setPosition(int x, int y) { SetWindowPos(m_hwnd, HWND_TOPMOST, x, y, m_width, m_height, SWP_SHOWWINDOW); }

    void setClickthrough(bool enabled) {
        LONG_PTR style = GetWindowLongPtr(m_hwnd, GWL_EXSTYLE);
        if (enabled) {
            style |= WS_EX_TRANSPARENT | WS_EX_LAYERED;
        } else {
            style &= ~(WS_EX_TRANSPARENT | WS_EX_LAYERED);
        }
        SetWindowLongPtr(m_hwnd, GWL_EXSTYLE, style);
    }

private:
    void initD3D() {
        D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
        UINT createDeviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

        D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, &featureLevel, 1, D3D11_SDK_VERSION, &m_device,
                          nullptr, &m_context);
    }

    void createWindow() {
        WNDCLASSEX wc = {};
        wc.cbSize = sizeof(WNDCLASSEX);
        wc.lpfnWndProc = DefWindowProc;
        wc.hInstance = GetModuleHandle(nullptr);
        wc.lpszClassName = "D3D11Overlay";
        RegisterClassEx(&wc);

        m_hwnd = CreateWindowEx(WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT, "D3D11Overlay", "Overlay", WS_POPUP, 0, 0, m_width,
                                m_height, nullptr, nullptr, wc.hInstance, nullptr);

        SetLayeredWindowAttributes(m_hwnd, RGB(0, 0, 0), 255, LWA_ALPHA);
        ShowWindow(m_hwnd, SW_SHOW);
    }

    void createResources() {
        // Create swapchain
        DXGI_SWAP_CHAIN_DESC scd = {};
        scd.BufferDesc.Width = m_width;
        scd.BufferDesc.Height = m_height;
        scd.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        scd.SampleDesc.Count = 1;
        scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        scd.BufferCount = 2;
        scd.OutputWindow = m_hwnd;
        scd.Windowed = TRUE;
        scd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
        scd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

        ComPtr<IDXGIDevice> dxgiDevice;
        m_device.As(&dxgiDevice);
        ComPtr<IDXGIAdapter> adapter;
        dxgiDevice->GetAdapter(&adapter);
        ComPtr<IDXGIFactory> factory;
        adapter->GetParent(IID_PPV_ARGS(&factory));
        factory->CreateSwapChain(m_device.Get(), &scd, &m_swapChain);

        // Create render target
        ComPtr<ID3D11Texture2D> backBuffer;
        m_swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
        m_device->CreateRenderTargetView(backBuffer.Get(), nullptr, &m_renderTargetView);
        m_context->OMSetRenderTargets(1, m_renderTargetView.GetAddressOf(), nullptr);

        // Create texture for overlay
        D3D11_TEXTURE2D_DESC texDesc = {};
        texDesc.Width = m_width;
        texDesc.Height = m_height;
        texDesc.MipLevels = 1;
        texDesc.ArraySize = 1;
        texDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        texDesc.SampleDesc.Count = 1;
        texDesc.Usage = D3D11_USAGE_DYNAMIC;
        texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        texDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        m_device->CreateTexture2D(&texDesc, nullptr, &m_texture);

        // Create shader resource view
        m_device->CreateShaderResourceView(m_texture.Get(), nullptr, &m_shaderResourceView);

        // Create viewport
        D3D11_VIEWPORT vp = {};
        vp.Width = (float)m_width;
        vp.Height = (float)m_height;
        vp.MaxDepth = 1.0f;
        m_context->RSSetViewports(1, &vp);

        // Create shaders and input layout
        createShaders();

        // Initialize canvas
        m_canvas = cv::Mat(m_height, m_width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    }

    void createShaders() {
        // Simple vertex shader
        const char *vsCode = R"(
            struct VS_INPUT {
                float2 pos : POSITION;
                float2 tex : TEXCOORD;
            };
            
            struct VS_OUTPUT {
                float4 pos : SV_POSITION;
                float2 tex : TEXCOORD;
            };
            
            VS_OUTPUT main(VS_INPUT input) {
                VS_OUTPUT output;
                output.pos = float4(input.pos, 0.0f, 1.0f);
                output.tex = input.tex;
                return output;
            }
        )";

        // Simple pixel shader
        const char *psCode = R"(
            Texture2D tex : register(t0);
            SamplerState samp : register(s0);
            
            float4 main(float4 pos : SV_POSITION, float2 tex : TEXCOORD) : SV_Target {
                return tex.Sample(samp, tex);
            }
        )";

        // Compile and create shaders
        ComPtr<ID3DBlob> vsBlob, psBlob;
        D3DCompile(vsCode, strlen(vsCode), nullptr, nullptr, nullptr, "main", "vs_4_0", 0, 0, &vsBlob, nullptr);
        D3DCompile(psCode, strlen(psCode), nullptr, nullptr, nullptr, "main", "ps_4_0", 0, 0, &psBlob, nullptr);

        m_device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &m_vertexShader);
        m_device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &m_pixelShader);

        m_context->VSSetShader(m_vertexShader.Get(), nullptr, 0);
        m_context->PSSetShader(m_pixelShader.Get(), nullptr, 0);
        m_context->PSSetShaderResources(0, 1, m_shaderResourceView.GetAddressOf());
    }

private:
    int m_width;
    int m_height;
    HWND m_hwnd = nullptr;
    cv::Mat m_canvas;

    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11DeviceContext> m_context;
    ComPtr<IDXGISwapChain> m_swapChain;
    ComPtr<ID3D11RenderTargetView> m_renderTargetView;
    ComPtr<ID3D11Texture2D> m_texture;
    ComPtr<ID3D11ShaderResourceView> m_shaderResourceView;
    ComPtr<ID3D11VertexShader> m_vertexShader;
    ComPtr<ID3D11PixelShader> m_pixelShader;
};