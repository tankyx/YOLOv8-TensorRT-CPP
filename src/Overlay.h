#pragma once
#include <Windows.h>
#include <d3d11.h>
#include <d3dx10.h>
#include <d3dx11.h>
#include <sstream>
#include <string>
#include <vector>
#include <xnamath.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dx11.lib")
#pragma comment(lib, "d3dx10.lib")

using namespace std;

struct Object {
    RECT rect;
    int label;
};

class Overlay {
public:
    Overlay();
    ~Overlay();
    bool Initialize();
    void Run();
    void CleanUp();
    void DrawDetectionsPen(const std::vector<Object> &detections);
    void DrawText(const std::string &text, int x, int y, COLORREF color, int fontSize);

private:
    HWND hijackNvidiaOverlay();
    bool InitializeDirect3d11App();
    void DrawScene();
    static LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

    HWND hwnd;
    IDXGISwapChain *SwapChain;
    ID3D11Device *d3d11Device;
    ID3D11DeviceContext *d3d11DevCon;
    ID3D11RenderTargetView *renderTargetView;
    ID3D11DepthStencilView *depthStencilView;
    ID3D11Texture2D *depthStencilBuffer;
    ID3D11VertexShader *VS;
    ID3D11PixelShader *PS;
    ID3D10Blob *VS_Buffer;
    ID3D10Blob *PS_Buffer;
    ID3D11InputLayout *vertLayout;
    ID3D11Buffer *cbPerObjectBuffer;
    XMMATRIX WVP;
    XMMATRIX camView;
    XMMATRIX camProjection;
    XMVECTOR camPosition;
    XMVECTOR camTarget;
    XMVECTOR camUp;
    float rot;

    HDC memDC;
    HFONT hFont;
    HPEN hPen;

    const std::vector<HBRUSH> BRUSH_LIST = {
        CreateSolidBrush(RGB(255, 0, 0)),     // Red
        CreateSolidBrush(RGB(255, 150, 150)), // Light Red
        CreateSolidBrush(RGB(0, 0, 255)),     // Blue
        CreateSolidBrush(RGB(150, 150, 255))  // Light Blue
    };
};
