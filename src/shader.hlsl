cbuffer ConstantBuffer : register(b0) { matrix WorldViewProjection; }

struct VS_INPUT {
    float3 Pos : POSITION;
    float2 Tex : TEXCOORD;
};

struct PS_INPUT {
    float4 Pos : SV_POSITION;
    float2 Tex : TEXCOORD;
};

PS_INPUT VS(VS_INPUT input) {
    PS_INPUT output;
    output.Pos = float4(input.Pos, 1.0f);
    output.Tex = input.Tex;
    return output;
}

Texture2D tex : register(t0);
SamplerState samLinear : register(s0);

float4 PS(PS_INPUT input) : SV_Target { return tex.Sample(samLinear, input.Tex); }
