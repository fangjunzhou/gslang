implementing math;

namespace utils {
    // Sigmoid (f(x) = 1 / (1 + exp(-x))).
    // @param x The input value.
    // @return The sigmoid of x.
    public float sigmoid(float x){return 1.0f / (1.0f + exp(-x));}
    public float2 sigmoid2(float2 x) { return 1.0f / (1.0f + exp(-x)); }
    public float3 sigmoid3(float3 x) { return 1.0f / (1.0f + exp(-x)); }
    public float4 sigmoid4(float4 x) { return 1.0f / (1.0f + exp(-x)); }

}