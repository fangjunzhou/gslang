module compute_texture;

uint2 g_size;
RWTexture2D g_output;

[shader("compute")]
[numthreads(16, 16, 1)]
void compute_main(uint3 tid: SV_DispatchThreadID) {
  float2 uv = (float2(tid.xy) + 0.5) / float2(g_size);
  float4 color = float4(uv.x, uv.y, 0.0, 1.0);
  g_output[tid.xy] = color;
}
