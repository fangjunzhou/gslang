module rasterizer;

import gaussian;
import camera;

Camera g_camera;
StructuredBuffer<Gaussian3D> g_gaussians;
RWTexture2D g_output;

[shader("compute")]
[numthreads(8, 8, 1)]
void render(uint3 tid: SV_DispatchThreadID) {
  // Compute the pixel coordinates.
  float2 uv = float2(tid.xy + 0.5) / float2(g_camera.sensorSize);
  // Render uv.
  g_output[tid.xy] = float4(uv, 0, 1);
}
