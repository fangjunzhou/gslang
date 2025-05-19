implementing "../renderer.slang";

[shader("compute")]
[numthreads(64, 1, 1)]
void projection(uint3 tid: SV_DispatchThreadID) {
  g_gaussian_2d[tid.x] = g_camera.toGaussian2D(g_gaussian_3d[tid.x]);
}
