implementing "../renderer.slang";

// Project 3D Gaussians to ray space and set cull flag.
[shader("compute")]
[numthreads(64, 1, 1)]
void projection(uint3 tid: SV_DispatchThreadID) {
  // Project the 3D Gaussian to screen space.
  g_gaussian_2d[tid.x] = g_camera.toGaussian2D(g_gaussian_3d[tid.x]);
  g_gaussian_2d[tid.x] = g_camera.toScreenSpace(g_gaussian_2d[tid.x]);
  // Cull the 2D Gaussian to the viewing frustum.
  g_cull_flag[tid.x] = g_gaussian_2d[tid.x].position.x >= 0.0 &&
                       g_gaussian_2d[tid.x].position.x <= 1.0 &&
                       g_gaussian_2d[tid.x].position.y >= 0.0 &&
                       g_gaussian_2d[tid.x].position.y <= 1.0;
}

// Cull 2D Gaussians based on the viewing frustum flag.
[shader("compute")]
[numthreads(64, 1, 1)]
void cull(uint3 tid: SV_DispatchThreadID) {
  if (g_cull_flag[tid.x]) {
    uint idx = g_cull_prefix[tid.x] - 1;
    g_gaussian_2d_culled[idx] = g_gaussian_2d[tid.x];
  }
}
