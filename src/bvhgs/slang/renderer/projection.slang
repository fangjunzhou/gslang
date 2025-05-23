implementing "../renderer.slang";

// Project 3D Gaussians to ray space and set cull flag.
[shader("compute")]
[numthreads(64, 1, 1)]
void project(uint tid: SV_DispatchThreadID) {
  // Project the 3D Gaussian to screen space.
  g_gaussian_2d[tid] = g_camera.toGaussian2D(g_gaussian_3d[tid]);
  // Cull the 2D Gaussian to the viewing frustum.
  // TODO: Reject Gaussians based on its intersection with viewing frustum.
  bool is_inside = g_gaussian_2d[tid].position.x >= 0.0 &&
                   g_gaussian_2d[tid].position.x <= 1.0 &&
                   g_gaussian_2d[tid].position.y >= 0.0 &&
                   g_gaussian_2d[tid].position.y <= 1.0 &&
                   g_gaussian_2d[tid].position.z > 0;
  g_inside_flag[tid] = is_inside ? 1 : 0;
}

// Cull 2D Gaussians based on the viewing frustum flag.
[shader("compute")]
[numthreads(64, 1, 1)]
void cull(uint tid: SV_DispatchThreadID) {
  if (g_inside_flag[tid] == 1) {
    uint idx = g_inside_offset[tid] - 1;
    g_gaussian_2d_culled[idx] = g_gaussian_2d[tid];
  }
}
