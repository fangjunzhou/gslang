implementing "../renderer.slang";

inline bool aabb_overlaps_unit_rect(float2 bbMin, float2 bbMax) {
  // Completely left / right / above / below
  if (bbMax.x < 0.0 || bbMin.x > 1.0) return false;
  if (bbMax.y < 0.0 || bbMin.y > 1.0) return false;
  return true;
}

[Differentiable]
Gaussian2D toGaussian2D(Gaussian3D g3d) {
  return g_camera.toGaussian2D(g3d);
}

// Project 3D Gaussians to ray space and set cull flag.
[shader("compute")]
[numthreads(64, 1, 1)]
void project(uint tid: SV_DispatchThreadID) {
  // Project the 3D Gaussian to screen space.
  Gaussian2D g2d = toGaussian2D(g_gaussian_3d[tid]);
  g_gaussian_2d[tid] = g2d;
  // Cull the 2D Gaussian to the viewing frustum.
  bool depthOK = (g2d.position.z >= 0.0) && (g2d.position.z <= 1.0);

  BoundingBox2D bb = g2d.boundingBox();

  bool keep = depthOK && aabb_overlaps_unit_rect(bb.min, bb.max);
  g_inside_flag[tid] = keep ? 1 : 0;
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

[shader("compute")]
[numthreads(64, 1, 1)]
void bwdCullProjection(uint tid: SV_DispatchThreadID) {
  // inverse cull (identity) then inverse projection to 3d
  if (g_inside_flag[tid] != 1) return;

  uint idx = g_inside_offset[tid] - 1;
  Gaussian2D g2d = d_gaussian_2d_culled[idx];

  // backprop the 2D Gaussian to the 3D Gaussian
  DifferentialPair<Gaussian3D> g3d_pair = diffPair(g_gaussian_3d[tid]);
  bwd_diff(toGaussian2D)(g3d_pair, g2d);

  d_gaussian_3d[tid] = d_gaussian_3d[tid] + g3d_pair.d;
}
