implementing "../renderer.slang";

[shader("compute")]
[numthreads(64, 1, 1)]
void gradDescentGaussian3D(uint tid: SV_DispatchThreadID, uniform float lr) {
  Gaussian3D gd = d_gaussian_3d[tid];
  g_gaussian_3d[tid] = g_gaussian_3d[tid] - (lr * gd);
  // Normalize the rotation quaternion.
  g_gaussian_3d[tid].rotation = normalize(g_gaussian_3d[tid].rotation);
}
