implementing "../renderer.slang";

[shader("compute")]
[numthreads(64, 1, 1)]
void markDuplicate(uint tid: SV_DispatchThreadID, uniform float threashold) {
  if (g_inside_flag[tid] == 0) return;
  uint idx = g_inside_offset[tid] - 1;

  // Get the positional gradient of the Gaussian point cloud.
  float3 posGrad = d_gaussian_2d_culled[idx].position;
  if (length(posGrad) > threashold) {
    // Mark the Gaussian as duplicate.
    g_duplicate_flag[tid] = 1;
  } else {
    // Mark the Gaussian as non-duplicate.
    g_duplicate_flag[tid] = 0;
  }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void densify(uint tid: SV_DispatchThreadID, uniform uint numSrc,
             uniform float underConstructionScale) {
  if (tid >= numSrc) return;
  g_gaussian_3d[tid] = g_gaussian_3d_src[tid];
  if (g_duplicate_flag[tid] == 1) {
    uint dst = numSrc + g_duplicate_prefix[tid] - 1;
    // Copy the Gaussian to the destination index.
    g_gaussian_3d[dst] = g_gaussian_3d_src[tid];
    // Move the Gaussian down the gradient.
    g_gaussian_3d[tid].position +=
        d_gaussian_3d[tid].position * underConstructionScale;
    // TODO: Handle over construction.
  }
}
