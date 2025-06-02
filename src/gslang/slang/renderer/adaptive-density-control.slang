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
             uniform float underConstructionGradScale,
             uniform float overConstructionShrinkScale,
             uniform float overConstructionGradScale, uniform float avgScale) {
  if (tid >= numSrc) return;
  g_gaussian_3d[tid] = g_gaussian_3d_src[tid];

  if (g_duplicate_flag[tid] == 1) {
    float scaleMag = linalg.eigen(g_gaussian_2d[tid].covariance).eigenvalues[0];
    uint dst = numSrc + g_duplicate_prefix[tid] - 1;

    if (scaleMag < avgScale) {
      // Copy the Gaussian to the destination index.
      g_gaussian_3d[dst] = g_gaussian_3d_src[tid];
      // Move the Gaussian down the gradient.
      g_gaussian_3d[tid].position +=
          d_gaussian_3d[tid].position * underConstructionGradScale;
      g_gaussian_3d[dst].position -=
          d_gaussian_3d[tid].position * underConstructionGradScale;
    } else {
      float logShrinkScale = log(overConstructionShrinkScale);
      g_gaussian_3d[tid].scale = g_gaussian_3d_src[tid].scale - logShrinkScale;

      g_gaussian_3d[dst] = g_gaussian_3d_src[tid];
      g_gaussian_3d[dst].scale = g_gaussian_3d_src[tid].scale - logShrinkScale;

      g_gaussian_3d[tid].position +=
          d_gaussian_3d[tid].position * overConstructionGradScale;
      g_gaussian_3d[dst].position -=
          d_gaussian_3d[tid].position * overConstructionGradScale;
    }
  }
}
