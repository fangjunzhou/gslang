implementing "../renderer.slang";

[shader("compute")]
[numthreads(64, 1, 1)]
void markDuplicate(uint tid: SV_DispatchThreadID, uniform float threashold) {
  // Get the positional gradient of the Gaussian point cloud.
  float posGradMag = length(d_gaussian_3d_pos[tid]);
  if (posGradMag > threashold) {
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
             uniform float overConstructionGradScale, uniform bool split) {
  if (tid >= numSrc) return;
  g_gaussian_3d[tid] = g_gaussian_3d_src[tid];

  if (g_duplicate_flag[tid] == 1) {
    uint dst = numSrc + g_duplicate_prefix[tid] - 1;
    float scale = length(exp(g_gaussian_3d[tid].scale));

    if (split) {
      // Copy the Gaussian to the destination index.
      g_gaussian_3d[dst] = g_gaussian_3d_src[tid];
      // Move the Gaussian down the gradient.
      g_gaussian_3d[tid].position += normalize(d_gaussian_3d_pos[tid]) *
                                     underConstructionGradScale * scale;
      g_gaussian_3d[dst].position -= normalize(d_gaussian_3d_pos[tid]) *
                                     underConstructionGradScale * scale;
      // g_gaussian_3d[tid].color = float3(10, 0, 0);
    } else {
      float logShrinkScale = log(overConstructionShrinkScale);
      g_gaussian_3d[tid].scale = g_gaussian_3d_src[tid].scale - logShrinkScale;
      g_gaussian_3d[dst] = g_gaussian_3d_src[tid];
      g_gaussian_3d[dst].scale = g_gaussian_3d_src[tid].scale - logShrinkScale;
      g_gaussian_3d[tid].position +=
          normalize(d_gaussian_3d_pos[tid]) * overConstructionGradScale * scale;
      g_gaussian_3d[dst].position -=
          normalize(d_gaussian_3d_pos[tid]) * overConstructionGradScale * scale;
      // g_gaussian_3d[tid].color = float3(0, 0, 10);
    }
  }

  // Clear positional gradient for the next iteration.
  d_gaussian_3d_pos[tid] = float3(0, 0, 0);
}
