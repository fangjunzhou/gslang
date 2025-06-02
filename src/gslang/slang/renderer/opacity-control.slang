implementing "../renderer.slang";
[shader("compute")]
[numthreads(64, 1, 1)]
void markKeep(uint tid: SV_DispatchThreadID, uniform uint numSrc,
              uniform float opacityThreshold) {
  if (tid >= numSrc) return;
  float opa = g_gaussian_3d_src[tid].opacity;
  if (opa < opacityThreshold) {
    g_keep_flag[tid] = 0;
  } else {
    g_keep_flag[tid] = 1;
  }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void removeGaussian(uint tid: SV_DispatchThreadID, uniform uint numSrc) {
  if (tid >= numSrc) return;

  if (g_keep_flag[tid] == 1u) {
    uint dst = g_keep_prefix[tid] - 1u;

    g_gaussian_3d[dst] = g_gaussian_3d_src[tid];
    d_gaussian_3d[dst] = d_gaussian_3d_src[tid];
    m_gaussian_3d[dst] = m_gaussian_3d_src[tid];
    v_gaussian_3d[dst] = v_gaussian_3d_src[tid];
  }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void setAllOpacity(uint tid: SV_DispatchThreadID, uniform uint numGaussians,
                   uniform float newOpacity) {
  if (tid >= numGaussians) return;
  g_gaussian_3d[tid].opacity = newOpacity;
}
