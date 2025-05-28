implementing "../renderer.slang";

static const float EPS = 1e-2f;

[Differentiable]
void calculateRadiance(Gaussian2D g, no_diff float2 uv, out float3 radiance,
                       out float alpha) {
  // Evaluate the Gaussian at the given UV coordinate.
  float sig = g.eval(uv);
  // Calculate the radiance contribution.
  alpha = 1 - exp(-sig * g.opacity);
  radiance = alpha * g.color;
}

// Duplicate the screen space Gaussian for each tile.
[shader("compute")]
[numthreads(64, 1, 1)]
void duplicateGaussian(uint tid: SV_DispatchThreadID) {
  uint gid = g_gaussian_table[tid].val;
  g_gaussian_2d_sorted[tid] = g_gaussian_2d_culled[gid];
}

[shader("compute")]
[numthreads(8, 8, 1)]
void rasterize(uint2 tid: SV_DispatchThreadID) {
  if (tid.x >= g_camera.sensorSize.x || tid.y >= g_camera.sensorSize.y) return;
  // Convert pixel coordinate to screen space uv.
  float2 uv = (float2(tid) + 0.5f) / float2(g_camera.sensorSize);
  uv.y = 1 - uv.y;  // Flip y coordinate to match the screen.
  // Tile size.
  float2 tileSize = float2(1, 1) / float2(TILE_SHAPE);
  // Tile id.
  uint2 tile = uint2(uv / tileSize);
  uint tileId = tile.y * TILE_SHAPE.x + tile.x;
  // Get the offset and length of the tile.
  uint tileLength = g_tile_hist[tileId];
  uint tileOffs = g_tile_offs[tileId];
  // Rasterize all the 2D gaussians in the tile.
  // Accumulate radiance.
  float3 radiance = float3(0, 0, 0);
  // Attenuation factor.
  float attenuation = 1;
  float depth = 1;
  // Loop over the gaussians in the tile.
  for (uint i = tileOffs; i < tileOffs + tileLength; i++) {
    // Get the gaussian.
    Gaussian2D g = g_gaussian_2d_sorted[i];
    float3 gaussianRadiance;
    float alpha;
    // Evaluate the Gaussian and calculate radiance contribution.
    calculateRadiance(g, uv, gaussianRadiance, alpha);
    radiance += attenuation * gaussianRadiance;
    // Accumulate the attenuation.
    attenuation *= (1 - alpha);
    // If the attenuation is zero, we can stop.
    if (attenuation <= EPS) {
      depth = g.position.z;
      break;
    }
  }
  // Write the radiance to the render target.
  g_render_target[tid] = float4(radiance, 1);
  g_depth_target[tid] = float4(float3(depth), 1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bwdDuplicateGaussian(uint tid: SV_DispatchThreadID) {
  // TODO: Implement the backward pass for duplicating the Gaussian.
}

[shader("compute")]
[numthreads(8, 8, 1)]
void bwdRasterize(uint2 tid: SV_DispatchThreadID) {
  if (tid.x >= g_camera.sensorSize.x || tid.y >= g_camera.sensorSize.y) return;
  // Convert pixel coordinate to screen space uv.
  float2 uv = (float2(tid) + 0.5f) / float2(g_camera.sensorSize);
  uv.y = 1 - uv.y;  // Flip y coordinate to match the screen.
  // Tile size.
  float2 tileSize = float2(1, 1) / float2(TILE_SHAPE);
  // Tile id.
  uint2 tile = uint2(uv / tileSize);
  uint tileId = tile.y * TILE_SHAPE.x + tile.x;
  // Get the offset and length of the tile.
  uint tileLength = g_tile_hist[tileId];
  uint tileOffs = g_tile_offs[tileId];
  // Rasterize all the 2D gaussians in the tile.
  // Accumulate radiance.
  float3 radiance = float3(0, 0, 0);
  // Attenuation factor.
  float attenuation = 1;
  float depth = 1;
  // Loop over the gaussians in the tile.
  for (uint i = tileOffs; i < tileOffs + tileLength; i++) {
    // Get the gaussian.
    Gaussian2D g = g_gaussian_2d_sorted[i];
    DifferentialPair<Gaussian2D> g_pair = diffPair(g);
    float3 gaussianRadiance;
    float alpha;
    // Evaluate the Gaussian and calculate radiance contribution.
    calculateRadiance(g, uv, gaussianRadiance, alpha);
    float alphaGrad = 0;
    if (i < tileOffs + tileLength - 1) {
      // Calculate the gradient to the next radiance.
      float3 gaussianRadianceNext;
      float alphaNext;
      calculateRadiance(g_gaussian_2d_sorted[i + 1], uv, gaussianRadianceNext,
                        alphaNext);
      float attenuationNextGrad =
          dot(gaussianRadianceNext, d_render_target[tid].xyz);
      alphaGrad = -attenuationNextGrad * attenuation;
    }
    bwd_diff(calculateRadiance)(
        g_pair, uv, d_render_target[tid].xyz * attenuation, alphaGrad);
    d_a_gaussian_2d_sorted[i].add(g_pair.d);
    radiance += attenuation * gaussianRadiance;
    // Accumulate the attenuation.
    attenuation = attenuation * (1 - alpha);
    // If the attenuation is zero, we can stop.
    if (attenuation <= EPS) {
      depth = g.position.z;
      break;
    }
  }
  // Write the radiance to the render target.
  g_render_target[tid] = float4(radiance, 1);
  g_depth_target[tid] = float4(float3(depth), 1);
}
