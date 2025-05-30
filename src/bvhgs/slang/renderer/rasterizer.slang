implementing "../renderer.slang";

static const float EPS = 1e-2f;

[Differentiable]
float calculateAlpha(Gaussian2D g, no_diff float2 uv) {
  // Evaluate the Gaussian at the given UV coordinate.
  float sig = g.eval(uv);
  // Calculate the radiance contribution.
  float alpha = 1 - exp(-sig * g.opacity);
  return alpha;
}

[Differentiable]
float3 calculateRadiance(Gaussian2D g, float alpha) {
  // Calculate the radiance contribution.
  float3 radiance = g.color * alpha;
  return radiance;
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
  g_num_rendered_gaussians[tid] = tileLength;
  // Loop over the gaussians in the tile.
  for (uint i = tileOffs; i < tileOffs + tileLength; i++) {
    // Get the gaussian.
    Gaussian2D g = g_gaussian_2d_sorted[i];
    float3 gaussianRadiance;
    // Evaluate the Gaussian and calculate radiance contribution.
    float alpha = calculateAlpha(g, uv);
    gaussianRadiance = calculateRadiance(g, alpha);
    radiance += attenuation * gaussianRadiance;
    // Accumulate the attenuation.
    attenuation *= (1 - alpha);
    // If the attenuation is zero, we can stop.
    if (attenuation <= EPS) {
      depth = g.position.z;
      g_num_rendered_gaussians[tid] = i - tileOffs + 1;
      break;
    }
  }
  // Write the radiance to the render target.
  g_render_target[tid] = float4(radiance, 1);
  g_depth_target[tid] = float4(float3(depth), 1);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void bwdRasterize(uint2 tid: SV_DispatchThreadID, uint2 gid: SV_GroupID) {
  if (tid.x >= g_camera.sensorSize.x || tid.y >= g_camera.sensorSize.y) return;
  // Convert pixel coordinate to screen space uv.
  float2 uv = (float2(tid) + 0.5f) / float2(g_camera.sensorSize);
  uv.y = 1 - uv.y;  // Flip y coordinate to match the screen.
  // Tile size.
  float2 tileSize = float2(1, 1) / float2(TILE_SHAPE);
  // Tile id.
  uint2 tile = uint2(uv / tileSize);
  uint tileId = tile.y * TILE_SHAPE.x + tile.x;

  // Get the group start and end tile ids.
  float2 groupStartUv = (float2(gid) * float2(8, 4) + float2(0.5, 0.5)) /
                        float2(g_camera.sensorSize);
  float2 groupEndUv =
      (float2(gid) * float2(8, 4) + float2(8, 4) - float2(0.5, 0.5)) /
      float2(g_camera.sensorSize);
  uint2 groupStartTile = uint2(groupStartUv / tileSize);
  uint2 groupEndTile = uint2(groupEndUv / tileSize);
  // Check if thread group crosses tile boundaries.
  bool crossesTileBoundary = (groupStartTile.x != groupEndTile.x) ||
                             (groupStartTile.y != groupEndTile.y);

  // Get the offset and length of the tile.
  uint tileLength = g_tile_hist[tileId];
  uint tileOffs = g_tile_offs[tileId];
  // Rasterize all the 2D gaussians in the tile.
  // Accumulate radiance.
  float3 radiance = float3(0, 0, 0);
  float3 dRadiance = d_render_target[tid].xyz;
  // Attenuation factor.
  float attenuation = 1;
  float depth = 1;
  // Loop over the gaussians in the tile.
  uint n = g_num_rendered_gaussians[tid];
  if (n > tileLength) n = tileLength;  // Safety check to avoid invalid n values
  uint maxN = WaveActiveMax(n);
  for (uint idx = 0; idx < maxN; idx++) {
    // Calculate index in reverse order, with safety check
    uint reverse_idx = maxN - idx - 1;  // This ensures we stay within bounds
    uint i = tileOffs + reverse_idx;
    if (reverse_idx >= n)
      continue;  // Skip if index exceeds the number of rendered gaussians

    // Get the gaussian.
    Gaussian2D g = g_gaussian_2d_sorted[i];
    DifferentialPair<Gaussian2D> g_pair = diffPair(g);

    // Evaluate the Gaussian and calculate radiance contribution.
    float alpha = calculateAlpha(g, uv);
    float3 gaussianRadiance = calculateRadiance(g, alpha);

    float3 dGaussianRadiance = attenuation * dRadiance;
    DifferentialPair<float> alphaPair = diffPair(alpha);
    bwd_diff(calculateRadiance)(g_pair, alphaPair, dGaussianRadiance);

    alphaPair = diffPair(alphaPair.p,
                         alphaPair.d - dot(dRadiance, radiance) * attenuation);

    Gaussian2D.Differential g_pair_d = g_pair.d;

    bwd_diff(calculateAlpha)(g_pair, uv, alphaPair.d);

    g_pair = diffPair(g_pair.p, g_pair.d + g_pair_d);

    float[11] d = g_pair.d.toArray();
    for (uint j = 0; j < 11; j++) {
      // Atomic add the gradient if crosses tile boundaries.
      if (crossesTileBoundary) {
        d_a_gaussian_2d_sorted[i][(j + tid.x) % 11].add(d[(j + tid.x) % 11]);
      } else {
        // Waveactivesum the gradient if not crossing tile boundaries.
        float sum = WaveActiveSum(d[j]);
        if (WaveIsFirstLane()) {
          d_a_gaussian_2d_sorted[i][j].add(sum);
        }
      }
    }

    radiance = radiance * (1 - alpha);
    radiance += gaussianRadiance;

    attenuation = attenuation * (1 - alpha);
  }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bwdDuplicateGaussian(uint tid: SV_DispatchThreadID) {
  uint gid = g_gaussian_table[tid].val;
  for (uint j = 0; j < 11; j++) {
    d_a_gaussian_2d_culled[gid][(j + tid.x) % 11].add(
        d_a_gaussian_2d_sorted[tid][(j + tid.x) % 11].load());
  }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void extractCulledGaussianGrad(uint tid: SV_DispatchThreadID) {
  // Extract the gradient of the culled Gaussian.
  d_gaussian_2d_culled[tid] =
      Gaussian2D.fromAtomic(d_a_gaussian_2d_culled[tid]);
}
