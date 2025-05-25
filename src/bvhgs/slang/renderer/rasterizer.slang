implementing "../renderer.slang";

static const float EPS = 1e-6f;

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
  // Loop over the gaussians in the tile.
  for (uint i = tileOffs; i < tileOffs + tileLength; i++) {
    // Get the gaussian.
    Gaussian2D g = g_gaussian_2d_sorted[i];
    // Evaluate the gaussian.
    float sig = g.eval(uv);
    // Radiance contribution.
    float alpha = 1 - exp(-sig * g.opacity);
    radiance += attenuation * alpha * g.color;
    // Accumulate the attenuation.
    attenuation *= (1 - alpha);
    // If the attenuation is zero, we can stop.
    if (attenuation <= EPS) {
      break;
    }
  }
  // Write the radiance to the render target.
  g_render_target[tid] = float4(radiance, 1);
}

