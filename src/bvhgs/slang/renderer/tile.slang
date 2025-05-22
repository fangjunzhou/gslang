implementing "../renderer.slang";

[shader("compute")]
[numthreads(64, 1, 1)]
void computeTile(uint tid: SV_DispatchThreadID) {
  // Get the bounding box of the gaussian.
  BoundingBox2D bbox = g_gaussian_2d_culled[tid].boundingBox();
  float2 bMin = bbox.min;
  float2 bMax = bbox.max;
  // Tile size.
  float2 tileSize = float2(1, 1) / float2(TILE_SHAPE);
  // Starting and ending tile.
  uint2 minTile = uint2(bMin / tileSize);
  uint2 maxTile = uint2(bMax / tileSize);
  g_num_tile[tid] = (maxTile.x - minTile.x) * (maxTile.y - minTile.y);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void buildGaussianTable(uint tid: SV_DispatchThreadID) {
  // TODO: Build (tile id, depth) to gaussian idx loopup table.
  Gaussian2D g = g_gaussian_2d_culled[tid];
  uint tableIdx;
  if (tid == 0) {
    tableIdx = 0;
  } else {
    tableIdx = g_num_tile_prefix[tid - 1];
  }
  // Gaussian depth.
  uint depth = uint(g.position.z * 16384);
  // Get the bounding box of the gaussian.
  BoundingBox2D bbox = g_gaussian_2d_culled[tid].boundingBox();
  float2 bMin = bbox.min;
  float2 bMax = bbox.max;
  // Tile size.
  float2 tileSize = float2(1, 1) / float2(TILE_SHAPE);
  // Starting and ending tile.
  uint2 minTile = uint2(bMin / tileSize);
  uint2 maxTile = uint2(bMax / tileSize);
  // Loop over the tiles.
  for (uint y = minTile.y; y < maxTile.y; y++) {
    for (uint x = minTile.x; x < maxTile.x; x++) {
      // Get the tile id.
      uint tileId = y * TILE_SHAPE.x + x;
      // Sort key.
      uint64_t sortKey = (uint64_t(tileId) << 32) | depth;
      // Add the gaussian to the table.
      g_gaussian_table[tableIdx] = Tuple<uint64_t, uint>(sortKey, tid);
      tableIdx++;
    }
  }
}
}
