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
}
