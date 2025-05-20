implementing "../renderer.slang";

[shader("compute")]
[numthreads(64, 1, 1)]
void computeTile(uint tid: SV_DispatchThreadID) {
  // TODO: Implement num_tile buffer compute.
}

[shader("compute")]
[numthreads(64, 1, 1)]
void buildGaussianTable(uint tid: SV_DispatchThreadID) {
  // TODO: Build (tile id, depth) to gaussian idx loopup table.
}
