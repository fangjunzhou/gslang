implementing "../tests.slang";

[shader("compute")]
[numthreads(16, 16, 1)]
void renderGaussian2D(uint2 tid: SV_DispatchThreadID,
                      uniform RWTexture2D<float> renderTarget,
                      uniform Gaussian2D g) {
  // Get the dimensions of the render target.
  uint2 rtSize;
  renderTarget.GetDimensions(rtSize.x, rtSize.y);
  float2 uv = (float2(tid) + 0.5f) / float2(rtSize);
  // Evaluate the Gaussian.
  float val = g.eval(uv);
  // Write the Gaussian to the render target.
  renderTarget[tid] = val;
}
