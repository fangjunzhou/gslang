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

[shader("compute")]
[numthreads(16, 16, 1)]
void loadGaussian3D(uint tid: SV_DispatchThreadID,
                    uniform RWStructuredBuffer<Gaussian3D> gaussians,
                    uniform RWStructuredBuffer<float3> positions,
                    uniform RWStructuredBuffer<float4> rotations,
                    uniform RWStructuredBuffer<float3> scales) {
  // Load the Gaussian parameters from the buffers.
  float3 position = positions[tid];
  float4 rotation = rotations[tid];
  float3 scale = scales[tid];
  // Create the Gaussian object.
  Gaussian3D g = Gaussian3D(position, rotation, scale);
  // Store the Gaussian in the buffer.
  gaussians[tid] = g;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void loadGaussian2D(uint tid: SV_DispatchThreadID,
                    uniform RWStructuredBuffer<Gaussian2D> gaussians,
                    uniform RWStructuredBuffer<float3> positions,
                    uniform RWStructuredBuffer<float2x2> covariances) {
  // Load the Gaussian parameters from the buffers.
  float3 position = positions[tid];
  float2x2 covariance = covariances[tid];
  // Create the Gaussian object.
  Gaussian2D g = Gaussian2D(position, covariance);
  // Store the Gaussian in the buffer.
  gaussians[tid] = g;
}
