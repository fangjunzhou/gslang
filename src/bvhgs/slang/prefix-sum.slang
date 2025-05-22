module prefix_sum;

static const uint WAVE = 32;

/* ------------ Kernel 1  :  in-wave scan + write partial sums ------------ */
[shader("compute")]
[numthreads(WAVE, 1, 1)]
void wave_scan(uint tid: SV_DispatchThreadID, StructuredBuffer<uint> src,
               RWStructuredBuffer<uint> dst, RWStructuredBuffer<uint> partial,
               uniform uint n) {
  uint val = (tid < n) ? src[tid] : 0u;

  uint excl = WavePrefixSum(val);
  uint incl = excl + val;

  // Store inclusive prefix sum for valid elements
  if (tid < n) dst[tid] = incl;

  // Lane (WAVE-1) writes the sum of this wave to "partial"
  if (tid % WAVE == WAVE - 1) {
    uint blockIdx = tid / WAVE;
    partial[blockIdx] = incl;
  }
}

[shader("compute")]
[numthreads(WAVE, 1, 1)]
void add_offset(uint tid: SV_DispatchThreadID, RWStructuredBuffer<uint> dst,
                StructuredBuffer<uint> partial, uniform uint n) {
  if (tid >= n) return;

  uint blockIdx = tid / WAVE;
  uint offset = (blockIdx == 0) ? 0u : partial[blockIdx - 1];
  dst[tid] += offset;
}
