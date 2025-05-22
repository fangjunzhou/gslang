module prefix_sum;

static const uint WAVE = 32;

/* ------------ Kernel 1  :  in-wave scan + write partial sums ------------ */
[shader("compute")]
[numthreads(WAVE, 1, 1)]
void wave_scan(
    uint3 gtid: SV_DispatchThreadID,
    StructuredBuffer<uint> src,
    RWStructuredBuffer<uint> dst,
    RWStructuredBuffer<uint> partial,
    uniform uint n){

    uint idx = gtid.x;

    uint val = (idx < n) ? src[idx] : 0u;

    uint excl = WavePrefixSum(val);
    uint incl = excl + val;

    // Store inclusive prefix sum for valid elements
    if (idx < n) dst[idx] = incl;

    // Lane (WAVE-1) writes the sum of this wave to "partial"
    if (WaveGetLaneIndex() == WAVE - 1){
        uint blockIdx = idx / WAVE;
        partial[blockIdx] = incl;
    }
}

[shader("compute")]
[numthreads(WAVE, 1, 1)]
void add_offset(
    uint3 gtid: SV_DispatchThreadID,
    RWStructuredBuffer<uint> dst,
    StructuredBuffer<uint> partial,
    uniform uint n){
    uint idx = gtid.x;
    if (idx >= n) return;

    uint blockIdx = idx / WAVE;
    uint offset = (blockIdx == 0) ? 0u : partial[blockIdx - 1];
    dst[idx] += offset;
}
