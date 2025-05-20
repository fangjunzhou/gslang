module prefix_sum;

[shader("compute")]
[numthreads(64, 1, 1)]
void upSweep(uint tid: SV_DispatchThreadID, StructuredBuffer<int> src,
             RWStructuredBuffer<int> dst, uniform uint depth) {
  dst[tid] = src[tid];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void downSweep(uint tid: SV_DispatchThreadID, StructuredBuffer<int> src,
               RWStructuredBuffer<int> dst, uniform uint depth) {
  dst[tid] = src[tid];
}
