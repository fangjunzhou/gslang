module radix_sort;

struct RadixState {
  uint shift;
  RWStructuredBuffer<Tuple<uint64_t, uint>> src;
  RWStructuredBuffer<Tuple<uint64_t, uint>> dst;
  RWStructuredBuffer<uint> hist;
  RWStructuredBuffer<uint> offs;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void clearHist(uint tid: SV_DispatchThreadID, RadixState state) {
  state.hist[tid] = 0;
  state.offs[tid] = 0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void buildHist(uint tid: SV_DispatchThreadID, RadixState state) {}

[shader("compute")]
[numthreads(64, 1, 1)]
void scatter(uint tid: SV_DispatchThreadID, RadixState state) {}
