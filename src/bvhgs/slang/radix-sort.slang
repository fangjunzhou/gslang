module radix_sort;

struct RadixState {
  uint shift;
  uint mask;
  RWStructuredBuffer<Tuple<uint64_t, uint>> src;
  RWStructuredBuffer<Tuple<uint64_t, uint>> dst;
  RWStructuredBuffer<Atomic<uint>> hist;
  RWStructuredBuffer<Atomic<uint>> offs;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void clearHist(uint tid: SV_DispatchThreadID, RadixState state) {
  if (tid <= state.mask) {
    state.hist[tid].store(0);
    state.offs[tid].store(0);
  }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void buildHist(uint tid: SV_DispatchThreadID, RadixState state) {
  uint64_t key = state.src[tid]._0;
  uint digit = uint((key >> state.shift) & state.mask);
  state.hist[digit].add(1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void scatter(uint tid: SV_DispatchThreadID, RadixState state) {
  Tuple<uint64_t, uint> kv = state.src[tid];
  uint digit = uint((kv._0 >> state.shift) & state.mask);
  uint newOffset = state.offs[digit].add(1);
  state.dst[newOffset - 1] = kv;
}