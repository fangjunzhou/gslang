module radix_sort;

struct RadixState {
  uint shift;
  uint mask;
  RWStructuredBuffer<Tuple<uint64_t, uint>> src;
  RWStructuredBuffer<Tuple<uint64_t, uint>> dst;
  RWStructuredBuffer<uint> hist;
  RWStructuredBuffer<uint> offs;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void clearHist(uint tid: SV_DispatchThreadID, RadixState state) {
  if (tid <= state.mask) {
    state.hist[tid] = 0;
    state.offs[tid] = 0;
  }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void buildHist(uint tid: SV_DispatchThreadID, RadixState state, uniform uint numElements) {
  if (tid < numElements) {
    uint64_t key = state.src[tid]._0;
    uint digit = uint((key >> state.shift) & state.mask);
    InterlockedAdd(state.hist[digit], 1);
  }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void scatter(uint tid: SV_DispatchThreadID, RadixState state, uniform uint numElements) {
  if (tid < numElements) {
    Tuple<uint64_t, uint> kv = state.src[tid];
    uint digit = uint((kv._0 >> state.shift) & state.mask);

    uint outIndex;
    InterlockedAdd(state.offs[digit], 1, outIndex);
    state.dst[outIndex] = kv;
  }
}