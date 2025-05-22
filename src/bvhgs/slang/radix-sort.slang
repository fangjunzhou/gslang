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
  uint index = tid;
  while (index < numElements) {
    uint64_t key = state.src[index]._0;
    uint digit = (key >> state.shift) & state.mask;
    InterlockedAdd(state.hist[digit], 1);
    index += 64;
  }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void scatter(uint tid: SV_DispatchThreadID, RadixState state, uniform uint numElements) {
  uint index = tid;
  while (index < numElements) {
    Tuple<uint64_t, uint> kv = state.src[index];
    uint digit = (kv._0 >> state.shift) & state.mask;

    uint offset;
    InterlockedAdd(state.offs[digit], 1, offset);

    state.dst[offset] = kv;
    index += 64;
  }
}