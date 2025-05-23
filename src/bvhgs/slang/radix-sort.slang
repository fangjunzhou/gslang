module radix_sort;

public struct KVPair {
  public uint64_t key;
  public uint val;

  public __init(uint64_t k, uint v) {
    key = k;
    val = v;
  }
};

struct RadixState {
  uint shift;
  uint mask;
  RWStructuredBuffer<KVPair> src;
  RWStructuredBuffer<KVPair> dst;
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
  uint64_t key = state.src[tid].key;
  uint digit = uint((key >> state.shift) & state.mask);
  state.hist[digit].add(1);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void scatter(uint tid: SV_DispatchThreadID, RadixState state) {
  KVPair kv = state.src[tid];
  uint digit = uint((kv.key >> state.shift) & state.mask);
  uint newOffset = state.offs[digit].add(1);
  state.dst[newOffset - 1] = kv;
}

