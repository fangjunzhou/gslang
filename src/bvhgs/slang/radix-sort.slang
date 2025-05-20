module radix_sort;

struct RadixState {
  uint bitPerPass;
  RWStructuredBuffer<Tuple<uint64_t, uint>> src;
  RWStructuredBuffer<Tuple<uint64_t, uint>> dst;
  RWStructuredBuffer<uint> hist;
  RWStructuredBuffer<uint> offs;

  property radix : uint64_t {
    get { return 1 << this.bitPerPass; }
  }

  property mask : uint64_t {
    get { return this.radix - 1; }
  }
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
