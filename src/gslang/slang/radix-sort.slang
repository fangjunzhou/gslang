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
  uint bucket;
  uint bufSize;
  uint numThreads;
  uint entriesPerThread;
  RWStructuredBuffer<KVPair> src;
  RWStructuredBuffer<KVPair> dst;
  RWStructuredBuffer<uint> hist;  // Global offset
  RWStructuredBuffer<uint> offs;
  RWStructuredBuffer<uint> globalOffs;  // Size of the histogram
};

// Clear histogram and offset
[shader("compute")]
[numthreads(256, 1, 1)]
void clearHist(uint tid: SV_DispatchThreadID, uniform RadixState state) {
  // Clear the histogram and offsets for each thread
  if (tid < state.numThreads) {
    uint startIdx = tid * state.bucket;
    uint endIdx = startIdx + state.bucket;
    for (uint i = startIdx; i < endIdx; i++) {
      state.hist[i] = 0;
    }
  } 
}

// Build histogram for the current pass
[shader("compute")]
[numthreads(256, 1, 1)]
void buildHist(uint tid: SV_DispatchThreadID, uniform RadixState state) {
  if (tid >= state.numThreads) {
    return;
  }
  // calculate mask from bucket
  uint mask = state.bucket - 1;

  // locate src start and end
  uint startIdx = tid * state.entriesPerThread;
  uint endIdx = startIdx + state.entriesPerThread;
  if (endIdx > state.bufSize) {
    endIdx = state.bufSize;
  }

  // locate histOffs start and end
  uint histStartIdx = tid * state.bucket;

  // read src, update histOffs in for loop
  for (uint i = startIdx; i < endIdx; i++) {
    uint digit = uint(state.src[i].key >> state.shift) & mask;
    state.hist[histStartIdx + digit] += 1; 
  }
}

// wave scan
static const uint WAVE = 32;
[shader("compute")]
[numthreads(WAVE, 1, 1)]
void waveScan(uint tid: SV_DispatchThreadID, uniform RadixState state, RWStructuredBuffer<uint> hist, RWStructuredBuffer<uint> offs, RWStructuredBuffer<uint> partial, uniform uint n) {
  uint bucket = state.bucket;
  uint numWaves = (n + WAVE - 1) / WAVE;
  for (uint i = 0; i < bucket; i++) {
    uint idx = tid * bucket + i;
    uint val = (tid < n) ? hist[idx] : 0u;
    uint excl = WavePrefixSum(val);
    if (tid < n) {
      offs[idx] = excl;
    }
    if (tid % WAVE == WAVE - 1) {
      uint blockIdx = tid / WAVE;
      partial[state.bucket * blockIdx + i] = excl + val;
    }
  }
}

// add offset
[shader("compute")]
[numthreads(WAVE, 1, 1)]
void addOffset(uint tid: SV_DispatchThreadID, uniform RadixState state, RWStructuredBuffer<uint> offs, StructuredBuffer<uint> partial, uniform uint n) {
  uint bucket = state.bucket;
  if (tid >= n) return;
  uint blockIdx = tid / WAVE;
  for (uint i = 0; i < bucket; i++) {
    uint idx = tid * bucket + i;
    uint offset = partial[state.bucket * blockIdx + i];
    offs[idx] += offset;
  }
}


[shader("compute")]
[numthreads(256, 1, 1)]
void scatter(uint tid: SV_DispatchThreadID, uniform RadixState state, ) {
  if (tid >= state.numThreads) {
    return;
  }
  uint mask = state.bucket - 1;
  uint startIdx = tid * state.entriesPerThread;
  uint endIdx = startIdx + state.entriesPerThread;
  if (startIdx >= state.bufSize) {
    return;
  }
  if (endIdx > state.bufSize) {
    endIdx = state.bufSize;
  }
  uint histStartIdx = tid * state.bucket;

  for (uint i = startIdx; i < endIdx; i++) {
    uint digit = uint(state.src[i].key >> state.shift) & mask;
    uint localOffset = state.offs[histStartIdx + digit];
    uint globalOffset = state.globalOffs[digit];
    state.dst[globalOffset + localOffset] = state.src[i];
    state.offs[histStartIdx + digit] += 1;
  }
}
