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
  RWStructuredBuffer<uint> histOffs;  // Global offset
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
      state.histOffs[i] = 0;
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
    uint digit = (state.src[i].key >> state.shift) & mask;
    state.histOffs[histStartIdx + digit] += 1;
    
  }

}

// wave scan
static const uint WAVE = 32;
[shader("compute")]
[numthreads(WAVE, 1, 1)]
void waveScan(uint tid: SV_DispatchThreadID, uniform RadixState state, RWStructuredBuffer<uint> histOffs, RWStructuredBuffer<uint> partial) {
  uint n = state.numThreads;
  uint bucket = state.bucket;
  uint numWaves = (n + WAVE - 1) / WAVE;
  for (uint i = 0; i < bucket; i++) {
    uint idx = tid * bucket + i;
    uint val = (tid < n) ? histOffs[idx] : 0u;
    uint excl = WavePrefixSum(val);
    if (tid < n) {
      histOffs[idx] = excl;
    }
    if (tid % WAVE == WAVE - 1) {
      uint blockIdx = tid / WAVE;
      partial[i * numWaves + blockIdx] = excl + val;
    }
  }
}

// add offset
[shader("compute")]
[numthreads(WAVE, 1, 1)]
void addOffset(uint tid: SV_DispatchThreadID, uniform RadixState state, RWStructuredBuffer<uint> histOffs, StructuredBuffer<uint> partial) {
  uint n = state.numThreads;
  uint bucket = state.bucket;
  uint numWaves = (n + WAVE - 1) / WAVE;
  if (tid >= n) return;
  uint blockIdx = tid / WAVE;
  for (uint i = 0; i < bucket; i++) {
    uint idx = tid * bucket + i;
    uint offset = (blockIdx == 0) ? 0u : partial[i * numWaves + blockIdx];
    histOffs[idx] += offset;
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

  uint localCounters[256];
  for (uint i = 0; i < state.bucket; i++) {
    localCounters[i] = 0;
  }

  for (uint i = startIdx; i < endIdx; i++) {
    uint digit = (state.src[i].key >> state.shift) & mask;
    uint globalOffset = state.histOffs[histStartIdx + digit];
    uint histOffset = localCounters[digit];
    state.dst[globalOffset + histOffset] = state.src[i];
    localCounters[digit] += 1;
  }
}