
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
  uint bufSize;
  RWStructuredBuffer<KVPair> src;
  RWStructuredBuffer<KVPair> dst;
  RWStructuredBuffer<Atomic<uint>> hist;  // Global histogram
  RWStructuredBuffer<Atomic<uint>> offs;  // Global offset
};

// Clear histogram and offset
[shader("compute")]
[numthreads(256, 1, 1)]
void clearHist(uint tid: SV_DispatchThreadID, uniform RadixState state) {
  if (tid < 256) {
    state.hist[tid].store(0);
    state.offs[tid].store(0);
  }
}

// Build histogram for the current pass
[shader("compute")]
[numthreads(256, 1, 1)]
void buildHist(uint tid: SV_DispatchThreadID, uniform RadixState state) {
  if (tid >= state.bufSize) return;
  uint64_t key = state.src[tid].key;
  uint digit = uint((key >> state.shift) & state.mask);
  state.hist[digit].add(1);
}

groupshared KVPair localBuffer[64];  // stores 64 KVPairs temporarily
groupshared Atomic<uint> localCounts[256];  // number of elements in each bin in this group
groupshared uint localOffsets[256];  // offset for each bin within the global output array
[shader("compute")]
[numthreads(64, 1, 1)]  
void scatter(uint3 gid: SV_GroupID, uint3 tid: SV_GroupThreadID, 
             uint gtid: SV_GroupIndex, uniform RadixState state,
             uniform uint binOffset, uniform uint binSize) {

  // clear 256 threads
  for (uint i = gtid; i < 256; i += 64) {
    localCounts[i].store(0);
    localOffsets[i] = 0;
  }
  GroupMemoryBarrierWithGroupSync();  // sync across all threads

  // global thread ID
  uint globalIdx = gid.x * 64 + tid.x;

  // each thread reads one element, stores KVPair
  if (globalIdx < binSize && binOffset + globalIdx < state.bufSize) {
    KVPair kv = state.src[binOffset + globalIdx];
    uint digit = uint((kv.key >> state.shift) & state.mask);
    localCounts[digit].add(1);
    localBuffer[gtid] = kv;
  }

  GroupMemoryBarrierWithGroupSync();

  // thread 0 computes global offset for each digit
  if (gtid == 0) {
    for (uint i = 0; i < 256; i++) {
      uint count = localCounts[i].load();
      if (count > 0) {
        // atomically add group’s count into the global offs buffer.
        localOffsets[i] = state.offs[i].add(count);
      }
    }
  }

  GroupMemoryBarrierWithGroupSync();
  // each thread writes its KVPair into the final destination
  if (globalIdx < binSize && binOffset + globalIdx < state.bufSize) {
    uint d = uint((localBuffer[gtid].key >> state.shift) & state.mask);
    // compute this thread’s rank with those have the same digit
    uint rank = 0;
    for (uint i = 0; i < gtid; i++) {
      if (uint((localBuffer[i].key >> state.shift) & state.mask) == d) {
        rank++;
      }
    }
    // write to the destination slot
    uint dstIdx = localOffsets[d] + rank;
    if (dstIdx < state.bufSize) {
      state.dst[dstIdx] = localBuffer[gtid];
    }
  }
}

