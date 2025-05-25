module radix_sort;

// Number of buckets per pass
static const uint NUM_BINS = 16;
// local histogram counts
groupshared Atomic<uint> localCounts[NUM_BINS];

public struct KVPair
{
  uint64_t key;
  uint val;
  __init(uint64_t k, uint v) { key = k; val = v; }
};

struct RadixState
{
  uint shift;
  uint mask;
  uint bufSize;
  uint histSize;   // number of local histogram entries
  uint numGroups;  // number of blocks
  uint blockSize;  // threads per group

  RWStructuredBuffer<KVPair> src;
  RWStructuredBuffer<KVPair> dst;

  RWStructuredBuffer<uint> localHist;
  RWStructuredBuffer<uint> offs;
  RWStructuredBuffer<uint> groupTotals;
  RWStructuredBuffer<uint> groupOffsets;
  RWStructuredBuffer<uint> globalHist;  // final histogram
};

[shader("compute")]
[numthreads(64, 1, 1)]
void clearAll(uint3 tid: SV_DispatchThreadID, RadixState state)
{
  uint idx = tid.x;
  if (idx < state.histSize) {
    state.localHist[idx] = 0;
    state.offs[idx] = 0;
  }
  if (idx < NUM_BINS)
    state.globalHist[idx] = 0;
}

// build per-block histogram
[shader("compute")]
[numthreads(64, 1, 1)]
void buildLocal(uint3 gid: SV_GroupID,         
                uint3 ltid: SV_GroupThreadID,  
                RadixState state)
{
    uint blockSize = state.blockSize;  
    uint base      = gid.x * blockSize;  

    // initialize local counts to zero
    if (ltid.x < NUM_BINS) {
        localCounts[ltid.x].store(0, MemoryOrder::Relaxed);
    }
    GroupMemoryBarrierWithGroupSync();

    // each thread handle 10 elements
    uint start = base + ltid.x;
    const uint stride = 64;
    const uint perThread = 10;
    for (uint i = 0; i < perThread; ++i) {
      uint idx = start + i * stride;
      if (idx >= state.bufSize) break;
      KVPair kv = state.src[idx];
      uint d = (kv.key >> state.shift) & state.mask;
      localCounts[d] += 1;
    }
    GroupMemoryBarrierWithGroupSync();

    // write local histogram counts
    if (ltid.x < NUM_BINS) {
        uint cnt = localCounts[ltid.x].load(MemoryOrder::Relaxed);
        state.localHist[gid.x * NUM_BINS + ltid.x] = cnt;
    }
}
groupshared uint lds[64];

uint validCountInGroup(uint globalBase, uint total, uint groupSize)
{
  return (total > globalBase) ? min(groupSize, total - globalBase) : 0;
}

// scan the local histogram in each group
[shader("compute")]
[numthreads(64, 1, 1)]
void scanGroup(uint3 tid: SV_DispatchThreadID,
               uint3 ltid: SV_GroupThreadID,  // local thread ID
               uint3 gid: SV_GroupID,
               RadixState state)
{
  // each group handles up to 64 histogram entries
  uint groupBase = gid.x * 64;
  uint validCount = validCountInGroup(groupBase, state.histSize, 64);

  // load the histogram count
  uint val = (ltid.x < validCount) ? state.localHist[groupBase + ltid.x] : 0;
  lds[ltid.x] = val;
  GroupMemoryBarrierWithGroupSync();

  // inclusive scan
  for (uint ofs = 1; ofs < 64; ofs <<= 1) {
    uint add = (ltid.x >= ofs) ? lds[ltid.x - ofs] : 0;
    GroupMemoryBarrierWithGroupSync();
    lds[ltid.x] += add;
    GroupMemoryBarrierWithGroupSync();
  }
  // exclusive scan
  if (ltid.x < validCount)
    state.offs[groupBase + ltid.x] = lds[ltid.x] - val;

  // writes the total count into groupTotals
  if (ltid.x == validCount - 1)
    state.groupTotals[gid.x] = lds[ltid.x];
}

// prefix-sum the array of groupTotals into groupOffsets
[shader("compute")]
[numthreads(64, 1, 1)]
void scanGroupTotals(uint3 tid: SV_DispatchThreadID,
                     uint3 ltid: SV_GroupThreadID,
                     RadixState state)
{
  // load each group’s total
  uint val = (ltid.x < state.numGroups) ? state.groupTotals[ltid.x] : 0;
  lds[ltid.x] = val;

  // inclusive scan
  for (uint ofs = 1; ofs < 64; ofs <<= 1) {
    uint add = (ltid.x >= ofs) ? lds[ltid.x - ofs] : 0;
    lds[ltid.x] += add;
  }
  // exclusive scan
  if (ltid.x < state.numGroups)
    state.groupOffsets[ltid.x] = lds[ltid.x] - val;
}

// add each group’s base offset back into every per-bin offset
[shader("compute")]
[numthreads(64, 1, 1)]
void addGroupOffsets(uint3 tid: SV_DispatchThreadID,
                     uint3 gid: SV_GroupID,
                     RadixState state)
{
  // add groupOffsets[gid] to each bin offset
  if (tid.x < state.histSize)
        state.offs[tid.x] += state.groupOffsets[gid.x];
}

// scatter each block’s elements
[shader("compute")]
[numthreads(64, 1, 1)]
void scatterBlock(uint3 tid: SV_DispatchThreadID,
                  RadixState state)
{
  uint base = tid.x * 64;
  if (base >= state.bufSize) return;

  // copy the per-bin starting offsets into an array
  uint cursor[NUM_BINS];
  uint offsBase = tid.x * NUM_BINS;

  for (uint b = 0; b < NUM_BINS; ++b)
        cursor[b] = state.offs[offsBase + b];

  // 64 slots
  for (uint i = 0; i < 64; ++i) {
    uint idx = base + i;
    if (idx >= state.bufSize) break;
    KVPair kv = state.src[idx];
    uint d = uint((kv.key >> state.shift) & state.mask);
    uint dstIx = cursor[d]++;
    state.dst[dstIx] = kv;
  }
}

// recompute the global histogram
[shader("compute")]
[numthreads(NUM_BINS, 1, 1)]
void tileHist(uint3 tid: SV_DispatchThreadID,
              RadixState state)
{
  uint b = tid.x;
  if (b >= NUM_BINS) return;

  uint numBlocks = (state.bufSize + state.blockSize - 1) / state.blockSize;
  uint sum = 0;
  // sum up every block’s count for bin b
  for (uint blk = 0; blk < numBlocks; ++blk)
    sum += state.localHist[blk * NUM_BINS + b];
  state.globalHist[b] = sum;
}