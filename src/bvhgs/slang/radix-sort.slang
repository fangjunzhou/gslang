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
  // 
}

// Build histogram for the current pass
[shader("compute")]
[numthreads(256, 1, 1)]
void buildHist(uint tid: SV_DispatchThreadID, uniform RadixState state) {
  // todo: 1. calculate mask from bucket
  //       2. locate src start and end
  //       3. locate histOffs start and end
  //       4. read src, update histOffs in for loop

}

// todo: add histogram wave scan kernel (tid,StructuredBuffer<uint> src, RWStructuredBuffer<uint> partial)
// [numthreads(32, 1, 1)]
// for i in range(bucket)

// todo: add offset kernel (tid, src, partial)
// [numthreads(32, 1, 1)]


[shader("compute")]
[numthreads(256, 1, 1)]
void scatter(uint3 tid: SV_GroupThreadID, uniform RadixState state, ) {
  // todo: // todo: 1. calculate mask from bucket
  //       2. locate src start and end
  //       3. locate histOffs start and end
  //       4. for entry in src
  //       5. shift bitwise mask
  //       6. find golbal offset and per-thread histOffset
  //       7. write to dst
  //       8. icrement per-thread histOffset

  
  
}