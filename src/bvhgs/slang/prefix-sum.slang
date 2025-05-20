module prefix_sum;

groupshared int partial[2];
[shader("compute")]
[numthreads(64, 1, 1)]
void wave_prefix_sum(uint tid: SV_DispatchThreadID, StructuredBuffer<int> src,
                RWStructuredBuffer<int> dst, uniform uint depth) {
  
  uint idx = tid.x;
  int val = src[idx];

  
  uint laneId = WaveGetLaneIndex(); // Get lane index within the current wave (0–31)
  uint waveSize = WaveGetLaneCount();  // (should be 32)
  uint waveIndex = idx / waveSize;  // (0 or 1 for 64 threads)

  int sumExcl = WavePrefixSum(val);
  int inWaveIncl = sumExcl + val;

  // Let last lane in each wave store its wave’s total into partial[]
  if (laneId == waveSize - 1){
    partial[waveIndex] = inWaveIncl;
  }

 
  GroupMemoryBarrierWithGroupSync();

  // Only lanes 0 and 1 of wave 0 need to scan partial[0..1]
  if (waveIndex == 0 && laneId < 2){
    int p = partial[laneId];
    int pExcl = WavePrefixSum(p);
    partial[laneId] = pExcl + p; 
  }

 
  GroupMemoryBarrierWithGroupSync();

  // Threads in wave 0 have no offset; wave 1 adds partial[0]
  int groupOffset = (waveIndex > 0) ? partial[waveIndex - 1] : 0;
  dst[idx] = inWaveIncl + groupOffset;;
}
