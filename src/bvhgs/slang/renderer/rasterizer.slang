implementing "../renderer.slang";

[shader("compute")]
[numthreads(8, 8, 1)]
void rasterize(uint3 tid: SV_DispatchThreadID) {}

