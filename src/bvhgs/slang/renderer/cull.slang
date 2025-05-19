implementing "../renderer.slang";

[shader("compute")]
[numthreads(64, 1, 1)]
void cull(uint3 tid: SV_DispatchThreadID) {}
