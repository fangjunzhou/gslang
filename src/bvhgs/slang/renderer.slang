module renderer;

import camera;
import gaussian;

// Camera parameters.
Camera g_camera;
// Gaussian point cloud in world space.
StructuredBuffer<Gaussian3D> g_gaussian_3d;
// Gaussian point cloud in ray space.
RWStructuredBuffer<Gaussian2D> g_gaussian_2d;
// Culled 2D gaussian point cloud in ray space.
// TODO: Use parallel prefix sum.
RWStructuredBuffer<bool> g_cull_flag;
RWStructuredBuffer<uint> g_cull_prefix;
RWStructuredBuffer<Gaussian2D> g_gaussian_2d_culled;

// Render target.
RWTexture2D<float4> g_render_target;

__include "./renderer/project.slang";
__include "./renderer/rasterizer.slang";
