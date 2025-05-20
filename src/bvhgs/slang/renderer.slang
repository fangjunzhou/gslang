module renderer;

import camera;
import gaussian;

// Camera parameters.
Camera g_camera;
// Gaussian point cloud in world space.
StructuredBuffer<Gaussian3D> g_gaussian_3d;
// Gaussian point cloud in screen space.
RWStructuredBuffer<Gaussian2D> g_gaussian_2d;
// Culled 2D gaussian point cloud in screen space.
RWStructuredBuffer<int> g_cull_flag;
RWStructuredBuffer<int> g_cull_prefix;
RWStructuredBuffer<Gaussian2D> g_gaussian_2d_culled;
// Gaussian tile.
RWStructuredBuffer<int> num_tile;
RWStructuredBuffer<int> num_tile_prefix;
RWStructuredBuffer<Tuple<uint64_t, uint>> gaussian_table;

// Render target.
RWTexture2D<float4> g_render_target;

__include "./renderer/projection.slang";
__include "./renderer/tile.slang";
__include "./renderer/rasterizer.slang";
