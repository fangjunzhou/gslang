module renderer;

import radix_sort;
import camera;
import bounding_box;
import gaussian;

// Camera parameters.
Camera g_camera;
// Gaussian point cloud in world space.
StructuredBuffer<Gaussian3D> g_gaussian_3d;
// Gaussian point cloud in screen space.
RWStructuredBuffer<Gaussian2D> g_gaussian_2d;
// Culled 2D gaussian point cloud in screen space.
RWStructuredBuffer<uint> g_inside_flag;
RWStructuredBuffer<uint> g_inside_offset;
RWStructuredBuffer<Gaussian2D> g_gaussian_2d_culled;
// Gaussian tile.
static const uint2 TILE_SHAPE = uint2(16, 16);
static const uint TILE_RADIX = 1 << 8;
RWStructuredBuffer<uint> g_num_tile;
RWStructuredBuffer<uint> g_num_tile_prefix;
RWStructuredBuffer<KVPair> g_gaussian_table;
RWStructuredBuffer<uint> g_tile_offs;
RWStructuredBuffer<Gaussian2D> g_gaussian_2d_sorted;

// Render target.
RWTexture2D<float4> g_render_target;

__include "./renderer/projection.slang";
__include "./renderer/tile.slang";
__include "./renderer/rasterizer.slang";
