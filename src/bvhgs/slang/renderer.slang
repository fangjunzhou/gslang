module renderer;

import radix_sort;
import camera;
import bounding_box;
import gaussian;

// Camera parameters.
Camera g_camera;
// Gaussian point cloud in world space.
RWStructuredBuffer<Gaussian3D> g_gaussian_3d;
// Gaussian point cloud in screen space.
RWStructuredBuffer<Gaussian2D> g_gaussian_2d;
// Culled 2D gaussian point cloud in screen space.
RWStructuredBuffer<uint> g_inside_flag;
RWStructuredBuffer<uint> g_inside_offset;
RWStructuredBuffer<Gaussian2D> g_gaussian_2d_culled;
// Gaussian tile.
static const uint2 TILE_SHAPE = uint2(16, 16);
static const uint TILE_RADIX = 1 << 8;
RWStructuredBuffer<uint> g_num_tiles;
RWStructuredBuffer<uint> g_num_tiles_prefix;
RWStructuredBuffer<KVPair> g_gaussian_table;
RWStructuredBuffer<Atomic<uint>> g_tile_hist_atomic;
RWStructuredBuffer<uint> g_tile_hist;
RWStructuredBuffer<uint> g_tile_offs;
RWStructuredBuffer<Gaussian2D> g_gaussian_2d_sorted;

// Render target.
RWTexture2D<float4> g_render_target;
RWTexture2D<float4> g_depth_target;
RWTexture2D<uint> g_num_rendered_gaussians;

// Render target gradient.
RWTexture2D<float4> d_render_target;
// Sorted screen space gaussian gradient.
RWStructuredBuffer<Atomic<float>[11]> d_a_gaussian_2d_sorted;
RWStructuredBuffer<Gaussian2D> d_gaussian_2d_sorted;
// Culled screen space gaussian gradient.
RWStructuredBuffer<Atomic<float>[11]> d_a_gaussian_2d_culled;
RWStructuredBuffer<Gaussian2D> d_gaussian_2d_culled;
// World space gaussian gradient.
RWStructuredBuffer<Gaussian3D> d_gaussian_3d;

__include "./renderer/projection.slang";
__include "./renderer/tile.slang";
__include "./renderer/rasterizer.slang";
__include "./renderer/optimizer.slang";
