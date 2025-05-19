module renderer;

import camera;
import gaussian;

// Camera parameters.
Camera g_camera;
// Gaussian point cloud in world space.
StructuredBuffer<Gaussian3D> g_gaussian_3d;
// Gaussian point cloud in ray space.
RWStructuredBuffer<Gaussian2D> g_gaussian_2d;

// Render target.
RWTexture2D<float4> g_render_target;

__include "./renderer/project.slang";
__include "./renderer/cull.slang";
__include "./renderer/rasterizer.slang";
