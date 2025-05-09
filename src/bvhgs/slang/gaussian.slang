module gaussian;

struct Gaussian3D {
  float3 position;
  float4 rotation;
  float3 scale;

  float3 color;
  float opacity;

  // TODO: Spherical harmonics.

  __init(float3 position = float3(0, 0, 0),
         float4 rotation = float4(0, 0, 0, 1), float3 scale = float3(1, 1, 1),
         float3 color = float3(1, 1, 1), float opacity = 1) {
    this.position = position;
    this.rotation = rotation;
    this.scale = scale;

    this.color = color;
    this.opacity = opacity;
  }
};

// TODO: Implement Gaussian2D.
