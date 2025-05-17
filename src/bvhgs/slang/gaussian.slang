module gaussian;

public
struct Gaussian3D : IDifferentiable {
  float3 position;
  float4 rotation;
  float3 scale;

  float3 color;
  float opacity;

  public 
  property float3 Position {
    get { return this.position; }
  }

  public
  property float3 Color {
      get { return this.color; }
  }

  public
  property float Opacity {
      get { return this.opacity; }
  }
  // TODO: Spherical harmonics.

 public
  __init(float3 position = float3(0, 0, 0),
         float4 rotation = float4(0, 0, 0, 1), float3 scale = float3(1, 1, 1),
         float3 color = float3(1, 0, 1), float opacity = 1) {
    this.position = position;
    this.rotation = rotation;
    this.scale = scale;

    this.color = color;
    this.opacity = opacity;
  }

  [Differentiable] public float3x3 covariance() {
    // TODO: Implement covariance matrix.
  }
};

public
struct Gaussian2D : IDifferentiable {
  // Gaussian position in normalized device coordinates.
  float3 position;
  // Gaussian covariance in normalized device coordinates.
  float2x2 covariance;

  // Gaussian color.
  float3 color;
  // Gaussian opacity.
  float opacity;

  // Gaussian depth in normalized device coordinates.
 public
  property float depth {
    get { return this.position.z; }
  }



 public
  __init(float3 position = float3(0, 0, 0),
         float2x2 covariance = float2x2(1, 0, 0, 1),
         float3 color = float3(1, 0, 1), float opacity = 1) {
    this.position = position;
    this.covariance = covariance;

    this.color = color;
    this.opacity = opacity;
  }

  // Evaluate the Gaussian at a given point in normalized device coordinates.
  [Differentiable] public float3 eval(float2 uv) {
    // TODO: Implement Gaussian2D evaluation.
  }
};
