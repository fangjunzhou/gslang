module gaussian;

public
struct Gaussian3D : IDifferentiable {
 public
  float3 position;
 public
  float4 rotation;
 public
  float3 scale;

 public
  float3 color;
 public
  float opacity;

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
 public
  float3 position;
  // Gaussian covariance in normalized device coordinates.
 public
  float2x2 covariance;

  // Gaussian color.
 public
  float3 color;
  // Gaussian opacity.
 public
  float opacity;

 const float PI = 3.14159265358979323846;
 float2x2 cachedInvCov;
 float cachedDet;
 float cachedNorm;
 const float EPSILON = 1e-6;
 
 public
  __init(float3 position = float3(0, 0, 0),
         float2x2 covariance = float2x2(1, 0, 0, 1),
         float3 color = float3(1, 0, 1), float opacity = 1) {
    this.position = position;
    this.covariance = covariance;

    this.color = color;
    this.opacity = opacity;

    //cache to avoid recomputing
    float a = this.covariance[0][0];
    float b = this.covariance[0][1];
    float c = this.covariance[1][0];
    float d = this.covariance[1][1];

    this.cachedDet = a * d - b * c;
    this.cachedDet = max(this.cachedDet, EPSILON);
    this.cachedNorm = 1 / (2 * PI * sqrt(this.cachedDet));

    this.cachedInvCov = float2x2(d / this.cachedDet, -b / this.cachedDet, -c / this.cachedDet, a / this.cachedDet);
  }

  // Evaluate the Gaussian at a given point in normalized device coordinates.
  [Differentiable] public float3 eval(float2 uv) {
    // TODO: Implement Gaussian2D evaluation.
    //opacity * color * (1/(2pi*sqrt(det(cov)))) * exp(-0.5 * (uv - position)^T * inv(cov) * (uv - position))
    float2 diff = uv - this.position.xy;
    float val = -0.5 * dot(diff, mul(this.cachedInvCov, diff));
    return this.opacity * this.color * this.cachedNorm * exp(val);
  }
};
