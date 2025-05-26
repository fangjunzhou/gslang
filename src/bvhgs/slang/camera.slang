module camera;

import math;
import gaussian;

static const float EPS = 1e-6;

public struct Camera {
  no_diff float4 _rotation;
  no_diff float3 _translation;
  no_diff uint2 _sensorSize;
  no_diff float _focalLength;

  no_diff float _nearPlane;
  no_diff float _farPlane;

  public __init(float4 rotation = float4(0, 0, 0, 1),
                float3 translation = float3(0, 0, 0),
                uint2 sensorSize = uint2(1920, 1080), float focalLength = 100.0,
                float nearPlane = 0.1, float farPlane = 1000.0) {
    this._rotation = rotation;
    this._translation = translation;
    this._sensorSize = sensorSize;
    this._focalLength = focalLength;
    this._nearPlane = nearPlane;
    this._farPlane = farPlane;
  }

  // Rotation matrix in affine viewing transform.
  property W : float3x3 {
    [Differentiable]
    get { return quat.asRotMat(this._rotation); }
  }

  property S : float2x2 {
    [Differentiable]
    get {
      float2 fSensorSize = float2(this._sensorSize) / this._focalLength;
      return float2x2(1 / fSensorSize.x, 0, 0, -1 / fSensorSize.y);
    }
  }

  // Trnslation vector in affine viewing transform.
  property d : float3 {
    [Differentiable]
    get { return this._translation; }
  }

  public property sensorSize : uint2 {
    get { return this._sensorSize; }
  }

  public property focalLength : float {
    get { return this._focalLength; }
  }

  // Convert a point in world space to view space.
  // @param p The point in world space.
  // @return The point in view space.
  [Differentiable]
  float3 toView(float3 p) { return mul(this.W, p) + this.d; }

  [Differentiable]
  float2 toScreen(float2 p) { return mul(this.S, p) + float2(0.5, 0.5); }

  // Convert a point in view space to ray space.
  // @param p The point in view space.
  // @return The point in ray space.
  [Differentiable]
  float3 toRay(float3 p) {
    float z = max(p.z, EPS);
    float inv_z = 1.0 / z;
    float sign = sign(p.z);
    return float3(p.x * inv_z, p.y * inv_z, p.z);
  }

  // Jacobian of the projective transformation.
  // @param p The point in view space.
  // @return The Jacobian of the projective transformation.
  [Differentiable]
  float3x3 toRayJ(float3 p) {
    float len = length(p);
    len = max(len, EPS);
    float sign = sign(p.z);

    float3 col0 = float3(1.0 / p.z, 0, (p.x / len) * sign);

    float3 col1 = float3(0, 1.0 / p.z, (p.y / len) * sign);

    float3 col2 = float3(0, 0, 0);

    return transpose(float3x3(col0, col1, col2));
  }

  // Convert a 3D Gaussian to a 2D Gaussian.
  // @param g The 3D Gaussian.
  // @return The 2D Gaussian.
  [Differentiable]
  public Gaussian2D toGaussian2D(Gaussian3D g) {
    float3 u = this.toView(g.position);

    // Vk'
    float3x3 Vrk = g.covariance();

    // to ray space
    float3 x = this.toRay(u);
    float depth = x.z;

    // normalize depth
    depth = (depth - this._nearPlane) / (this._farPlane - this._nearPlane);

    float3x3 J = this.toRayJ(u);
    float3x3 T = mul(J, this.W);
    float3x3 Vk = mul(T, mul(Vrk, transpose(T)));
    float2x2 conv2d = float2x2(Vk[0][0], Vk[0][1], Vk[1][0], Vk[1][1]);

    float3 s = float3(this.toScreen(x.xy), depth);
    float2x2 sCov = mul(this.S, mul(conv2d, transpose(this.S)));

    float3 viewDir = normalize(-u);
    float3 sh = g.sphericalHarmonics(viewDir);
    float3 color = g.colorSig() + sh;
    float opacity = g.opacitySig();

    return Gaussian2D(s, sCov, color, opacity);
  }
};
