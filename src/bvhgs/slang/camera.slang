module camera;

import math;
import gaussian;
import math;

static const float EPS = 1e-6;

public struct Camera {
  float4 rotation;
  float3 translation;
  float2 sensorSize;
  float focalLength;

  // Rotation matrix in affine viewing transform.
  property W : float3x3 {
    get { return quat.asRotMat(this.rotation); }
  }

  // Trnslation vector in affine viewing transform.
  property d : float3 {
    get { return this.translation; }
  }

  // Convert a point in world space to view space.
  // @param p The point in world space.
  // @return The point in view space.
  float3 toView(float3 p) { return mul(this.W, p) + this.d; };

  // Convert a point in view space to ray space.
  // @param p The point in view space.
  // @return The point in ray space.
  float3 toRay(float3 p) {
    float z = max(p.z, EPS);
    float inv_z = 1.0 / z;
    return float3(p.x * inv_z, p.y * inv_z, length(p));
  };

  // Jacobian of the projective transformation.
  // @param p The point in view space.
  // @return The Jacobian of the projective transformation.
  float3x3 toRayJ(float3 p) {
    float len = length(p);
    len = max(len, EPS);

    float3 col0 = float3(1.0 / p.z, 0, p.x / len);

    float3 col1 = float3(0, 1.0 / p.z, p.y / len);

    float3 col2 = float3(-p.x / (p.z * p.z), -p.y / (p.z * p.z), p.z / len);

    return float3x3(col0, col1, col2);
  };

  // Convert a 3D Gaussian to a 2D Gaussian.
  // @param g The 3D Gaussian.
  // @return The 2D Gaussian.
  Gaussian2D toGaussian2D(Gaussian3D g) {
    float3 u = this.toView(g.position);

    // Vk'
    float3x3 Vobj = g.covariance();
    float3x3 Vcam = mul(this.W, mul(Vobj, transpose(this.W)));
    
    // to ray space
    float3 x = this.toRay(u);
    float3x3 J = this.toRayJ(u);
    float3x3 Vk = mul(J, mul(Vcam, transpose(J)));

    float2x2 conv2d = float2x2(Vk[0][0], Vk[0][1], Vk[1][0], Vk[1][1]);

    float3 viewDir = normalize(-u);
    float3 sh = g.sphericalHarmonics(viewDir);
    float3 color = g.color + sh;

    return Gaussian2D(float3(x), conv2d, utils.sigmoid3(color), utils.sigmoid(g.opacity));
  };
};
