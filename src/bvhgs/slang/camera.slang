module camera;

import math;
import gaussian;

public
struct Camera {
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
  float3 toView(float3 p) { return this.W * p + this.d; }

  // Convert a point in view space to ray space.
  // @param p The point in view space.
  // @return The point in ray space.
  float3 toRay(float3 p) {
    // TODO: Implement projective transformation.
  }

  // Jacobian of the projective transformation.
  // @param p The point in view space.
  // @return The Jacobian of the projective transformation.
  float3x3 toRayJ(float3 p) {
    // TODO: Imlement analytical Jacobian.
    return float3x3(0);
  }
};
