implementing math;

namespace quat {

// Quaternion multiplication.
// @param q1 First quaternion.
// @param q2 Second quaternion.
// @return The product of the two quaternions.
[Differentiable]
public float4 mul(float4 q1, float4 q2) {
  // Scalar.
  float w = q1.w * q2.w - dot(q1.xyz, q2.xyz);
  // Vector.
  float3 v = q1.w * q2.xyz + q2.w * q1.xyz + cross(q1.xyz, q2.xyz);
  return float4(v, w);
}

// Complex conjugate.
// @param q Quaternion.
// @return The conjugate of the quaternion.
[Differentiable]
public float4 conj(float4 q) { return float4(-q.xyz, q.w); }

// Quaternion inverse.
// @param q Quaternion.
// @return The inverse of the quaternion. This function works for both
// normalized and non-normalized quaternions by using the squared norm for
// robustness.
[Differentiable]
public float4 inv(float4 q) { return conj(q) / dot(q, q); }

// Convert a axis-angle representation to a quaternion.
// @param v Axis-angle representation with v as the axis of rotation and the
// length of v as the angle of rotation.
// @return The quaternion representation of the axis-angle representation.
[Differentiable]
public float4 fromAxisAngle(float3 v) {
  float theta = length(v);
  if (theta > 0) {
    v = normalize(v) * sin(theta / 2);
  }
  float w = cos(theta / 2);
  return float4(v, w);
}

// Convert a quaternion to a axis-angle representation.
// @param q Quaternion.
// @return The axis-angle representation of the quaternion.
[Differentiable]
public float3 asAxisAngle(float4 q) {
  float theta = 2 * acos(q.w);
  float norm = length(q.xyz);
  float3 res = float3(0, 0, 0);
  if (norm > 0) {
    res = q.xyz / norm * theta;
  }
  return res;
}

// Rotate a vector by a quaternion.
// @param q Quaternion.
// @param v Vector to rotate.
// @return The rotated vector.
[Differentiable]
public float3 rotate(float4 q, float3 v) {
  float4 qv = float4(v, 0);
  return mul(mul(q, qv), conj(q)).xyz;
}

// Convert a quaternion to a rotation matrix.
// @param q Quaternion.
// @return The rotation matrix representation of the quaternion.
[Differentiable]
public float3x3 asRotMat(float4 q) {
  // TODO: Check correctness and efficiency.
  float3 rx = rotate(q, float3(1, 0, 0));
  float3 ry = rotate(q, float3(0, 1, 0));
  float3 rz = rotate(q, float3(0, 0, 1));
  return transpose(float3x3(rx, ry, rz));
}

}  // namespace quat
