implementing math;

namespace quat {

public
float4 mul(float4 q1, float4 q2) {
  // Scalar.
  float w = q1.w * q2.w - dot(q1.xyz, q2.xyz);
  // Vector.
  float3 v = q1.w * q2.xyz + q2.w * q1.xyz + cross(q1.xyz, q2.xyz);
  return float4(v, w);
}

}  // namespace quat
