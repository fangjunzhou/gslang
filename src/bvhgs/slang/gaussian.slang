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

  // Spherical harmonics.
  float3 sh10, sh11, sh12;
  float3 sh20, sh21, sh22, sh23, sh24;
  float3 sh30, sh31, sh32, sh33, sh34, sh35, sh36;
  const float PI = 3.14159265358979323846;
  const float c0 = sqrt(3.0 / PI);
  const float c1 = sqrt(15.0 / PI);
  const float c2 = sqrt(5.0 / PI);
  const float c3 = sqrt(35.0 / (2.0 * PI));
  const float c4 = sqrt(105.0 / PI);
  const float c5 = sqrt(21.0 / (2.0 * PI));
  const float c6 = sqrt(7.0 / PI);

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

  [Differentiable] public float3x3 covariance(){
      // TODO: Implement covariance matrix.
  }

      /**
       * Computes the spherical harmonics (SH) value for a given direction.
       *
       * @param dir A 3D unit vector representing the direction in Cartesian
       * coordinates. It is expected to be normalized (i.e., |dir| = 1).
       * @return A float3 value representing the SH evaluation for the given
       * direction. Each component corresponds to the weighted sum of SH basis
       * functions for the respective color channel (e.g., RGB).
       *
       * The SH coefficients (sh10, sh11, ..., sh36) are ordered by degree (l)
       * and order (m) as follows:
       *   - l = 1: sh10 (m = -1), sh11 (m = 0), sh12 (m = 1)
       *   - l = 2: sh20 (m = -2), sh21 (m = -1), sh22 (m = 0), sh23 (m = 1),
       * sh24 (m = 2)
       *   - l = 3: sh30 (m = -3), sh31 (m = -2), ..., sh36 (m = 3)
       *
       * The output is computed as a linear combination of SH basis functions
       * (y_lm) and their corresponding coefficients (sh_lm). The basis
       * functions are precomputed using the input direction and constants (c0,
       * c1, ...).
       */
          [Differentiable] public float3 sphericalHarmonics(float3 dir) {
    float y10 = 0.5 * c0 * dir.y;
    float y11 = 0.5 * c0 * dir.z;
    float y12 = 0.5 * c0 * dir.x;

    float y20 = 0.5 * c1 * dir.x * dir.y;
    float y21 = 0.5 * c1 * dir.y * dir.z;
    float y22 = 0.25 * c2 * (3 * dir.z * dir.z - 1);
    float y23 = 0.5 * c1 * dir.x * dir.z;
    float y24 = 0.25 * c1 * (dir.x * dir.x - dir.y * dir.y);

    float y30 = 0.25 * c3 * dir.y * (3 * dir.x * dir.x - dir.y * dir.y);
    float y31 = 0.5 * c4 * dir.x * dir.y * dir.z;
    float y32 = 0.25 * c5 * dir.y * (5 * dir.z * dir.z - 1);
    float y33 = 0.25 * c6 * (5 * dir.z * dir.z * dir.z - 3 * dir.z);
    float y34 = 0.25 * c5 * dir.x * (5 * dir.z * dir.z - 1);
    float y35 = 0.25 * c4 * (dir.x * dir.x - dir.y * dir.y) * dir.z;
    float y36 = 0.25 * c3 * dir.x * (dir.x * dir.x - 3 * dir.y * dir.y);

    return y10 * this.sh10 + y11 * this.sh11 + y12 * this.sh12 +
           y20 * this.sh20 + y21 * this.sh21 + y22 * this.sh22 +
           y23 * this.sh23 + y24 * this.sh24 + y30 * this.sh30 +
           y31 * this.sh31 + y32 * this.sh32 + y33 * this.sh33 +
           y34 * this.sh34 + y35 * this.sh35 + y36 * this.sh36;
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

    // cache to avoid recomputing
    float a = this.covariance[0][0];
    float b = this.covariance[0][1];
    float c = this.covariance[1][0];
    float d = this.covariance[1][1];

    this.cachedDet = a * d - b * c;
    this.cachedDet = max(this.cachedDet, EPSILON);
    this.cachedNorm = 1 / (2 * PI * sqrt(this.cachedDet));

    this.cachedInvCov = float2x2(d / this.cachedDet, -b / this.cachedDet,
                                 -c / this.cachedDet, a / this.cachedDet);
  }

  // Evaluate the Gaussian at a given point in normalized device coordinates.
  [Differentiable] public float3 eval(float2 uv) {
    // TODO: Implement Gaussian2D evaluation.
    // opacity * color * (1/(2pi*sqrt(det(cov)))) * exp(-0.5 * (uv - position)^T
    // * inv(cov) * (uv - position))
    float2 diff = uv - this.position.xy;
    float val = -0.5 * dot(diff, mul(this.cachedInvCov, diff));
    return this.opacity * this.color * this.cachedNorm * exp(val);
  }
};
