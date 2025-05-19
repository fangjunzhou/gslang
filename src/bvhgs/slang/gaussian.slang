module gaussian;

static const float PI = 3.14159265358979323846;
static const float EPS = 1e-6;
static const float C0 = sqrt(3.0 / PI);
static const float C1 = sqrt(15.0 / PI);
static const float C2 = sqrt(5.0 / PI);
static const float C3 = sqrt(35.0 / (2.0 * PI));
static const float C4 = sqrt(105.0 / PI);
static const float C5 = sqrt(21.0 / (2.0 * PI));
static const float C6 = sqrt(7.0 / PI);
static const int SH_COUNT = 15;

public struct Gaussian3D : IDifferentiable {
  public float3 position;
  public float4 rotation;
  public float3 scale;

  public float3 color;
  public float opacity;

  // Spherical harmonics.
  public float3 sh[SH_COUNT];
  private static float3 defaultSH[SH_COUNT] = {
    float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0),
    float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0),
    float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0),
    float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0)
  };

  public __init(float3 position = float3(0, 0, 0),
                float4 rotation = float4(0, 0, 0, 1),
                float3 scale = float3(1, 1, 1), float3 color = float3(1, 0, 1),
                float opacity = 1, float3 shCoeffs[SH_COUNT] = defaultSH, ) {
    this.position = position;
    this.rotation = rotation;
    this.scale = scale;
    this.sh = shCoeffs;
    this.color = color;
    this.opacity = opacity;
  }

  [Differentiable]
  public float3x3 covariance() {
    // TODO: Implement covariance matrix.
    return float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
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
   * functions are precomputed using the input direction and constants (C0,
   * c1, ...).
   */
  [Differentiable]
  public float3 sphericalHarmonics(float3 dir) {
    float y[SH_COUNT];

    y[0] = 0.5 * C0 * dir.y;
    y[1] = 0.5 * C0 * dir.z;
    y[2] = 0.5 * C0 * dir.x;

    y[3] = 0.5 * C1 * dir.x * dir.y;
    y[4] = 0.5 * C1 * dir.y * dir.z;
    y[5] = 0.25 * C2 * (3 * dir.z * dir.z - 1);
    y[6] = 0.5 * C1 * dir.x * dir.z;
    y[7] = 0.25 * C1 * (dir.x * dir.x - dir.y * dir.y);

    y[8] = 0.25 * C3 * dir.y * (3 * dir.x * dir.x - dir.y * dir.y);
    y[9] = 0.5 * C4 * dir.x * dir.y * dir.z;
    y[10] = 0.25 * C5 * dir.y * (5 * dir.z * dir.z - 1);
    y[11] = 0.25 * C6 * (5 * dir.z * dir.z * dir.z - 3 * dir.z);
    y[12] = 0.25 * C5 * dir.x * (5 * dir.z * dir.z - 1);
    y[13] = 0.25 * C4 * (dir.x * dir.x - dir.y * dir.y) * dir.z;
    y[14] = 0.25 * C3 * dir.x * (dir.x * dir.x - 3 * dir.y * dir.y);

    float3 result = float3(0, 0, 0);
    for (int i = 0; i < SH_COUNT; ++i) {
      result += y[i] * this.sh[i];
    }
    return result;
  }
};

public struct Gaussian2D : IDifferentiable {
  // Gaussian position in normalized device coordinates.
  public float3 position;
  // Gaussian covariance in normalized device coordinates.
  public float2x2 covariance;

  // Gaussian color.
  public float3 color;
  // Gaussian opacity.
  public float opacity;

  float2x2 cachedInvCov;
  float cachedDet;
  float cachedNorm;

  [Differentiable]
  public __init(float3 position = float3(0, 0, 0),
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
    this.cachedDet = max(this.cachedDet, EPS);
    this.cachedNorm = 1 / (2 * PI * sqrt(this.cachedDet));

    this.cachedInvCov = float2x2(d / this.cachedDet, -b / this.cachedDet,
                                 -c / this.cachedDet, a / this.cachedDet);
  }

  // Evaluate the Gaussian at a given point in normalized device coordinates.
  [Differentiable]
  public float3 eval(float2 uv) {
    // TODO: Implement Gaussian2D evaluation.
    // opacity * color * (1/(2pi*sqrt(det(cov)))) * exp(-0.5 * (uv - position)^T
    // * inv(cov) * (uv - position))
    float2 diff = uv - this.position.xy;
    float val = -0.5 * dot(diff, mul(this.cachedInvCov, diff));
    return this.opacity * this.color * this.cachedNorm * exp(val);
  }
};
