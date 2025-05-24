module gaussian;

import math;
import bounding_box;

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

static const float BBOX_SIZE = 3;

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
  public float3 scaleExp() { return exp(this.scale); }
  [Differentiable]
  public float3 colorSig() { return utils::sigmoid3(this.color); }
  [Differentiable]
  public float opacitySig() { return utils::sigmoid(this.opacity); }

  [Differentiable]
  public float3x3 covariance() {
    float3 scaleExp = exp(this.scale);
    float3x3 R = quat::asRotMat(rotation);
    float3x3 S = float3x3(float3(scaleExp.x, 0, 0), float3(0, scaleExp.y, 0),
                          float3(0, 0, scaleExp.z));
    return mul(R, mul(S, mul(transpose(S), transpose(R))));
  }
  [Differentiable]
  private float3x3 inv3x3(float3x3 m) {
    float3 c0 = m[0];
    float3 c1 = m[1];
    float3 c2 = m[2];
    float3 adj0 = cross(c1, c2);
    float3 adj1 = cross(c2, c0);
    float3 adj2 = cross(c0, c1);

    float det = dot(c0, adj0);
    return float3x3(adj0, adj1, adj2) / det;
  }
  [Differentiable]
  public float4 eval(float3 pos, float3 dir) {
    float3 d = pos - this.position;
    float3x3 covInv = inv3x3(covariance());
    float exponent = -dot(d, mul(covInv, d));
    float rho = exp(exponent);

    float alpha = this.opacitySig() * rho;

    float3 baseCol = this.colorSig() * rho;
    float3 shCol = this.sphericalHarmonics(normalize(dir));
    float3 finalCol = baseCol + shCol;

    return float4(finalCol, alpha);
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

    float det = a * d - b * c;

    this.cachedInvCov = float2x2(d / det, -b / det, -c / det, a / det);
  }

  // Evaluate the Gaussian at a given point in normalized device coordinates.
  [Differentiable]
  public float eval(float2 uv) {
    // exp(-0.5 * (uv - position)^T * inv(cov) * (uv - position))
    float2 diff = uv - this.position.xy;
    float val = -0.5 * dot(diff, mul(this.cachedInvCov, diff));
    return exp(val);
  }

  // Get the bounding box of a 2D Gaussian.
  // @return A tuple containing the minimum and maximum coordinates of the
  // bounding box.
  public BoundingBox2D boundingBox() {
    // Get the eigenvalues and eigenvectors of the covariance matrix.
    linalg.eigen2x2 eig = linalg.eigen(this.covariance);
    float2 eigenVals = eig.eigenvalues;
    float2x2 eigenVecs = eig.eigenvectors;
    // Convert to semi-axes.
    eigenVecs[0] = normalize(eigenVecs[0]) * sqrt(eigenVals.x) * BBOX_SIZE;
    eigenVecs[1] = normalize(eigenVecs[1]) * sqrt(eigenVals.y) * BBOX_SIZE;

    // Approximate the bounding box of the Gaussian with four axis-aligned
    // rectangles.
    float2[4] axis = { eigenVecs[0], eigenVecs[1], -eigenVecs[0],
                       -eigenVecs[1] };
    for (int i = 0; i < 4; ++i) {
      axis[i] = this.position.xy + axis[i];
    }
    // Calculate the bounding box.
    float2 bMin = axis[0];
    float2 bMax = axis[0];
    for (int i = 1; i < 4; ++i) {
      bMin.x = min(bMin.x, axis[i].x);
      bMin.y = min(bMin.y, axis[i].y);
      bMax.x = max(bMax.x, axis[i].x);
      bMax.y = max(bMax.y, axis[i].y);
    }
    // Return the bounding box as a tuple.
    return BoundingBox2D(bMin, bMax);
  }
};
