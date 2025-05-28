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

  public __init(inout Atomic<float>[11] atomicGaussian) {
    this.position = float3(atomicGaussian[0].load(), atomicGaussian[1].load(),
                           atomicGaussian[2].load());
    this.covariance =
        float2x2(atomicGaussian[3].load(), atomicGaussian[4].load(),
                 atomicGaussian[5].load(), atomicGaussian[6].load());
    this.color = float3(atomicGaussian[7].load(), atomicGaussian[8].load(),
                        atomicGaussian[9].load());
    this.opacity = atomicGaussian[10].load();

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

  public Atomic<float>[11] toAtomic() {
    Atomic<float>[11] atomicGaussian;
    atomicGaussian[0] = this.position.x;
    atomicGaussian[1] = this.position.y;
    atomicGaussian[2] = this.position.z;
    atomicGaussian[3] = this.covariance[0][0];
    atomicGaussian[4] = this.covariance[0][1];
    atomicGaussian[5] = this.covariance[1][0];
    atomicGaussian[6] = this.covariance[1][1];
    atomicGaussian[7] = this.color.x;
    atomicGaussian[8] = this.color.y;
    atomicGaussian[9] = this.color.z;
    atomicGaussian[10] = this.opacity;
    return atomicGaussian;
  }

  public float[11] toArray() {
    float[11] arr;
    arr[0] = this.position.x;
    arr[1] = this.position.y;
    arr[2] = this.position.z;
    arr[3] = this.covariance[0][0];
    arr[4] = this.covariance[0][1];
    arr[5] = this.covariance[1][0];
    arr[6] = this.covariance[1][1];
    arr[7] = this.color.x;
    arr[8] = this.color.y;
    arr[9] = this.color.z;
    arr[10] = this.opacity;
    return arr;
  }

};

// Overload the addition and subtraction operators for Gaussian2D.
public Gaussian2D operator +(Gaussian2D a, Gaussian2D b) {
  return Gaussian2D(a.position + b.position, a.covariance + b.covariance,
                    a.color + b.color, a.opacity + b.opacity);
}
public Gaussian2D operator-(Gaussian2D a, Gaussian2D b) {
  return Gaussian2D(a.position - b.position, a.covariance - b.covariance,
                    a.color - b.color, a.opacity - b.opacity);
}
public Gaussian2D operator *(Gaussian2D g, float s) {
  return Gaussian2D(g.position * s, g.covariance * (s * s), g.color * s,
                    g.opacity * s);
}
public Gaussian2D operator*(float s, Gaussian2D g) {
  return Gaussian2D(g.position * s, g.covariance * (s * s), g.color * s,
                    g.opacity * s);
}
public Gaussian2D operator /(Gaussian2D g, float s) {
  return Gaussian2D(g.position / s, g.covariance / (s * s), g.color / s,
                    g.opacity / s);
}



// public struct Gaussian2DAtomic {
//   public Atomic<float>[3] position;

//   public Atomic<float> [2][2] covariance;
//   public Atomic<float>[3] color;
//   public Atomic<float> opacity;

//   __init(Gaussian2D g) {
//     this.position[0] = g.position.x;
//     this.position[1] = g.position.y;
//     this.position[2] = g.position.z;
//     this.covariance[0][0] = g.covariance[0][0];
//     this.covariance[0][1] = g.covariance[0][1];
//     this.covariance[1][0] = g.covariance[1][0];
//     this.covariance[1][1] = g.covariance[1][1];
//     this.color[0] = g.color.x;
//     this.color[1] = g.color.y;
//     this.color[2] = g.color.z;
//     this.opacity = g.opacity;
//   }

//   [mutating]
//   public void add(Gaussian2D g) {
//     this.position[0].add(g.position.x);
//     this.position[1].add(g.position.y);
//     this.position[2].add(g.position.z);
//     this.covariance[0][0].add(g.covariance[0][0]);
//     this.covariance[0][1].add(g.covariance[0][1]);
//     this.covariance[1][0].add(g.covariance[1][0]);
//     this.covariance[1][1].add(g.covariance[1][1]);
//     this.color[0].add(g.color.x);
//     this.color[1].add(g.color.y);
//     this.color[2].add(g.color.z);
//     this.opacity.add(g.opacity);
//   }

//   [mutating]
//   public Gaussian2D toGaussian2D() {
//     float3 position = float3(this.position[0].load(),
//     this.position[1].load(),
//                              this.position[2].load());
//     float2x2 covariance =
//         float2x2(this.covariance[0][0].load(), this.covariance[0][1].load(),
//                  this.covariance[1][0].load(), this.covariance[1][1].load());
//     float3 color = float3(this.color[0].load(), this.color[1].load(),
//                           this.color[2].load());
//     float opacity = this.opacity.load();
//     return Gaussian2D(position, covariance, color, opacity);
//   }
// };
