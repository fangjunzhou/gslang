implementing "../math.slang";

namespace linalg {

static const float EPS = 1e-6;

// Eigenvalue decomposition result of a 2x2 matrix.
public struct eigen2x2 {
  // The eigenvalues of the matrix.
  public float2 eigenvalues;
  // The eigenvectors of the matrix.
  public float2x2 eigenvectors;
}

public eigen2x2 eigen(float2x2 m) {
  float a = m[0][0];
  float b = m[0][1];
  float c = m[1][0];
  float d = m[1][1];

  // Special case for diagonal matrix with identical values
  // This directly handles cases like [0.1, 0, 0, 0.1]
  if (abs(b) < EPS && abs(c) < EPS && abs(a - d) < EPS) {
    // Return the eigenvalues (both equal to a) and identity matrix as
    // eigenvectors
    return eigen2x2(float2(a, a), float2x2(1, 0, 0, 1));
  }

  float trace = a + d;
  float det = a * d - b * c;

  float discriminant = trace * trace - 4 * det;
  if (discriminant < 0) {
    // Return small positive values instead of zeros to avoid NaNs
    return eigen2x2(float2(EPS, EPS), float2x2(1, 0, 0, 1));
  }

  float sqrtDiscriminant = sqrt(discriminant);
  float lambda1 = (trace + sqrtDiscriminant) / 2;
  float lambda2 = (trace - sqrtDiscriminant) / 2;

  // Ensure eigenvalues are at least EPS to avoid sqrting very small numbers
  lambda1 = max(lambda1, EPS);
  lambda2 = max(lambda2, EPS);

  // Special case for matrices with repeated eigenvalues (including identity
  // matrix) In this case, any two orthogonal vectors are valid eigenvectors
  if (abs(discriminant) < EPS) {
    // For identity or scalar matrices (where eigenvalues are the same),
    // we can choose any orthogonal basis as eigenvectors
    return eigen2x2(float2(lambda1, lambda2), float2x2(1, 0, 0, 1));
  }

  float2 v1;
  if (abs(b) > EPS) {
    v1 = float2(b, lambda1 - a);
  } else if (abs(c) > EPS) {
    v1 = float2(lambda1 - d, c);
  } else {
    // If both b and c are zero, the matrix is diagonal
    // In this case, the eigenvectors are the standard basis vectors
    if (abs(a - lambda1) < EPS) {
      v1 = float2(1, 0);
    } else {
      v1 = float2(0, 1);
    }
  }

  if (length(v1) < EPS) {
    v1 = float2(1, 0);
  } else {
    v1 = normalize(v1);
  }

  float2 v2;
  if (abs(b) > EPS) {
    v2 = float2(b, lambda2 - a);
  } else if (abs(c) > EPS) {
    v2 = float2(lambda2 - d, c);
  } else {
    // If both b and c are zero, the matrix is diagonal
    if (abs(a - lambda2) < EPS) {
      v2 = float2(1, 0);
    } else {
      v2 = float2(0, 1);
    }
  }

  if (length(v2) < EPS) {
    v2 = float2(0,
                1);  // Use a different default than v1 to ensure orthogonality
  } else {
    v2 = normalize(v2);
  }

  // Always ensure orthogonality between eigenvectors
  float dot_product = v1.x * v2.x + v1.y * v2.y;
  if (abs(dot_product) > EPS) {
    // Gram-Schmidt orthogonalization
    v2 = v2 - dot_product * v1;

    // If v2 became too small after orthogonalization, choose a perpendicular
    // vector
    if (length(v2) < EPS) {
      v2 = float2(-v1.y, v1.x);  // This is guaranteed to be perpendicular to v1
    }
    v2 = normalize(v2);
  }

  // Double-check orthogonality again
  dot_product = v1.x * v2.x + v1.y * v2.y;
  if (abs(dot_product) > 0.01) {
    // If still not orthogonal enough, force it
    v2 = float2(-v1.y, v1.x);
  }

  float2x2 eigenvectors = float2x2(v1, v2);

  return eigen2x2(float2(lambda1, lambda2), eigenvectors);
}

}
