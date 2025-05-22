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

  float trace = a + d;
  float det = a * d - b * c;

  float discriminant = trace * trace - 4 * det;
  if (discriminant < 0) {
    return eigen2x2(float2(0, 0), float2x2(0, 0, 0, 0));
  }

  float sqrtDiscriminant = sqrt(discriminant);
  float lambda1 = (trace + sqrtDiscriminant) / 2;
  float lambda2 = (trace - sqrtDiscriminant) / 2;

  float2 v1;
  if (abs(b) > abs(a - lambda1)) {
    v1 = float2(b, lambda1 - a);
  } else {
    v1 = float2(lambda1 - d, c);
  }
  if (length(v1) < EPS) {
    v1 = float2(1, 0);
  } else {
    v1 = normalize(v1);
  }

  float2 v2;
  if (abs(b) > abs(a - lambda2)) {
    v2 = float2(b, lambda2 - a);
  } else {
    v2 = float2(lambda2 - d, c);
  }
  if (length(v2) < EPS) {
    v2 = float2(1, 0);
  } else {
    v2 = normalize(v2);
  }

  float2x2 eigenvectors = float2x2(v1, v2);

  return eigen2x2(float2(lambda1, lambda2), eigenvectors);
}

}
