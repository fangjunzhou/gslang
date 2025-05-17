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

 //Spherical harmonics.
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

  [Differentiable] public float3x3 covariance() {
    // TODO: Implement covariance matrix.
  }

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
      float y33 = 0.25 * c6 * (5 * dir.z * dir.z - 3 * dir.z);
      float y34 = 0.25 * c5 * dir.x * (5 * dir.z * dir.z - 1);
      float y35 = 0.25 * c4 * (dir.x * dir.x - dir.y * dir.y) * dir.z;
      float y36 = 0.25 * c3 * dir.x * (dir.x * dir.x - 3 * dir.y * dir.y);

      return
      y10 * this.sh10 +
             y11 * this.sh11 +
             y12 * this.sh12 +
             y20 * this.sh20 +
             y21 * this.sh21 +
             y22 * this.sh22 +
             y23 * this.sh23 +
             y24 * this.sh24 +
             y30 * this.sh30 +
             y31 * this.sh31 +
             y32 * this.sh32 +
             y33 * this.sh33 +
             y34 * this.sh34 +
             y35 * this.sh35 +
             y36 * this.sh36;
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

 public
  __init(float3 position = float3(0, 0, 0),
         float2x2 covariance = float2x2(1, 0, 0, 1),
         float3 color = float3(1, 0, 1), float opacity = 1) {
    this.position = position;
    this.covariance = covariance;

    this.color = color;
    this.opacity = opacity;
  }

  // Evaluate the Gaussian at a given point in normalized device coordinates.
  [Differentiable] public float3 eval(float2 uv) {
    // TODO: Implement Gaussian2D evaluation.
  }
};
