struct Gaussian : IDifferentiable {
  float rotation;
  float2 scale;

  __init() {
    this.rotation = 0;
    this.scale = float2(1, 1);
  }

  [Differentiable] float2x2 covariance() {
    float2x2 rotationMatrix =
        float2x2(cos(rotation), -sin(rotation), sin(rotation), cos(rotation));
    float2x2 scaleMatrix = float2x2(scale.x, 0, 0, scale.y);
    return rotationMatrix * scaleMatrix * transpose(scaleMatrix) *
           transpose(rotationMatrix);
  }
};

[Differentiable] float2x2 covarianceWrapper(Gaussian g) {
  return g.covariance();
}

StructuredBuffer<Gaussian> gaussians;
RWStructuredBuffer<Gaussian.Differential> gaussianDiffs;
RWStructuredBuffer<float2x2> covariance;
StructuredBuffer<float2x2> covarianceDiff;

[shader("compute")][numthreads(4, 1, 1)] void computeCov(
    uint3 tid : SV_DispatchThreadID) {
  uint index = tid.x;
  float2x2 cov = covarianceWrapper(gaussians[index]);
  covariance[index] = cov;
};

[shader("compute")][numthreads(4, 1, 1)] void computeCovDiff(
    uint3 tid : SV_DispatchThreadID) {
  uint index = tid.x;
  DifferentialPair<Gaussian> diffPair = diffPair(gaussians[index]);
  float2x2 covDiff = covarianceDiff[index];
  bwd_diff(covarianceWrapper)(diffPair, covDiff);
  gaussianDiffs[index] = diffPair.d;
};
