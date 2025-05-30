implementing "../renderer.slang";

static const float EPS_ADAMW = 1e-8f;

[shader("compute")]
[numthreads(64, 1, 1)]
void gradDescentGaussian3D(uint tid: SV_DispatchThreadID, uniform float lr,
                           uniform float beta1, uniform float beta2,
                           uniform float weightDecay,
                           uniform uint step) {
  // Gaussian3D gd = d_gaussian_3d[tid];
  // g_gaussian_3d[tid] = g_gaussian_3d[tid] - (lr * gd);
  // // Normalize the rotation quaternion.
  // g_gaussian_3d[tid].rotation = normalize(g_gaussian_3d[tid].rotation);


  Gaussian3D grad = d_gaussian_3d[tid];
  Gaussian3D m_old = m_gaussian_3d[tid]; //first moment
  Gaussian3D v_old = v_gaussian_3d[tid]; //second moment
  Gaussian3D param = g_gaussian_3d[tid];

  m_gaussian_3d[tid] = m_old * beta1 + grad * (1.0 - beta1);

  Gaussian3D grad_sq;
  grad_sq.position = grad.position * grad.position;
  grad_sq.rotation = grad.rotation * grad.rotation;
  grad_sq.color = grad.color * grad.color;
  grad_sq.opacity = grad.opacity * grad.opacity;
  v_gaussian_3d[tid] = v_old * beta2 + grad_sq * (1.0 - beta2);

  float bias_c1 = 1.0 - pow(beta1, float(step));
  float bias_c2 = 1.0 - pow(beta2, float(step));
  Gaussian3D m_hat = m_gaussian_3d[tid] / bias_c1;
  Gaussian3D v_hat = v_gaussian_3d[tid] / bias_c2;

  Gaussian3D update;
  update.position = m_hat.position / (sqrt(v_hat.position) + EPS_ADAMW);
  update.rotation = m_hat.rotation / (sqrt(v_hat.rotation) + EPS_ADAMW);
  update.color = m_hat.color / (sqrt(v_hat.color) + EPS_ADAMW);
  update.opacity = m_hat.opacity / (sqrt(v_hat.opacity) + EPS_ADAMW);

  param.position -= param.position * (lr * weightDecay);
  param.rotation -= param.rotation * (lr * weightDecay);
  param.color -= param.color * (lr * weightDecay);
  param.opacity  -= param.opacity  * (lr * weightDecay);

  param = param - update * lr;
  param.rotation = normalize(param.rotation);

  g_gaussian_3d[tid] = param;

}
