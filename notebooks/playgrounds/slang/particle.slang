import sgl.device.print;

struct Particle {
  float3 position;
  float3 velocity;

  __init(float3 position = float3(0, 0, 0), float3 velocity = float3(0, 0, 0)) {
    this.position = position;
    this.velocity = velocity;
  }

  void info(int id) { print("Particle {}", id); }
};
