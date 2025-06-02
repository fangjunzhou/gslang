module bounding_box;

public struct BoundingBox2D {
  public float2 min;
  public float2 max;
  
  public __init(float2 min = float2(0, 0), float2 max = float2(1, 1)) {
    this.min = min;
    this.max = max;
  }
};
