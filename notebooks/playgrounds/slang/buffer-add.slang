import "slangpy";

struct Pixel {
  float r;
  float g;
  float b;
};

Pixel add(Pixel a, Pixel b) {
  Pixel res;
  res.r = a.r + b.r;
  res.g = a.g + b.g;
  res.b = a.b + b.b;
  return res;
}

struct ColorU8 {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;
};

ColorU8 toColor(Pixel p) {
  ColorU8 res;
  res.r = uint8_t(clamp(p.r, 0, 1) * 255);
  res.g = uint8_t(clamp(p.g, 0, 1) * 255);
  res.b = uint8_t(clamp(p.b, 0, 1) * 255);
  res.a = 255;
  return res;
}
