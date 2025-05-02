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
