#pragma once
#include "cuda_runtime_api.h"
struct dim3 {
  unsigned x, y, z;
  constexpr dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};
