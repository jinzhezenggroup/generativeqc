#pragma once

#include <cuda_runtime_api.h>

namespace generativeqc::integrals {

/** Only partial local incidence produces a nonzero frozen coordinate
 * derivative. A simultaneous translation of all mathematical centers
 * vanishes, even when several incidences name the same physical atom. */
inline __host__ __device__ bool frozen_one_electron_active(const long long* atoms, unsigned centers,
                                                           long long atom) {
  if (atom < 0) return true;
  unsigned moved = 0;
  for (unsigned center = 0; center < centers; ++center) moved += atoms[center] == atom;
  return moved != 0 && moved != centers;
}

}  // namespace generativeqc::integrals
