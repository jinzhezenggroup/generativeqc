#pragma once

#include "scf/cuda/cartesian_angular.cuh"

namespace generativeqc::scf::cuda_execution {

/** Match the simplex ordering of the resident public-AO Hermite transforms. */
__device__ inline Angular md_hermite_angular(unsigned angular, unsigned index) {
  unsigned first = 0;
  while (index >= (angular - first + 1) * (angular - first + 2) / 2) {
    index -= (angular - first + 1) * (angular - first + 2) / 2;
    ++first;
  }
  unsigned second = 0;
  while (index >= angular - first - second + 1) {
    index -= angular - first - second + 1;
    ++second;
  }
  return {first, second, index};
}

}  // namespace generativeqc::scf::cuda_execution
