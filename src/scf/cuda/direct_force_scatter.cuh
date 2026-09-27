#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace vibeqc::scf::cuda_execution {

/** Preserve first-occurrence atom order while collapsing repeated shell centers. */
__device__ __forceinline__ unsigned direct_force_unique_center_atoms(
    const std::int32_t* center_atoms, std::int32_t* unique_center_atoms) {
  unsigned unique_center_count = 0;
  for (unsigned center = 0; center < 4; ++center) {
    bool duplicate_center = false;
    for (unsigned previous = 0; previous < unique_center_count; ++previous) {
      duplicate_center = duplicate_center || center_atoms[center] == unique_center_atoms[previous];
    }
    if (!duplicate_center) {
      unique_center_atoms[unique_center_count++] = center_atoms[center];
    }
  }
  return unique_center_count;
}

/**
 * Recover the omitted fourth shell-center derivative and accumulate atom forces.
 *
 * Generated low-order force roots carry independent centers 0/1/2. Translation
 * invariance supplies center 3. Repeated shell centers are accumulated into one
 * atom contribution, and the final unique atom is recovered from the negative
 * sum so the quartet contribution remains exactly translation-balanced.
 */
template <class Gradient>
__device__ __forceinline__ void scatter_direct_force_independent_gradient(
    const std::int32_t* center_atoms, const std::int32_t* unique_center_atoms,
    unsigned unique_center_count, const Gradient& gradient, double* forces) {
  if (unique_center_count <= 1U) return;

  double derivative_sum[3]{};
  for (unsigned atom = 0; atom + 1 < unique_center_count; ++atom) {
    const std::int64_t coordinate = static_cast<std::int64_t>(unique_center_atoms[atom]) * 3;
    for (unsigned axis = 0; axis < 3; ++axis) {
      double derivative = 0.0;
      double fourth_derivative = 0.0;
      for (unsigned center = 0; center < 3; ++center) {
        const double value = gradient.center[center][axis];
        fourth_derivative -= value;
        if (center_atoms[center] == unique_center_atoms[atom]) {
          derivative += value;
        }
      }
      if (center_atoms[3] == unique_center_atoms[atom]) {
        derivative += fourth_derivative;
      }
      derivative_sum[axis] += derivative;
      if (derivative != 0.0) {
        atomicAdd(forces + coordinate + axis, -derivative);
      }
    }
  }

  const std::int64_t final_coordinate =
      static_cast<std::int64_t>(unique_center_atoms[unique_center_count - 1]) * 3;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (derivative_sum[axis] != 0.0) {
      atomicAdd(forces + final_coordinate + axis, derivative_sum[axis]);
    }
  }
}

}  // namespace vibeqc::scf::cuda_execution
