#pragma once

#include "generated_df_md_j.cuh"
#include "scf/cuda/df_md_j.hpp"
#include "scf/cuda/hermite_recurrence.cuh"
#include "scf/cuda/md_hermite_index.cuh"

namespace generativeqc::scf::cuda_execution {

/** Preparation-only reuse of the existing Cartesian Hermite recurrence. Its
 * first dimension admits f and its second dimension admits g; auxiliary g is
 * intentionally placed in the second slot. Shared E avoids recomputing each
 * axis for every public component. No CPU integral oracle participates. */
static __global__ void df_md_prepare_transforms(DfMdView view) {
  auto& primitive = view.primitives[blockIdx.x];
  __shared__ HermiteCoefficients<double> coefficients[3];
  if (threadIdx.x == 0) {
    primitive.exponent = primitive.alpha + primitive.beta;
    // Relative shifts avoid cancellation from adding weighted absolute centers.
    const double fraction = primitive.beta / primitive.exponent;
    primitive.product = {primitive.first.x + fraction * (primitive.second.x - primitive.first.x),
                         primitive.first.y + fraction * (primitive.second.y - primitive.first.y),
                         primitive.first.z + fraction * (primitive.second.z - primitive.first.z)};
    const double p[3]{primitive.product.x, primitive.product.y, primitive.product.z};
    const double a[3]{primitive.first.x, primitive.first.y, primitive.first.z};
    const double b[3]{primitive.second.x, primitive.second.y, primitive.second.z};
    for (unsigned axis = 0; axis < 3; ++axis)
      fill_hermite(primitive.first_angular, primitive.second_angular, p[axis], a[axis], b[axis],
                   primitive.alpha, primitive.beta, coefficients[axis]);
  }
  __syncthreads();
  const unsigned angular = primitive.first_angular + primitive.second_angular;
  const unsigned hermites = df_md_count(angular);
  const unsigned components = primitive.first_count * primitive.second_count;
  for (unsigned item = threadIdx.x; item < components * hermites; item += blockDim.x) {
    const auto component = item / hermites;
    const auto& first = view.first_aos[primitive.first_ao + component / primitive.second_count];
    const auto& second = view.second_aos[primitive.second_ao + component % primitive.second_count];
    const auto d = md_hermite_angular(angular, item % hermites);
    double value = 0;
    for (unsigned i = 0; i < first.count; ++i)
      for (unsigned j = 0; j < second.count; ++j) {
        const auto a = first.angular[i], b = second.angular[j];
        if (d.x > a.x + b.x || d.y > a.y + b.y || d.z > a.z + b.z) continue;
        value += first.coefficients[i] * second.coefficients[j] *
                 coefficients[0].at(a.x, b.x, d.x) * coefficients[1].at(a.y, b.y, d.y) *
                 coefficients[2].at(a.z, b.z, d.z);
      }
    view.transforms[primitive.transform_offset + item] = primitive.weight * value;
  }
  __syncthreads();
  if (view.prepare_norms && threadIdx.x == 0) {
    double maximum = 0;
    for (unsigned component = 0; component < components; ++component) {
      double norm = 0;
      for (unsigned d = 0; d < hermites; ++d) {
        const auto e = fabs(view.transforms[primitive.transform_offset + component * hermites + d]);
        if (e != 0)
          norm = __dadd_ru(norm, __dmul_ru(e, df_md_hermite_norm(primitive.exponent, angular, d)));
      }
      maximum = !isfinite(norm) ? INFINITY : fmax(maximum, norm);
    }
    primitive.component_norm_bound = maximum;
  }
}

}  // namespace generativeqc::scf::cuda_execution
