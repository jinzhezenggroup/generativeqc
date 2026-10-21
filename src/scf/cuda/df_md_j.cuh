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
__global__ void df_md_prepare_transforms(DfMdView view) {
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
}

/** Precontract the public density (or auxiliary potential) before radial work.
 * Off-diagonal shell pairs sum both orientations, preserving nonsymmetric D.
 * The matching projection below writes both J orientations without this factor.
 * This pass overwrites every Hermite density and clears the complete potential. */
__global__ void df_md_precontract(DfMdView view, const double* input) {
  const auto primitive = view.primitives[blockIdx.x];
  const auto hermites = df_md_count(primitive.first_angular + primitive.second_angular);
  for (unsigned d = threadIdx.x; d < hermites; d += blockDim.x) {
    double value = 0;
    for (unsigned i = 0; i < primitive.first_count; ++i)
      for (unsigned j = 0; j < primitive.second_count; ++j) {
        const auto first = primitive.first_ao + i, second = primitive.second_ao + j;
        double weight = view.auxiliary ? input[second] : input[first * view.nbf + second];
        if (!view.auxiliary && !primitive.same_shell) weight += input[second * view.nbf + first];
        value += weight * view.transforms[primitive.transform_offset +
                                          (i * primitive.second_count + j) * hermites + d];
      }
    view.density[primitive.hermite_offset + d] = value;
    view.potential[primitive.hermite_offset + d] = 0;
  }
}

/** Project one primitive's Hermite potential into a caller-cleared public
 * vector/matrix. Primitive overlaps require FP64 atomic accumulation, including
 * signed contractions. This is the exact transpose of the preparation map,
 * with the distinct-shell output symmetry described above. */
__global__ void df_md_project(DfMdView view, double* output) {
  const auto primitive = view.primitives[blockIdx.x];
  const auto hermites = df_md_count(primitive.first_angular + primitive.second_angular);
  for (unsigned c = threadIdx.x; c < primitive.first_count * primitive.second_count;
       c += blockDim.x) {
    double value = 0;
    for (unsigned d = 0; d < hermites; ++d)
      value += view.transforms[primitive.transform_offset + c * hermites + d] *
               view.potential[primitive.hermite_offset + d];
    const auto first = primitive.first_ao + c / primitive.second_count;
    const auto second = primitive.second_ao + c % primitive.second_count;
    atomicAdd(output + (view.auxiliary ? second : first * view.nbf + second), value);
    if (!view.auxiliary && !primitive.same_shell)
      atomicAdd(output + second * view.nbf + first, value);
  }
}

/** Eight source lanes share radial work across all target components. A block
 * exclusively owns one target primitive, so Hermite accumulation needs no
 * atomics and has a fixed source traversal. Angular source groups must be
 * launched sequentially on the owning stream. Padding lanes are initialized
 * and participate in every barrier/shuffle; no partial warp uses stale roots.
 * This complete unscreened path is a qualification seam, not a default policy. */
template <unsigned PairAngular, unsigned AuxiliaryAngular, bool Charge>
__global__ void df_md_contract(DfMdView pairs, std::size_t pair_begin, std::size_t pair_count,
                               DfMdView auxiliary, std::size_t auxiliary_begin,
                               std::size_t auxiliary_count) {
  constexpr unsigned sources = 8, components = 16, threads = sources * components;
  constexpr unsigned target_count = df_md_count(Charge ? AuxiliaryAngular : PairAngular);
  constexpr unsigned rows = (target_count + components - 1) / components;
  __shared__ double radial[sources * df_md_count(PairAngular + AuxiliaryAngular)];
  __shared__ double scale[sources];
  const unsigned lane = threadIdx.x % sources, component_lane = threadIdx.x / sources;
  const auto target_view = Charge ? auxiliary : pairs;
  const auto source_view = Charge ? pairs : auxiliary;
  const auto target = target_view.primitives[(Charge ? auxiliary_begin : pair_begin) + blockIdx.x];
  const auto begin = Charge ? pair_begin : auxiliary_begin;
  const auto count = Charge ? pair_count : auxiliary_count;
  double values[rows]{};
  for (std::size_t offset = 0; offset < count; offset += sources) {
    const bool active = offset + lane < count;
    DfMdPrimitive source{};
    if (active) source = source_view.primitives[begin + offset + lane];
    if (component_lane == 0) {
      scale[lane] = 0;
      if (active) {
        const auto pair = Charge ? source : target, aux = Charge ? target : source;
        const double sum = pair.exponent + aux.exponent;
        const double rho = pair.exponent * (aux.exponent / sum);
        df_md_radial<PairAngular + AuxiliaryAngular, sources>(rho, pair.product, aux.product,
                                                              radial + lane);
        scale[lane] = 34.986836655249725 / (pair.exponent * aux.exponent * sqrt(sum));
      }
    }
    __syncthreads();
    if (active)
      for (unsigned row = 0; row < rows; ++row) {
        const auto c = component_lane + row * components;
        if (c >= target_count) continue;
        const auto* input = source_view.density + source.hermite_offset;
        if constexpr (Charge)
          values[row] +=
              scale[lane] * df_md_auxiliary_charge<PairAngular, AuxiliaryAngular, sources>(
                                radial + lane, input, c);
        else
          values[row] += scale[lane] * df_md_pair_potential<PairAngular, AuxiliaryAngular, sources>(
                                           radial + lane, input, c);
      }
    __syncthreads();
  }
  for (unsigned row = 0; row < rows; ++row) {
    double value = values[row];
    for (unsigned stride = sources / 2; stride; stride /= 2)
      value += __shfl_down_sync(0xffffffffU, value, stride, sources);
    const auto c = component_lane + row * components;
    if (!lane && c < target_count) target_view.potential[target.hermite_offset + c] += value;
  }
  static_assert(threads == 128);  // Launch contract: four complete warps.
}

}  // namespace generativeqc::scf::cuda_execution
