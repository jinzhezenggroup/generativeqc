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
  bool active = false;
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
    active = active || value != 0;
  }
  // This is an exact zero test on the already produced FP64 density, not a
  // tolerance or a changed integral cutoff. Keep NaNs active so a failure is
  // never hidden. Every input replay overwrites the flag on the owning stream.
  const bool any_active = __syncthreads_or(active);
  if (threadIdx.x == 0) view.primitives[blockIdx.x].density_active = any_active;
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

/** Source lanes share radial work across all target components. A block
 * exclusively owns one target primitive, so Hermite accumulation needs no
 * atomics and has a fixed source traversal. Angular source groups must be
 * launched sequentially on the owning stream. Padding lanes are initialized
 * and participate in every barrier/shuffle; no partial warp uses stale roots.
 * The prepared owner admits this unscreened path within its qualified default
 * domain; df_md_plan owns the capacity and angular fallback decisions. */
template <unsigned PairAngular, unsigned AuxiliaryAngular, bool Charge, bool Response = false>
__global__ void df_md_contract(DfMdView pairs, std::size_t pair_begin, std::size_t pair_count,
                               DfMdView auxiliary, std::size_t auxiliary_begin,
                               std::size_t auxiliary_count) {
  constexpr unsigned pair_order = PairAngular + (Response && !Charge);
  constexpr unsigned auxiliary_order = AuxiliaryAngular + (Response && Charge);
  constexpr unsigned target_count = df_md_count(Charge ? auxiliary_order : pair_order);
  // Low-order targets cannot use sixteen component lanes: for an s target,
  // fifteen lanes wait while one contracts. Give those lanes independent
  // sources instead. Every source group remains a power-of-two subwarp, and
  // the complete block stays at 128 threads, including padding components.
  // Higher targets retain the original bounded radial shared-memory layout.
  constexpr unsigned sources = target_count <= 4 ? 32 : target_count <= 10 ? 16 : 8;
  constexpr unsigned components = 128 / sources, threads = sources * components;
  constexpr unsigned rows = (target_count + components - 1) / components;
  __shared__ double radial[sources * df_md_count(pair_order + auxiliary_order)];
  __shared__ double scale[sources];
  const unsigned lane = threadIdx.x % sources, component_lane = threadIdx.x / sources;
  const auto target_view = Charge ? auxiliary : pairs;
  const auto source_view = Charge ? pairs : auxiliary;
  const auto target = target_view.primitives[(Charge ? auxiliary_begin : pair_begin) + blockIdx.x];
  const auto begin = Charge ? pair_begin : auxiliary_begin;
  const auto count = Charge ? pair_count : auxiliary_count;
  double values[rows]{};
  for (std::size_t offset = 0; offset < count; offset += sources) {
    bool active = offset + lane < count;
    DfMdPrimitive source{};
    if (active) {
      source = source_view.primitives[begin + offset + lane];
      active = source.density_active;
    }
    if (component_lane == 0) {
      scale[lane] = 0;
      if (active) {
        const auto pair = Charge ? source : target, aux = Charge ? target : source;
        const double sum = pair.exponent + aux.exponent;
        const double rho = pair.exponent * (aux.exponent / sum);
        df_md_radial<pair_order + auxiliary_order, sources>(rho, pair.product, aux.product,
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
          values[row] += scale[lane] * df_md_auxiliary_charge<pair_order, auxiliary_order, sources>(
                                           radial + lane, input, c);
        else
          values[row] += scale[lane] * df_md_pair_potential<pair_order, auxiliary_order, sources>(
                                           radial + lane, input, c);
      }
    __syncthreads();
  }
  for (unsigned row = 0; row < rows; ++row) {
    double value = values[row];
    for (unsigned stride = sources / 2; stride; stride /= 2)
      value += __shfl_down_sync(0xffffffffU, value, stride, sources);
    const auto c = component_lane + row * components;
    if (!lane && c < target_count) {
      auto* output = Response ? target_view.response_potential : target_view.potential;
      output[(Response ? target.response_offset : target.hermite_offset) + c] += value;
    }
  }
  static_assert(threads == 128);  // Launch contract: four complete warps.
}

/** Contract a raised Hermite potential directly into atom derivatives. The
 * compiler owns the center derivative algebra; this kernel owns component
 * projection and reductions. Swapping centers puts every raised power in the
 * recurrence's larger second dimension. No derivative tensor is materialized.
 * Each block adds its complete primitive contribution to the caller's existing
 * gradient, so metric and other physical terms can share the output stream. */
template <bool First>
__global__ void df_md_project_derivative(DfMdView view, const double* input, double coefficient,
                                         double* gradient) {
  const auto primitive = view.primitives[blockIdx.x];
  __shared__ HermiteCoefficients<double> coefficients[3];
  __shared__ double partials[128];
  const auto first = First ? primitive.second : primitive.first;
  const auto second = First ? primitive.first : primitive.second;
  const auto alpha = First ? primitive.beta : primitive.alpha;
  const auto beta = First ? primitive.alpha : primitive.beta;
  if (threadIdx.x == 0) {
    const double p[3]{primitive.product.x, primitive.product.y, primitive.product.z};
    const double a[3]{first.x, first.y, first.z}, b[3]{second.x, second.y, second.z};
    for (unsigned axis = 0; axis < 3; ++axis)
      fill_hermite(First ? primitive.second_angular : primitive.first_angular,
                   (First ? primitive.first_angular : primitive.second_angular) + 1, p[axis],
                   a[axis], b[axis], alpha, beta, coefficients[axis]);
  }
  __syncthreads();
  const unsigned angular = primitive.first_angular + primitive.second_angular + 1;
  const unsigned hermites = df_md_count(angular);
  const unsigned components = primitive.first_count * primitive.second_count;
  for (unsigned axis = 0; axis < 3; ++axis) {
    double value = 0;
    for (unsigned item = threadIdx.x; item < components * hermites; item += blockDim.x) {
      const auto component = item / hermites;
      const auto i = primitive.first_ao + component / primitive.second_count;
      const auto j = primitive.second_ao + component % primitive.second_count;
      const auto& original_first = view.first_aos[i];
      const auto& original_second = view.second_aos[j];
      const auto& a = First ? original_second : original_first;
      const auto& b = First ? original_first : original_second;
      const auto h = md_hermite_angular(angular, item % hermites);
      double derivative = 0;
      for (unsigned x = 0; x < a.count; ++x)
        for (unsigned y = 0; y < b.count; ++y)
          derivative += a.coefficients[x] * b.coefficients[y] *
                        df_md_second_center_transform(coefficients, a.angular[x], b.angular[y], h,
                                                      axis, beta);
      double weight = view.auxiliary ? input[j] : input[i * view.nbf + j];
      if (!view.auxiliary && !primitive.same_shell) weight += input[j * view.nbf + i];
      value += weight * derivative *
               view.response_potential[primitive.response_offset + item % hermites];
    }
    partials[threadIdx.x] = value;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
      if (threadIdx.x < stride) partials[threadIdx.x] += partials[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0)
      atomicAdd(gradient + 3 * (First ? primitive.first_atom : primitive.second_atom) + axis,
                coefficient * primitive.weight * partials[0]);
    __syncthreads();
  }
}

}  // namespace generativeqc::scf::cuda_execution
