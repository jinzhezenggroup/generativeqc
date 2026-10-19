"""Emit an opt-in, bounded masked-Hermite residual Coulomb consumer.

Sorting *oriented* AO pairs makes each original FP64 Schwarz predicate a
suffix, without assuming symmetric bounds or density. Canonical AO pairs own
the E coefficients; directed consumers replace the eightfold ERI scatter.
Native SCF retains admission, policy, storage and the incumbent fallback.
"""

from __future__ import annotations

from .md_j_masked_packed_cuda import emit_md_j_masked_packed
from .md_j_masked_tile_cuda import emit_md_j_masked_tile

_SOURCE = r"""
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define GQC_MASKED_HD __host__ __device__
#else
#define GQC_MASKED_HD
#endif

namespace generativeqc::scf::cuda_execution {

/** An oriented density/output entry uses its canonical AO pair's transform.
 * Reverse AO bounds and alpha/beta density entries remain independent. */
struct MdMaskedEntry {
  unsigned component, first, second;
  double bound, weight;
};

template <unsigned Capacity>
struct MdMaskedPlan {
  static constexpr unsigned capacity = Capacity;
  MdMaskedEntry entries[Capacity];
  unsigned count{}, groups{}, group[Capacity]{}, cutoff[Capacity]{};
};

/** Geometry-bound sorted directions; density weights are refreshed per replay.
 * A sentinel offset excludes unsupported pairs before any cache consumption. */
struct MdMaskedResidentPair {
  std::size_t suffix_offset{static_cast<std::size_t>(-1)};
  unsigned count{};
  bool valid{};
  std::uint8_t directions[18]{};
};

/** Reject invalid bounds rather than extending the suffix proof to NaNs,
 * infinities or negative values. Stable sorting also fixes equal-bound order. */
template <class Plan>
GQC_MASKED_HD bool md_masked_sort(Plan& plan) {
  for (unsigned index = 0; index < plan.count; ++index) {
    if (!std::isfinite(plan.entries[index].bound) || plan.entries[index].bound < 0.0)
      return false;
    const auto entry = plan.entries[index];
    unsigned position = index;
    while (position && plan.entries[position - 1].bound > entry.bound) {
      plan.entries[position] = plan.entries[position - 1];
      --position;
    }
    plan.entries[position] = entry;
  }
  return true;
}

/** Expand canonical AO pairs into independent directed entries. The generic
 * pair descriptor keeps this mathematical map executable in host oracle tests. */
template <class Plan, class Pair>
GQC_MASKED_HD bool md_masked_pair(Plan& plan, const Pair& pair,
                                 std::size_t nbf, std::size_t offset, const double* bounds,
                                 const double* density, const double* beta, bool unrestricted) {
  plan.count = 0;
  for (unsigned component = 0;
       component < static_cast<unsigned>(pair.first_count * pair.second_count); ++component) {
    const unsigned first = pair.first_ao + component / pair.second_count;
    const unsigned second = pair.second_ao + component % pair.second_count;
    if (first < second) continue;
    for (unsigned reverse = 0; reverse < (first == second ? 1U : 2U); ++reverse) {
      if (plan.count == plan.capacity) return false;
      const auto row = reverse ? second : first, column = reverse ? first : second;
      const auto index = offset + row * nbf + column;
      plan.entries[plan.count++] = {
          component, row, column, bounds[index],
          density ? density[index] + (unrestricted ? beta[index] : 0.0) : 0.0};
    }
  }
  return md_masked_sort(plan);
}

/** Copy only immutable directed entries, never another task's cutoff groups. */
template <unsigned Capacity, class Pair>
GQC_MASKED_HD bool md_masked_cached_pair(MdMaskedPlan<Capacity>& output,
                                        const MdMaskedResidentPair& resident, const Pair& pair,
                                        std::size_t nbf, std::size_t offset, const double* bounds) {
  if (!resident.valid || resident.count > Capacity) return false;
  output.count = resident.count;
  for (unsigned index = 0; index < output.count; ++index) {
    const unsigned code = resident.directions[index], component = code / 2;
    const unsigned first = pair.first_ao + component / pair.second_count;
    const unsigned second = pair.second_ao + component % pair.second_count;
    const unsigned row = code & 1U ? second : first, column = code & 1U ? first : second;
    output.entries[index] = {component, row, column, bounds[offset + row * nbf + column], 0.0};
  }
  return true;
}

/** Cache only the stable direction permutation, not duplicated bounds/weights. */
template <class Pair>
GQC_MASKED_HD void md_masked_prepare_resident(MdMaskedResidentPair& resident, const Pair& pair,
                                              std::size_t nbf, std::size_t offset,
                                              const double* bounds) {
  MdMaskedPlan<18> plan;
  resident.valid = md_masked_pair(plan, pair, nbf, offset, bounds, nullptr, nullptr, false);
  resident.count = resident.valid ? plan.count : 0;
  for (unsigned index = 0; index < resident.count; ++index) {
    const auto& entry = plan.entries[index];
    const auto canonical_first = pair.first_ao + entry.component / pair.second_count;
    resident.directions[index] = 2 * entry.component + (entry.first != canonical_first);
  }
}

/** Refresh in sorted order without changing geometry-bound masks or E indices. */
template <unsigned Capacity>
GQC_MASKED_HD void md_masked_weights(MdMaskedPlan<Capacity>& plan, std::size_t nbf,
                                    std::size_t offset, const double* density,
                                    const double* beta, bool unrestricted) {
  for (unsigned index = 0; index < plan.count; ++index) {
    auto& entry = plan.entries[index];
    const auto location = offset + entry.first * nbf + entry.second;
    entry.weight = density[location] + (unrestricted ? beta[location] : 0.0);
  }
}

/** Search with the incumbent multiplication/comparison, never epsilon / B.
 * This preserves equality, zero thresholds, underflow and overflow decisions.
 * Empty masks use a sentinel and never acquire a Hermite projection. */
template <class Output, class Input>
GQC_MASKED_HD void md_masked_cutoffs(Output& output, const Input& input,
                                    double screening) {
  output.groups = 0;
  for (unsigned index = 0; index < output.count; ++index) {
    unsigned begin = 0, end = input.count;
    while (begin < end) {
      const unsigned middle = begin + (end - begin) / 2;
      if (output.entries[index].bound * input.entries[middle].bound < screening)
        begin = middle + 1;
      else
        end = middle;
    }
    unsigned group = 0;
    if (begin == input.count) {
      output.group[index] = Output::capacity;
      continue;
    }
    while (group < output.groups && output.cutoff[group] != begin) ++group;
    if (group == output.groups) output.cutoff[output.groups++] = begin;
    output.group[index] = group;
  }
}

/** Dense host fixtures and native compact transforms share the same algebra. */
template <unsigned Hermites>
GQC_MASKED_HD double md_masked_coefficient(const double* transform, unsigned component,
                                          unsigned derivative) {
  return transform[component * Hermites + derivative];
}

template <unsigned Hermites, class Transform>
GQC_MASKED_HD double md_masked_coefficient(Transform transform, unsigned component,
                                          unsigned derivative) {
  return transform(component, derivative);
}

/** One thread per Hermite derivative builds all suffixes for one primitive.
 * No subtraction of large prefix sums is needed for a small retained tail. */
template <unsigned Hermites, class Input, class Transform>
GQC_MASKED_HD void md_masked_suffix(const Input& input,
                                   Transform transform, unsigned derivative,
                                   double* suffix) {
  double value = 0.0;
  suffix[input.count * Hermites + derivative] = value;
  for (unsigned index = input.count; index > 0; --index) {
    const auto entry = input.entries[index - 1];
    value += entry.weight * md_masked_coefficient<Hermites>(transform, entry.component, derivative);
    suffix[(index - 1) * Hermites + derivative] = value;
  }
}

/** Grouped density/R contraction shared by native CUDA and compiled host gates.
 * Radial supplies the existing R_(t+u) with the canonical ket parity sign. */
template <unsigned OutputHermites, unsigned InputHermites, class Output, class Radial>
GQC_MASKED_HD double md_masked_project(const Output& output,
                                      const double* suffix, unsigned element, Radial radial) {
  const unsigned group = element / OutputHermites, derivative = element % OutputHermites;
  double value = 0.0;
  for (unsigned input = 0; input < InputHermites; ++input)
    value += radial(derivative, input) * suffix[output.cutoff[group] * InputHermites + input];
  return value;
}

/** Read one requested tail without constructing every unused suffix. Preserve
 * the reverse accumulation order and avoid cancellation from prefix subtraction. */
template <unsigned Hermites, class Input, class Transform>
GQC_MASKED_HD double md_masked_tail(const Input& input, Transform transform,
                                   unsigned derivative, unsigned cutoff) {
  double value = 0.0;
  for (unsigned index = input.count; index > cutoff; --index) {
    const auto entry = input.entries[index - 1];
    value += entry.weight * md_masked_coefficient<Hermites>(transform, entry.component, derivative);
  }
  return value;
}

template <unsigned Hermites, class Output, class Transform>
GQC_MASKED_HD double md_masked_output(const Output& output,
                                     Transform transform, const double* projected,
                                     unsigned index) {
  if (output.group[index] == Output::capacity) return 0.0;
  double value = 0.0;
  for (unsigned derivative = 0; derivative < Hermites; ++derivative)
    value += md_masked_coefficient<Hermites>(transform, output.entries[index].component, derivative) *
             projected[output.group[index] * Hermites + derivative];
  return value;
}

/** Bound the grouped radial work and number of scratch projections. Setup
 * cost is deliberately not a speed claim; promotion requires endpoint timing. */
template <unsigned BraHermites, unsigned KetHermites>
GQC_MASKED_HD bool md_masked_work_admitted(unsigned groups, unsigned bra_components,
                                          unsigned ket_components, unsigned maximum_groups) {
  const unsigned incumbent = bra_components * KetHermites * (BraHermites + ket_components);
  return groups <= maximum_groups && groups * BraHermites * KetHermites < incumbent;
}

}  // namespace generativeqc::scf::cuda_execution

#ifdef __CUDACC__
#include "scf/cuda/coulomb_auxiliary.cuh"
#include "scf/cuda/direct_md_j.hpp"
#include "scf/cuda/md_hermite_index.cuh"
#include "scf/cuda/md_source_tiles.cuh"

namespace generativeqc::scf::cuda_execution {

/** Native owns dense/compact E layout; the consumer never assumes a stride. */
struct MdMaskedTransform {
  MdJView md;
  const MdJPrimitive& primitive;
  const MdJPair& pair;
  __device__ double operator()(unsigned component, unsigned derivative) const {
    return md_j_transform(md, primitive, pair, component, derivative);
  }
};

/** Prepared once after the immutable Schwarz bounds and transforms exist. */
static __global__ void md_masked_prepare_pairs(DeviceBatch batch, MdJView md, const double* bounds) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= md.pair_count) return;
  auto& resident = md.masked_pairs[index];
  if (resident.suffix_offset == static_cast<std::size_t>(-1)) return;
  const auto pair = md.pairs[index];
  const auto offset = static_cast<std::size_t>(pair.system) * batch.nbf * batch.nbf;
  md_masked_prepare_resident(resident, pair, batch.nbf, offset, bounds);
}

/** One shared directed plan feeds all primitives of an input pair. Every replay
 * overwrites all suffixes before the residual consumer runs on the same stream. */
static __global__ void md_masked_refresh(DeviceBatch batch, MdJView md, std::size_t system_begin,
                                  std::size_t system_end, bool unrestricted, double screening,
                                  const double* bounds, const double* density, const double* beta) {
  const auto pair_index = static_cast<std::size_t>(blockIdx.x);
  const auto pair = md.pairs[pair_index];
  const auto& resident = md.masked_pairs[pair_index];
  if (!resident.valid || pair.system < system_begin || pair.system >= system_end ||
      md.maximum_bounds[pair_index] * *md.maximum_bound < screening) return;
  __shared__ __align__(8) unsigned char plan_storage[sizeof(MdMaskedPlan<18>)];
  auto& plan = *reinterpret_cast<MdMaskedPlan<18>*>(plan_storage);
  if (threadIdx.x == 0) {
    md_masked_cached_pair(plan, resident, pair, batch.nbf,
        static_cast<std::size_t>(pair.system) * batch.nbf * batch.nbf, bounds);
    md_masked_weights(plan, batch.nbf,
        static_cast<std::size_t>(pair.system) * batch.nbf * batch.nbf,
        density, beta, unrestricted);
    if (md.work_counts) {
      const auto primitives = pair.primitive_end - pair.primitive_begin;
      atomicAdd(&md.work_counts->preproject_primitives[pair.angular],
                static_cast<unsigned long long>(primitives));
      atomicAdd(&md.work_counts->preproject_summands[pair.angular],
                static_cast<unsigned long long>(primitives) * plan.count * pair.hermites);
    }
  }
  __syncthreads();
  const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const auto stride = (2 * pair.first_count * pair.second_count + 1) * pair.hermites;
  for (auto primitive = pair.primitive_begin + warp; primitive < pair.primitive_end;
       primitive += blockDim.x / 32) {
    auto* suffix = md.masked_suffixes + resident.suffix_offset +
                   (primitive - pair.primitive_begin) * stride;
    const auto source = md.primitives[primitive];
    if (pair.angular == 1 && lane < 4)
      md_masked_suffix<4>(plan, MdMaskedTransform{md, source, pair}, lane, suffix);
    if (pair.angular == 2 && lane < 10)
      md_masked_suffix<10>(plan, MdMaskedTransform{md, source, pair}, lane, suffix);
  }
}

__device__ inline const double* md_masked_resident_suffix(MdJView md, const MdJPair& pair,
                                                   unsigned pair_index, unsigned primitive) {
  const auto stride = (2 * pair.first_count * pair.second_count + 1) * pair.hermites;
  return md.masked_suffixes + md.masked_pairs[pair_index].suffix_offset +
         (primitive - pair.primitive_begin) * stride;
}

/** Fixed s/p/d-pair scratch, independent of molecular AO/quartet counts. */
template <unsigned BraAngular, unsigned KetAngular, bool Preprojected = false>
struct MdMaskedWorkspace {
  static_assert(BraAngular == 2 && (KetAngular == 1 || KetAngular == 2));
  static constexpr unsigned bra_capacity = 18, ket_capacity = KetAngular == 1 ? 6 : 18;
  static constexpr unsigned bra_hermites = md_j_hermite_count(BraAngular);
  static constexpr unsigned ket_hermites = md_j_hermite_count(KetAngular);
  MdMaskedPlan<bra_capacity> bra;
  MdMaskedPlan<ket_capacity> ket;
  bool admitted;
  CoulombAuxiliary<double, BraAngular + KetAngular> auxiliary;
  double factor;
  double bra_suffix[Preprojected ? 1 : (bra_capacity + 1) * bra_hermites];
  double ket_suffix[Preprojected ? 1 : (ket_capacity + 1) * ket_hermites];
  double bra_projected[bra_capacity * bra_hermites];
  double ket_projected[ket_capacity * ket_hermites];
  double bra_values[bra_capacity], ket_values[ket_capacity];
};

/** Reversing the directed consumer transposes R/E, not the canonical ket
 * parity. This is the same reciprocal identity used by uniform MD-J. */
template <unsigned BraAngular, unsigned KetAngular, bool Reverse>
struct MdMaskedRadial {
  CoulombAuxiliary<double, BraAngular + KetAngular>& auxiliary;
  __device__ double operator()(unsigned output, unsigned input) const {
    const auto bra = md_hermite_angular(BraAngular, Reverse ? input : output);
    const auto ket = md_hermite_angular(KetAngular, Reverse ? output : input);
    const double sign = (ket.x + ket.y + ket.z) & 1U ? -1.0 : 1.0;
    return sign * auxiliary.at(0, bra.x + ket.x, bra.y + ket.y, bra.z + ket.z);
  }
};

/** A warp handles two independent directed AO consumers. For a diagonal shell
 * task only the forward consumer is needed: it already covers every ordered
 * output/input pair, including identical pairs exactly once. This replaces,
 * rather than approximates, canonical-quartet deduplication and orbit scatter.
 * Reject complex masks before publishing anything; the incumbent owns misses. */
template <unsigned BraAngular, unsigned KetAngular, bool CountWork, bool Preprojected = false>
__device__ bool md_masked_contract_source(DeviceBatch batch, MdJView md, MdSourceTask task,
                                          bool unrestricted, double screening,
                                          const double* bounds, const double* density,
                                          const double* beta, double* coulomb, double* workspace) {
  using Scratch = MdMaskedWorkspace<BraAngular, KetAngular, Preprojected>;
  auto& scratch = *reinterpret_cast<Scratch*>(workspace);
  constexpr auto bra_hermites = Scratch::bra_hermites, ket_hermites = Scratch::ket_hermites;
  const unsigned lane = threadIdx.x % 32;
  const auto bra_pair = md.pairs[task.bra], ket_pair = md.pairs[task.ket];
  const std::size_t offset = static_cast<std::size_t>(bra_pair.system) * batch.nbf * batch.nbf;
  const bool reverse = task.bra != task.ket;
  if (lane == 0) {
    scratch.admitted = std::isfinite(screening) && screening >= 0.0;
    if constexpr (Preprojected) {
      scratch.admitted = scratch.admitted &&
          md_masked_cached_pair(scratch.bra, md.masked_pairs[task.bra], bra_pair,
                                  batch.nbf, offset, bounds) &&
          md_masked_cached_pair(scratch.ket, md.masked_pairs[task.ket], ket_pair,
                                  batch.nbf, offset, bounds);
    } else {
      scratch.admitted = scratch.admitted &&
          md_masked_pair(scratch.bra, bra_pair, batch.nbf, offset, bounds, density, beta, unrestricted) &&
          md_masked_pair(scratch.ket, ket_pair, batch.nbf, offset, bounds, density, beta, unrestricted);
    }
    if (scratch.admitted) {
      md_masked_cutoffs(scratch.bra, scratch.ket, screening);
      md_masked_cutoffs(scratch.ket, scratch.bra, screening);
      const auto groups = scratch.bra.groups + (reverse ? scratch.ket.groups : 0);
      const unsigned bra_components = bra_pair.first_count * bra_pair.second_count;
      const unsigned ket_components = ket_pair.first_count * ket_pair.second_count;
      scratch.admitted = md_masked_work_admitted<bra_hermites, ket_hermites>(
          groups, bra_components, ket_components, KetAngular == 1 ? 4U : 8U);
      if constexpr (CountWork) {
        if (scratch.admitted) {
          constexpr unsigned angular = 5 * BraAngular + KetAngular;
          const auto products = static_cast<unsigned long long>(
              bra_pair.primitive_end - bra_pair.primitive_begin) *
              (ket_pair.primitive_end - ket_pair.primitive_begin);
          atomicAdd(&md.work_counts->masked_tasks[angular], 1ULL);
          atomicAdd(&md.work_counts->masked_groups[angular], static_cast<unsigned long long>(groups));
          atomicAdd(&md.work_counts->masked_summands[angular],
                    products * groups * bra_hermites * ket_hermites);
        }
      }
    }
  }
  __syncwarp();
  const bool admitted = scratch.admitted;
  // A rejected task immediately aliases this storage as incumbent scratch.
  // Every lane must finish reading the decision before that overwrite starts.
  __syncwarp();
  if (!admitted) return false;
  if (lane < scratch.bra.count) scratch.bra_values[lane] = 0.0;
  if (lane < scratch.ket.count) scratch.ket_values[lane] = 0.0;
  for (auto bra_index = bra_pair.primitive_begin; bra_index < bra_pair.primitive_end; ++bra_index) {
    const auto bra = md.primitives[bra_index];
    const double* bra_suffix = scratch.bra_suffix;
    if constexpr (Preprojected) {
      bra_suffix = md_masked_resident_suffix(md, bra_pair, task.bra, bra_index);
    } else {
      if (reverse && lane < bra_hermites)
        md_masked_suffix<bra_hermites>(scratch.bra, MdMaskedTransform{md, bra, bra_pair},
                                      lane, scratch.bra_suffix);
    }
    for (auto ket_index = ket_pair.primitive_begin; ket_index < ket_pair.primitive_end; ++ket_index) {
      const auto ket = md.primitives[ket_index];
      if (lane == 0) {
        const double rho = bra.exponent * ket.exponent / (bra.exponent + ket.exponent);
        fill_coulomb<BraAngular + KetAngular>(rho, bra.product, ket.product, scratch.auxiliary);
        scratch.factor = 2.0 * pow(kPi, 2.5) /
            (bra.exponent * ket.exponent * sqrt(bra.exponent + ket.exponent));
      }
      const double* ket_suffix = scratch.ket_suffix;
      if constexpr (Preprojected) {
        ket_suffix = md_masked_resident_suffix(md, ket_pair, task.ket, ket_index);
      } else {
        if (lane < ket_hermites)
          md_masked_suffix<ket_hermites>(scratch.ket, MdMaskedTransform{md, ket, ket_pair},
                                        lane, scratch.ket_suffix);
      }
      __syncwarp();
      for (unsigned element = lane; element < scratch.bra.groups * bra_hermites; element += 32)
        scratch.bra_projected[element] = md_masked_project<bra_hermites, ket_hermites>(
            scratch.bra, ket_suffix, element,
            MdMaskedRadial<BraAngular, KetAngular, false>{scratch.auxiliary});
      if (reverse)
        for (unsigned element = lane; element < scratch.ket.groups * ket_hermites; element += 32)
          scratch.ket_projected[element] = md_masked_project<ket_hermites, bra_hermites>(
              scratch.ket, bra_suffix, element,
              MdMaskedRadial<BraAngular, KetAngular, true>{scratch.auxiliary});
      __syncwarp();
      if (lane < scratch.bra.count)
        scratch.bra_values[lane] += scratch.factor * md_masked_output<bra_hermites>(
            scratch.bra, MdMaskedTransform{md, bra, bra_pair}, scratch.bra_projected, lane);
      if (reverse && lane < scratch.ket.count)
        scratch.ket_values[lane] += scratch.factor * md_masked_output<ket_hermites>(
            scratch.ket, MdMaskedTransform{md, ket, ket_pair}, scratch.ket_projected, lane);
      __syncwarp();
    }
  }
  if (lane < scratch.bra.count && scratch.bra.group[lane] != Scratch::bra_capacity) {
    const auto entry = scratch.bra.entries[lane];
    atomicAdd(coulomb + offset + entry.first * batch.nbf + entry.second, scratch.bra_values[lane]);
  }
  if (reverse && lane < scratch.ket.count && scratch.ket.group[lane] != Scratch::ket_capacity) {
    const auto entry = scratch.ket.entries[lane];
    atomicAdd(coulomb + offset + entry.first * batch.nbf + entry.second, scratch.ket_values[lane]);
  }
  __syncwarp();
  return true;
}

}  // namespace generativeqc::scf::cuda_execution
#endif

#undef GQC_MASKED_HD
"""


def emit_md_j_masked_header() -> str:
    """Return deterministic header bytes without importing/probing the runtime."""
    return _SOURCE.lstrip().replace(
        "\n#endif\n\n#undef GQC_MASKED_HD",
        emit_md_j_masked_tile()
        + emit_md_j_masked_packed()
        + "\n#endif\n\n#undef GQC_MASKED_HD",
    )
