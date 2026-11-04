#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <weighted_eri.cuh>

#include "integrals/range_moments.hpp"
#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_force_density.cuh"
#include "scf/cuda/direct_force_profile.cuh"
#include "scf/cuda/direct_force_scatter.cuh"
#include "scf/cuda/direct_force_sources.hpp"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"

namespace generativeqc::scf::cuda_execution {

// A constant keeps the failure marker usable by both NVCC and CuMetal without
// invoking a host-only nan()/numeric_limits function from a device function.
inline constexpr double kInvalidDirectForceMoment = std::numeric_limits<double>::quiet_NaN();
inline constexpr double kMaximumSeparableForceComponent = std::numeric_limits<double>::max() / 16.0;

/** Prove the exact incumbent AO Schwarz mask without enumerating its product.
 * FP64 multiplication is monotone for finite nonnegative factors, including
 * underflow and overflow. Distinct shell pairs enumerate a Cartesian product;
 * the same-pair triangle contains its minimum-factor diagonal as well. Invalid
 * factors use the original comparisons, preserving negative/NaN/Inf behavior.
 * This changes screening work, never its tolerance or admitted AO domain. */
__device__ inline bool direct_force_complete_schwarz_mask(
    const DeviceBatch& batch, std::size_t first_pair, std::size_t second_pair,
    const DirectShellAoQuartetLayout& layout, std::size_t system_ao_begin,
    std::size_t physical_offset, double screening_tolerance, const double* schwarz_bounds,
    ScalarSeparableDirectForceWork* work) {
  if (layout.quartet_count == 0U) {
    if (work) atomicAdd(&work->screening_pair_proofs, 1ULL);
    return true;
  }
  const std::size_t dimension = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t pairs[2] = {first_pair, second_pair};
  double minimum[2]{};
  bool valid_minimum = true;
  for (unsigned side = 0; side < 2U && valid_minimum; ++side) {
    const auto count = shell_ao_pair_count(batch, pairs[side]);
    for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
      std::size_t first{}, second{};
      decode_shell_ao_pair(batch, pairs[side], ordinal, system_ao_begin, first, second);
      const double bound = schwarz_bounds[physical_offset + matrix_index(first, second, dimension)];
      if (work) atomicAdd(&work->screening_pair_terms, 1ULL);
      if (!isfinite(bound) || bound < 0.0) {
        valid_minimum = false;
        break;
      }
      if (ordinal == 0U || bound < minimum[side]) minimum[side] = bound;
    }
  }
  if (valid_minimum) {
    if (work) atomicAdd(&work->screening_pair_proofs, 1ULL);
    return !(minimum[0] * minimum[1] < screening_tolerance);
  }
  if (work) atomicAdd(&work->screening_pair_fallbacks, 1ULL);
  for (std::size_t ordinal = 0; ordinal < layout.quartet_count; ++ordinal) {
    std::size_t raw_ao[4];
    decode_shell_ao_quartet(batch, first_pair, second_pair, layout, ordinal, system_ao_begin,
                            raw_ao);
    if (work) atomicAdd(&work->checked_ao_quartets, 1ULL);
    if (schwarz_bounds[physical_offset + matrix_index(raw_ao[0], raw_ao[1], dimension)] *
            schwarz_bounds[physical_offset + matrix_index(raw_ao[2], raw_ao[3], dimension)] <
        screening_tolerance)
      return false;
  }
  return true;
}

/** Bind one low-order class to radial-moment-parametric compiler force roots. */
template <unsigned ShellClass>
struct LowOrderSourceRoots {
  static constexpr unsigned angular_order = direct_shell_class_angular_order(ShellClass);
  static_assert(angular_order <= 3U);
  static constexpr unsigned component_count =
      ShellClass == kSsssShellClass                                    ? 1U
      : ShellClass == kPsssShellClass                                  ? 3U
      : ShellClass == kDsssShellClass                                  ? 6U
      : ShellClass == kFsssShellClass                                  ? 10U
      : ShellClass == kPppsShellClass                                  ? 27U
      : ShellClass == kDspsShellClass || ShellClass == kDpssShellClass ? 18U
                                                                       : 9U;

  __device__ static __forceinline__ generated_weighted_eri::IndependentGradientF32 evaluate_mixed(
      const generated_weighted_eri::Geometry& geometry, const double* weights) {
    if constexpr (ShellClass == kSsssShellClass)
      return generated_weighted_eri::ssss_force_fp32(geometry, weights);
    else if constexpr (ShellClass == kPsssShellClass)
      return generated_weighted_eri::psss_force_fp32(geometry, weights);
    else if constexpr (ShellClass == kPspsShellClass)
      return generated_weighted_eri::psps_force_fp32(geometry, weights);
    else if constexpr (ShellClass == kPpssShellClass)
      return generated_weighted_eri::ppss_force_fp32(geometry, weights);
    else if constexpr (ShellClass == kDsssShellClass)
      return generated_weighted_eri::dsss_force_fp32(geometry, weights);
    else if constexpr (ShellClass == kPppsShellClass)
      return generated_weighted_eri::ppps_force_fp32(geometry, weights);
    else if constexpr (ShellClass == kDspsShellClass)
      return generated_weighted_eri::dsps_force_fp32(geometry, weights);
    else if constexpr (ShellClass == kDpssShellClass)
      return generated_weighted_eri::dpss_force_fp32(geometry, weights);
    else if constexpr (ShellClass == kFsssShellClass)
      return generated_weighted_eri::fsss_force_fp32(geometry, weights);
  }

  __device__ static __forceinline__ generated_weighted_eri::IndependentGradient evaluate(
      const generated_weighted_eri::Geometry& geometry, const double* weights) {
    if constexpr (ShellClass == kSsssShellClass) {
      return generated_weighted_eri::ssss_force(geometry, weights);
    } else if constexpr (ShellClass == kPsssShellClass) {
      return generated_weighted_eri::psss_force(geometry, weights);
    } else if constexpr (ShellClass == kPspsShellClass) {
      return generated_weighted_eri::psps_force(geometry, weights);
    } else if constexpr (ShellClass == kPpssShellClass) {
      return generated_weighted_eri::ppss_force(geometry, weights);
    } else if constexpr (ShellClass == kDsssShellClass) {
      return generated_weighted_eri::dsss_force(geometry, weights);
    } else if constexpr (ShellClass == kPppsShellClass) {
      return generated_weighted_eri::ppps_force(geometry, weights);
    } else if constexpr (ShellClass == kDspsShellClass) {
      return generated_weighted_eri::dsps_force(geometry, weights);
    } else if constexpr (ShellClass == kDpssShellClass) {
      return generated_weighted_eri::dpss_force(geometry, weights);
    } else {
      static_assert(ShellClass == kFsssShellClass);
      return generated_weighted_eri::fsss_force(geometry, weights);
    }
  }

  /** The scalar ssss root is already smaller than a relative-frame rewrite. */
  __device__ static __forceinline__ generated_weighted_eri::IndependentGradient evaluate_relative(
      const generated_weighted_eri::Geometry& geometry, const double* weights) {
    if constexpr (ShellClass == kSsssShellClass)
      return evaluate(geometry, weights);
    else if constexpr (ShellClass == kPsssShellClass)
      return generated_weighted_eri::psss_force_relative(geometry, weights);
    else if constexpr (ShellClass == kPspsShellClass)
      return generated_weighted_eri::psps_force_relative(geometry, weights);
    else if constexpr (ShellClass == kPpssShellClass)
      return generated_weighted_eri::ppss_force_relative(geometry, weights);
    else if constexpr (ShellClass == kDsssShellClass)
      return generated_weighted_eri::dsss_force_relative(geometry, weights);
    else if constexpr (ShellClass == kPppsShellClass)
      return generated_weighted_eri::ppps_force_relative(geometry, weights);
    else if constexpr (ShellClass == kDspsShellClass)
      return generated_weighted_eri::dsps_force_relative(geometry, weights);
    else if constexpr (ShellClass == kDpssShellClass)
      return generated_weighted_eri::dpss_force_relative(geometry, weights);
    else {
      static_assert(ShellClass == kFsssShellClass);
      return generated_weighted_eri::fsss_force_relative(geometry, weights);
    }
  }
};

/** The ordinary instantiation retains no pair-array stack frame. */
template <class Roots, bool Enabled>
struct ScalarDirectForcePairInputs {};

template <class Roots>
struct ScalarDirectForcePairInputs<Roots, true> {
  double bra[Roots::bra_pair_components]{};
  double ket[Roots::ket_pair_components]{};
};

/**
 * Consume one screened shell task into full-range J/K or one LR exchange force.
 *
 * Only immutable shell geometry, AO traversal and radial moments are shared. Each
 * source retains its own density weights, generated force evaluation and
 * primitive reduction order. Output mode changes only the coefficient/output
 * layout; Combined and Separate consume the same shell and primitive traversal.
 * Optional resident storage contains the canonical bra's complete primitive
 * domain. A missing or mismatched lease falls back to the global pair cache.
 *
 * LR uses the same generated roots: with rho and omega fixed under nuclear
 * displacement, its radial moments satisfy dM_n/dT = -M_(n+1), just as Boys
 * values do. One moment ladder therefore supplies every AO component and
 * independent center derivative, instead of repeating a Dual3 recurrence for
 * each AO quartet and atom. Higher angular orders keep their existing owner.
 */
// The default root family preserves the qualified order-zero-through-three
// specialization. Other callers may bind compiler-owned external-weight roots
// while retaining exactly this density, orientation and scatter contract.
template <bool Unrestricted, unsigned ShellClass, bool LongRange = false,
          class Roots = LowOrderSourceRoots<ShellClass>,
          DirectForceOutputMode Mode = DirectForceOutputMode::Separate, bool ResidentBra = false,
          bool Mixed = false, bool RelativeResponse = false, bool PairSeparable = false>
__device__ inline __noinline__ bool contract_two_electron_force_low_order_sources_impl(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    double coulomb_coefficient, double exchange_coefficient, double omega = 0.0,
    const PrimitivePairData* resident_first_pairs = nullptr,
    std::int64_t resident_first_pair_count = 0) {
  static_assert(!Mixed || !RelativeResponse);
  static_assert(!PairSeparable || (!Mixed && !RelativeResponse && !LongRange));
  using Sources = DirectForceSources<Mode>;
  constexpr unsigned source_count = LongRange ? 1U : Sources::count;
  if (task.tile != 0U) return true;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return true;
  if ((LongRange || coulomb_coefficient == 0.0) && exchange_coefficient == 0.0) return true;

  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair], batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair], batch.shell_pair_second[second_pair]};
  unsigned canonical_raw_slot[4];
  generated_weighted_eri::canonicalize_direct_shell_slots(batch.shell_angular, raw_shell,
                                                          canonical_raw_slot);
  std::int32_t canonical_shell[4], center_atoms[4], unique_center_atoms[4];
  std::int32_t unique_order_atoms[4];
  std::size_t component_begin[4];
  unsigned component_count[4];
  const std::size_t dimension = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = dimension * dimension;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * dimension;
#pragma unroll
  for (unsigned center = 0; center < 4; ++center) {
    canonical_shell[center] = raw_shell[canonical_raw_slot[center]];
    center_atoms[center] = batch.shell_atoms[canonical_shell[center]];
    unique_order_atoms[center] =
        ShellClass == kPsssShellClass ? batch.shell_atoms[raw_shell[center]] : center_atoms[center];
    component_begin[center] =
        static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[center]]) -
        system_ao_begin;
    const unsigned angular = batch.shell_angular[canonical_shell[center]];
    component_count[center] = (angular + 1U) * (angular + 2U) / 2U;
  }
  // PSSS's existing worker restores the last unique atom in raw shell order.
  // Retain that order even when the canonical p shell moves to the first slot.
  const unsigned unique_center_count =
      direct_force_unique_center_atoms(unique_order_atoms, unique_center_atoms);
  if (unique_center_count == 1U) return true;

  const DirectShellAoQuartetLayout layout =
      direct_shell_ao_quartet_layout(batch, first_pair, second_pair);
  auto* diagnostic = direct_force_class_work(batch, ShellClass);
  // Deterministic sparse sampling avoids timing every primitive or repeating
  // a class-filtered GPU scan. Clock reads occur only in an explicit census.
  const bool sampled = diagnostic && ((first_pair * 131U + second_pair) & 255U) == 0U;
  const unsigned long long started = sampled ? clock64() : 0ULL;
  bool full_mask = true;
  double weights[source_count][PairSeparable ? 1U : Roots::component_count]{};
  ScalarDirectForcePairInputs<Roots, PairSeparable> pair_weights;
  bool source_active[source_count]{};
  const std::size_t canonical_pair[2] = {canonical_raw_slot[0] < 2U ? first_pair : second_pair,
                                         canonical_raw_slot[2] < 2U ? first_pair : second_pair};
  auto* pair_work = diagnostic ? &diagnostic->scalar_pair : nullptr;
  if constexpr (PairSeparable) {
    if (pair_work) atomicAdd(&pair_work->attempted_shells, 1ULL);
    // A partial four-index mask is not separable. Prove its completeness before
    // either density projection; failure still retries the exact strict worker.
    if (!direct_force_complete_schwarz_mask(batch, first_pair, second_pair, layout, system_ao_begin,
                                            physical_offset, screening_tolerance, schwarz_bounds,
                                            pair_work)) {
      if (pair_work) atomicAdd(&pair_work->mask_rejections, 1ULL);
      return false;
    }
    double* outputs[2] = {pair_weights.bra, pair_weights.ket};
    const unsigned capacities[2] = {Roots::bra_pair_components, Roots::ket_pair_components};
    bool nonzero[2]{};
    for (unsigned side = 0; side < 2U; ++side) {
      const unsigned slot = 2U * side;
      const auto pair = canonical_pair[side];
      const bool reverse = batch.shell_pair_first[pair] != canonical_shell[slot];
      const auto count = shell_ao_pair_count(batch, pair);
      for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
        std::size_t raw_first{}, raw_second{};
        decode_shell_ao_pair(batch, pair, ordinal, system_ao_begin, raw_first, raw_second);
        const auto first = reverse ? raw_second : raw_first;
        const auto second = reverse ? raw_first : raw_second;
        const auto output = (first - component_begin[slot]) * component_count[slot + 1U] + second -
                            component_begin[slot + 1U];
        if (first < component_begin[slot] || second < component_begin[slot + 1U] ||
            first - component_begin[slot] >= component_count[slot] ||
            second - component_begin[slot + 1U] >= component_count[slot + 1U] ||
            output >= capacities[side]) {
          if (pair_work) atomicAdd(&pair_work->numerical_rejections, 1ULL);
          return false;
        }
        if (pair_work)
          atomicAdd(side == 0U ? &pair_work->bra_density_terms : &pair_work->ket_density_terms,
                    1ULL);
        const double scale =
            side == 0U ? coulomb_coefficient * (first_pair == second_pair ? 0.5 : 1.0) : 1.0;
        if (!direct_force_pair_weight<Unrestricted, true>(
                dimension, physical_offset, spin_offset, density, batch.direct_ao_coefficients,
                system_ao_begin, raw_first, raw_second, scale, outputs[side][output])) {
          if (pair_work) atomicAdd(&pair_work->numerical_rejections, 1ULL);
          return false;
        }
        nonzero[side] = nonzero[side] || outputs[side][output] != 0.0;
      }
    }
    source_active[0] = nonzero[0] && nonzero[1];
  } else
    for (std::size_t ordinal = 0; ordinal < layout.quartet_count; ++ordinal) {
      std::size_t raw_ao[4];
      decode_shell_ao_quartet(batch, first_pair, second_pair, layout, ordinal, system_ao_begin,
                              raw_ao);
      if (schwarz_bounds[physical_offset + matrix_index(raw_ao[0], raw_ao[1], dimension)] *
              schwarz_bounds[physical_offset + matrix_index(raw_ao[2], raw_ao[3], dimension)] <
          screening_tolerance) {
        full_mask = false;
        continue;
      }
      unsigned output = 0U;
#pragma unroll
      for (unsigned center = 0; center < 4; ++center) {
        const std::size_t canonical_ao = raw_ao[canonical_raw_slot[center]];
        if (canonical_ao < component_begin[center]) return true;
        const auto component = static_cast<unsigned>(canonical_ao - component_begin[center]);
        if (component >= component_count[center]) return true;
        output = output * component_count[center] + component;
      }
      if (output >= Roots::component_count) return true;
#pragma unroll
      for (unsigned source = 0; source < source_count; ++source) {
        const auto coefficients =
            LongRange ? DirectForceSourceCoefficients{0.0, exchange_coefficient}
                      : Sources::coefficients(source, coulomb_coefficient, exchange_coefficient);
        const double coulomb = coefficients.coulomb;
        const double exchange = coefficients.exchange;
        if (coulomb == 0.0 && exchange == 0.0) continue;
        const double coefficient = direct_force_density_coefficient_scaled<Unrestricted>(
            dimension, physical_offset, spin_offset, density, raw_ao[0], raw_ao[1], raw_ao[2],
            raw_ao[3], coulomb, exchange, batch.direct_force_seed);
        if constexpr (ShellClass != kPsssShellClass) {
          if (coefficient == 0.0) continue;
        }
        const double weight =
            direct_force_component_weight(batch.direct_ao_coefficients, system_ao_begin, raw_ao[0],
                                          raw_ao[1], raw_ao[2], raw_ao[3], coefficient);
        if constexpr (Roots::angular_order <= 1U) {
          weights[source][output] = weight;
        } else {
          weights[source][output] += weight;
        }
        source_active[source] = true;
      }
    }
  if constexpr (ShellClass == kPsssShellClass) {
#pragma unroll
    for (unsigned source = 0; source < source_count; ++source) {
      source_active[source] =
          weights[source][0] != 0.0 || weights[source][1] != 0.0 || weights[source][2] != 0.0;
    }
  }
  bool any_source = false;
#pragma unroll
  for (unsigned source = 0; source < source_count; ++source)
    any_source = any_source || source_active[source];
  if (!any_source) {
    if constexpr (PairSeparable) {
      if (pair_work) atomicAdd(&pair_work->accepted_shells, 1ULL);
    }
    return true;
  }
  if (diagnostic) {
    atomicAdd(&diagnostic->executed_shells, 1ULL);
    profile_direct_force_primitive_products(batch, diagnostic, first_pair, second_pair);
  }

  const Vec3<double> position[4] = {atom_position<double>(batch, center_atoms[0], -1),
                                    atom_position<double>(batch, center_atoms[1], -1),
                                    atom_position<double>(batch, center_atoms[2], -1),
                                    atom_position<double>(batch, center_atoms[3], -1)};
  const bool reverse_first = batch.shell_pair_first[canonical_pair[0]] != canonical_shell[0];
  const bool reverse_second = batch.shell_pair_first[canonical_pair[1]] != canonical_shell[2];
  const std::int64_t first_begin = batch.shell_pair_primitive_offsets[canonical_pair[0]];
  const std::int64_t first_end = batch.shell_pair_primitive_offsets[canonical_pair[0] + 1];
  const std::int64_t second_begin = batch.shell_pair_primitive_offsets[canonical_pair[1]];
  const std::int64_t second_end = batch.shell_pair_primitive_offsets[canonical_pair[1] + 1];
  // The resident scheduler supplies the canonical p-s bra. Exact capacity
  // matching protects primitive indexing; partial storage never drops work.
  const bool use_resident = ResidentBra && resident_first_pairs != nullptr &&
                            resident_first_pair_count == first_end - first_begin;
  using LocalGradient = std::conditional_t<Mixed, generated_weighted_eri::IndependentGradientF32,
                                           generated_weighted_eri::IndependentGradient>;
  LocalGradient result[source_count]{};
  unsigned long long primitive_sources = 0;
  for (std::int64_t first_primitive = first_begin; first_primitive < first_end; ++first_primitive) {
    const PrimitivePairData first_data = use_resident
                                             ? resident_first_pairs[first_primitive - first_begin]
                                             : batch.shell_primitive_pairs[first_primitive];
    for (std::int64_t second_primitive = second_begin; second_primitive < second_end;
         ++second_primitive) {
      if constexpr (PairSeparable) {
        if (pair_work) atomicAdd(&pair_work->primitive_products, 1ULL);
      }
      const PrimitivePairData second_data = batch.shell_primitive_pairs[second_primitive];
      generated_weighted_eri::Geometry geometry;
      const double boys_argument = generated_weighted_eri::make_direct_cached_geometry(
          first_data, second_data, reverse_first, reverse_second, position[0], position[1],
          position[2], position[3], geometry);
      if constexpr (LongRange) {
        const bool valid = generativeqc::integrals::range_moments(
            Roots::angular_order + 1U, boys_argument, geometry.rho,
            generativeqc::integrals::CoulombRange::Long, omega, geometry.boys);
        // Preserve the existing nonfinite-result failure contract. An invalid
        // ladder must not leave uninitialized moments or silently omit work.
        if (!valid) {
          for (unsigned order = 0; order <= Roots::angular_order + 1U; ++order)
            geometry.boys[order] = kInvalidDirectForceMoment;
        }
      } else {
        boys_values<Roots::angular_order + 1U>(boys_argument, geometry.boys);
      }
#pragma unroll
      for (unsigned source = 0; source < source_count; ++source) {
        if (!source_active[source]) continue;
        const auto primitive = [&] {
          if constexpr (PairSeparable)
            return Roots::evaluate_separable(geometry, pair_weights.bra, pair_weights.ket);
          else if constexpr (RelativeResponse)
            return Roots::evaluate_relative(geometry, weights[source]);
          else if constexpr (Mixed)
            return Roots::evaluate_mixed(geometry, weights[source]);
          else
            return Roots::evaluate(geometry, weights[source]);
        }();
        if constexpr (Mixed) ++primitive_sources;
#pragma unroll
        for (unsigned center = 0; center < 3; ++center) {
#pragma unroll
          for (unsigned axis = 0; axis < 3; ++axis) {
            result[source].center[center][axis] += primitive.center[center][axis];
          }
        }
      }
    }
  }
  if (diagnostic && !PairSeparable) {
    const auto elapsed = sampled ? clock64() - started : 0ULL;
    auto& scalar = diagnostic->scalar;
    atomicAdd(&scalar.calls, 1ULL);
    if (sampled) {
      atomicAdd(&scalar.sampled_calls, 1ULL);
      atomicAdd(&scalar.sampled_thread_cycles, elapsed);
    }
    if (!LongRange && !Mixed && full_mask && exchange_coefficient == 0.0 &&
        batch.direct_force_seed == nullptr) {
      atomicAdd(&scalar.separable_calls, 1ULL);
      atomicAdd(&scalar.separable_primitive_products,
                static_cast<unsigned long long>(first_end - first_begin) *
                    static_cast<unsigned long long>(second_end - second_begin));
      if (sampled) {
        atomicAdd(&scalar.separable_sampled_calls, 1ULL);
        atomicAdd(&scalar.separable_sampled_thread_cycles, elapsed);
      }
    }
  }
  if constexpr (Mixed) {
    auto* work = batch.direct_force_precision_work
                     ? batch.direct_force_precision_work + 3 * static_cast<std::size_t>(system)
                     : nullptr;
    if (work) {
      atomicAdd(work, 1ULL);
      atomicAdd(work + 1, primitive_sources);
    }
  }
  if constexpr (Mixed || RelativeResponse || PairSeparable) {
    bool finite = true;
    for (unsigned source = 0; source < source_count; ++source)
      for (unsigned center = 0; center < 3; ++center)
        for (unsigned axis = 0; axis < 3; ++axis)
          finite = finite && isfinite(result[source].center[center][axis]);
    if constexpr (PairSeparable) {
      // The scatter folds at most six signed copies per axis (including
      // translation recovery). A conservative bound validates that entire
      // publication before its first atomic, without duplicating atom policy.
      for (unsigned center = 0; center < 3; ++center)
        for (unsigned axis = 0; axis < 3; ++axis)
          finite =
              finite && ::fabs(result[0].center[center][axis]) <= kMaximumSeparableForceComponent;
    }
    if (!finite) {
      // Nothing has reached the atom arrays yet. Retry exactly this task once
      // with the retained FP64 graph; never clip inputs or publish partial forces.
      if constexpr (Mixed) {
        auto* work = batch.direct_force_precision_work
                         ? batch.direct_force_precision_work + 3 * static_cast<std::size_t>(system)
                         : nullptr;
        if (work) atomicAdd(work + 2, 1ULL);
      }
      if constexpr (PairSeparable) {
        if (pair_work) atomicAdd(&pair_work->numerical_rejections, 1ULL);
      }
      return false;
    }
  }
  if constexpr (PairSeparable) {
    if (pair_work) atomicAdd(&pair_work->accepted_shells, 1ULL);
  }
#pragma unroll
  for (unsigned source = 0; source < source_count; ++source) {
    if (!source_active[source]) continue;
    generated_weighted_eri::IndependentGradient widened{};
#pragma unroll
    for (unsigned center = 0; center < 3; ++center)
#pragma unroll
      for (unsigned axis = 0; axis < 3; ++axis)
        widened.center[center][axis] = result[source].center[center][axis];
    scatter_direct_force_independent_gradient(center_atoms, unique_center_atoms,
                                              unique_center_count, widened,
                                              Sources::output(forces, batch.total_atoms, source));
  }
  return true;
}

/** Select arithmetic per launch, outside the shared scientific task body.
 * A failed candidate returns before the strict call so their large private
 * recurrence/weight frames need not be live simultaneously on the CUDA stack. */
template <bool Unrestricted, unsigned ShellClass, bool LongRange = false,
          class Roots = LowOrderSourceRoots<ShellClass>,
          DirectForceOutputMode Mode = DirectForceOutputMode::Separate, bool ResidentBra = false>
__device__ inline void contract_two_electron_force_low_order_sources_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* forces,
    double coulomb_coefficient, double exchange_coefficient, double omega = 0.0,
    const PrimitivePairData* resident_first_pairs = nullptr,
    std::int64_t resident_first_pair_count = 0) {
  if constexpr (!LongRange) {
    if constexpr (requires(const generated_weighted_eri::Geometry& geometry,
                           const double* weights) {
                    Roots::evaluate_separable(geometry, weights, weights);
                  }) {
      if (batch.direct_force_pair_separable && !batch.direct_force_fp32 &&
          batch.direct_force_seed == nullptr && exchange_coefficient == 0.0 &&
          coulomb_coefficient != 0.0) {
        if (contract_two_electron_force_low_order_sources_impl<
                Unrestricted, ShellClass, false, Roots, Mode, ResidentBra, false, false, true>(
                batch, task, screening_tolerance, schwarz_bounds, density, active, forces,
                coulomb_coefficient, exchange_coefficient, omega, resident_first_pairs,
                resident_first_pair_count))
          return;
        // A declined task has published nothing. Retry the original strict
        // nuclear graph exactly once, not another experimental factorization.
        contract_two_electron_force_low_order_sources_impl<Unrestricted, ShellClass, false, Roots,
                                                           Mode, ResidentBra, false, false>(
            batch, task, screening_tolerance, schwarz_bounds, density, active, forces,
            coulomb_coefficient, exchange_coefficient, omega, resident_first_pairs,
            resident_first_pair_count);
        return;
      }
    }
    if (batch.direct_force_fp32 && batch.direct_force_seed == nullptr) {
      if (contract_two_electron_force_low_order_sources_impl<Unrestricted, ShellClass, LongRange,
                                                             Roots, Mode, ResidentBra, true>(
              batch, task, screening_tolerance, schwarz_bounds, density, active, forces,
              coulomb_coefficient, exchange_coefficient, omega, resident_first_pairs,
              resident_first_pair_count))
        return;
    }
    if constexpr (Roots::angular_order != 0U &&
                  requires(const generated_weighted_eri::Geometry& geometry,
                           const double* weights) {
                    Roots::evaluate_relative(geometry, weights);
                  }) {
      // Retry the original nuclear graph before any atom scatter if the
      // relative factorization or its primitive reduction becomes nonfinite.
      if (batch.direct_force_relative_response && !batch.direct_force_fp32 &&
          contract_two_electron_force_low_order_sources_impl<Unrestricted, ShellClass, LongRange,
                                                             Roots, Mode, ResidentBra, false, true>(
              batch, task, screening_tolerance, schwarz_bounds, density, active, forces,
              coulomb_coefficient, exchange_coefficient, omega, resident_first_pairs,
              resident_first_pair_count))
        return;
    }
  }
  contract_two_electron_force_low_order_sources_impl<Unrestricted, ShellClass, LongRange, Roots,
                                                     Mode, ResidentBra, false>(
      batch, task, screening_tolerance, schwarz_bounds, density, active, forces,
      coulomb_coefficient, exchange_coefficient, omega, resident_first_pairs,
      resident_first_pair_count);
}

/** Dispatch only the closed low-order domain for the selected radial owner. */
template <bool Unrestricted, bool LongRange = false,
          DirectForceOutputMode Mode = DirectForceOutputMode::Separate>
__device__ inline void contract_two_electron_force_low_order_sources(
    unsigned shell_class, const DeviceBatch& batch, ActiveShellQuartetTile task,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* forces, double coulomb_coefficient,
    double exchange_coefficient, double omega = 0.0) {
#define GENERATIVEQC_LOW_ORDER_SOURCES_CASE(ShellClass)                                        \
  case ShellClass:                                                                             \
    contract_two_electron_force_low_order_sources_task<Unrestricted, ShellClass, LongRange,    \
                                                       LowOrderSourceRoots<ShellClass>, Mode>( \
        batch, task, screening_tolerance, schwarz_bounds, density, active, forces,             \
        coulomb_coefficient, exchange_coefficient, omega);                                     \
    break
  switch (shell_class) {
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kSsssShellClass);
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kPsssShellClass);
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kPspsShellClass);
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kPpssShellClass);
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kDsssShellClass);
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kPppsShellClass);
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kDspsShellClass);
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kDpssShellClass);
    GENERATIVEQC_LOW_ORDER_SOURCES_CASE(kFsssShellClass);
    default:
      break;
  }
#undef GENERATIVEQC_LOW_ORDER_SOURCES_CASE
}

}  // namespace generativeqc::scf::cuda_execution
