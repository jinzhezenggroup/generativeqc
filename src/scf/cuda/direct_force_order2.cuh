#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <weighted_eri.cuh>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_force_density.cuh"
#include "scf/cuda/direct_force_scatter.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/direct_native_gradient_types.cuh"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct force order2 contraction helpers.
// Borrow immutable metadata and density/output views; host plans own lifetime.

namespace vibeqc::scf::cuda_execution {

/**
 * Contract one canonical order-two shell class through compiler-owned Weighted IntegralIR math.
 *
 * Queueing, screening, density folding and atom accumulation remain native scheduler concerns.
 * This adapter only binds the existing primitive-pair representation to the generated geometry
 * vocabulary; no independent ERI/gradient recurrence is retained here.
 */
template <unsigned TargetShellClass>
__device__ inline __noinline__ generated_weighted_eri::IndependentGradient
contracted_eri_cartesian_source_order2_generated_weighted_gradient(
    const DeviceBatch& batch, std::size_t first_shell_pair, std::size_t second_shell_pair,
    std::int32_t first_shell, std::int32_t second_shell, std::int32_t third_shell,
    std::int32_t fourth_shell, const double* component_weight) {
  static_assert(TargetShellClass == kPspsShellClass || TargetShellClass == kPpssShellClass ||
                TargetShellClass == kDsssShellClass);

  const Vec3<double> first = atom_position<double>(batch, batch.shell_atoms[first_shell], -1);
  const Vec3<double> second = atom_position<double>(batch, batch.shell_atoms[second_shell], -1);
  const Vec3<double> third = atom_position<double>(batch, batch.shell_atoms[third_shell], -1);
  const Vec3<double> fourth = atom_position<double>(batch, batch.shell_atoms[fourth_shell], -1);
  const bool first_pair_matches_canonical_order =
      batch.shell_pair_first[first_shell_pair] == first_shell;
  const bool second_pair_matches_canonical_order =
      batch.shell_pair_first[second_shell_pair] == third_shell;
  const std::int64_t first_pair_begin = batch.shell_pair_primitive_offsets[first_shell_pair];
  const std::int64_t first_pair_end = batch.shell_pair_primitive_offsets[first_shell_pair + 1];
  const std::int64_t second_pair_begin = batch.shell_pair_primitive_offsets[second_shell_pair];
  const std::int64_t second_pair_end = batch.shell_pair_primitive_offsets[second_shell_pair + 1];

  generated_weighted_eri::IndependentGradient result{};
  for (std::int64_t first_primitive = first_pair_begin; first_primitive < first_pair_end;
       ++first_primitive) {
    const PrimitivePairData first_pair = batch.shell_primitive_pairs[first_primitive];
    for (std::int64_t second_primitive = second_pair_begin; second_primitive < second_pair_end;
         ++second_primitive) {
      const PrimitivePairData second_pair = batch.shell_primitive_pairs[second_primitive];
      generated_weighted_eri::Geometry geometry;
      const double boys_argument = generated_weighted_eri::make_direct_cached_geometry(
          first_pair, second_pair, !first_pair_matches_canonical_order,
          !second_pair_matches_canonical_order, first, second, third, fourth, geometry);
      boys_values<3>(boys_argument, geometry.boys);

      generated_weighted_eri::IndependentGradient primitive{};
      if constexpr (TargetShellClass == kPspsShellClass) {
        primitive = generated_weighted_eri::psps_force(geometry, component_weight);
      } else if constexpr (TargetShellClass == kPpssShellClass) {
        primitive = generated_weighted_eri::ppss_force(geometry, component_weight);
      } else {
        primitive = generated_weighted_eri::dsss_force(geometry, component_weight);
      }
#pragma unroll
      for (unsigned center = 0; center < 3; ++center) {
#pragma unroll
        for (unsigned axis = 0; axis < 3; ++axis) {
          result.center[center][axis] += primitive.center[center][axis];
        }
      }
    }
  }
  return result;
}

/**
 * Evaluate one exact order-two AO-quartet gradient through compiler-owned force roots.
 *
 * The generic force fallback supplies a one-hot Cartesian component weight. Shell/pair
 * canonicalization and ERI/derivative algebra use the same compiler-owned helpers as the
 * generated PSPS/PPSS/DSSS production consumers above.
 */
__device__ inline CartesianQuartetGradient
contracted_eri_cartesian_source_order2_generated_gradient(const DeviceBatch& batch,
                                                          std::int32_t system, std::int32_t i,
                                                          std::int32_t j, std::int32_t k,
                                                          std::int32_t l) {
  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t raw_ao[4] = {
      static_cast<std::size_t>(i),
      static_cast<std::size_t>(j),
      static_cast<std::size_t>(k),
      static_cast<std::size_t>(l),
  };
  const std::int32_t raw_shell[4] = {
      batch.direct_ao_shells[system_ao_begin + raw_ao[0]],
      batch.direct_ao_shells[system_ao_begin + raw_ao[1]],
      batch.direct_ao_shells[system_ao_begin + raw_ao[2]],
      batch.direct_ao_shells[system_ao_begin + raw_ao[3]],
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[raw_shell[0]], batch.shell_angular[raw_shell[1]],
      batch.shell_angular[raw_shell[2]], batch.shell_angular[raw_shell[3]]);

  if (shell_class != kPspsShellClass && shell_class != kPpssShellClass &&
      shell_class != kDsssShellClass) {
    return {};
  }
  unsigned canonical_raw_slot[4];
  generated_weighted_eri::canonicalize_direct_shell_slots(batch.shell_angular, raw_shell,
                                                          canonical_raw_slot);

  const std::int32_t canonical_shell[4] = {
      raw_shell[canonical_raw_slot[0]],
      raw_shell[canonical_raw_slot[1]],
      raw_shell[canonical_raw_slot[2]],
      raw_shell[canonical_raw_slot[3]],
  };
  const std::size_t canonical_pair[2] = {
      system_shell_pair_index(batch, system, canonical_shell[0], canonical_shell[1]),
      system_shell_pair_index(batch, system, canonical_shell[2], canonical_shell[3]),
  };

  const std::size_t component_begin[3] = {
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[0]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[1]]) - system_ao_begin,
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[2]]) - system_ao_begin,
  };
  const unsigned first_component =
      static_cast<unsigned>(raw_ao[canonical_raw_slot[0]] - component_begin[0]);
  unsigned output = first_component;
  if (shell_class == kPspsShellClass) {
    const unsigned third_component =
        static_cast<unsigned>(raw_ao[canonical_raw_slot[2]] - component_begin[2]);
    if (first_component >= 3U || third_component >= 3U) return {};
    output = first_component * 3U + third_component;
  } else if (shell_class == kPpssShellClass) {
    const unsigned second_component =
        static_cast<unsigned>(raw_ao[canonical_raw_slot[1]] - component_begin[1]);
    if (first_component >= 3U || second_component >= 3U) return {};
    output = first_component * 3U + second_component;
  } else if (first_component >= 6U) {
    return {};
  }

  double component_weight[9]{};
  component_weight[output] =
      direct_force_component_weight(batch.direct_ao_coefficients, system_ao_begin, raw_ao[0],
                                    raw_ao[1], raw_ao[2], raw_ao[3], 1.0);

  generated_weighted_eri::IndependentGradient gradient{};
  if (shell_class == kPspsShellClass) {
    gradient = contracted_eri_cartesian_source_order2_generated_weighted_gradient<kPspsShellClass>(
        batch, canonical_pair[0], canonical_pair[1], canonical_shell[0], canonical_shell[1],
        canonical_shell[2], canonical_shell[3], component_weight);
  } else if (shell_class == kPpssShellClass) {
    gradient = contracted_eri_cartesian_source_order2_generated_weighted_gradient<kPpssShellClass>(
        batch, canonical_pair[0], canonical_pair[1], canonical_shell[0], canonical_shell[1],
        canonical_shell[2], canonical_shell[3], component_weight);
  } else {
    gradient = contracted_eri_cartesian_source_order2_generated_weighted_gradient<kDsssShellClass>(
        batch, canonical_pair[0], canonical_pair[1], canonical_shell[0], canonical_shell[1],
        canonical_shell[2], canonical_shell[3], component_weight);
  }

  CartesianQuartetGradient result{};
#pragma unroll
  for (unsigned axis = 0; axis < 3; ++axis) {
    double fourth = 0.0;
#pragma unroll
    for (unsigned canonical = 0; canonical < 3; ++canonical) {
      const double value = gradient.center[canonical][axis];
      result.center[canonical_raw_slot[canonical]][axis] = value;
      fourth -= value;
    }
    result.center[canonical_raw_slot[3]][axis] = fourth;
  }
  return result;
}

/** Evaluate and write one complete density-weighted psps force shell task. */
template <bool Unrestricted>
__device__ inline __noinline__ void contract_two_electron_force_psps_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, double coulomb_coefficient,
    double exchange_coefficient, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  // A psps shell quartet has at most nine Cartesian AO quartets and therefore
  // always fits in the first compact tile.
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[raw_shell[0]], batch.shell_angular[raw_shell[1]],
      batch.shell_angular[raw_shell[2]], batch.shell_angular[raw_shell[3]]);
  if (shell_class != kPspsShellClass) return;
  if ((generated_shell_class_mask & (std::uint64_t{1} << kPspsShellClass)) != 0U) {
    return;
  }

  unsigned canonical_raw_slot[4];
  generated_weighted_eri::canonicalize_direct_shell_slots(batch.shell_angular, raw_shell,
                                                          canonical_raw_slot);
  const std::int32_t canonical_shell[4] = {
      raw_shell[canonical_raw_slot[0]],
      raw_shell[canonical_raw_slot[1]],
      raw_shell[canonical_raw_slot[2]],
      raw_shell[canonical_raw_slot[3]],
  };

  const std::int32_t canonical_center_atoms[4] = {
      batch.shell_atoms[canonical_shell[0]],
      batch.shell_atoms[canonical_shell[1]],
      batch.shell_atoms[canonical_shell[2]],
      batch.shell_atoms[canonical_shell[3]],
  };
  std::int32_t unique_center_atoms[4];
  const unsigned unique_center_count =
      direct_force_unique_center_atoms(canonical_center_atoms, unique_center_atoms);
  if (unique_center_count == 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_p_ao_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[0]]) - system_ao_begin;
  const std::size_t second_p_ao_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[2]]) - system_ao_begin;
  const DirectShellAoQuartetLayout ao_quartet_layout =
      direct_shell_ao_quartet_layout(batch, first_pair, second_pair);

  double component_weight[9]{};
  bool any_component = false;
  for (std::size_t ordinal = 0; ordinal < ao_quartet_layout.quartet_count; ++ordinal) {
    std::size_t raw_ao[4];
    decode_shell_ao_quartet(batch, first_pair, second_pair, ao_quartet_layout, ordinal,
                            system_ao_begin, raw_ao);
    if (schwarz_bounds[physical_offset + matrix_index(raw_ao[0], raw_ao[1], n)] *
            schwarz_bounds[physical_offset + matrix_index(raw_ao[2], raw_ao[3], n)] <
        screening_tolerance) {
      continue;
    }
    const double density_coefficient = direct_force_density_coefficient_scaled<Unrestricted>(
        n, physical_offset, spin_offset, density, raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3],
        coulomb_coefficient, exchange_coefficient);
    if (density_coefficient == 0.0) continue;
    const unsigned first_axis =
        static_cast<unsigned>(raw_ao[canonical_raw_slot[0]] - first_p_ao_begin);
    const unsigned second_axis =
        static_cast<unsigned>(raw_ao[canonical_raw_slot[2]] - second_p_ao_begin);
    if (first_axis >= 3 || second_axis >= 3) return;
    component_weight[first_axis * 3 + second_axis] +=
        direct_force_component_weight(batch.direct_ao_coefficients, system_ao_begin, raw_ao[0],
                                      raw_ao[1], raw_ao[2], raw_ao[3], density_coefficient);
    any_component = true;
  }
  if (!any_component) return;

  const auto gradient =
      contracted_eri_cartesian_source_order2_generated_weighted_gradient<kPspsShellClass>(
          batch, first_pair, second_pair, canonical_shell[0], canonical_shell[1],
          canonical_shell[2], canonical_shell[3], component_weight);
  scatter_direct_force_independent_gradient(canonical_center_atoms, unique_center_atoms,
                                            unique_center_count, gradient, forces);
}

/** Evaluate one closed ppss or dsss shell task over its exact AO domain. */
template <bool Unrestricted, unsigned TargetShellClass>
__device__ inline __noinline__ void contract_two_electron_force_pair_order2_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, double coulomb_coefficient,
    double exchange_coefficient, const std::uint8_t* active, double* forces,
    std::uint64_t generated_shell_class_mask) {
  static_assert(TargetShellClass == kPpssShellClass || TargetShellClass == kDsssShellClass);
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active[system] == 0) return;

  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[raw_shell[0]], batch.shell_angular[raw_shell[1]],
      batch.shell_angular[raw_shell[2]], batch.shell_angular[raw_shell[3]]);
  if (shell_class != TargetShellClass) return;
  if ((generated_shell_class_mask & (std::uint64_t{1} << TargetShellClass)) != 0U) {
    return;
  }

  unsigned canonical_raw_slot[4];
  generated_weighted_eri::canonicalize_direct_shell_slots(batch.shell_angular, raw_shell,
                                                          canonical_raw_slot);
  const std::int32_t canonical_shell[4] = {
      raw_shell[canonical_raw_slot[0]],
      raw_shell[canonical_raw_slot[1]],
      raw_shell[canonical_raw_slot[2]],
      raw_shell[canonical_raw_slot[3]],
  };
  const std::size_t canonical_pair[2] = {
      canonical_raw_slot[0] < 2U ? first_pair : second_pair,
      canonical_raw_slot[2] < 2U ? first_pair : second_pair,
  };

  const std::int32_t canonical_center_atoms[4] = {
      batch.shell_atoms[canonical_shell[0]],
      batch.shell_atoms[canonical_shell[1]],
      batch.shell_atoms[canonical_shell[2]],
      batch.shell_atoms[canonical_shell[3]],
  };
  std::int32_t unique_center_atoms[4];
  const unsigned unique_center_count =
      direct_force_unique_center_atoms(canonical_center_atoms, unique_center_atoms);
  if (unique_center_count == 1) return;

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t first_component_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[0]]) - system_ao_begin;
  const std::size_t second_component_begin =
      static_cast<std::size_t>(batch.shell_direct_ao_offsets[canonical_shell[1]]) - system_ao_begin;
  const DirectShellAoQuartetLayout ao_quartet_layout =
      direct_shell_ao_quartet_layout(batch, first_pair, second_pair);

  double component_weight[9]{};
  bool any_component = false;
  for (std::size_t ordinal = 0; ordinal < ao_quartet_layout.quartet_count; ++ordinal) {
    std::size_t raw_ao[4];
    decode_shell_ao_quartet(batch, first_pair, second_pair, ao_quartet_layout, ordinal,
                            system_ao_begin, raw_ao);
    if (schwarz_bounds[physical_offset + matrix_index(raw_ao[0], raw_ao[1], n)] *
            schwarz_bounds[physical_offset + matrix_index(raw_ao[2], raw_ao[3], n)] <
        screening_tolerance) {
      continue;
    }
    const double density_coefficient = direct_force_density_coefficient_scaled<Unrestricted>(
        n, physical_offset, spin_offset, density, raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3],
        coulomb_coefficient, exchange_coefficient);
    if (density_coefficient == 0.0) continue;
    const unsigned first_component =
        static_cast<unsigned>(raw_ao[canonical_raw_slot[0]] - first_component_begin);
    unsigned output = first_component;
    if constexpr (TargetShellClass == kPpssShellClass) {
      const unsigned second_component =
          static_cast<unsigned>(raw_ao[canonical_raw_slot[1]] - second_component_begin);
      if (first_component >= 3U || second_component >= 3U) return;
      output = first_component * 3U + second_component;
    } else if (first_component >= 6U) {
      return;
    }
    component_weight[output] +=
        direct_force_component_weight(batch.direct_ao_coefficients, system_ao_begin, raw_ao[0],
                                      raw_ao[1], raw_ao[2], raw_ao[3], density_coefficient);
    any_component = true;
  }
  if (!any_component) return;

  const auto gradient =
      contracted_eri_cartesian_source_order2_generated_weighted_gradient<TargetShellClass>(
          batch, canonical_pair[0], canonical_pair[1], canonical_shell[0], canonical_shell[1],
          canonical_shell[2], canonical_shell[3], component_weight);

  scatter_direct_force_independent_gradient(canonical_center_atoms, unique_center_atoms,
                                            unique_center_count, gradient, forces);
}

}  // namespace vibeqc::scf::cuda_execution
