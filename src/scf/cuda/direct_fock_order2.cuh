#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <weighted_eri.cuh>

#include "generated_direct_order2_shell.cuh"
#include "scf/cuda/direct_fock_accumulation.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/direct_screening.cuh"
#include "scf/cuda/packed_basis.hpp"

// Retained direct fock order2 contraction helpers.
// Borrow immutable metadata and density/output views; host plans own lifetime.

namespace generativeqc::scf::cuda_execution {

/** One canonical shell slot and its position in the original quartet. */
struct Order2SourceSlot {
  std::int32_t shell;
  unsigned original;
};

/** Map raw AO slots to the fused vector's canonical Cartesian product. */
__device__ __forceinline__ unsigned order2_component_index(const DeviceBatch& batch,
                                                           const Order2SourceSlot (&slots)[4],
                                                           const std::size_t (&raw_ao)[4],
                                                           std::size_t system_ao_begin) {
  unsigned output = 0;
#pragma unroll
  for (unsigned slot = 0; slot < 4; ++slot) {
    const unsigned angular = batch.shell_angular[slots[slot].shell];
    const unsigned component_count = (angular + 1) * (angular + 2) / 2;
    const std::size_t local_begin =
        static_cast<std::size_t>(batch.shell_direct_ao_offsets[slots[slot].shell]) -
        system_ao_begin;
    const unsigned component = static_cast<unsigned>(raw_ao[slots[slot].original] - local_begin);
    output = output * component_count + component;
  }
  return output;
}

/** Evaluate and scatter one complete psps, ppss, or dsss shell task. */
template <bool Unrestricted>
__device__ inline __noinline__ void contract_fock_direct_order2_task(
    const DeviceBatch& batch, ActiveShellQuartetTile task, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  if (task.tile != 0U) return;
  const std::size_t first_pair = task.first_pair;
  const std::size_t second_pair = task.second_pair;
  const std::int32_t system = batch.shell_pair_systems[first_pair];
  if (active != nullptr && active[system] == 0) return;

  const std::int32_t raw_shell[4] = {
      batch.shell_pair_first[first_pair],
      batch.shell_pair_second[first_pair],
      batch.shell_pair_first[second_pair],
      batch.shell_pair_second[second_pair],
  };
  const unsigned shell_class = direct_quartet_shell_class_device(
      batch.shell_angular[raw_shell[0]], batch.shell_angular[raw_shell[1]],
      batch.shell_angular[raw_shell[2]], batch.shell_angular[raw_shell[3]]);
  if (shell_class != 2U && shell_class != 3U && shell_class != 6U) return;
  // Generated order-two workers execute before this runtime fallback.
  if (generated_fock_shell_class_mask != nullptr &&
      ((*generated_fock_shell_class_mask & (std::uint64_t{1} << shell_class)) != 0U)) {
    return;
  }

  unsigned canonical_raw_slot[4];
  generated_weighted_eri::canonicalize_direct_shell_slots(batch.shell_angular, raw_shell,
                                                          canonical_raw_slot);
  Order2SourceSlot slots[4] = {
      {raw_shell[canonical_raw_slot[0]], canonical_raw_slot[0]},
      {raw_shell[canonical_raw_slot[1]], canonical_raw_slot[1]},
      {raw_shell[canonical_raw_slot[2]], canonical_raw_slot[2]},
      {raw_shell[canonical_raw_slot[3]], canonical_raw_slot[3]},
  };

  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const DirectShellAoQuartetLayout ao_quartet_layout =
      direct_shell_ao_quartet_layout(batch, first_pair, second_pair);

  unsigned active_component_mask = 0;
  for (std::size_t ordinal = 0; ordinal < ao_quartet_layout.quartet_count; ++ordinal) {
    std::size_t raw_ao[4];
    decode_shell_ao_quartet(batch, first_pair, second_pair, ao_quartet_layout, ordinal,
                            system_ao_begin, raw_ao);
    if (!direct_ao_quartet_survives_schwarz(schwarz_bounds, physical_offset, n, raw_ao[0],
                                            raw_ao[1], raw_ao[2], raw_ao[3], screening_tolerance)) {
      continue;
    }
    active_component_mask |= 1U << order2_component_index(batch, slots, raw_ao, system_ao_begin);
  }
  if (active_component_mask == 0) return;

  Order2IntegralVector integral{};
  switch (shell_class) {
    case 2:
      integral = contracted_eri_cartesian_source_order2_shell<1, 0, 1, 0>(
          batch, slots[0].shell, slots[1].shell, slots[2].shell, slots[3].shell,
          active_component_mask);
      break;
    case 3:
      integral = contracted_eri_cartesian_source_order2_shell<1, 1, 0, 0>(
          batch, slots[0].shell, slots[1].shell, slots[2].shell, slots[3].shell,
          active_component_mask);
      break;
    default:
      integral = contracted_eri_cartesian_source_order2_shell<2, 0, 0, 0>(
          batch, slots[0].shell, slots[1].shell, slots[2].shell, slots[3].shell,
          active_component_mask);
      break;
  }

  for (std::size_t ordinal = 0; ordinal < ao_quartet_layout.quartet_count; ++ordinal) {
    std::size_t raw_ao[4];
    decode_shell_ao_quartet(batch, first_pair, second_pair, ao_quartet_layout, ordinal,
                            system_ao_begin, raw_ao);
    const unsigned component = order2_component_index(batch, slots, raw_ao, system_ao_begin);
    if ((active_component_mask & (1U << component)) == 0 || integral.component[component] == 0.0) {
      continue;
    }
    accumulate_direct_fock_integral<Unrestricted>(n, physical_offset, spin_offset, density, fock,
                                                  raw_ao[0], raw_ao[1], raw_ao[2], raw_ao[3],
                                                  integral.component[component]);
  }
}

}  // namespace generativeqc::scf::cuda_execution
