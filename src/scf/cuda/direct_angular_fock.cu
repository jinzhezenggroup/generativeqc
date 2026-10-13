#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/direct_angular_fock.hpp"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_fock_order2.cuh"
#include "scf/cuda/direct_fock_quartet.cuh"
#include "scf/cuda/direct_metadata.hpp"
#include "scf/cuda/packed_basis.hpp"

namespace generativeqc::scf::cuda_execution {

/** Bounded class-major counterpart of the fixed-topology whole-shell owner.
 * A 64-bit grid-stride domain cannot overflow a 32-bit queue head. All lanes
 * retire the shared recurrence before the next task is published, including
 * inactive systems and shell/AO-screened tasks. No AO tile repeats preparation.
 */
template <bool Unrestricted, unsigned AngularOrder>
__global__ void bounded_materialized_fock_kernel(DeviceBatch batch,
                                                 const GeneratedShellPairStream* topology_pointer,
                                                 unsigned bra_class, unsigned ket_class,
                                                 double screening_tolerance,
                                                 const double* schwarz_bounds,
                                                 const double* density, const std::uint8_t* active,
                                                 double* fock, unsigned long long* work_count) {
  constexpr unsigned lanes = canonical_materialized_component_lanes(AngularOrder);
  constexpr unsigned slots = canonical_materialized_component_slots(AngularOrder);
  __shared__ MaterializedDirectPairRecurrence<AngularOrder> recurrence;
  __shared__ ActiveShellQuartetTile task;
  __shared__ bool keep;
  const auto& topology = *topology_pointer;
  const std::size_t stride = static_cast<std::size_t>(topology.batch_size) + 1U;
  const auto* density_bounds =
      reinterpret_cast<const ShellPairDensityBounds*>(topology.shell_pair_density_bounds);
  for (std::int32_t system = 0; system < topology.batch_size; ++system) {
    if (active && !active[system]) continue;
    const auto bra_begin = topology.pair_class_offsets[bra_class * stride + system];
    const auto ket_begin = topology.pair_class_offsets[ket_class * stride + system];
    const std::size_t bra_count =
        topology.pair_class_offsets[bra_class * stride + system + 1U] - bra_begin;
    const std::size_t ket_count =
        topology.pair_class_offsets[ket_class * stride + system + 1U] - ket_begin;
    const bool same_class = bra_class == ket_class;
    const auto count = same_class ? bra_count * (bra_count + 1U) / 2U : bra_count * ket_count;
    for (std::size_t ordinal = blockIdx.x; ordinal < count; ordinal += gridDim.x) {
      if (threadIdx.x == 0) {
        std::size_t bra{}, ket{};
        if (same_class)
          decode_lower_triangle(ordinal, bra, ket);
        else {
          bra = ordinal / ket_count;
          ket = ordinal % ket_count;
        }
        const auto first = topology.pair_order[bra_begin + bra];
        const auto second = topology.pair_order[ket_begin + ket];
        task = {max(first, second), min(first, second), 0U};
        keep = direct_shell_quartet_survives_screening<Unrestricted, DirectScreeningPurpose::Fock>(
            batch, task.first_pair, task.second_pair, screening_tolerance,
            topology.shell_pair_bounds, density_bounds, nullptr);
        if (keep && work_count) atomicAdd(work_count, 1ULL);
      }
      __syncthreads();
      if (keep)
        contract_materialized_direct_pair_fock<Unrestricted, AngularOrder, slots, lanes>(
            batch, task, screening_tolerance, schwarz_bounds, density, active, fock, nullptr,
            recurrence);
      __syncthreads();
    }
  }
}

cudaError_t launch_bounded_materialized_fock(
    bool unrestricted, unsigned shell_class, cudaStream_t stream, unsigned worker_blocks,
    DeviceBatch batch, const GeneratedShellPairStream* topology, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* fock,
    unsigned long long* work_count) {
  if (shell_class >= detail::kDirectQuartetShellClassCount || !worker_blocks || !topology ||
      !batch.shell_primitive_pairs || !batch.shell_pair_primitive_offsets)
    return cudaErrorInvalidValue;
  const auto pairs = detail::decode_direct_triangular_class(shell_class);
  const auto order = detail::direct_quartet_shell_class_angular_order(shell_class);
#define GENERATIVEQC_MATERIALIZED_FOCK_CASE(order_value)                                         \
  case order_value:                                                                              \
    if (unrestricted)                                                                            \
      bounded_materialized_fock_kernel<true, order_value>                                        \
          <<<worker_blocks, canonical_materialized_component_lanes(order_value), 0, stream>>>(   \
              batch, topology, pairs[0], pairs[1], screening_tolerance, schwarz_bounds, density, \
              active, fock, work_count);                                                         \
    else                                                                                         \
      bounded_materialized_fock_kernel<false, order_value>                                       \
          <<<worker_blocks, canonical_materialized_component_lanes(order_value), 0, stream>>>(   \
              batch, topology, pairs[0], pairs[1], screening_tolerance, schwarz_bounds, density, \
              active, fock, work_count);                                                         \
    break
  switch (order) {
    GENERATIVEQC_MATERIALIZED_FOCK_CASE(3);
    GENERATIVEQC_MATERIALIZED_FOCK_CASE(4);
    GENERATIVEQC_MATERIALIZED_FOCK_CASE(5);
    GENERATIVEQC_MATERIALIZED_FOCK_CASE(6);
    GENERATIVEQC_MATERIALIZED_FOCK_CASE(7);
    GENERATIVEQC_MATERIALIZED_FOCK_CASE(8);
    GENERATIVEQC_MATERIALIZED_FOCK_CASE(9);
    GENERATIVEQC_MATERIALIZED_FOCK_CASE(10);
    GENERATIVEQC_MATERIALIZED_FOCK_CASE(11);
    GENERATIVEQC_MATERIALIZED_FOCK_CASE(12);
    default:
      return cudaErrorInvalidValue;
  }
#undef GENERATIVEQC_MATERIALIZED_FOCK_CASE
  return cudaPeekAtLastError();
}

/** Fixed-capacity wrapper retained for high-register angular orders. */
template <bool Unrestricted, unsigned AngularOrder, typename EvalScalar = double>
__global__ void build_fock_direct_quartet_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, double screening_tolerance,
    const double* schwarz_bounds, const double* density, const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  contract_fock_direct_quartet_subtile<Unrestricted, AngularOrder, EvalScalar>(
      batch, active_shell_quartet_tile_count, active_shell_quartet_tiles, screening_tolerance,
      schwarz_bounds, density, active, fock, generated_fock_shell_class_mask,
      static_cast<std::size_t>(blockIdx.x), threadIdx.x);
}

/** The first packet consumes all components of an admitted shell quartet.
 * Compaction admits every packet together on the same precision/source route.
 * Generated-class exclusions and independent AO screening remain unchanged. */
template <bool Unrestricted, unsigned AngularOrder>
__global__ void build_fock_direct_pair_materialized_kernel(
    DeviceBatch batch, const std::uint32_t* count, const ActiveShellQuartetTile* tasks,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* fock, const std::uint64_t* generated_mask) {
  if (blockIdx.x >= *count) return;
  const auto task = tasks[blockIdx.x];
  if (task.tile != 0U) return;
  // At fixed total angular order, balanced shell angular momenta maximize
  // the product of Cartesian component counts. This bounds per-lane register
  // slots without allocating the 40-slot ffff maximum for every lower order.
  constexpr unsigned slots = [] {
    unsigned components = 1;
    for (unsigned shell = 0; shell < 4; ++shell) {
      const unsigned angular = AngularOrder / 4 + (shell < AngularOrder % 4);
      components *= (angular + 1) * (angular + 2) / 2;
    }
    return (components + detail::kDirectQuartetTileSize - 1) / detail::kDirectQuartetTileSize;
  }();
  __shared__ MaterializedDirectPairRecurrence<AngularOrder> shared;
  contract_materialized_direct_pair_fock<Unrestricted, AngularOrder, slots>(
      batch, task, screening_tolerance, schwarz_bounds, density, active, fock, generated_mask,
      shared);
}

/** Pack exact ssss shell tasks across all lanes of one worker warp. */
template <bool Unrestricted, unsigned AngularOrder>
__global__ void build_fock_direct_quartet_packed_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* fock) {
  static_assert(AngularOrder < kPackedSsssAngularOrderCount);
  const unsigned lane = threadIdx.x;
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  while (true) {
    std::uint32_t packed_begin = 0;
    if (lane == 0) {
      packed_begin = atomicAdd(task_head, static_cast<std::uint32_t>(warpSize));
    }
    packed_begin = __shfl_sync(0xffffffffU, packed_begin, 0);
    if (packed_begin >= work_count) return;
    const std::uint32_t packed_item = packed_begin + lane;
    if (packed_item < work_count) {
      // Order zero has one subtile and one exact logical tile per shell
      // quartet, so packed_item is also its compact subtile index.
      contract_fock_direct_quartet_subtile<Unrestricted, AngularOrder>(
          batch, active_shell_quartet_tile_count, active_shell_quartet_tiles, screening_tolerance,
          schwarz_bounds, density, active, fock, nullptr, packed_item, 0U);
    }
  }
}

/** Consume complete order-two shell tasks, one independent task per lane. */
template <bool Unrestricted>
__global__ void build_fock_direct_order2_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  const unsigned lane = threadIdx.x;
  const std::uint32_t work_count = *active_shell_quartet_tile_count;
  while (true) {
    std::uint32_t packed_begin = 0;
    if (lane == 0) {
      packed_begin = atomicAdd(task_head, static_cast<std::uint32_t>(warpSize));
    }
    packed_begin = __shfl_sync(0xffffffffU, packed_begin, 0);
    if (packed_begin >= work_count) return;
    const std::uint32_t packed_item = packed_begin + lane;
    if (packed_item < work_count) {
      contract_fock_direct_order2_task<Unrestricted>(batch, active_shell_quartet_tiles[packed_item],
                                                     screening_tolerance, schwarz_bounds, density,
                                                     active, fock, generated_fock_shell_class_mask);
    }
  }
}

/** Consume only the active compacted Fock domain from a device queue. */
template <bool Unrestricted, unsigned AngularOrder, typename EvalScalar = double>
__global__ void build_fock_direct_quartet_persistent_kernel(
    DeviceBatch batch, const std::uint32_t* active_shell_quartet_tile_count,
    const ActiveShellQuartetTile* active_shell_quartet_tiles, std::uint32_t* task_head,
    double screening_tolerance, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  static_assert(AngularOrder < detail::kDirectQuartetAngularOrderCount);
  const unsigned lane = threadIdx.x % warpSize;
  constexpr std::uint32_t subtiles_per_tile =
      static_cast<std::uint32_t>(detail::direct_quartet_subtiles_per_tile(AngularOrder));
  const std::uint32_t work_count = *active_shell_quartet_tile_count * subtiles_per_tile;
  while (true) {
    std::uint32_t active_subtile = 0;
    if (lane == 0) active_subtile = atomicAdd(task_head, 1U);
    active_subtile = __shfl_sync(0xffffffffU, active_subtile, 0);
    if (active_subtile >= work_count) return;
    contract_fock_direct_quartet_subtile<Unrestricted, AngularOrder, EvalScalar>(
        batch, active_shell_quartet_tile_count, active_shell_quartet_tiles, screening_tolerance,
        schwarz_bounds, density, active, fock, generated_fock_shell_class_mask, active_subtile,
        threadIdx.x);
  }
}

template <bool Unrestricted, typename EvalScalar = double, unsigned AngularOrder = 0>
void launch_angular_fock_quartets(
    cudaStream_t stream,
    const std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>& offsets,
    DeviceBatch batch, const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles, const std::uint32_t* generic_order5_tile_count,
    const ActiveShellQuartetTile* generic_order5_tiles, std::uint32_t* persistent_task_heads,
    unsigned persistent_worker_blocks, double screening_tolerance, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  if constexpr (AngularOrder < detail::kDirectQuartetAngularOrderCount) {
    if (capacities[AngularOrder] != 0) {
      const std::uint32_t* order_tile_count = active_tile_counts + AngularOrder;
      const ActiveShellQuartetTile* order_tiles = active_tiles + offsets[AngularOrder];
      if constexpr (AngularOrder == kGenericOrderFiveAngularOrder &&
                    std::is_same_v<EvalScalar, double>) {
        // Every order-five class currently enabled for generated Fock owns an
        // exact queue. Avoid making the generic persistent worker claim all
        // six subtiles only to decode the class and return.
        order_tile_count = generic_order5_tile_count;
        order_tiles = generic_order5_tiles;
      }
      bool materialized = false;
      if constexpr (AngularOrder >= 5 && std::is_same_v<EvalScalar, double>) {
        // Component-specific reachable closures and reassociated Hermite sums
        // retain their own source contracts. Missing pair storage also keeps
        // the bounded incumbent, without allocation or selection during replay.
        if (batch.direct_pair_materialized_values && batch.shell_primitive_pairs &&
            batch.shell_pair_primitive_offsets && !(batch.direct_coulomb_reachable & 1U) &&
            !(batch.direct_hermite_convolution & 1U)) {
          build_fock_direct_pair_materialized_kernel<Unrestricted, AngularOrder>
              <<<static_cast<unsigned>(capacities[AngularOrder]), detail::kDirectQuartetTileSize, 0,
                 stream>>>(batch, order_tile_count, order_tiles, screening_tolerance,
                           schwarz_bounds, density, active, fock, generated_fock_shell_class_mask);
          materialized = true;
        }
      }
      if (!materialized) {
        if constexpr (std::is_same_v<EvalScalar, MixedPrecisionFloat>) {
          // Mixed work begins at order three.  It keeps an independent queue and
          // persistent head so the ERI recurrence contains no per-warp precision
          // branch and low-order shell-fused workers remain unchanged.
          if constexpr (AngularOrder >= kMixedFockMinimumAngularOrder &&
                        AngularOrder < kPersistentFockAngularOrderCount) {
            const unsigned capacity_blocks = static_cast<unsigned>(
                capacities[AngularOrder] * detail::direct_quartet_subtiles_per_tile(AngularOrder));
            build_fock_direct_quartet_persistent_kernel<Unrestricted, AngularOrder, EvalScalar>
                <<<std::min(capacity_blocks, persistent_worker_blocks),
                   detail::kDirectQuartetThreads, 0, stream>>>(
                    batch, order_tile_count, order_tiles, persistent_task_heads + AngularOrder,
                    screening_tolerance, schwarz_bounds, density, active, fock,
                    generated_fock_shell_class_mask);
          } else if constexpr (AngularOrder >= kPersistentFockAngularOrderCount) {
            build_fock_direct_quartet_kernel<Unrestricted, AngularOrder, EvalScalar>
                <<<static_cast<unsigned>(capacities[AngularOrder] *
                                         detail::direct_quartet_subtiles_per_tile(AngularOrder)),
                   detail::kDirectQuartetThreads, 0, stream>>>(
                    batch, order_tile_count, order_tiles, screening_tolerance, schwarz_bounds,
                    density, active, fock, generated_fock_shell_class_mask);
          }
        } else if constexpr (AngularOrder < kPackedSsssAngularOrderCount) {
          const unsigned capacity_workers =
              static_cast<unsigned>((capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
                                    detail::kDirectQuartetThreads);
          build_fock_direct_quartet_packed_persistent_kernel<Unrestricted, AngularOrder>
              <<<std::min(capacity_workers, persistent_worker_blocks),
                 detail::kDirectQuartetThreads, 0, stream>>>(
                  batch, order_tile_count, order_tiles, persistent_task_heads + AngularOrder,
                  screening_tolerance, schwarz_bounds, density, active, fock);
        } else if constexpr (AngularOrder == kFusedOrderTwoAngularOrder) {
          const unsigned capacity_workers =
              static_cast<unsigned>((capacities[AngularOrder] + detail::kDirectQuartetThreads - 1) /
                                    detail::kDirectQuartetThreads);
          build_fock_direct_order2_persistent_kernel<Unrestricted>
              <<<std::min(capacity_workers, persistent_worker_blocks),
                 detail::kDirectQuartetThreads, 0, stream>>>(
                  batch, order_tile_count, order_tiles, persistent_task_heads + AngularOrder,
                  screening_tolerance, schwarz_bounds, density, active, fock,
                  generated_fock_shell_class_mask);
        } else if constexpr (AngularOrder < kPersistentFockAngularOrderCount) {
          const unsigned capacity_blocks = static_cast<unsigned>(
              capacities[AngularOrder] * detail::direct_quartet_subtiles_per_tile(AngularOrder));
          build_fock_direct_quartet_persistent_kernel<Unrestricted, AngularOrder>
              <<<std::min(capacity_blocks, persistent_worker_blocks), detail::kDirectQuartetThreads,
                 0, stream>>>(batch, order_tile_count, order_tiles,
                              persistent_task_heads + AngularOrder, screening_tolerance,
                              schwarz_bounds, density, active, fock,
                              generated_fock_shell_class_mask);
        } else {
          build_fock_direct_quartet_kernel<Unrestricted, AngularOrder>
              <<<static_cast<unsigned>(capacities[AngularOrder] *
                                       detail::direct_quartet_subtiles_per_tile(AngularOrder)),
                 detail::kDirectQuartetThreads, 0, stream>>>(
                  batch, order_tile_count, order_tiles, screening_tolerance, schwarz_bounds,
                  density, active, fock, generated_fock_shell_class_mask);
        }
      }
    }
    launch_angular_fock_quartets<Unrestricted, EvalScalar, AngularOrder + 1>(
        stream, capacities, offsets, batch, active_tile_counts, active_tiles,
        generic_order5_tile_count, generic_order5_tiles, persistent_task_heads,
        persistent_worker_blocks, screening_tolerance, schwarz_bounds, density, active, fock,
        generated_fock_shell_class_mask);
  }
}

void dispatch_angular_fock_quartets(
    bool unrestricted, bool mixed_precision, cudaStream_t stream,
    const std::array<std::size_t, detail::kDirectQuartetAngularOrderCount>& capacities,
    const std::array<std::uint32_t, detail::kDirectQuartetAngularOrderCount + 1>& offsets,
    DeviceBatch batch, const std::uint32_t* active_tile_counts,
    const ActiveShellQuartetTile* active_tiles, const std::uint32_t* generic_order5_tile_count,
    const ActiveShellQuartetTile* generic_order5_tiles, std::uint32_t* persistent_task_heads,
    unsigned persistent_worker_blocks, double screening_tolerance, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* fock,
    const std::uint64_t* generated_fock_shell_class_mask) {
  if (unrestricted) {
    if (mixed_precision) {
      launch_angular_fock_quartets<true, MixedPrecisionFloat>(
          stream, capacities, offsets, batch, active_tile_counts, active_tiles,
          generic_order5_tile_count, generic_order5_tiles, persistent_task_heads,
          persistent_worker_blocks, screening_tolerance, schwarz_bounds, density, active, fock,
          generated_fock_shell_class_mask);
    } else {
      launch_angular_fock_quartets<true, double>(
          stream, capacities, offsets, batch, active_tile_counts, active_tiles,
          generic_order5_tile_count, generic_order5_tiles, persistent_task_heads,
          persistent_worker_blocks, screening_tolerance, schwarz_bounds, density, active, fock,
          generated_fock_shell_class_mask);
    }
  } else {
    if (mixed_precision) {
      launch_angular_fock_quartets<false, MixedPrecisionFloat>(
          stream, capacities, offsets, batch, active_tile_counts, active_tiles,
          generic_order5_tile_count, generic_order5_tiles, persistent_task_heads,
          persistent_worker_blocks, screening_tolerance, schwarz_bounds, density, active, fock,
          generated_fock_shell_class_mask);
    } else {
      launch_angular_fock_quartets<false, double>(
          stream, capacities, offsets, batch, active_tile_counts, active_tiles,
          generic_order5_tile_count, generic_order5_tiles, persistent_task_heads,
          persistent_worker_blocks, screening_tolerance, schwarz_bounds, density, active, fock,
          generated_fock_shell_class_mask);
    }
  }
}

}  // namespace generativeqc::scf::cuda_execution
