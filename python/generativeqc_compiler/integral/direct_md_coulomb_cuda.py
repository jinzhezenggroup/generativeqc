"""Pair-precontracted McMurchie-Davidson J using shared Direct Hermite algebra.

Density contraction and AO projection occur once per primitive pair, outside
quartet traversal. The initial inventory is bounded by the independently
qualified order-two Coulomb workspace, rather than method or molecule names.
Native code owns storage, lifetime, launch and prepared-provider selection.
"""

from __future__ import annotations

from .ir import KernelConsumer, build_integral_ir
from .production_cost import shell_class_index, shell_pair_class
from .shell_spec import FUSED_SHELL_SPECS, ShellClassSpec


def direct_md_coulomb_specs() -> tuple[ShellClassSpec, ...]:
    """Return exact value classes accepted by the shared order-two algebra."""
    return tuple(
        spec
        for spec in FUSED_SHELL_SPECS
        if build_integral_ir(spec, (KernelConsumer.FOCK,)).value_coulomb_order <= 2
    )


def emit_direct_md_coulomb_header() -> str:
    """Emit FP64 pair transforms and symmetric, bounded J-only traversal.

    Gaussian decay and signed primitive coefficients belong to the existing
    primitive-pair cache. Shared Wick expansions deliberately omit that decay,
    so it is applied exactly once in each pair transform. Ket Hermite parity
    belongs to the Coulomb interaction, including its reciprocal contribution.
    """
    specs = direct_md_coulomb_specs()
    mask = sum(1 << shell_class_index(spec) for spec in specs)
    pairs = tuple((i, j) for i in range(3) for j in range(3) if i + j <= 2)
    prepare_cases = "\n".join(
        f"    case {i * 3 + j}: direct_md_pair_transform<{i}, {j}, Prepare>("
        "batch, pair, density, density_hermite, potential_hermite, coulomb); break;"
        for i, j in pairs
    )
    launch_cases = []
    for spec in specs:
        first = shell_pair_class(*spec.angular[:2])
        second = shell_pair_class(*spec.angular[2:])
        high, low = max(first, second), min(first, second)
        # Pair-class ordering also defines the primitive-cache order consumed
        # by the stream. Hermite pair states are invariant to within-pair swaps.
        first_order = sum(spec.angular[:2])
        second_order = sum(spec.angular[2:])
        if first < second:
            first_order, second_order = second_order, first_order
        launch_cases.append(
            f"    case {shell_class_index(spec)}: "
            f"direct_md_j_stream<{high}, {low}, {first_order}, {second_order}>"
            "<<<workers, 32, 0, stream>>>(topology, primitive_pairs, primitive_offsets, "
            "density_hermite, potential_hermite, screening, head, work_count); break;"
        )
    return (
        r"""#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include "generated_direct_eri_order2.cuh"
#include "generated_direct_pair_order2.cuh"
#include "scf/generated_shell_task.hpp"

// Generated from direct_md_coulomb_cuda.py. Native code owns prepared storage.
namespace generativeqc::scf::cuda_execution {

inline constexpr unsigned kDirectMDHermiteWidth = 10U;
inline constexpr std::uint64_t kDirectMDCoulombMask = """
        + str(mask)
        + r"""ULL;

/** Compact total-degree ordering of the existing two-bit Hermite states. */
__device__ __forceinline__ unsigned direct_md_state(unsigned item) {
  switch (item) {
    case 0: return 0; case 1: return 1; case 2: return 4; case 3: return 16;
    case 4: return 2; case 5: return 5; case 6: return 17; case 7: return 8;
    case 8: return 20; default: return 32;
  }
}

__device__ __forceinline__ unsigned direct_md_index(unsigned state) {
  switch (state) {
    case 0: return 0; case 1: return 1; case 4: return 2; case 16: return 3;
    case 2: return 4; case 5: return 5; case 17: return 6; case 8: return 7;
    case 20: return 8; default: return 9;
  }
}

template <unsigned Order>
__host__ __device__ constexpr unsigned direct_md_width() {
  static_assert(Order <= 2);
  return (Order + 1) * (Order + 2) * (Order + 3) / 6;
}

/** Reuse canonical AO powers, angular normalization and cached signed weights.
 * Different shell pairs consume D_ij+D_ji; same-shell pairs consume the full
 * Cartesian block once. The reciprocal projection publishes both off-shell
 * orientations and never assumes a symmetric input density. */
template <unsigned First, unsigned Second, bool Prepare>
__device__ void direct_md_pair_transform(
    DeviceBatch batch, std::uint32_t pair, const double* density,
    double* density_hermite, double* potential_hermite, double* coulomb) {
  constexpr unsigned Order = First + Second;
  constexpr unsigned Width = direct_md_width<Order>();
  constexpr unsigned Terms = 1U << Order;
  const auto first_shell = batch.shell_pair_first[pair];
  const auto second_shell = batch.shell_pair_second[pair];
  const auto system = batch.shell_pair_systems[pair];
  const auto first_ao = batch.shell_direct_ao_offsets[first_shell];
  const auto second_ao = batch.shell_direct_ao_offsets[second_shell];
  const std::size_t base = static_cast<std::size_t>(system) * batch.direct_nbf;
  const std::size_t matrix_base = base * batch.direct_nbf;
  const Vec3<double> first = atom_position<double>(batch, batch.shell_atoms[first_shell], -1);
  const Vec3<double> second = atom_position<double>(batch, batch.shell_atoms[second_shell], -1);
  const auto begin = batch.shell_pair_primitive_offsets[pair];
  const auto end = batch.shell_pair_primitive_offsets[pair + 1U];
  for (auto primitive = begin + threadIdx.x; primitive < end; primitive += blockDim.x) {
    const auto geometry = batch.shell_primitive_pairs[primitive];
    double transformed[Width]{};
    const auto offset = static_cast<std::size_t>(primitive) * kDirectMDHermiteWidth;
    for (unsigned i = 0; i < order2_shell_component_count<First>(); ++i) {
      const auto angular_i = order2_shell_component<First>(i);
      const std::size_t ai = static_cast<std::size_t>(first_ao) + i;
      for (unsigned j = 0; j < order2_shell_component_count<Second>(); ++j) {
        const auto angular_j = order2_shell_component<Second>(j);
        const std::size_t aj = static_cast<std::size_t>(second_ao) + j;
        const std::size_t ij = matrix_base + (ai - base) * batch.direct_nbf + aj - base;
        const std::size_t ji = matrix_base + (aj - base) * batch.direct_nbf + ai - base;
        const double coefficient = geometry.weighted_coefficient *
            batch.direct_ao_coefficients[ai] * batch.direct_ao_coefficients[aj];
        const auto expansion = make_low_order_pair_expansion<First, Second>(
            geometry.exponent_sum, geometry.product_center, first, angular_i, second, angular_j);
        if constexpr (Prepare) {
          const double value = coefficient *
              (density[ij] + (first_shell == second_shell ? 0.0 : density[ji]));
          for (unsigned term = 0; term < Terms; ++term) {
            const auto item = expansion.terms[term];
            transformed[direct_md_index(item.derivative_state)] += value * item.coefficient;
          }
        } else {
          double value = 0.0;
          for (unsigned term = 0; term < Terms; ++term) {
            const auto item = expansion.terms[term];
            value += item.coefficient * potential_hermite[offset + direct_md_index(item.derivative_state)];
          }
          value *= coefficient;
          if (value != 0.0) {
            atomicAdd(coulomb + ij, value);
            if (first_shell != second_shell) atomicAdd(coulomb + ji, value);
          }
        }
      }
    }
    if constexpr (Prepare) {
      for (unsigned item = 0; item < Width; ++item) {
        density_hermite[offset + item] = transformed[item];
        potential_hermite[offset + item] = 0.0;
      }
    }
  }
}

template <bool Prepare>
__global__ void direct_md_pair_transform_kernel(
    DeviceBatch batch, const double* density, double* density_hermite,
    double* potential_hermite, double* coulomb, const std::uint8_t* active) {
  const std::uint32_t pair = blockIdx.x;
  if (pair >= batch.total_shell_pairs) return;
  const auto system = batch.shell_pair_systems[pair];
  if (active != nullptr && active[system] == 0U) return;
  const auto first = batch.shell_angular[batch.shell_pair_first[pair]];
  const auto second = batch.shell_angular[batch.shell_pair_second[pair]];
  if (first + second > 2U) return;
  switch (3U * first + second) {
"""
        + prepare_cases
        + r"""
  }
}

/** Density is already contracted in pair space. Only the Coulomb tensor and
 * its two pair contractions are evaluated inside the primitive-quartet loop.
 * Half weight for an identical shell pair accounts for reciprocal updates;
 * all primitive products remain present, including unequal signed weights. */
template <unsigned FirstOrder, unsigned SecondOrder>
__device__ void direct_md_contract_quartet(
    std::uint32_t first_pair, std::uint32_t second_pair,
    const PrimitivePairData* primitive_pairs, const std::int64_t* primitive_offsets,
    const double* density_hermite, double* potential_hermite) {
  constexpr unsigned FirstWidth = direct_md_width<FirstOrder>();
  constexpr unsigned SecondWidth = direct_md_width<SecondOrder>();
  constexpr unsigned Order = FirstOrder + SecondOrder;
  static_assert(Order <= 2);
  const double symmetry = first_pair == second_pair ? 0.5 : 1.0;
  for (auto first = primitive_offsets[first_pair]; first < primitive_offsets[first_pair + 1U]; ++first) {
    const auto p = primitive_pairs[first];
    const auto poffset = static_cast<std::size_t>(first) * kDirectMDHermiteWidth;
    double first_result[FirstWidth]{};
    for (auto second = primitive_offsets[second_pair]; second < primitive_offsets[second_pair + 1U]; ++second) {
      const auto q = primitive_pairs[second];
      const auto qoffset = static_cast<std::size_t>(second) * kDirectMDHermiteWidth;
      const double rho = p.exponent_sum * q.exponent_sum / (p.exponent_sum + q.exponent_sum);
      const Vec3<double> difference{p.product_center.x - q.product_center.x,
                                    p.product_center.y - q.product_center.y,
                                    p.product_center.z - q.product_center.z};
      double boys[3]{};
      boys_values<Order>(rho * (difference.x * difference.x + difference.y * difference.y +
                               difference.z * difference.z), boys);
      const double prefactor = symmetry * 2.0 * pow(kPi, 2.5) /
          (p.exponent_sum * q.exponent_sum * sqrt(p.exponent_sum + q.exponent_sum));
      double second_result[SecondWidth]{};
      for (unsigned i = 0; i < FirstWidth; ++i) {
        for (unsigned j = 0; j < SecondWidth; ++j) {
          const unsigned state = direct_md_state(j);
          const unsigned degree = (state & 3U) + ((state >> 2U) & 3U) + ((state >> 4U) & 3U);
          const double value = prefactor * ((degree & 1U) ? -1.0 : 1.0) *
              low_order_coulomb(direct_md_state(i) + state, rho, difference, boys);
          first_result[i] += value * density_hermite[qoffset + j];
          second_result[j] += value * density_hermite[poffset + i];
        }
      }
      for (unsigned j = 0; j < SecondWidth; ++j)
        if (second_result[j] != 0.0) atomicAdd(potential_hermite + qoffset + j, second_result[j]);
    }
    for (unsigned i = 0; i < FirstWidth; ++i)
      if (first_result[i] != 0.0) atomicAdd(potential_hermite + poffset + i, first_result[i]);
  }
}

/** Use the existing Schwarz-descending class topology and finite worker heads.
 * Equal-class streams admit each unordered shell pair product once. Density
 * transformations are outside traversal; no quartet list or AO ERI tensor is
 * materialized. Pure J retains the existing geometry-only screening contract. */
template <unsigned FirstClass, unsigned SecondClass, unsigned FirstOrder, unsigned SecondOrder>
__global__ __launch_bounds__(32) void direct_md_j_stream(
    const detail::GeneratedShellPairStream* topology_pointer,
    const PrimitivePairData* primitive_pairs, const std::int64_t* primitive_offsets,
    const double* density_hermite, double* potential_hermite, double screening,
    std::uint32_t* head, unsigned long long* work_count) {
  const auto& topology = *topology_pointer;
  const std::size_t stride = static_cast<std::size_t>(topology.batch_size) + 1U;
  const auto bra_begin = topology.pair_class_offsets[FirstClass * stride];
  const auto bra_end = topology.pair_class_offsets[FirstClass * stride + topology.batch_size];
  __shared__ std::uint32_t ordinal;
  while (true) {
    if (threadIdx.x == 0U) ordinal = atomicAdd(head, 1U);
    __syncthreads();
    if (ordinal >= bra_end - bra_begin) return;
    const auto bra = topology.pair_order[bra_begin + ordinal];
    const auto system = topology.shell_pair_systems[bra];
    if (topology.active != nullptr && topology.active[system] == 0U) continue;
    const auto ket_begin = topology.pair_class_offsets[SecondClass * stride + system];
    auto low = ket_begin;
    auto high = topology.pair_class_offsets[SecondClass * stride + system + 1U];
    const double bra_bound = topology.shell_pair_bounds[bra];
    while (low < high) {
      const auto middle = low + (high - low) / 2U;
      if (bra_bound * topology.shell_pair_bounds[topology.pair_order[middle]] < screening)
        high = middle;
      else low = middle + 1U;
    }
    for (auto ket_index = ket_begin + threadIdx.x; ket_index < low; ket_index += blockDim.x) {
      const auto ket = topology.pair_order[ket_index];
      if constexpr (FirstClass == SecondClass) if (ket > bra) continue;
      if (work_count != nullptr) atomicAdd(work_count, 1ULL);
      direct_md_contract_quartet<FirstOrder, SecondOrder>(bra, ket, primitive_pairs,
          primitive_offsets, density_hermite, potential_hermite);
    }
    __syncthreads();
  }
}

inline cudaError_t launch_direct_md_j_class(
    unsigned shell_class, unsigned workers, cudaStream_t stream,
    const detail::GeneratedShellPairStream* topology, const PrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_offsets, const double* density_hermite,
    double* potential_hermite, double screening, std::uint32_t* head,
    unsigned long long* work_count) {
  if (workers == 0U) return cudaSuccess;
  switch (shell_class) {
"""
        + "\n".join(launch_cases)
        + r"""
    default: return cudaErrorNotSupported;
  }
  return cudaPeekAtLastError();
}

}  // namespace generativeqc::scf::cuda_execution
"""
    )
