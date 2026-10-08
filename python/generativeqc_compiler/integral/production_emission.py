"""Emit deterministic CUDA production source from accepted kernel selections."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from generativeqc_compiler.common.cuda_target import cuda_target_info

from .capabilities import (
    CAPABILITY_LOCAL_PACKED_STREAMING_FOCK,
    CAPABILITY_MIXED_FOCK,
    CAPABILITY_STREAMING_FOCK,
)
from .cuda_emitter import emit_shell_class_fused_cuda
from .cuda_lowering import emit_ppps_resident_bra_rys3_cuda
from .cuda_schedule import ScheduleIR, ScheduleKind
from .fused_schedule import build_fused_shell_plan
from .ir import KernelConsumer
from .production_cost import _STABLE_AOT_SHARD_MAP_VERSION, shell_class_index
from .production_exchange_queue import exchange_streaming_worker
from .production_profile import _profile_identifier
from .production_registry import _stable_selection_order
from .production_selection import KernelSelection, _selection_integral
from .shell_spec import ShellClassSpec, shell_pair_class
from .signature import GeneratedKernelArgument, GeneratedKernelSignature

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .production_profile import ResolvedProductionProfile

_PRODUCTION_PRELUDE = r"""#include "scf/generated_shell_task.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <limits>

template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) values[order] = 0.0;
  // Switch low orders to stable upward recurrence before the generic cutoff;
  // this removes long alternating series from dominant s/p/d quartets.
  constexpr double series_threshold = MaximumOrder == 0 ? 1.0e-8
      : MaximumOrder == 1 ? 0.25
      : MaximumOrder == 2 ? 0.75
      : MaximumOrder == 3 ? 1.25
      : MaximumOrder == 4 ? 2.0
                          : 6.0;
  if (argument < series_threshold) {
    double term = 1.0;
    double sum = 0.0;
    for (unsigned k = 0; k < 80U; ++k) {
      sum += term /
          static_cast<double>(2U * MaximumOrder + 2U * k + 1U);
      term *= -argument / static_cast<double>(k + 1U);
      if (fabs(term) < 1.0e-18) break;
    }
    values[MaximumOrder] = sum;
    const double exponential = exp(-argument);
    for (unsigned order = MaximumOrder; order > 0U; --order) {
      values[order - 1U] =
          (2.0 * argument * values[order] + exponential) /
          static_cast<double>(2U * order - 1U);
    }
    return;
  }
  values[0] = 0.5 * sqrt(3.14159265358979323846 / argument) *
      erf(sqrt(argument));
  const double exponential = exp(-argument);
  for (unsigned order = 1; order <= MaximumOrder; ++order) {
    values[order] =
        ((2.0 * static_cast<double>(order) - 1.0) * values[order - 1U] -
         exponential) /
        (2.0 * argument);
  }
}

"""


def _launch_wrapper(
    spec: ShellClassSpec,
    symbol: str | None = None,
) -> str:
    """Emit a stable C ABI wrapper around one generated persistent kernel."""

    class_name = spec.name[0].upper() + spec.name[1:]
    return f"""
static_assert(sizeof(Generated{class_name}ShellTask) ==
              sizeof(generativeqc::scf::detail::GeneratedShellTask));
static_assert(alignof(Generated{class_name}ShellTask) ==
              alignof(generativeqc::scf::detail::GeneratedShellTask));
static_assert(offsetof(Generated{class_name}ShellTask, primitive_begin) ==
              offsetof(generativeqc::scf::detail::GeneratedShellTask, primitive_begin));
static_assert(offsetof(Generated{class_name}ShellTask, primitive_end) ==
              offsetof(generativeqc::scf::detail::GeneratedShellTask, primitive_end));
static_assert(offsetof(Generated{class_name}ShellTask, ao_begin) ==
              offsetof(generativeqc::scf::detail::GeneratedShellTask, ao_begin));
static_assert(offsetof(Generated{class_name}ShellTask, ao_coefficient_begin) ==
              offsetof(generativeqc::scf::detail::GeneratedShellTask,
                       ao_coefficient_begin));
static_assert(offsetof(Generated{class_name}ShellTask, density_offset) ==
              offsetof(generativeqc::scf::detail::GeneratedShellTask, density_offset));
static_assert(offsetof(Generated{class_name}ShellTask, spin_offset) ==
              offsetof(generativeqc::scf::detail::GeneratedShellTask, spin_offset));
static_assert(offsetof(Generated{class_name}ShellTask, matrix_order) ==
              offsetof(generativeqc::scf::detail::GeneratedShellTask, matrix_order));
static_assert(offsetof(Generated{class_name}ShellTask, shell_pair) ==
              offsetof(generativeqc::scf::detail::GeneratedShellTask, shell_pair));
static_assert(
    offsetof(Generated{class_name}ShellTask, reversed_shell_pair_mask) ==
    offsetof(generativeqc::scf::detail::GeneratedShellTask,
             reversed_shell_pair_mask));
static_assert(offsetof(Generated{class_name}ShellTask, shell) ==
              offsetof(generativeqc::scf::detail::GeneratedShellTask, shell));
static_assert(offsetof(Generated{class_name}ShellTask, atom) ==
              offsetof(generativeqc::scf::detail::GeneratedShellTask, atom));
static_assert(sizeof(Generated{class_name}PrimitivePairData) ==
              sizeof(generativeqc::scf::detail::GeneratedPrimitivePairData));
static_assert(alignof(Generated{class_name}PrimitivePairData) ==
              alignof(generativeqc::scf::detail::GeneratedPrimitivePairData));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, exponent_sum) ==
    offsetof(generativeqc::scf::detail::GeneratedPrimitivePairData, exponent_sum));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, reduced_exponent) ==
    offsetof(generativeqc::scf::detail::GeneratedPrimitivePairData, reduced_exponent));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, product_center) ==
    offsetof(generativeqc::scf::detail::GeneratedPrimitivePairData, product_center));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, weighted_coefficient) ==
    offsetof(generativeqc::scf::detail::GeneratedPrimitivePairData,
             weighted_coefficient));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, first_product_scale) ==
    offsetof(generativeqc::scf::detail::GeneratedPrimitivePairData,
             first_product_scale));
static_assert(
    offsetof(Generated{class_name}PrimitivePairData, second_product_scale) ==
    offsetof(generativeqc::scf::detail::GeneratedPrimitivePairData,
             second_product_scale));

extern "C" cudaError_t {symbol or f"generativeqc_launch_generated_{spec.name}"}(
    cudaStream_t stream, bool unrestricted, unsigned worker_blocks,
    const void* tasks, const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients,
    const void* atom_positions, double screening_tolerance,
    const double* schwarz_bounds, const double* density, double* forces,
    const std::uint32_t* task_count, std::uint32_t* task_head) {{
  if (worker_blocks == 0U) return cudaSuccess;
  const auto* typed_tasks =
      static_cast<const Generated{class_name}ShellTask*>(tasks);
  const auto* typed_positions =
      static_cast<const Generated{class_name}Vec3*>(atom_positions);
  const auto* typed_primitive_pairs =
      static_cast<const Generated{class_name}PrimitivePairData*>(
          primitive_pairs);
  if (unrestricted) {{
    generated_{spec.name}_shell_class_force_uhf_persistent_kernel<<<
        worker_blocks, kGenerated{class_name}BlockThreads, 0, stream>>>(
        typed_tasks, typed_primitive_pairs, primitive_pair_offsets,
        ao_coefficients, typed_positions, screening_tolerance, schwarz_bounds,
        density, forces, task_offset, task_count, task_head);
  }} else {{
    generated_{spec.name}_shell_class_force_rhf_persistent_kernel<<<
        worker_blocks, kGenerated{class_name}BlockThreads, 0, stream>>>(
        typed_tasks, typed_primitive_pairs, primitive_pair_offsets,
        ao_coefficients, typed_positions, screening_tolerance, schwarz_bounds,
        density, forces, task_offset, task_count, task_head);
  }}
  return cudaPeekAtLastError();
}}
"""


def _fock_launch_wrapper(
    spec: ShellClassSpec,
    symbol: str | None = None,
) -> str:
    """Emit the stable C ABI wrapper for one generated Fock worker."""

    class_name = spec.name[0].upper() + spec.name[1:]
    return f"""
extern "C" cudaError_t {symbol or f"generativeqc_launch_generated_{spec.name}_fock"}(
    cudaStream_t stream, bool unrestricted, unsigned worker_blocks,
    const void* tasks, const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients,
    const void* atom_positions, double screening_tolerance,
    const double* schwarz_bounds, const double* density, double* fock,
    const std::uint32_t* task_count, std::uint32_t* task_head) {{
  if (worker_blocks == 0U) return cudaSuccess;
  const auto* typed_tasks =
      static_cast<const Generated{class_name}ShellTask*>(tasks);
  const auto* typed_positions =
      static_cast<const Generated{class_name}Vec3*>(atom_positions);
  const auto* typed_primitive_pairs =
      static_cast<const Generated{class_name}PrimitivePairData*>(
          primitive_pairs);
  if (unrestricted) {{
    generated_{spec.name}_shell_class_fock_uhf_persistent_kernel<<<
        worker_blocks, kGenerated{class_name}FockBlockThreads, 0, stream>>>(
        typed_tasks, typed_primitive_pairs, primitive_pair_offsets,
        ao_coefficients, typed_positions, screening_tolerance, schwarz_bounds,
        density, fock, task_offset, task_count, task_head);
  }} else {{
    generated_{spec.name}_shell_class_fock_rhf_persistent_kernel<<<
        worker_blocks, kGenerated{class_name}FockBlockThreads, 0, stream>>>(
        typed_tasks, typed_primitive_pairs, primitive_pair_offsets,
        ao_coefficients, typed_positions, screening_tolerance, schwarz_bounds,
        density, fock, task_offset, task_count, task_head);
  }}
  return cudaPeekAtLastError();
}}
"""


def _mixed_fock_launch_wrapper(
    spec: ShellClassSpec,
    symbol: str | None = None,
) -> str:
    """Emit the stable C ABI wrapper for one generated mixed Fock worker."""

    class_name = spec.name[0].upper() + spec.name[1:]
    return f"""
extern "C" cudaError_t {symbol or f"generativeqc_launch_generated_{spec.name}_mixed_fock"}(
    cudaStream_t stream, bool unrestricted, unsigned worker_blocks,
    const void* tasks, const std::uint32_t* task_offset,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients,
    const void* atom_positions, double screening_tolerance,
    const double* schwarz_bounds, const double* density, double* fock,
    const std::uint32_t* task_count, std::uint32_t* task_head) {{
  if (worker_blocks == 0U) return cudaSuccess;
  const auto* typed_tasks =
      static_cast<const Generated{class_name}ShellTask*>(tasks);
  const auto* typed_positions =
      static_cast<const Generated{class_name}Vec3*>(atom_positions);
  const auto* typed_primitive_pairs =
      static_cast<const Generated{class_name}PrimitivePairData*>(
          primitive_pairs);
  if (unrestricted) {{
    generated_{spec.name}_shell_class_mixed_fock_uhf_persistent_kernel<<<
        worker_blocks, kGenerated{class_name}MixedFockBlockThreads, 0, stream>>>(
        typed_tasks, typed_primitive_pairs, primitive_pair_offsets,
        ao_coefficients, typed_positions, screening_tolerance, schwarz_bounds,
        density, fock, task_offset, task_count, task_head);
  }} else {{
    generated_{spec.name}_shell_class_mixed_fock_rhf_persistent_kernel<<<
        worker_blocks, kGenerated{class_name}MixedFockBlockThreads, 0, stream>>>(
        typed_tasks, typed_primitive_pairs, primitive_pair_offsets,
        ao_coefficients, typed_positions, screening_tolerance, schwarz_bounds,
        density, fock, task_offset, task_count, task_head);
  }}
  return cudaPeekAtLastError();
}}
"""


def _streaming_fock_schedule(selection: KernelSelection) -> ScheduleIR:
    """Resolve the value-path schedule used by streaming Fock lowering."""

    spec = selection.spec
    schedule = selection.fock_schedule or selection.schedule
    if (
        selection.fock_schedule is None
        and schedule.kind == ScheduleKind.SUBGROUP_TASKS
        and selection.recurrence in ("rys3", "rys4", "rys5")
    ):
        # Uniform subgroup warps are a force-only Rys experiment.  Mirror the
        # value emitter's established component-lane fallback exactly so the
        # streaming wrapper calls the Fock task with its real block geometry.
        angular_order = sum(spec.angular)
        value_state_count = (
            (angular_order + 1) * (angular_order + 2) * (angular_order + 3) // 6
        )
        block_threads = (max(spec.component_count, value_state_count) + 31) // 32 * 32
        return ScheduleIR(
            kind=ScheduleKind.COMPONENT_LANES,
            block_threads=block_threads,
            component_tile=spec.component_count,
            tasks_per_warp=1,
            shared_coulomb=True,
            pair_orientation=schedule.pair_orientation,
            pair_storage=schedule.pair_storage,
            unroll_pair_terms=schedule.unroll_pair_terms,
            minimum_blocks_per_sm=(
                2 if selection.recurrence in ("rys4", "rys5") else 0
            ),
            warp_size=schedule.warp_size,
        )
    return schedule


def _streaming_fock_internal_signature(
    selection: KernelSelection,
) -> GeneratedKernelSignature:
    """Derive the specialized compiler-owned streaming-kernel ABI.

    The public registry ABI remains stable. Only the generated device/kernel
    boundary is specialized, and launch forwarding is rendered from this same
    manifest so a removed parameter cannot remain in host-side packing.
    """

    spec = selection.spec
    class_name = spec.name[0].upper() + spec.name[1:]
    schedule = _streaming_fock_schedule(selection)
    first_pair_class = shell_pair_class(*spec.angular[:2])
    second_pair_class = shell_pair_class(*spec.angular[2:])
    head_name = (
        "task_head"
        if (
            schedule.kind == ScheduleKind.COMPONENT_LANES
            and first_pair_class != second_pair_class
        )
        else "bra_head"
    )
    signature = GeneratedKernelSignature(
        (
            GeneratedKernelArgument(
                "const generativeqc::scf::detail::GeneratedShellPairStream*",
                "topology_pointer",
                "topology",
            ),
            GeneratedKernelArgument(
                f"const Generated{class_name}PrimitivePairData*",
                "primitive_pairs",
                "typed_primitive_pairs",
            ),
            GeneratedKernelArgument("const std::int64_t*", "primitive_pair_offsets"),
            GeneratedKernelArgument("const double*", "ao_coefficients"),
            GeneratedKernelArgument(
                f"const Generated{class_name}Vec3*",
                "atom_positions",
                "typed_positions",
            ),
            GeneratedKernelArgument("double", "screening_tolerance"),
            GeneratedKernelArgument("bool", "mixed_precision_enabled"),
            GeneratedKernelArgument("double", "fp64_threshold"),
            GeneratedKernelArgument("const double*", "schwarz_bounds"),
            GeneratedKernelArgument("const double*", "density"),
            GeneratedKernelArgument("double*", "fock"),
            GeneratedKernelArgument("std::uint32_t*", head_name, "bra_head"),
            GeneratedKernelArgument("unsigned long long*", "fp64_work_count"),
            GeneratedKernelArgument("unsigned long long*", "fp32_work_count"),
        )
    )
    if selection.has_capability(CAPABILITY_MIXED_FOCK):
        return signature
    return signature.without(
        "mixed_precision_enabled",
        "fp64_threshold",
        "fp32_work_count",
    )


def _streaming_fock_source(
    selection: KernelSelection, *, include_work_buckets: bool = True
) -> str:
    """Emit fixed-storage shell-pair enumeration for dominant Fock classes."""

    if not selection.has_capability(CAPABILITY_STREAMING_FOCK):
        return ""

    spec = selection.spec
    schedule = _streaming_fock_schedule(selection)

    class_name = spec.name[0].upper() + spec.name[1:]
    first_pair_class = shell_pair_class(*spec.angular[:2])
    second_pair_class = shell_pair_class(*spec.angular[2:])
    high_pair_class = max(first_pair_class, second_pair_class)
    low_pair_class = min(first_pair_class, second_pair_class)
    shell_class = shell_class_index(selection.spec)
    first_angular, second_angular, third_angular, fourth_angular = spec.angular
    density_pair_classes = tuple(
        dict.fromkeys(
            (
                first_pair_class,
                second_pair_class,
                shell_pair_class(first_angular, third_angular),
                shell_pair_class(first_angular, fourth_angular),
                shell_pair_class(second_angular, third_angular),
                shell_pair_class(second_angular, fourth_angular),
            )
        )
    )

    def pair_class_density_bound(pair_classes: tuple[int, ...]) -> str:
        bound = (
            "topology.system_pair_density_bounds["
            "static_cast<std::size_t>(system) * 10U + "
            f"{pair_classes[0]}U]"
        )
        for pair_class in pair_classes[1:]:
            bound = (
                f"fmax({bound}, "
                "topology.system_pair_density_bounds["
                "static_cast<std::size_t>(system) * 10U + "
                f"{pair_class}U])"
            )
        return bound

    system_density_bound = pair_class_density_bound(density_pair_classes)
    coulomb_density_pair_classes = tuple(
        dict.fromkeys((first_pair_class, second_pair_class))
    )
    coulomb_system_density_bound = pair_class_density_bound(
        coulomb_density_pair_classes
    )
    system_density_bound = (
        "(topology.system_pair_density_bounds == nullptr ? 1.0 : "
        "(topology.fock_consumer == "
        "generativeqc::scf::detail::GeneratedFockConsumer::Coulomb"
        f" ? {coulomb_system_density_bound} : {system_density_bound}))"
    )
    prefix = f"generated_{spec.name}"
    supports_mixed_fock = selection.has_capability(CAPABILITY_MIXED_FOCK)
    internal_signature = _streaming_fock_internal_signature(selection)
    internal_parameters = internal_signature.parameter_list()
    internal_arguments = internal_signature.argument_list()
    retained_state = (
        "mixed_precision_enabled && contribution_bound < fp64_threshold ? 3U : 1U"
        if supports_mixed_fock
        else "1U"
    )
    precision_parameters = (
        """    std::uint32_t state, unsigned long long* fp64_work_count,
    unsigned long long* fp32_work_count"""
        if supports_mixed_fock
        else "    unsigned long long* fp64_work_count"
    )
    precision_body = (
        """  unsigned long long* counter =
      state == 3U ? fp32_work_count : fp64_work_count;
  if (counter != nullptr) atomicAdd(counter, 1ULL);"""
        if supports_mixed_fock
        else "  if (fp64_work_count != nullptr) atomicAdd(fp64_work_count, 1ULL);"
    )

    def record_precision(state: str) -> str:
        arguments = (
            f"{state}, fp64_work_count, fp32_work_count"
            if supports_mixed_fock
            else "fp64_work_count"
        )
        return f"{prefix}_record_fock_precision({arguments});"

    common = f"""
/** Return the packed shell-pair ordinal for two shells in one system. */
__device__ __forceinline__ std::size_t {prefix}_stream_pair_index(
    const generativeqc::scf::detail::GeneratedShellPairStream& topology,
    std::int32_t system, std::int32_t first_shell,
    std::int32_t second_shell) {{
  const std::size_t shell_begin = static_cast<std::size_t>(
      topology.system_shell_offsets[system]);
  const std::size_t first = static_cast<std::size_t>(first_shell) - shell_begin;
  const std::size_t second =
      static_cast<std::size_t>(second_shell) - shell_begin;
  const std::size_t high = first > second ? first : second;
  const std::size_t low = first > second ? second : first;
  return static_cast<std::size_t>(topology.system_shell_pair_offsets[system]) +
      high * (high + 1U) / 2U + low;
}}

/** Apply the exact bounded RHF/UHF Fock screening predicate. */
template <bool Unrestricted>
__device__ __forceinline__ bool {prefix}_stream_survives(
    const generativeqc::scf::detail::GeneratedShellPairStream& topology,
    std::uint32_t first_pair, std::uint32_t second_pair,
    double screening_tolerance, double* contribution_bound) {{
  const double quartet_bound = topology.shell_pair_bounds[first_pair] *
      topology.shell_pair_bounds[second_pair];
  if (quartet_bound < screening_tolerance) return false;
  const std::int32_t system = topology.shell_pair_systems[first_pair];
  if (topology.active != nullptr && topology.active[system] == 0U) return false;
  // Optional/manual topologies without density bounds retain the legacy
  // Schwarz-only contract. Production generated-J owners publish these bounds.
  if (topology.shell_pair_density_bounds == nullptr) {{
    if (contribution_bound != nullptr) *contribution_bound = quartet_bound;
    return true;
  }}
  const auto ab = topology.shell_pair_density_bounds[first_pair];
  const auto cd = topology.shell_pair_density_bounds[second_pair];
  if (topology.fock_consumer == generativeqc::scf::detail::GeneratedFockConsumer::Coulomb) {{
    const double contribution =
        quartet_bound * fmax(ab.coulomb, cd.coulomb);
    if (contribution_bound != nullptr) *contribution_bound = contribution;
    return contribution >= screening_tolerance;
  }}
  const std::int32_t first_shell = topology.shell_pair_first[first_pair];
  const std::int32_t second_shell = topology.shell_pair_second[first_pair];
  const std::int32_t third_shell = topology.shell_pair_first[second_pair];
  const std::int32_t fourth_shell = topology.shell_pair_second[second_pair];
  const std::size_t ac_pair = {prefix}_stream_pair_index(
      topology, system, first_shell, third_shell);
  const std::size_t ad_pair = {prefix}_stream_pair_index(
      topology, system, first_shell, fourth_shell);
  const std::size_t bc_pair = {prefix}_stream_pair_index(
      topology, system, second_shell, third_shell);
  const std::size_t bd_pair = {prefix}_stream_pair_index(
      topology, system, second_shell, fourth_shell);
  const auto ac = topology.shell_pair_density_bounds[ac_pair];
  const auto ad = topology.shell_pair_density_bounds[ad_pair];
  const auto bc = topology.shell_pair_density_bounds[bc_pair];
  const auto bd = topology.shell_pair_density_bounds[bd_pair];
  const bool exchange_only =
      topology.fock_consumer ==
      generativeqc::scf::detail::GeneratedFockConsumer::Exchange ||
      topology.fock_consumer ==
      generativeqc::scf::detail::GeneratedFockConsumer::HartreeFockExchange;
  double density_bound = exchange_only ? 0.0 : fmax(ab.coulomb, cd.coulomb);
  if constexpr (Unrestricted) {{
    const double exchange_bound = fmax(
        fmax(fmax(ac.exchange_alpha, ac.exchange_beta),
             fmax(ad.exchange_alpha, ad.exchange_beta)),
        fmax(fmax(bc.exchange_alpha, bc.exchange_beta),
             fmax(bd.exchange_alpha, bd.exchange_beta)));
    density_bound =
        exchange_only ? exchange_bound : fmax(density_bound, exchange_bound);
  }} else {{
    const double exchange_bound = fmax(
        fmax(ac.exchange_alpha, ad.exchange_alpha),
        fmax(bc.exchange_alpha, bd.exchange_alpha));
    density_bound = exchange_only
        ? exchange_bound
        : fmax(density_bound, 0.5 * exchange_bound);
  }}
  const double contribution = quartet_bound * density_bound;
  if (contribution_bound != nullptr) *contribution_bound = contribution;
  return contribution >= screening_tolerance;
}}

/** Find the first coarse-screened ket in a Schwarz-descending class segment. */
__device__ __forceinline__ std::uint32_t {prefix}_stream_coarse_ket_end(
    const generativeqc::scf::detail::GeneratedShellPairStream& topology,
    std::uint32_t bra_pair, std::uint32_t ket_begin, std::uint32_t ket_end,
    double system_density_bound, double screening_tolerance) {{
  const std::int32_t system = topology.shell_pair_systems[bra_pair];
  if (topology.active != nullptr && topology.active[system] == 0U) return ket_begin;
  const double bra_bound = topology.shell_pair_bounds[bra_pair];
  std::uint32_t low = ket_begin;
  std::uint32_t high = ket_end;
  while (low < high) {{
    const std::uint32_t middle = low + (high - low) / 2U;
    const std::uint32_t ket_pair = topology.pair_order[middle];
    const bool past_tail =
        bra_bound * topology.shell_pair_bounds[ket_pair] *
            system_density_bound < screening_tolerance;
    if (past_tail) {{
      high = middle;
    }} else {{
      low = middle + 1U;
    }}
  }}
  return low;
}}

/** Count the arithmetic route actually selected for one retained quartet. */
__device__ __forceinline__ void {prefix}_record_fock_precision(
{precision_parameters}) {{
{precision_body}
}}

/** Canonicalize one pair product into the stable generated task ABI. */
__device__ __forceinline__ void {prefix}_stream_populate_task(
    const generativeqc::scf::detail::GeneratedShellPairStream& topology,
    std::uint32_t first_pair, std::uint32_t second_pair,
    Generated{class_name}ShellTask& task) {{
  std::int32_t shells[4] = {{
      topology.shell_pair_first[first_pair],
      topology.shell_pair_second[first_pair],
      topology.shell_pair_first[second_pair],
      topology.shell_pair_second[second_pair],
  }};
  std::uint32_t shell_pairs[2] = {{first_pair, second_pair}};
  std::uint32_t reversed_mask = 0U;
  if (topology.shell_angular[shells[0]] <
      topology.shell_angular[shells[1]]) {{
    const std::int32_t swap = shells[0];
    shells[0] = shells[1];
    shells[1] = swap;
    reversed_mask |= 1U;
  }}
  if (topology.shell_angular[shells[2]] <
      topology.shell_angular[shells[3]]) {{
    const std::int32_t swap = shells[2];
    shells[2] = shells[3];
    shells[3] = swap;
    reversed_mask |= 2U;
  }}
  const unsigned first_class =
      static_cast<unsigned>(topology.shell_angular[shells[0]]) *
          (static_cast<unsigned>(topology.shell_angular[shells[0]]) + 1U) /
          2U +
      static_cast<unsigned>(topology.shell_angular[shells[1]]);
  const unsigned second_class =
      static_cast<unsigned>(topology.shell_angular[shells[2]]) *
          (static_cast<unsigned>(topology.shell_angular[shells[2]]) + 1U) /
          2U +
      static_cast<unsigned>(topology.shell_angular[shells[3]]);
  if (first_class < second_class) {{
    const std::int32_t first = shells[0];
    const std::int32_t second = shells[1];
    shells[0] = shells[2];
    shells[1] = shells[3];
    shells[2] = first;
    shells[3] = second;
    const std::uint32_t pair_swap = shell_pairs[0];
    shell_pairs[0] = shell_pairs[1];
    shell_pairs[1] = pair_swap;
    reversed_mask = ((reversed_mask & 1U) << 1U) |
        ((reversed_mask & 2U) >> 1U);
  }}
  const std::int32_t system = topology.shell_pair_systems[first_pair];
  const std::size_t matrix_order = topology.matrix_order;
  const std::size_t matrix_size = matrix_order * matrix_order;
  const std::size_t system_ao_begin =
      static_cast<std::size_t>(system) * matrix_order;
#pragma unroll
  for (unsigned center = 0U; center < 4U; ++center) {{
    const std::int32_t shell = shells[center];
    task.primitive_begin[center] = static_cast<std::uint64_t>(
        topology.shell_primitive_offsets[shell]);
    task.primitive_end[center] = static_cast<std::uint64_t>(
        topology.shell_primitive_offsets[shell + 1]);
    const std::size_t ao_begin = static_cast<std::size_t>(
        topology.shell_direct_ao_offsets[shell]);
    task.ao_begin[center] = static_cast<std::uint64_t>(
        ao_begin - system_ao_begin);
    task.ao_coefficient_begin[center] =
        static_cast<std::uint64_t>(ao_begin);
    task.shell[center] = static_cast<std::uint32_t>(shell);
    task.atom[center] = static_cast<std::uint32_t>(
        topology.shell_atoms[shell]);
  }}
  task.density_offset = static_cast<std::uint64_t>(
      static_cast<std::size_t>(system) * matrix_size);
  task.spin_offset = static_cast<std::uint64_t>(
      static_cast<std::size_t>(system) * 2U * matrix_size);
  task.matrix_order = topology.matrix_order;
  task.shell_pair[0] = shell_pairs[0];
  task.shell_pair[1] = shell_pairs[1];
  const std::uint32_t consumer_bit =
      topology.fock_consumer ==
              generativeqc::scf::detail::GeneratedFockConsumer::Coulomb
          ? kGenerated{class_name}CoulombConsumerBit
          : (topology.fock_consumer ==
                     generativeqc::scf::detail::GeneratedFockConsumer::Exchange
                 ? kGenerated{class_name}ExchangeConsumerBit
                 : (topology.fock_consumer ==
                            generativeqc::scf::detail::GeneratedFockConsumer::HartreeFockExchange
                        ? kGenerated{class_name}ExchangeConsumerBit |
                              kGenerated{class_name}CoulombConsumerBit
                        : 0U));
  task.reversed_shell_pair_mask = reversed_mask | consumer_bit;
}}
"""

    if schedule.kind in (ScheduleKind.PACKED_TASKS, ScheduleKind.SUBGROUP_TASKS):
        packed = schedule.kind == ScheduleKind.PACKED_TASKS
        worker_options = {
            "prefix": prefix,
            "class_name": class_name,
            "internal_parameters": internal_parameters,
            "shell_class": shell_class,
            "high_pair_class": high_pair_class,
            "low_pair_class": low_pair_class,
            "system_density_bound": system_density_bound,
            "block_threads": schedule.block_threads,
            "width": 32 if packed else schedule.tasks_per_block,
            "packed": packed,
            "local_lane_state": selection.has_capability(
                CAPABILITY_LOCAL_PACKED_STREAMING_FOCK
            ),
            "supports_mixed_fock": supports_mixed_fock,
            "retained_state": retained_state,
            "record_precision": record_precision,
        }
        worker = exchange_streaming_worker(**worker_options)
    elif (
        schedule.kind == ScheduleKind.COMPONENT_LANES
        and high_pair_class != low_pair_class
    ):
        # A component-lane CTA already spends the full block on one shell
        # quartet. Claim pair products directly instead of pinning one CTA to
        # a bra and serially draining its entire ket segment. This exposes the
        # same two-dimensional shell-pair concurrency used by GPU4PySCF while
        # retaining the accepted per-quartet Rys/component implementation.
        same_pair_class = str(high_pair_class == low_pair_class).lower()
        worker = f"""
template <bool Unrestricted>
__device__ __forceinline__ void {prefix}_streaming_fock(
{internal_parameters}) {{
  static_assert(kGenerated{class_name}FockBlockThreads ==
                {schedule.block_threads}U);
  __shared__ Generated{class_name}ShellTask stream_task[1];
  __shared__ std::uint32_t candidate_ordinal;
  __shared__ std::uint32_t stream_state;
  const auto& topology = *topology_pointer;
  if (topology.generated_overflow != nullptr &&
      topology.generated_overflow[{shell_class}U] == 0U) return;
  const std::size_t stride = static_cast<std::size_t>(topology.batch_size) + 1U;
  while (true) {{
    if (threadIdx.x == 0U) {{
      candidate_ordinal = atomicAdd(task_head, 1U);
      std::uint64_t remaining = candidate_ordinal;
      // State 2 means the global pair-product domain is exhausted, 1 is an
      // exact retained quartet, and 0 is a screened candidate.
      stream_state = 2U;
      for (std::int32_t system = 0; system < topology.batch_size; ++system) {{
        const std::uint32_t high_begin = topology.pair_class_offsets[
            {high_pair_class}U * stride + static_cast<std::size_t>(system)];
        const std::uint32_t high_end = topology.pair_class_offsets[
            {high_pair_class}U * stride + static_cast<std::size_t>(system) + 1U];
        const std::uint32_t low_begin = topology.pair_class_offsets[
            {low_pair_class}U * stride + static_cast<std::size_t>(system)];
        const std::uint32_t low_end = topology.pair_class_offsets[
            {low_pair_class}U * stride + static_cast<std::size_t>(system) + 1U];
        const std::uint64_t high_count = high_end - high_begin;
        const std::uint64_t low_count = low_end - low_begin;
        const std::uint64_t system_candidates = {same_pair_class}
            ? high_count * (high_count + 1U) / 2U
            : high_count * low_count;
        if (remaining >= system_candidates) {{
          remaining -= system_candidates;
          continue;
        }}

        std::uint64_t high_local = 0U;
        std::uint64_t low_local = 0U;
        if constexpr ({same_pair_class}) {{
          while ((high_local + 1U) * (high_local + 2U) / 2U <= remaining) {{
            ++high_local;
          }}
          low_local = remaining - high_local * (high_local + 1U) / 2U;
        }} else {{
          high_local = remaining / low_count;
          low_local = remaining - high_local * low_count;
        }}
        const std::uint32_t bra_pair = topology.pair_order[
            high_begin + static_cast<std::uint32_t>(high_local)];
        const std::uint32_t ket_pair = topology.pair_order[
            low_begin + static_cast<std::uint32_t>(low_local)];
        const bool past_schwarz_tail =
            topology.shell_pair_bounds[bra_pair] *
                topology.shell_pair_bounds[ket_pair] *
                {system_density_bound} < screening_tolerance;
        double contribution_bound = 0.0;
        const bool keep = !past_schwarz_tail &&
            {prefix}_stream_survives<Unrestricted>(
                topology, bra_pair, ket_pair, screening_tolerance,
                &contribution_bound);
        stream_state = keep ? {retained_state} : 0U;
        // This cursor flattens all bra/ket products and then all systems.
        // A screened ket tail is local to this bra; later bras/systems can
        // still survive. Only domain exhaustion may retire this worker.
        if (keep) {{
          {record_precision("stream_state")}
          {prefix}_stream_populate_task(
              topology, bra_pair, ket_pair, stream_task[0]);
        }}
        break;
      }}
    }}
    __syncthreads();
    if (stream_state == 2U) return;
    if (stream_state == 1U) {{
      {prefix}_shell_class_fock_task<Unrestricted>(
          stream_task, primitive_pairs, primitive_pair_offsets,
          ao_coefficients, atom_positions, screening_tolerance,
          schwarz_bounds, density, fock, 0U);
    }}
    {
            f'''if (stream_state == 3U) {{
      {prefix}_shell_class_mixed_fock_task<Unrestricted>(
          stream_task, primitive_pairs, primitive_pair_offsets,
          ao_coefficients, atom_positions, screening_tolerance,
          schwarz_bounds, density, fock, 0U);
    }}'''
            if supports_mixed_fock
            else ""
        }
    __syncthreads();
  }}
}}
"""
    else:
        worker = f"""
template <bool Unrestricted>
__device__ __forceinline__ void {prefix}_streaming_fock(
{internal_parameters}) {{
  static_assert(kGenerated{class_name}FockBlockThreads ==
                {schedule.block_threads}U);
  __shared__ Generated{class_name}ShellTask stream_task[1];
  __shared__ std::uint32_t stream_state;
  __shared__ std::uint32_t bra_ordinal;
  const auto& topology = *topology_pointer;
  if (topology.generated_overflow != nullptr &&
      topology.generated_overflow[{shell_class}U] == 0U) return;
  const std::size_t stride = static_cast<std::size_t>(topology.batch_size) + 1U;
  const std::uint32_t bra_begin = topology.pair_class_offsets[
      {high_pair_class}U * stride];
  const std::uint32_t bra_end = topology.pair_class_offsets[
      {high_pair_class}U * stride + topology.batch_size];
  while (true) {{
    if (threadIdx.x == 0U) bra_ordinal = atomicAdd(bra_head, 1U);
    __syncthreads();
    if (bra_ordinal >= bra_end - bra_begin) return;
    const std::uint32_t bra_pair =
        topology.pair_order[bra_begin + bra_ordinal];
    const std::int32_t system = topology.shell_pair_systems[bra_pair];
    const std::uint32_t ket_begin = topology.pair_class_offsets[
        {low_pair_class}U * stride + system];
    const std::uint32_t ket_end = topology.pair_class_offsets[
        {low_pair_class}U * stride + system + 1U];
    const double system_density_bound = {system_density_bound};
    for (std::uint32_t ket_ordinal = ket_begin; ket_ordinal < ket_end;
         ++ket_ordinal) {{
      if (threadIdx.x == 0U) {{
        const std::uint32_t ket_pair = topology.pair_order[ket_ordinal];
        const bool past_schwarz_tail =
            topology.shell_pair_bounds[bra_pair] *
                topology.shell_pair_bounds[ket_pair] *
                system_density_bound < screening_tolerance;
        bool keep = !past_schwarz_tail;
        if (keep && {str(high_pair_class == low_pair_class).lower()}) {{
          keep = bra_pair >= ket_pair;
        }}
        double contribution_bound = 0.0;
        if (keep) {{
          keep = {prefix}_stream_survives<Unrestricted>(
              topology, bra_pair, ket_pair, screening_tolerance,
              &contribution_bound);
        }}
        stream_state = past_schwarz_tail ? 2U : (keep ? {retained_state} : 0U);
        if (keep) {{
          {record_precision("stream_state")}
          {prefix}_stream_populate_task(
              topology, bra_pair, ket_pair, stream_task[0]);
        }}
      }}
      __syncthreads();
      if (stream_state == 2U) break;
      if (stream_state == 1U) {{
        {prefix}_shell_class_fock_task<Unrestricted>(
            stream_task, primitive_pairs, primitive_pair_offsets,
            ao_coefficients, atom_positions, screening_tolerance,
            schwarz_bounds, density, fock, 0U);
      }}
      {
            f'''if (stream_state == 3U) {{
        {prefix}_shell_class_mixed_fock_task<Unrestricted>(
            stream_task, primitive_pairs, primitive_pair_offsets,
            ao_coefficients, atom_positions, screening_tolerance,
            schwarz_bounds, density, fock, 0U);
      }}'''
            if supports_mixed_fock
            else ""
        }
      __syncthreads();
    }}
  }}
}}
"""

    kernels = f"""
extern "C" __global__ __launch_bounds__(kGenerated{class_name}FockBlockThreads)
void {prefix}_shell_class_fock_rhf_streaming_kernel(
{internal_parameters}) {{
  {prefix}_streaming_fock<false>(
      {internal_arguments});
}}

extern "C" __global__ __launch_bounds__(kGenerated{class_name}FockBlockThreads)
void {prefix}_shell_class_fock_uhf_streaming_kernel(
{internal_parameters}) {{
  {prefix}_streaming_fock<true>(
      {internal_arguments});
}}
"""
    if include_work_buckets and schedule.kind in (
        ScheduleKind.PACKED_TASKS,
        ScheduleKind.SUBGROUP_TASKS,
    ):
        work_worker = exchange_streaming_worker(**worker_options, work_aware=True)
        work_worker = work_worker.replace(
            f"void {prefix}_streaming_fock(", f"void {prefix}_work_streaming_fock("
        )
        work_kernels = kernels.replace(
            "_streaming_kernel(", "_work_streaming_kernel("
        ).replace(f"{prefix}_streaming_fock<", f"{prefix}_work_streaming_fock<")
        return common + worker + kernels + work_worker + work_kernels
    return common + worker + kernels


def _streaming_fock_launch_wrapper(
    selection: KernelSelection,
    symbol: str | None = None,
    *,
    work_aware: bool = False,
) -> str:
    """Emit the stable host wrapper adapting to a specialized internal ABI."""

    spec = selection.spec
    class_name = spec.name[0].upper() + spec.name[1:]
    kernel_infix = "work_" if work_aware else ""
    internal_arguments = _streaming_fock_internal_signature(selection).argument_list(
        wrapper=True
    )
    return f"""
extern "C" cudaError_t {symbol or f"generativeqc_launch_generated_{spec.name}_streaming_fock"}(
    cudaStream_t stream, bool unrestricted, unsigned worker_blocks,
    const void* shell_pair_stream,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, bool mixed_precision_enabled,
    double fp64_threshold, const double* schwarz_bounds,
    const double* density, double* fock, std::uint32_t* bra_head,
    unsigned long long* fp64_work_count,
    unsigned long long* fp32_work_count) {{
  if (worker_blocks == 0U) return cudaSuccess;
  const auto* topology = static_cast<
      const generativeqc::scf::detail::GeneratedShellPairStream*>(
          shell_pair_stream);
  const auto* typed_positions =
      static_cast<const Generated{class_name}Vec3*>(atom_positions);
  const auto* typed_primitive_pairs =
      static_cast<const Generated{class_name}PrimitivePairData*>(
          primitive_pairs);
  if (unrestricted) {{
    generated_{spec.name}_shell_class_fock_uhf_{kernel_infix}streaming_kernel<<<
        worker_blocks, kGenerated{class_name}FockBlockThreads, 0, stream>>>(
        {internal_arguments});
  }} else {{
    generated_{spec.name}_shell_class_fock_rhf_{kernel_infix}streaming_kernel<<<
        worker_blocks, kGenerated{class_name}FockBlockThreads, 0, stream>>>(
        {internal_arguments});
  }}
  return cudaPeekAtLastError();
}}
"""


def _ppps_resident_launch_wrapper(symbol: str | None = None) -> str:
    """Emit the stable opaque-pointer C ABI for resident ppps force work.

    The generated CUDA source owns a profile-scoped descriptor type so that
    multiple architecture shards can coexist in one fat binary.  The host
    descriptor lives in ``generated_shell_task.hpp``; these size/alignment and
    field-offset assertions make a mismatch fail at AOT compilation instead
    of silently corrupting a resident launch.
    """

    return f"""
static_assert(sizeof(GeneratedPppsResidentTask) ==
              sizeof(generativeqc::scf::detail::GeneratedPppsResidentTask));
static_assert(alignof(GeneratedPppsResidentTask) ==
              alignof(generativeqc::scf::detail::GeneratedPppsResidentTask));
static_assert(offsetof(GeneratedPppsResidentTask, bra_pair) ==
              offsetof(generativeqc::scf::detail::GeneratedPppsResidentTask,
                       bra_pair));
static_assert(offsetof(GeneratedPppsResidentTask, ket_begin) ==
              offsetof(generativeqc::scf::detail::GeneratedPppsResidentTask,
                       ket_begin));
static_assert(offsetof(GeneratedPppsResidentTask, ket_count) ==
              offsetof(generativeqc::scf::detail::GeneratedPppsResidentTask,
                       ket_count));
static_assert(sizeof(GeneratedPppsPrimitivePairData) ==
              sizeof(generativeqc::scf::detail::GeneratedPrimitivePairData));
static_assert(alignof(GeneratedPppsPrimitivePairData) ==
              alignof(generativeqc::scf::detail::GeneratedPrimitivePairData));

extern "C" cudaError_t {symbol or "generativeqc_launch_ppps_resident"}(
    cudaStream_t stream, bool unrestricted, const void* resident_tasks,
    const void* ket_tasks,
    const std::int64_t* primitive_pair_offsets, const void* primitive_pairs,
    const double* ao_coefficients, const void* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* forces, unsigned block_threads,
    std::size_t task_count) {{
  if (task_count == 0U) return cudaSuccess;
  if (block_threads != 32U && block_threads != 64U &&
      block_threads != 128U && block_threads != 256U) {{
    return cudaErrorInvalidValue;
  }}
  if (task_count > static_cast<std::size_t>(
          std::numeric_limits<unsigned>::max())) return cudaErrorInvalidValue;
  const auto* typed_resident_tasks =
      static_cast<const GeneratedPppsResidentTask*>(resident_tasks);
  const auto* typed_ket_tasks =
      static_cast<const GeneratedPppsShellTask*>(ket_tasks);
  const auto* typed_positions =
      static_cast<const GeneratedPppsVec3*>(atom_positions);
  const auto* typed_primitive_pairs =
      static_cast<const GeneratedPppsPrimitivePairData*>(primitive_pairs);
  if (unrestricted) {{
    generated_ppps_resident_bra_force_uhf_kernel<<<
        static_cast<unsigned>(task_count),
        block_threads, 0, stream>>>(
        typed_resident_tasks, typed_ket_tasks, typed_primitive_pairs,
        primitive_pair_offsets, ao_coefficients, typed_positions,
        screening_tolerance, schwarz_bounds, density, forces, task_count);
  }} else {{
    generated_ppps_resident_bra_force_rhf_kernel<<<
        static_cast<unsigned>(task_count),
        block_threads, 0, stream>>>(
        typed_resident_tasks, typed_ket_tasks, typed_primitive_pairs,
        primitive_pair_offsets, ao_coefficients, typed_positions,
        screening_tolerance, schwarz_bounds, density, forces, task_count);
  }}
  return cudaPeekAtLastError();
}}
"""


def _emit_ppps_resident_source(selection: KernelSelection) -> str:
    """Emit only the resident ppps tail for an opted-in production row."""

    if selection.resident_force_recurrence is None:
        return ""
    # The ordinary ppps source is emitted immediately before this tail.  A
    # A subset/Wick row needs the resident Rys evaluator appended.  The scalar
    # thread-task Rys3 force worker owns that exact symbol already, while the
    # uniform-warp worker deliberately uses a schedule-qualified root symbol
    # and therefore still needs the resident evaluator beside it.
    ordinary_owns_resident_roots = (
        selection.recurrence == "rys3"
        and selection.schedule.kind == ScheduleKind.THREAD_TASKS
    )
    resident_integral = _selection_integral(selection, recurrence="rys3")
    return emit_ppps_resident_bra_rys3_cuda(
        include_shared_definitions=False,
        include_rys3_roots=not ordinary_owns_resident_roots,
        integral=resident_integral,
    )


def emit_production_shard(
    specifications: Iterable[ShellClassSpec | KernelSelection],
) -> str:
    """Emit one CUDA TU containing a deterministic subset of accepted classes."""

    selections = _stable_selection_order(specifications)
    body = [
        f"// Stable AOT shard map version: {_STABLE_AOT_SHARD_MAP_VERSION}\n",
        _PRODUCTION_PRELUDE,
    ]
    for selection in selections:
        integral = _selection_integral(selection)
        plan = build_fused_shell_plan(
            selection.spec,
            integral=integral,
            schedule=selection.schedule,
            target=cuda_target_info(selection.architecture),
        )
        body.append(
            emit_shell_class_fused_cuda(
                selection.spec,
                plan,
                fock_schedule=selection.fock_schedule,
                capabilities=selection.capabilities,
            )
        )
        if (
            KernelConsumer.FORCE in integral.consumers
            or not integral.recurrence.startswith("rys")
        ):
            body.append(_launch_wrapper(selection.spec))
        if KernelConsumer.FOCK in selection.consumers:
            body.append(_fock_launch_wrapper(selection.spec))
            if selection.has_capability(CAPABILITY_MIXED_FOCK):
                body.append(_mixed_fock_launch_wrapper(selection.spec))
            if selection.has_capability(CAPABILITY_STREAMING_FOCK):
                body.append(_streaming_fock_source(selection))
                body.append(_streaming_fock_launch_wrapper(selection))
                if _streaming_fock_schedule(selection).kind in (
                    ScheduleKind.PACKED_TASKS,
                    ScheduleKind.SUBGROUP_TASKS,
                ):
                    body.append(
                        _streaming_fock_launch_wrapper(
                            selection,
                            f"generativeqc_launch_generated_{selection.spec.name}_work_streaming_fock",
                            work_aware=True,
                        )
                    )
        if selection.resident_force_recurrence is not None:
            body.append(_emit_ppps_resident_source(selection))
            body.append(_ppps_resident_launch_wrapper())
    if not selections:
        body.append("// Empty deterministic shard reserved for stable CMake outputs.\n")
    return "".join(body)


def _strip_emitter_includes(source: str) -> str:
    """Remove global standard includes before placing source in a namespace."""

    return re.sub(
        r"^#include <(?:cstddef|cstdint)>\n",
        "",
        source,
        flags=re.MULTILINE,
    )


def _scope_profile_identifiers(
    source: str,
    selection: KernelSelection,
    identifier: str,
) -> str:
    """Apply the emitter's two structured identifier roots to one profile.

    Generated code owns a lower-case CUDA symbol root and a CamelCase type
    root. Rewriting only those declared roots keeps profile isolation explicit
    and avoids unconstrained shell-name/string substitution.
    """

    # Profile scoping applies to generated CUDA types, never to the stable
    # runtime ABI included from ``generated_shell_task.hpp``.  Protect the
    # qualified host descriptor while rewriting the shared ``GeneratedPpps``
    # prefix, then restore it verbatim.
    host_resident_task = "generativeqc::scf::detail::GeneratedPppsResidentTask"
    host_resident_task_placeholder = "GENERATIVEQC_STABLE_PPPS_RESIDENT_TASK_ABI"
    source = source.replace(host_resident_task, host_resident_task_placeholder)
    class_name = selection.spec.name[0].upper() + selection.spec.name[1:]
    profile_class = "".join(part.capitalize() for part in identifier.split("_"))
    return (
        source.replace(
            f"generated_{selection.spec.name}",
            f"generated_{identifier}_{selection.spec.name}",
        )
        .replace(
            f"Generated{class_name}",
            f"Generated{profile_class}{class_name}",
        )
        .replace(host_resident_task_placeholder, host_resident_task)
    )


def emit_profile_shard(
    profile: ResolvedProductionProfile,
    selections: Iterable[KernelSelection],
    *,
    variant: str = "",
) -> str:
    """Emit one architecture-namespaced shard with collision-free symbols."""

    items = tuple(selections)
    identifier = _profile_identifier(profile.target.architecture) + variant
    namespace = f"generativeqc::scf::generated::profile_{identifier}"
    body = [
        f"// Stable AOT shard map version: {_STABLE_AOT_SHARD_MAP_VERSION}\n",
        # Alternatives share their incumbent TU's global helper definitions.
        _PRODUCTION_PRELUDE if not variant else "",
        f"\nnamespace {namespace} {{\n",
    ]
    for selection in items:
        integral = _selection_integral(selection)
        plan = build_fused_shell_plan(
            selection.spec,
            integral=integral,
            schedule=selection.schedule,
            target=profile.target,
        )
        source = emit_shell_class_fused_cuda(
            selection.spec,
            plan,
            fock_schedule=selection.fock_schedule,
            capabilities=selection.capabilities,
        )
        force_symbol = (
            f"generativeqc_launch_{identifier}_generated_{selection.spec.name}"
        )
        body.append(
            _scope_profile_identifiers(
                _strip_emitter_includes(source), selection, identifier
            )
        )
        force_wrapper = _scope_profile_identifiers(
            _launch_wrapper(selection.spec, force_symbol),
            selection,
            identifier,
        ).replace(
            _scope_profile_identifiers(force_symbol, selection, identifier),
            force_symbol,
        )
        if (
            KernelConsumer.FORCE in selection.consumers
            or not selection.recurrence.startswith("rys")
        ):
            body.append(force_wrapper)
        if KernelConsumer.FOCK in selection.consumers:
            fock_symbol = f"{force_symbol}_fock"
            fock_wrapper = _scope_profile_identifiers(
                _fock_launch_wrapper(
                    selection.spec,
                    fock_symbol,
                ),
                selection,
                identifier,
            ).replace(
                _scope_profile_identifiers(fock_symbol, selection, identifier),
                fock_symbol,
            )
            body.append(fock_wrapper)
            if selection.has_capability(CAPABILITY_MIXED_FOCK):
                mixed_fock_symbol = f"{force_symbol}_mixed_fock"
                mixed_fock_wrapper = _scope_profile_identifiers(
                    _mixed_fock_launch_wrapper(
                        selection.spec,
                        mixed_fock_symbol,
                    ),
                    selection,
                    identifier,
                ).replace(
                    _scope_profile_identifiers(
                        mixed_fock_symbol, selection, identifier
                    ),
                    mixed_fock_symbol,
                )
                body.append(mixed_fock_wrapper)
            if selection.has_capability(CAPABILITY_STREAMING_FOCK):
                body.append(
                    _scope_profile_identifiers(
                        _streaming_fock_source(
                            selection, include_work_buckets=not variant
                        ),
                        selection,
                        identifier,
                    )
                )
                streaming_symbol = f"{force_symbol}_streaming_fock"
                streaming_wrapper = _scope_profile_identifiers(
                    _streaming_fock_launch_wrapper(
                        selection,
                        streaming_symbol,
                    ),
                    selection,
                    identifier,
                ).replace(
                    _scope_profile_identifiers(streaming_symbol, selection, identifier),
                    streaming_symbol,
                )
                body.append(streaming_wrapper)
                if not variant and _streaming_fock_schedule(selection).kind in (
                    ScheduleKind.PACKED_TASKS,
                    ScheduleKind.SUBGROUP_TASKS,
                ):
                    work_symbol = f"{force_symbol}_work_streaming_fock"
                    body.append(
                        _scope_profile_identifiers(
                            _streaming_fock_launch_wrapper(
                                selection, work_symbol, work_aware=True
                            ),
                            selection,
                            identifier,
                        ).replace(
                            _scope_profile_identifiers(
                                work_symbol, selection, identifier
                            ),
                            work_symbol,
                        )
                    )
        if selection.resident_force_recurrence is not None:
            resident_source = _scope_profile_identifiers(
                _strip_emitter_includes(_emit_ppps_resident_source(selection)),
                selection,
                identifier,
            )
            body.append(resident_source)
            resident_symbol = f"generativeqc_launch_{identifier}_ppps_resident"
            resident_wrapper = _scope_profile_identifiers(
                _ppps_resident_launch_wrapper(resident_symbol),
                selection,
                identifier,
            ).replace(
                _scope_profile_identifiers(resident_symbol, selection, identifier),
                resident_symbol,
            )
            body.append(resident_wrapper)
    if not items:
        body.append("// Portable profile: generic CUDA kernels remain active.\n")
    body.append(f"\n}}  // namespace {namespace}\n")
    return "".join(body)
