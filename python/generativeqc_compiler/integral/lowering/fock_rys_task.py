"""Emit lane-local Rys values through the existing packed Fock/queue ABI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..rys import (
    emit_rys1_roots_cuda,
    emit_rys2_roots_cuda,
    emit_rys_value_root_body_cuda,
)
from .common import _emitted_component_names, _generic_task_component_setup
from .fock_tiled import _emit_packed_fock_consumer_cuda, _packed_restricted_k_block

if TYPE_CHECKING:
    from ..fused_schedule import FusedShellPlan
    from ..shell_spec import ShellClassSpec


def emit_rys_task_support_cuda(plan: FusedShellPlan) -> str:
    """Own exact value roots without addressed component HRR or force state."""
    roots = plan.kernel.integral.required_rys_roots
    prefix = f"generated_dppp_rys{roots}"
    if roots == 1:
        return emit_rys1_roots_cuda(symbol_prefix=prefix)
    if roots != 2:
        raise ValueError("task-parallel Rys support requires one or two value roots")
    # A lane owns the whole quartet: inlining keeps the four outputs and live
    # accumulators in registers instead of a device-call/addressed-array frame.
    return emit_rys2_roots_cuda(
        symbol_prefix=prefix, high_accuracy=True, forceinline=True
    )


def emit_rys_task_fock_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
    minimum_blocks_per_sm: int,
    *,
    k_block: bool,
) -> str:
    """Evaluate each admitted quartet once, independently in one warp lane.

    Only the surrounding admission queue uses warp collectives. Primitive
    geometry, quadrature and every Cartesian component stay lane-local, while
    screening, pair orientation and spin/symmetry scatter retain their owners.
    """
    setup = _generic_task_component_setup(spec).replace("shared.task", "task")
    names = _emitted_component_names(spec)
    storage, contraction = (
        _packed_restricted_k_block(spec, setup, names, maximum_doubles=64)
        if k_block
        else ("", "")
    )
    roots = plan.kernel.integral.required_rys_roots
    body = emit_rys_value_root_body_cuda(spec, plan.kernel.integral)
    # Keep ordinary/persistent entry points byte-identical to the packed ABI.
    # Only their value worker changes; the discarded prefix is not emitted.
    packed = _emit_packed_fock_consumer_cuda(
        spec, plan, minimum_blocks_per_sm, lane_private=True
    )
    marker = 'extern "C" __global__'
    begin = packed.find(marker)
    if begin < 0:
        raise RuntimeError("packed Fock launch marker changed unexpectedly")
    kernels = packed[begin:]
    return f"""
struct GeneratedDpppPackedFockLaneStorage {{
{storage}}};

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_packed_fock_lane(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance, const double* schwarz_bounds,
    const double* density, double* fock, std::size_t task_index,
    GeneratedDpppPackedFockLaneStorage& storage) {{
  const GeneratedDpppShellTask& task = tasks[task_index];
  std::uint32_t retained_components = 0U;
#pragma unroll
  for (unsigned component = 0U; component < kGeneratedDpppComponentCount;
       ++component) {{
{setup}
    const std::size_t matrix_order = static_cast<std::size_t>(task.matrix_order);
    if (unique_ket_component && (schwarz_bounds == nullptr ||
        schwarz_bounds[task.density_offset +
            generated_dppp_matrix_index(i, j, matrix_order)] *
        schwarz_bounds[task.density_offset +
            generated_dppp_matrix_index(k, l, matrix_order)] >= screening_tolerance)) {{
      retained_components |= 1U << component;
    }}
  }}
  if (retained_components == 0U) return;
  const GeneratedDpppVec3 first = atom_positions[task.atom[0]];
  const GeneratedDpppVec3 second = atom_positions[task.atom[1]];
  const GeneratedDpppVec3 third = atom_positions[task.atom[2]];
  const GeneratedDpppVec3 fourth = atom_positions[task.atom[3]];
  const double abx = second.x - first.x;
  const double aby = second.y - first.y;
  const double abz = second.z - first.z;
  const double cdx = fourth.x - third.x;
  const double cdy = fourth.y - third.y;
  const double cdz = fourth.z - third.z;
  double component_integrals[kGeneratedDpppComponentCount]{{}};
  for (std::int64_t bra = primitive_pair_offsets[task.shell_pair[0]];
       bra < primitive_pair_offsets[task.shell_pair[0] + 1U]; ++bra) {{
    const GeneratedDpppPrimitivePairData first_pair = primitive_pairs[bra];
    const double p = first_pair.exponent_sum;
    const double pax = first_pair.product_center.x - first.x;
    const double pay = first_pair.product_center.y - first.y;
    const double paz = first_pair.product_center.z - first.z;
    for (std::int64_t ket = primitive_pair_offsets[task.shell_pair[1]];
         ket < primitive_pair_offsets[task.shell_pair[1] + 1U]; ++ket) {{
      const GeneratedDpppPrimitivePairData second_pair = primitive_pairs[ket];
      const double q = second_pair.exponent_sum;
      const double qcx = second_pair.product_center.x - third.x;
      const double qcy = second_pair.product_center.y - third.y;
      const double qcz = second_pair.product_center.z - third.z;
      const double dx = first_pair.product_center.x - second_pair.product_center.x;
      const double dy = first_pair.product_center.y - second_pair.product_center.y;
      const double dz = first_pair.product_center.z - second_pair.product_center.z;
      const double inverse_sum = 1.0 / (p + q);
      const double prefactor = 34.986836655249725 * first_pair.weighted_coefficient *
          second_pair.weighted_coefficient / (p * q * sqrt(p + q));
      double roots_weights[{2 * roots}];
      generated_dppp_rys{roots}_roots(
          p * q * inverse_sum * (dx * dx + dy * dy + dz * dz), roots_weights, 1U);
#pragma unroll
      for (unsigned root_index = 0U; root_index < {roots}U; ++root_index) {{
        const double root_over_sum = roots_weights[2U * root_index] * inverse_sum;
        const double root_bra = root_over_sum * q;
        const double root_ket = root_over_sum * p;
        const double b10 = 0.5 / p * (1.0 - root_bra);
        const double b00 = 0.5 * root_over_sum;
        const double b01 = 0.5 / q * (1.0 - root_ket);
        const double weighted_root = roots_weights[2U * root_index + 1U] * prefactor;
        const double c0x = pax - dx * root_bra;
        const double c0y = pay - dy * root_bra;
        const double c0z = paz - dz * root_bra;
        const double cpx = qcx + dx * root_ket;
        const double cpy = qcy + dy * root_ket;
        const double cpz = qcz + dz * root_ket;
{body}
      }}
    }}
  }}
#pragma unroll
  for (unsigned component = 0U; component < kGeneratedDpppComponentCount;
       ++component) {{
{setup}
    component_integrals[component] *=
        ao_coefficients[task.ao_coefficient_begin[0] + {names[0]}] *
        ao_coefficients[task.ao_coefficient_begin[1] + {names[1]}] *
        ao_coefficients[task.ao_coefficient_begin[2] + {names[2]}] *
        ao_coefficients[task.ao_coefficient_begin[3] + {names[3]}];
  }}
{contraction}
#pragma unroll
  for (unsigned component = 0U; component < kGeneratedDpppComponentCount;
       ++component) {{
{setup}
    const double integral = component_integrals[component];
    if (integral != 0.0) generated_dppp_accumulate_fock<Unrestricted>(
        task, density, fock, i, j, k, l, integral);
  }}
}}
{kernels}
"""
