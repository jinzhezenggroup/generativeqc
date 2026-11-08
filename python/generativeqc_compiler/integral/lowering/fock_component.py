"""Emit fixed-root component-lane Fock contractions from the selected shell plan."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..cuda_schedule import (
    ScheduleKind,
)
from ..rys import (
    build_rys_force_program,
    emit_rys2_roots_cuda,
    emit_rys3_roots_cuda,
    emit_rys4_roots_cuda,
    emit_rys5_roots_cuda,
    emit_rys_hrr_state_cuda,
)
from .common import (
    _component_axis_expression,
    _emitted_component_names,
    _generic_task_component_setup,
)

if TYPE_CHECKING:
    from ..fused_schedule import (
        FusedShellPlan,
    )
    from ..ir import IntegralIR
    from ..shell_spec import (
        ShellClassSpec,
    )


def emit_rys_value_support_cuda(spec: ShellClassSpec, integral: IntegralIR) -> str:
    """Own roots, geometry and HRR for a derivative-free Direct Fock plan.

    Strict tables are shared with existing compiler value lowerings. There is
    no force-program dependency, and addressed TRR bounds contain only values.
    Production admission and resource qualification remain separate decisions.
    """

    if integral.spec != spec or integral.derivative is not None:
        raise ValueError("Rys value support requires the matching value-only IR")
    nroots = integral.required_rys_roots
    emitters = {
        2: emit_rys2_roots_cuda,
        3: emit_rys3_roots_cuda,
        4: emit_rys4_roots_cuda,
        5: emit_rys5_roots_cuda,
    }
    if nroots not in emitters or integral.recurrence != f"rys{nroots}":
        raise ValueError(
            "Rys value support requires an exact two- through five-root IR"
        )
    roots = emitters[nroots](
        symbol_prefix=f"generated_dppp_rys{nroots}", high_accuracy=True
    )
    fields = "\n".join(
        f"  double {field};"
        for field in (
            "p",
            "q",
            "pax",
            "pay",
            "paz",
            "qcx",
            "qcy",
            "qcz",
            "abx",
            "aby",
            "abz",
            "cdx",
            "cdy",
            "cdz",
            "dx",
            "dy",
            "dz",
            "primitive_prefactor",
        )
    )
    hrr = emit_rys_hrr_state_cuda(
        nroots, sum(spec.angular[:2]) + 1, sum(spec.angular[2:]) + 1
    )
    return (
        roots
        + f"""
/** Geometry shared by the component lanes of a value-only primitive quartet. */
struct GeneratedDpppRys{nroots}Primitive {{
{fields}
}};

{hrr}"""
    )


def _emit_rys_axis_fill_cuda(bra_extent: int, ket_extent: int) -> str:
    """One TRR arithmetic emitter for private and CTA-shared addressed tables.

    Text factoring preserves the incumbent private function and its noinline
    boundary. Shared producers change storage ownership, never recurrence math.
    """

    return f"""  trr[0][0] = seed;
#pragma unroll
  for (unsigned bra = 1U; bra < {bra_extent}U; ++bra) {{
    double value = c0 * trr[bra - 1U][0];
    if (bra > 1U) value += (bra - 1U) * b10 * trr[bra - 2U][0];
    trr[bra][0] = value;
  }}
#pragma unroll
  for (unsigned ket = 1U; ket < {ket_extent}U; ++ket) {{
#pragma unroll
    for (unsigned bra = 0U; bra < {bra_extent}U; ++bra) {{
      double value = cp * trr[bra][ket - 1U];
      if (ket > 1U) value +=
          (ket - 1U) * b01 * trr[bra][ket - 2U];
      if (bra > 0U) value +=
          bra * b00 * trr[bra - 1U][ket - 1U];
      trr[bra][ket] = value;
    }}
  }}
"""


def _emit_rys_component_lane_fock_consumer_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
    minimum_blocks_per_sm: int,
    *,
    support_integral: IntegralIR | None = None,
) -> str:
    """Emit fixed-root Rys value contraction with one lane per component.

    A derivative-free plan owns value support directly. Legacy fused plans can
    still reuse the preceding force emitter's qualified tables and HRR bounds.
    Both routes retain the same screening, primitive-pair inputs and canonical
    RHF/UHF scatter; requesting values never implicitly requests derivatives.
    """

    recurrence_integral = (
        plan.kernel.integral if support_integral is None else support_integral
    )
    force_support = recurrence_integral.derivative is not None
    nroots = (
        build_rys_force_program(spec, integral=recurrence_integral).nroots
        if force_support
        else recurrence_integral.required_rys_roots
    )
    recurrence = f"rys{nroots}"
    if recurrence_integral.recurrence != recurrence:
        raise ValueError(
            f"component-lane Rys Fock for {spec.name} requires {recurrence}"
        )
    if (
        plan.schedule.kind != ScheduleKind.COMPONENT_LANES
        or plan.schedule.block_threads < spec.component_count
    ):
        raise ValueError(
            "component-lane Rys Fock requires one lane per Cartesian component"
        )
    if nroots not in ((3, 4) if force_support else (2, 3, 4, 5)):
        raise ValueError("component-lane Rys Fock has no support for this root count")
    if max(spec.angular) > (2 if force_support else 3) or spec.angular[3] > 1:
        raise ValueError(
            "component-lane Rys Fock requires s/p/d force support or s/p/d/f "
            "value support, with at most p angular momentum on the fourth center"
        )

    task_component_setup = _generic_task_component_setup(spec)
    component_names = _emitted_component_names(spec)
    symbol_tag = f"rys{nroots}"
    class_tag = f"Rys{nroots}"
    support_order = 2 if force_support else 1
    bra_extent = sum(spec.angular[:2]) + support_order
    ket_extent = sum(spec.angular[2:]) + support_order
    task_qualifier = "__noinline__" if spec.angular[1] == 2 else "__forceinline__"
    kernel_qualifier = (
        f"__launch_bounds__(kGeneratedDpppFockBlockThreads, {minimum_blocks_per_sm})"
    )
    bounds_comment = (
        "are inherited from the validated force HRR and include no dynamic bounds."
        if force_support
        else "follow the selected value IR and include no dynamic bounds."
    )

    def axis_count(center: int, axis: int) -> str:
        """Return one runtime Cartesian exponent from a component ordinal."""

        order = spec.angular[center]
        component_name = component_names[center]
        if order == 0:
            return "0U"
        if order == 1:
            return f"({component_name} == {axis}U)"
        if order == 3:
            return " + ".join(
                f"({_component_axis_expression(spec, center, quantum, component_name)} == {axis}U)"
                for quantum in range(order)
            )
        return (
            f"(generated_dppp_d_axes[{component_name}][0] == {axis}U) + "
            f"(generated_dppp_d_axes[{component_name}][1] == {axis}U)"
        )

    component_axis_counts = tuple(
        tuple(axis_count(center, axis) for axis in range(3)) for center in range(4)
    )
    source = f"""
/** Evaluate only the value needed by Fock, omitting force derivatives. */
__device__ __noinline__ double generated_dppp_{symbol_tag}_value_axis(
    unsigned a, unsigned b, unsigned c, unsigned d,
    double c0, double cp, double ab, double cd,
    double b10, double b00, double b01, double seed) {{
  // Runtime component ordinals require addressed storage.  The exact extents
  // {bounds_comment}
  volatile double trr[{bra_extent}][{ket_extent}];
{_emit_rys_axis_fill_cuda(bra_extent, ket_extent)}  return generated_dppp_{symbol_tag}_state(trr, a, b, c, d, ab, cd);
}}

template <bool Unrestricted>
__device__ {task_qualifier} void generated_dppp_shell_class_fock_task(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* fock,
    std::size_t task_index) {{
  struct Shared {{
    GeneratedDpppShellTask task;
    GeneratedDpppVec3 positions[4];
    GeneratedDppp{class_tag}Primitive primitive;
    double roots_weights[{2 * nroots}];
  }};
  __shared__ Shared shared;
  const unsigned lane = threadIdx.x;
  if (blockDim.x != kGeneratedDpppFockBlockThreads) return;
  if (lane == 0U) {{
    shared.task = tasks[task_index];
#pragma unroll
    for (unsigned center = 0U; center < 4U; ++center) {{
      shared.positions[center] = atom_positions[shared.task.atom[center]];
    }}
  }}
  __syncthreads();

  const bool component_lane = lane < kGeneratedDpppComponentCount;
  const unsigned component = component_lane ? lane : 0U;
{task_component_setup}
  const std::size_t matrix_order =
      static_cast<std::size_t>(shared.task.matrix_order);
  const bool retained_by_schwarz = component_lane && unique_ket_component &&
      (schwarz_bounds == nullptr ||
       schwarz_bounds[
           shared.task.density_offset +
           generated_dppp_matrix_index(i, j, matrix_order)] *
           schwarz_bounds[
               shared.task.density_offset +
               generated_dppp_matrix_index(k, l, matrix_order)] >=
           screening_tolerance);
  const double angular_coefficient = retained_by_schwarz
      ? ao_coefficients[
            shared.task.ao_coefficient_begin[0] + {component_names[0]}] *
        ao_coefficients[
            shared.task.ao_coefficient_begin[1] + {component_names[1]}] *
        ao_coefficients[
            shared.task.ao_coefficient_begin[2] + {component_names[2]}] *
        ao_coefficients[
            shared.task.ao_coefficient_begin[3] + {component_names[3]}]
      : 0.0;
  if (!__syncthreads_or(retained_by_schwarz)) return;

  const unsigned ax = {component_axis_counts[0][0]};
  const unsigned ay = {component_axis_counts[0][1]};
  const unsigned az = {component_axis_counts[0][2]};
  const unsigned bx = {component_axis_counts[1][0]};
  const unsigned by = {component_axis_counts[1][1]};
  const unsigned bz = {component_axis_counts[1][2]};
  const unsigned cx = {component_axis_counts[2][0]};
  const unsigned cy = {component_axis_counts[2][1]};
  const unsigned cz = {component_axis_counts[2][2]};
  const unsigned dx_order = {component_axis_counts[3][0]};
  const unsigned dy_order = {component_axis_counts[3][1]};
  const unsigned dz_order = {component_axis_counts[3][2]};
  double component_integral = 0.0;

  const std::int64_t first_pair_begin =
      primitive_pair_offsets[shared.task.shell_pair[0]];
  const std::int64_t first_pair_end =
      primitive_pair_offsets[shared.task.shell_pair[0] + 1U];
  const std::int64_t second_pair_begin =
      primitive_pair_offsets[shared.task.shell_pair[1]];
  const std::int64_t second_pair_end =
      primitive_pair_offsets[shared.task.shell_pair[1] + 1U];
  for (std::int64_t first_primitive = first_pair_begin;
       first_primitive < first_pair_end; ++first_primitive) {{
    for (std::int64_t second_primitive = second_pair_begin;
         second_primitive < second_pair_end; ++second_primitive) {{
      if (lane == 0U) {{
        const GeneratedDpppPrimitivePairData first_pair =
            primitive_pairs[first_primitive];
        const GeneratedDpppPrimitivePairData second_pair =
            primitive_pairs[second_primitive];
        GeneratedDppp{class_tag}Primitive& primitive = shared.primitive;
        primitive.p = first_pair.exponent_sum;
        primitive.q = second_pair.exponent_sum;
        primitive.pax = first_pair.product_center.x - shared.positions[0].x;
        primitive.pay = first_pair.product_center.y - shared.positions[0].y;
        primitive.paz = first_pair.product_center.z - shared.positions[0].z;
        primitive.qcx = second_pair.product_center.x - shared.positions[2].x;
        primitive.qcy = second_pair.product_center.y - shared.positions[2].y;
        primitive.qcz = second_pair.product_center.z - shared.positions[2].z;
        primitive.abx = shared.positions[1].x - shared.positions[0].x;
        primitive.aby = shared.positions[1].y - shared.positions[0].y;
        primitive.abz = shared.positions[1].z - shared.positions[0].z;
        primitive.cdx = shared.positions[3].x - shared.positions[2].x;
        primitive.cdy = shared.positions[3].y - shared.positions[2].y;
        primitive.cdz = shared.positions[3].z - shared.positions[2].z;
        primitive.dx = first_pair.product_center.x -
            second_pair.product_center.x;
        primitive.dy = first_pair.product_center.y -
            second_pair.product_center.y;
        primitive.dz = first_pair.product_center.z -
            second_pair.product_center.z;
        const double rho =
            primitive.p * primitive.q / (primitive.p + primitive.q);
        generated_dppp_{symbol_tag}_roots(
            rho * (primitive.dx * primitive.dx +
                   primitive.dy * primitive.dy +
                   primitive.dz * primitive.dz),
            shared.roots_weights, 1U);
        primitive.primitive_prefactor =
            34.986836655249725 * first_pair.weighted_coefficient *
            second_pair.weighted_coefficient /
            (primitive.p * primitive.q * sqrt(primitive.p + primitive.q));
      }}
      __syncthreads();
      if (retained_by_schwarz) {{
        const GeneratedDppp{class_tag}Primitive& primitive = shared.primitive;
#pragma unroll 1
        for (unsigned root_index = 0U; root_index < {nroots}U;
             ++root_index) {{
          const double root = shared.roots_weights[2U * root_index];
          const double weighted_root =
              shared.roots_weights[2U * root_index + 1U] *
              primitive.primitive_prefactor * angular_coefficient;
          const double root_over_sum = root / (primitive.p + primitive.q);
          const double root_bra = root_over_sum * primitive.q;
          const double root_ket = root_over_sum * primitive.p;
          const double b10 = 0.5 / primitive.p * (1.0 - root_bra);
          const double b00 = 0.5 * root_over_sum;
          const double b01 = 0.5 / primitive.q * (1.0 - root_ket);
          const double x = generated_dppp_{symbol_tag}_value_axis(
              ax, bx, cx, dx_order,
              primitive.pax - primitive.dx * root_bra,
              primitive.qcx + primitive.dx * root_ket,
              primitive.abx, primitive.cdx, b10, b00, b01, 1.0);
          const double y = generated_dppp_{symbol_tag}_value_axis(
              ay, by, cy, dy_order,
              primitive.pay - primitive.dy * root_bra,
              primitive.qcy + primitive.dy * root_ket,
              primitive.aby, primitive.cdy, b10, b00, b01, 1.0);
          const double z = generated_dppp_{symbol_tag}_value_axis(
              az, bz, cz, dz_order,
              primitive.paz - primitive.dz * root_bra,
              primitive.qcz + primitive.dz * root_ket,
              primitive.abz, primitive.cdz, b10, b00, b01, weighted_root);
          component_integral += x * y * z;
        }}
      }}
      __syncthreads();
    }}
  }}
  if (retained_by_schwarz && component_integral != 0.0) {{
    generated_dppp_accumulate_fock<Unrestricted>(
        shared.task, density, fock, i, j, k, l, component_integral);
  }}
}}

extern "C" __global__ {kernel_qualifier}
void generated_dppp_shell_class_fock_rhf_kernel(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* fock,
    std::size_t task_count) {{
  if (blockIdx.x >= task_count) return;
  generated_dppp_shell_class_fock_task<false>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, fock,
      static_cast<std::size_t>(blockIdx.x));
}}

extern "C" __global__ {kernel_qualifier}
void generated_dppp_shell_class_fock_uhf_kernel(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* fock,
    std::size_t task_count) {{
  if (blockIdx.x >= task_count) return;
  generated_dppp_shell_class_fock_task<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, fock,
      static_cast<std::size_t>(blockIdx.x));
}}

template <bool Unrestricted>
__device__ __forceinline__ void generated_dppp_shell_class_fock_persistent(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* fock,
    const std::uint32_t* task_offset,
    const std::uint32_t* task_count,
    std::uint32_t* task_head) {{
  __shared__ std::uint32_t shared_task_index;
  while (true) {{
    if (threadIdx.x == 0U) shared_task_index = atomicAdd(task_head, 1U);
    __syncthreads();
    const std::uint32_t task_index = shared_task_index;
    if (task_index >= *task_count) return;
    generated_dppp_shell_class_fock_task<Unrestricted>(
        tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
        atom_positions, screening_tolerance, schwarz_bounds, density, fock,
        static_cast<std::size_t>(*task_offset + task_index));
    __syncthreads();
  }}
}}

extern "C" __global__ {kernel_qualifier}
void generated_dppp_shell_class_fock_rhf_persistent_kernel(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* fock,
    const std::uint32_t* task_offset,
    const std::uint32_t* task_count,
    std::uint32_t* task_head) {{
  generated_dppp_shell_class_fock_persistent<false>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, fock,
      task_offset, task_count, task_head);
}}

extern "C" __global__ {kernel_qualifier}
void generated_dppp_shell_class_fock_uhf_persistent_kernel(
    const GeneratedDpppShellTask* tasks,
    const GeneratedDpppPrimitivePairData* primitive_pairs,
    const std::int64_t* primitive_pair_offsets,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    double screening_tolerance,
    const double* schwarz_bounds,
    const double* density,
    double* fock,
    const std::uint32_t* task_offset,
    const std::uint32_t* task_count,
    std::uint32_t* task_head) {{
  generated_dppp_shell_class_fock_persistent<true>(
      tasks, primitive_pairs, primitive_pair_offsets, ao_coefficients,
      atom_positions, screening_tolerance, schwarz_bounds, density, fock,
      task_offset, task_count, task_head);
}}
"""

    if not plan.schedule.fock_shared_rys_xy:
        return source
    if force_support:
        raise ValueError("shared Rys x/y does not support derivative recurrence tables")
    return _with_shared_rys_xy_task(
        source,
        bra_extent,
        ket_extent,
        nroots=nroots,
        weighted_z_spec=spec if plan.schedule.fock_shared_rys_weighted_z else None,
    )


def _with_shared_rys_xy_task(
    source: str,
    bra_extent: int,
    ket_extent: int,
    *,
    nroots: int,
    weighted_z_spec: ShellClassSpec | None = None,
) -> str:
    """Add a restricted task while preserving the private/UHF instantiation.

    The fixture capability does not replace UHF, mixed, streaming or work-K
    consumers. The RHF persistent queue and launch geometry remain incumbent.
    Axis extents and root count come from the same value IR as the private
    evaluator; sharing changes table ownership, not the recurrence algebra.
    """

    if nroots not in (3, 4):
        raise ValueError("shared Rys x/y supports three- and four-root values")
    if weighted_z_spec is not None and nroots != 3:
        raise ValueError("shared Rys weighted-Z retains its three-root qualification")

    marker = "template <bool Unrestricted>\n__device__"
    task_begin = source.index(marker)
    task_end = source.index('extern "C" __global__', task_begin)
    task = source[task_begin:task_end]
    task = (
        task.replace("template <bool Unrestricted>\n", "", 1)
        .replace(
            "generated_dppp_shell_class_fock_task(",
            "generated_dppp_shell_class_fock_shared_xy_task(",
            1,
        )
        .replace(
            "generated_dppp_accumulate_fock<Unrestricted>",
            "generated_dppp_accumulate_fock<false>",
        )
    )
    task = task.replace(
        f"    double roots_weights[{2 * nroots}];",
        f"    double roots_weights[{2 * nroots}];\n"
        f"    volatile double xy[{nroots}][2][{bra_extent}][{ket_extent}];",
        1,
    )
    task = task.replace(
        "  if (!__syncthreads_or(retained_by_schwarz)) return;",
        "  const unsigned retained_count = __syncthreads_count(retained_by_schwarz);\n"
        "  if (retained_count == 0U) return;\n"
        "  const bool share_xy = retained_count > 2U;",
        1,
    )
    producer = """      // Six fixed lanes own one complete root/axis table each, even when
      // that lane's Cartesian component was screened. All lanes publish/retire.
      if (share_xy && lane < 6U) {
        const unsigned root_index = lane / 2U;
        const unsigned axis = lane % 2U;
        const GeneratedDpppRys3Primitive& primitive = shared.primitive;
        const double root = shared.roots_weights[2U * root_index];
        const double root_over_sum = root / (primitive.p + primitive.q);
        const double root_bra = root_over_sum * primitive.q;
        const double root_ket = root_over_sum * primitive.p;
        const double b10 = 0.5 / primitive.p * (1.0 - root_bra);
        const double b00 = 0.5 * root_over_sum;
        const double b01 = 0.5 / primitive.q * (1.0 - root_ket);
        generated_dppp_rys3_fill_axis(
            shared.xy[root_index][axis],
            (axis == 0U ? primitive.pax : primitive.pay) -
                (axis == 0U ? primitive.dx : primitive.dy) * root_bra,
            (axis == 0U ? primitive.qcx : primitive.qcy) +
                (axis == 0U ? primitive.dx : primitive.dy) * root_ket,
            b10, b00, b01, 1.0);
      }
      __syncthreads();
"""
    producer = (
        producer.replace("Six fixed", "Six fixed" if nroots == 3 else "Eight fixed")
        .replace("lane < 6U", f"lane < {2 * nroots}U")
        .replace("Rys3Primitive", f"Rys{nroots}Primitive")
        .replace("generated_dppp_rys3_", f"generated_dppp_rys{nroots}_")
    )
    task = task.replace(
        "      __syncthreads();\n      if (retained_by_schwarz) {",
        "      __syncthreads();\n" + producer + "      if (retained_by_schwarz) {",
        1,
    )
    for axis, ordinal in (("x", 0), ("y", 1)):
        task = task.replace(
            f"          const double {axis} = generated_dppp_rys{nroots}_value_axis(",
            f"          const double {axis} = share_xy\n"
            f"              ? generated_dppp_rys{nroots}_state(\n"
            f"                    shared.xy[root_index][{ordinal}],\n"
            f"                    a{axis}, b{axis}, c{axis}, d{axis}_order,\n"
            f"                    primitive.ab{axis}, primitive.cd{axis})\n"
            f"              : generated_dppp_rys{nroots}_value_axis(",
            1,
        )
    guard = ""
    if weighted_z_spec is not None:
        task = _with_shared_rys_weighted_z_task(
            task, weighted_z_spec, bra_extent, ket_extent
        )
        guard = _emit_weighted_z_match_cuda()
    fill = f"""
/** Fill an unweighted root/axis table; HRR remains component-local. */
__device__ __forceinline__ void generated_dppp_rys{nroots}_fill_axis(
    volatile double (&trr)[{bra_extent}][{ket_extent}],
    double c0, double cp, double b10, double b00, double b01, double seed) {{
{_emit_rys_axis_fill_cuda(bra_extent, ket_extent)}}}

"""
    result = (
        source[:task_begin]
        + fill
        + guard
        + source[task_begin:task_end]
        + task
        + source[task_end:]
    )
    # The direct RHF wrapper owns a concrete restricted call. Persistent workers
    # keep their template and choose uniformly at compile time, leaving UHF exact.
    result = result.replace(
        "  generated_dppp_shell_class_fock_task<false>(",
        "  generated_dppp_shell_class_fock_shared_xy_task(",
        1,
    )
    call = "    generated_dppp_shell_class_fock_task<Unrestricted>("
    begin = result.index(call)
    end = result.index(";", begin) + 1
    private_call = result[begin:end]
    shared_call = private_call.replace(
        "generated_dppp_shell_class_fock_task<Unrestricted>",
        "generated_dppp_shell_class_fock_shared_xy_task",
    )
    result = (
        result[:begin]
        + (
            "    if constexpr (Unrestricted) {\n"
            + private_call
            + "\n    } else {\n"
            + shared_call
            + "\n    }"
        )
        + result[end:]
    )
    return result


def _emit_weighted_z_match_cuda() -> str:
    """The executable bit guard used by the companion and host controls."""
    return """
__device__ __forceinline__ bool generated_dppp_weighted_z_matches(
    double angular_coefficient, double representative) {
  return isfinite(representative) &&
      __double_as_longlong(angular_coefficient) ==
          __double_as_longlong(representative);
}

"""


def _with_shared_rys_weighted_z_task(
    task: str, spec: ShellClassSpec, bra_extent: int, ket_extent: int
) -> str:
    """Reuse identical weighted seeds within the single incumbent XY lifetime.

    The complete task votes before primitive work. A refusal keeps the original
    XY expressions and private Z, without another task call or shared struct.
    """
    first = spec.center_components[0]
    groups = tuple(
        0 if spec.name == "pppp" or label[0] == label[1] else 1 for label in first
    )
    expected = (0, 0, 0) if spec.name == "pppp" else (0, 1, 1, 0, 1, 0)
    if spec.name not in ("dppp", "pppp") or groups != expected:
        raise ValueError("weighted-Z representatives require the admitted Cartesian IR")
    representatives = tuple(groups.index(g) for g in range(max(groups) + 1))
    names = _emitted_component_names(spec)
    ngroups = len(representatives)
    task = task.replace(
        f"    volatile double xy[3][2][{bra_extent}][{ket_extent}];",
        f"    volatile double xy[3][2][{bra_extent}][{ket_extent}];\n"
        f"    volatile double weighted_z[3][{ngroups}][{bra_extent}][{ket_extent}];\n"
        f"    double weighted_z_alpha[{ngroups}];\n"
        "    unsigned weighted_z_finite;",
        1,
    )
    begin = task.index("      ? ao_coefficients[")
    end = task.index("\n      : 0.0;", begin)
    alpha = task[begin + len("      ? ") : end]
    assignments = []
    for group, representative in enumerate(representatives):
        expression = alpha
        for center, name in enumerate(names):
            expression = expression.replace(
                f"+ {name}]", f"+ {representative if center == 0 else 0}U]"
            )
        assignments.append(
            f"    shared.weighted_z_alpha[{group}] =\n        {expression};\n"
        )
    assignments.append(
        "    shared.weighted_z_finite = "
        + " && ".join(f"isfinite(shared.weighted_z_alpha[{g}])" for g in range(ngroups))
        + ";\n"
    )
    publish = "  }\n  __syncthreads();\n\n  const bool component_lane"
    task = task.replace(publish, "".join(assignments) + publish, 1)
    group_expression = "0U"
    if ngroups == 2:
        group_expression = (
            "("
            + " || ".join(
                f"{names[0]} == {i}U" for i, group in enumerate(groups) if group == 1
            )
            + ") ? 1U : 0U"
        )
    task = task.replace(
        "  const bool share_xy = retained_count > 2U;",
        "  const bool share_xy = retained_count > 2U;\n"
        f"  const unsigned weighted_z_group = {group_expression};\n"
        "  // One uniform vote before any primitive/root evaluation or Fock update.\n"
        "  const bool share_weighted_z = share_xy && !__syncthreads_or(\n"
        "      !shared.weighted_z_finite || (retained_by_schwarz &&\n"
        "       !generated_dppp_weighted_z_matches(\n"
        "           angular_coefficient, shared.weighted_z_alpha[weighted_z_group])));",
        1,
    )
    producer = f"""      // Producer ownership is independent of output screening.
      if (share_weighted_z && lane >= 6U && lane < {6 + 3 * ngroups}U) {{
        const unsigned root_index = (lane - 6U) / {ngroups}U;
        const unsigned group = (lane - 6U) % {ngroups}U;
        const GeneratedDpppRys3Primitive& primitive = shared.primitive;
        const double root = shared.roots_weights[2U * root_index];
        const double weighted_root =
            shared.roots_weights[2U * root_index + 1U] *
            primitive.primitive_prefactor * shared.weighted_z_alpha[group];
        const double root_over_sum = root / (primitive.p + primitive.q);
        const double root_bra = root_over_sum * primitive.q;
        const double root_ket = root_over_sum * primitive.p;
        const double b10 = 0.5 / primitive.p * (1.0 - root_bra);
        const double b00 = 0.5 * root_over_sum;
        const double b01 = 0.5 / primitive.q * (1.0 - root_ket);
        generated_dppp_rys3_fill_axis(
            shared.weighted_z[root_index][group],
            primitive.paz - primitive.dz * root_bra,
            primitive.qcz + primitive.dz * root_ket,
            b10, b00, b01, weighted_root);
      }}
"""
    task = task.replace(
        "      __syncthreads();\n      if (retained_by_schwarz) {",
        producer + "      __syncthreads();\n      if (retained_by_schwarz) {",
        1,
    )
    return task.replace(
        "          const double z = generated_dppp_rys3_value_axis(",
        "          const double z = share_weighted_z\n"
        "              ? generated_dppp_rys3_state(\n"
        "                    shared.weighted_z[root_index][weighted_z_group],\n"
        "                    az, bz, cz, dz_order, primitive.abz, primitive.cdz)\n"
        "              : generated_dppp_rys3_value_axis(",
        1,
    )
