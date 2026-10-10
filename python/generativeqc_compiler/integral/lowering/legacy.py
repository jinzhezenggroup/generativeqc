"""Resident-PPPS adapter to the general shell lowering.\n\nHistorical DPPP wrappers were retired after all repository callers migrated to\nthe generic fused-shell plan and emitter APIs.\n"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..cuda_schedule import (
    ScheduleIR,
    ScheduleKind,
)
from ..fused_schedule import (
    build_fused_shell_plan,
)
from ..ir import IntegralIR, build_integral_ir
from ..shell_spec import (
    FUSED_SHELL_SPEC_BY_NAME,
)
from .common import _specialize_dppp_identifiers
from .dispatch import emit_shell_class_fused_cuda
from .force_resident import _emit_ppps_resident_bra_rys3_force_consumer_cuda

if TYPE_CHECKING:
    from generativeqc_compiler.common.cuda_target import CudaTargetInfo


def emit_ppps_resident_bra_rys3_cuda(
    *,
    target: CudaTargetInfo,
    include_shared_definitions: bool = True,
    include_rys3_roots: bool = True,
    integral: IntegralIR | None = None,
) -> str:
    """Emit the scalar ``ppps`` resident-bra Rys3 worker.

    The generated kernel has a 256-thread launch bound and shared-memory
    capacity, while its task and bra-staging strides use the actual block
    dimension. Production can therefore compare 32/64/128/256-thread CTAs
    from one binary without changing scalar quartet ownership.

    ``target`` is the explicit CUDA compilation target; no device-specific
    fallback is inferred by this compatibility adapter.

    ``integral`` carries explicit derivative-center and translation-recovery
    metadata into both the ordinary shared-definition prefix and the resident
    force consumer. Omitting it preserves the canonical four-center ERI
    default (centers 0, 1, and 2 independent; center 3 recovered).

    ``include_shared_definitions`` keeps the standalone correctness harness
    self-contained.  Production AOT shards already contain the ordinary ppps
    shell definitions, so they request only the resident-specific tail to
    avoid duplicate CUDA type/function definitions.  A subset/Wick ordinary
    ppps row also lacks the Rys evaluator; ``include_rys3_roots`` therefore
    controls that one additional shared dependency independently.
    """

    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    selected_integral = integral or build_integral_ir(spec, recurrence="rys3")
    if selected_integral.spec != spec:
        raise ValueError("resident ppps integral spec does not match ppps")
    if selected_integral.recurrence != "rys3":
        raise ValueError("resident ppps lowering requires an rys3 integral")
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
    )
    plan = build_fused_shell_plan(
        spec,
        integral=selected_integral,
        schedule=schedule,
        target=target,
    )
    resident_tail = _emit_ppps_resident_bra_rys3_force_consumer_cuda(
        include_rys3_roots=include_rys3_roots,
        integral=selected_integral,
    )
    if include_shared_definitions:
        source = emit_shell_class_fused_cuda(spec, plan)
        # The existing Rys thread consumer starts with the attributed roots
        # table; cut before that table so the standalone source does not
        # retain an unused 27-entry shared weight helper from the one-task
        # prototype.
        marker = """/*
 * Three-root interpolation adapted from GPU4PySCF."""
        marker_index = source.find(marker)
        if marker_index < 0:
            raise RuntimeError("generated force task marker changed unexpectedly")
        prefix = source[:marker_index]
    else:
        prefix = ""
        # The resident tail references the normal generated ppps task/cache
        # helpers.  The production shard places it directly after the normal
        # ppps emitter, where those definitions are already available.
    return prefix + _specialize_dppp_identifiers(resident_tail, spec)
