"""Shared source-preserving builders for the retained history schedules.

Dot, Cholesky and correction fragments serve the candidate-selected production
consumers. Arithmetic statements always come from TensorIR; there are no copied
complete CPU/CUDA bodies or alternate scalar equations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .ordered_history import (
    OrderedHistoryProgram,
    RetainedHistorySchedule,
    require_history_schedule,
)
from .scalar_cpp import _identifier, emit_scalar_cpp_statement, native_scalar_read


@dataclass(frozen=True)
class ReturnFalseFailure:
    """A checked CPU dot returns immediately without publishing its result."""


@dataclass(frozen=True)
class ReturnStatusFailure:
    """Method-owned status/error binding; the emitter owns immediate return."""

    function: str
    state: str
    system: str
    message: str
    error: str
    message_line: str = "first"

    def __post_init__(self) -> None:
        _identifier(self.function, "status failure function")
        for value in (self.state, self.system, self.error):
            native_scalar_read(value)
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("status failure requires a nonempty diagnostic")
        if self.message_line not in ("first", "second"):
            raise ValueError("unsupported status diagnostic spelling")


@dataclass(frozen=True)
class CudaFlagFailure:
    """Mandatory error reporting and atomic invalidation, with no raw body.

    The enclosing phase chooses break-dot versus continue-to-barrier. Neither
    effect can be omitted or replaced by a thread-local return in this binding.
    """

    function: str
    error_buffer: str
    code: str
    valid: str

    def __post_init__(self) -> None:
        _identifier(self.function, "CUDA failure function")
        for value in (self.error_buffer, self.code, self.valid):
            native_scalar_read(value)


FailureEffect = ReturnFalseFailure | ReturnStatusFailure | CudaFlagFailure


def _failure_lines(
    schedule: RetainedHistorySchedule, phase: str, failure: FailureEffect
) -> tuple[str, ...]:
    if schedule.name == "compact-cpu":
        if phase == "dot" and isinstance(failure, ReturnFalseFailure):
            return ("return false;",)
        if phase in ("correction", "checked-value") and isinstance(
            failure, ReturnStatusFailure
        ):
            if failure.message_line == "second":
                return (
                    f"return {failure.function}({failure.state}, {failure.system},",
                    f"                              {json.dumps(failure.message)}, {failure.error});",
                )
            return (
                f"return {failure.function}({failure.state}, {failure.system}, {json.dumps(failure.message)},",
                f"                              {failure.error});",
            )
    elif isinstance(failure, CudaFlagFailure):
        lines = (
            f"{failure.function}({failure.error_buffer}, {failure.code});",
            f"atomicExch(&{failure.valid}, 0);",
        )
        return (*lines, "break;") if phase == "dot" else lines
    raise ValueError("failure effect does not implement the retained phase schedule")


def _statement(
    program: OrderedHistoryProgram,
    schedule: RetainedHistorySchedule,
    stage: str,
    target: str,
    bindings: dict[str, str],
    form: str = "=",
    read_accessors: tuple[str, ...] = (),
) -> str:
    return emit_scalar_cpp_statement(
        program.stage(stage),
        bindings=bindings,
        target=target,
        form=form,
        math_namespace=schedule.math_namespace,
        read_accessors=read_accessors,
    )


def emit_dot_loop(
    program: OrderedHistoryProgram,
    schedule: RetainedHistorySchedule,
    *,
    accumulator: str,
    left: str,
    right: str,
    index: str,
    extent: str,
    failure: FailureEffect,
    indent: int,
) -> str:
    """One common ordered dot, with retained return versus flag/break effects."""
    require_history_schedule(program, schedule)
    failure_lines = _failure_lines(schedule, "dot", failure)
    _identifier(index, "dot component index")
    native_scalar_read(extent)
    if type(indent) is not int or indent < 0:
        raise ValueError("dot indentation must be a nonnegative integer")
    pad = " " * indent
    update = _statement(
        program,
        schedule,
        "dot_update",
        accumulator,
        {"accumulator": accumulator, "left": left, "right": right},
        "+=",
    )
    return "\n".join(
        (
            f"{pad}for ({schedule.index_type} {index} = {schedule.zero}; {index} < {extent}; ++{index}) {{",
            f"{pad}  {update}",
            f"{pad}  if (!{schedule.math_namespace}isfinite({accumulator})) {{",
            *(f"{pad}    {line}" for line in failure_lines),
            f"{pad}  }}",
            f"{pad}}}",
        )
    )


def emit_cpu_dot(
    program: OrderedHistoryProgram, schedule: RetainedHistorySchedule
) -> str:
    require_history_schedule(program, schedule)
    if schedule.name != "compact-cpu":
        raise ValueError("standalone dot helper requires the compact CPU schedule")
    return "\n".join(
        (
            "bool dot_product(const double* first, const double* second, std::size_t count, double& result) {",
            "  double sum = 0.0;",
            emit_dot_loop(
                program,
                schedule,
                accumulator="sum",
                left="first[index]",
                right="second[index]",
                index="index",
                extent="count",
                failure=ReturnFalseFailure(),
                indent=2,
            ),
            "  result = sum;",
            "  return true;",
            "}",
        )
    )


def emit_cholesky(
    program: OrderedHistoryProgram, schedule: RetainedHistorySchedule
) -> str:
    """One strided recurrence builder; CPU binds its stride to compact dimension."""
    require_history_schedule(program, schedule)
    cpu = schedule.name == "compact-cpu"
    integer, zero, one, ld, ns = (
        schedule.index_type,
        schedule.zero,
        schedule.one,
        schedule.leading_dimension,
        schedule.math_namespace,
    )

    def matrix(row: str, column: str) -> str:
        return f"matrix[{row} * {ld} + {column}]"

    def subtract(left: str, right: str) -> str:
        return _statement(
            program,
            schedule,
            "subtract_product",
            "value",
            {"accumulator": "value", "left": left, "right": right},
            "-=",
        )

    def quotient(diagonal: str) -> str:
        return _statement(
            program,
            schedule,
            "divide",
            "value",
            {"value": "value", "diagonal": diagonal},
            "/=",
        )

    signature = f"bool cholesky_solve(double* matrix, double* right_hand_side, {integer} dimension"
    if not cpu:
        signature = (
            "__device__ "
            + signature
            + ",\n                               std::int64_t leading_dimension"
        )
    lines = [signature + ") {"]
    if cpu:
        lines.append(
            "  /* beta is symmetric positive definite because omega0^2 is added to I. */"
        )
    update = subtract(matrix("row", "inner"), matrix("column", "inner"))
    if not cpu:
        update = update.replace(" -= ", " -=\n            ")
    root = _statement(
        program, schedule, "sqrt", matrix("row", "column"), {"value": "value"}
    )
    lines.extend(
        (
            f"  for ({integer} row = {zero}; row < dimension; ++row) {{",
            f"    for ({integer} column = {zero}; column <= row; ++column) {{",
            f"      double value = {matrix('row', 'column')};",
            f"      for ({integer} inner = {zero}; inner < column; ++inner) {{",
            f"        {update}",
            "      }",
            f"      if (!{ns}isfinite(value)) {{",
            "        return false;",
            "      }",
            "      if (row == column) {",
            "        if (!(value > 0.0)) {",
            "          return false;",
            "        }",
            f"        {root}",
            "      } else {",
        )
    )
    diagonal = matrix("column", "column")
    if cpu:
        lines.append(f"        const double diagonal = {diagonal};")
        diagonal = "diagonal"
    lines.extend(
        (
            f"        {quotient(diagonal)}",
            f"        if (!{ns}isfinite(value)) {{",
            "          return false;",
            "        }",
            f"        {matrix('row', 'column')} = value;",
            "      }",
            "    }",
            "  }",
        )
    )
    if cpu:
        lines.append("")
    # Both substitutions share the same scalar fold and publication builder.
    for reverse in (False, True):
        lines.append(
            f"  for ({integer} reverse = dimension; reverse > {zero}; --reverse) {{"
            if reverse
            else f"  for ({integer} row = {zero}; row < dimension; ++row) {{"
        )
        if reverse:
            lines.append(f"    const {integer} row = reverse - {one};")
        lines.append("    double value = right_hand_side[row];")
        start, end = (f"row + {one}", "dimension") if reverse else (zero, "row")
        lines.append(
            f"    for ({integer} column = {start}; column < {end}; ++column) {{"
        )
        factor = matrix("column", "row") if reverse else matrix("row", "column")
        lines.extend(
            (
                f"      {subtract(factor, 'right_hand_side[column]')}",
                "    }",
                f"    {quotient(matrix('row', 'row'))}",
                f"    if (!{ns}isfinite(value)) {{",
                "      return false;",
                "    }",
                "    right_hand_side[row] = value;",
                "  }",
            )
        )
    return "\n".join((*lines, "  return true;", "}"))


@dataclass(frozen=True)
class CorrectionBindings:
    """Physical reads and failure destinations, never arithmetic bodies.

    The retained envelope supplies dimension/history_count, component/history,
    first_iteration/new_slot and, on CUDA, vector_begin/history_begin/system.
    """

    current: str
    damping: str
    residual: str
    coefficient: str
    mixed: str
    failure: FailureEffect
    history_slots: str = ""
    weight_accessor: str = ""
    vector_accessor: str = ""
    capacity: str = ""
    tentative_weight: str = ""
    weights: str = ""
    tentative_vectors: str = ""
    vectors: str = ""


def emit_correction(
    program: OrderedHistoryProgram,
    schedule: RetainedHistorySchedule,
    bindings: CorrectionBindings,
) -> str:
    """Common seeded fold with two retained physical bindings/effects."""
    require_history_schedule(program, schedule)
    cpu = schedule.name == "compact-cpu"
    failure_lines = _failure_lines(schedule, "correction", bindings.failure)
    for value in (
        bindings.current,
        bindings.damping,
        bindings.residual,
        bindings.coefficient,
        bindings.mixed,
    ):
        native_scalar_read(value)
    if cpu:
        native_scalar_read(bindings.history_slots)
        _identifier(bindings.weight_accessor, "history weight accessor")
        _identifier(bindings.vector_accessor, "history vector accessor")
        accessors = (bindings.weight_accessor, bindings.vector_accessor)
    else:
        for value in (
            bindings.capacity,
            bindings.tentative_weight,
            bindings.weights,
            bindings.tentative_vectors,
            bindings.vectors,
        ):
            native_scalar_read(value)
        accessors = ()
    integer, zero, ns = schedule.index_type, schedule.zero, schedule.math_namespace
    seed = _statement(
        program,
        schedule,
        "seed",
        "value",
        {
            "current": bindings.current,
            "damping": bindings.damping,
            "residual": bindings.residual,
        },
    )
    omega = f"{bindings.weight_accessor}(slot)" if cpu else "omega"
    vector = f"{bindings.vector_accessor}(slot)[component]" if cpu else "u[component]"
    correction = _statement(
        program,
        schedule,
        "correction",
        "value",
        {
            "accumulator": "value",
            "omega": omega,
            "coefficient": bindings.coefficient,
            "update": vector,
        },
        "-=",
        read_accessors=accessors,
    )
    start, step = (
        (zero, "++component") if cpu else ("threadIdx.x", "component += blockDim.x")
    )
    lines = [
        f"    for ({integer} component = {start}; component < dimension; {step}) {{"
    ]
    if not cpu:
        lines.append("      const std::int64_t index = vector_begin + component;")
    lines.extend(
        (
            f"      double {seed}",
            f"      for ({integer} history = {zero}; history < history_count; ++history) {{",
        )
    )
    if cpu:
        lines.append(
            f"        const std::size_t slot = static_cast<std::size_t>({bindings.history_slots}[history]);"
        )
    else:
        lines.extend(
            (
                "        const std::uint64_t represented = first_iteration + static_cast<std::uint64_t>(history);",
                "        const std::int64_t slot = static_cast<std::int64_t>(",
                f"            (represented - 1u) % static_cast<std::uint64_t>({bindings.capacity}));",
                "        const double omega =",
                f"            slot == new_slot ? {bindings.tentative_weight} : {bindings.weights}[system * {bindings.capacity} + slot];",
                "        const double* const u = slot == new_slot",
                f"                                    ? {bindings.tentative_vectors} + vector_begin",
                f"                                    : {bindings.vectors} + history_begin + slot * dimension;",
            )
        )
    lines.extend((f"        {correction}", "      }"))
    store = f"      {bindings.mixed} = value;"
    if not cpu:
        lines.append(store)
    lines.append(f"      if (!{ns}isfinite(value)) {{")
    lines.extend("        " + line for line in failure_lines)
    lines.append("      }")
    if cpu:
        lines.append(store)
    lines.append("    }")
    return "\n".join(lines)
