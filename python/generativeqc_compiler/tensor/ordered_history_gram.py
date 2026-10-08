"""One ordered Gram/RHS builder over the retained physical history schedules.

The same scalar stages, logical loops, diagonal shift and phase checks own both
backends. Physical overlay reads and checked CPU helper versus inline CUDA dot
are explicit schedule choices; method bindings provide no arithmetic bodies.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ordered_history import (
    OrderedHistoryProgram,
    RetainedHistorySchedule,
    require_history_schedule,
)
from .ordered_history_emit import (
    FailureEffect,
    _failure_lines,
    _statement,
    emit_dot_loop,
)
from .scalar_cpp import native_scalar_read


@dataclass(frozen=True)
class GramBindings:
    delta_f: str
    new_u: str
    residual: str
    df_history: str
    u_history: str
    weights: str
    new_weight: str
    capacity: str
    coefficients: str
    beta: str
    omega_zero: str
    coefficient_dot_failure: FailureEffect
    coefficient_product_failure: FailureEffect
    overlap_failure: FailureEffect
    matrix_failure: FailureEffect
    history_slots: str = ""
    weight_failure: FailureEffect | None = None

    def validate(self, schedule: RetainedHistorySchedule) -> None:
        for name in (
            "delta_f",
            "new_u",
            "residual",
            "df_history",
            "u_history",
            "weights",
            "new_weight",
            "capacity",
            "coefficients",
            "beta",
            "omega_zero",
        ):
            native_scalar_read(getattr(self, name))
        if schedule.name == "compact-cpu":
            native_scalar_read(self.history_slots)
        else:
            _failure_lines(schedule, "checked-value", self.weight_failure)
        for effect in (
            self.coefficient_dot_failure,
            self.coefficient_product_failure,
            self.overlap_failure,
            self.matrix_failure,
        ):
            _failure_lines(schedule, "checked-value", effect)


def _first_iteration() -> tuple[str, str]:
    return (
        "    const std::uint64_t first_iteration =",
        "        old_iteration - static_cast<std::uint64_t>(history_count) + 1u;",
    )


def emit_history_window(
    program: OrderedHistoryProgram,
    schedule: RetainedHistorySchedule,
    bindings: GramBindings,
) -> str:
    """Retain placement of count/new-slot setup, including CPU slot materialization."""
    require_history_schedule(program, schedule)
    bindings.validate(schedule)
    cpu = schedule.name == "compact-cpu"
    integer, capacity = schedule.index_type, bindings.capacity
    if not cpu:
        return "\n".join(
            (
                f"        history_count = static_cast<{integer}>(",
                f"            min(static_cast<std::uint64_t>({capacity}), old_iteration));",
                f"        new_slot = static_cast<{integer}>((old_iteration - 1u) %",
                f"                                             static_cast<std::uint64_t>({capacity}));",
            )
        )
    lines = [
        f"    const {integer} history_count = static_cast<{integer}>(",
        f"        std::min<std::uint64_t>(static_cast<std::uint64_t>({capacity}), old_iteration));",
        *_first_iteration(),
        f"    const {integer} new_slot =",
        f"        static_cast<{integer}>((old_iteration - 1u) % static_cast<std::uint64_t>({capacity}));",
        f"    for ({integer} history = 0u; history < history_count; ++history) {{",
        "      const std::uint64_t represented_iteration =",
        "          first_iteration + static_cast<std::uint64_t>(history);",
        f"      {bindings.history_slots}[history] = static_cast<std::int64_t>(",
        f"          (represented_iteration - 1u) % static_cast<std::uint64_t>({capacity}));",
        "    }",
        "",
    ]
    for name, tentative, history in (
        ("df_vector", bindings.delta_f, bindings.df_history),
        ("u_vector", bindings.new_u, bindings.u_history),
    ):
        lines.extend(
            (
                f"    const auto {name} = [&]({integer} slot) {{",
                f"      return slot == new_slot ? {tentative}",
                f"                              : {history} + history_offset + slot * dimension;",
                "    };",
            )
        )
    lines.extend(
        (
            f"    const auto slot_omega = [&]({integer} slot) {{",
            f"      return slot == new_slot ? {bindings.new_weight} : {bindings.weights}[system * {capacity} + slot];",
            "    };",
        )
    )
    return "\n".join(lines)


def _select_history(
    schedule: RetainedHistorySchedule, b: GramBindings, role: str
) -> list[str]:
    """Read one logical history row through its retained chronological overlay."""
    pad = "      " if role == "row" else "        "
    if schedule.name == "compact-cpu":
        return [
            f"{pad}const std::size_t {role}_slot = static_cast<std::size_t>({b.history_slots}[{role}]);",
            f"{pad}const double {role}_omega = slot_omega({role}_slot);",
        ]
    represented = f"first_iteration + static_cast<std::uint64_t>({role});"
    lines = (
        [f"{pad}const std::uint64_t represented_{role} = {represented}"]
        if role == "row"
        else [
            f"{pad}const std::uint64_t represented_{role} =",
            f"{pad}    {represented}",
        ]
    )
    lines.extend(
        (
            f"{pad}const std::int64_t {role}_slot = static_cast<std::int64_t>(",
            f"{pad}    (represented_{role} - 1u) % static_cast<std::uint64_t>({b.capacity}));",
        )
    )
    # These retained wrap/indent choices are source spelling only. Both views
    # select the tentative vector/weight at new_slot before touching ring data.
    if role == "row":
        lines.extend(
            (
                f"{pad}const double* const row_df = row_slot == new_slot",
                f"                                       ? {b.delta_f} + vector_begin",
                f"                                       : {b.df_history} + history_begin + row_slot * dimension;",
                f"{pad}const double row_omega =",
                f"{pad}    row_slot == new_slot ? {b.new_weight} : {b.weights}[system * {b.capacity} + row_slot];",
            )
        )
    else:
        lines.extend(
            (
                f"{pad}const double* const column_df =",
                f"{pad}    column_slot == new_slot ? {b.delta_f} + vector_begin",
                f"                                    : {b.df_history} + history_begin + column_slot * dimension;",
                f"{pad}const double column_omega = column_slot == new_slot",
                f"                                        ? {b.new_weight}",
                f"                                        : {b.weights}[system * {b.capacity} + column_slot];",
            )
        )
    return lines


def emit_history_gram(
    program: OrderedHistoryProgram,
    schedule: RetainedHistorySchedule,
    bindings: GramBindings,
) -> str:
    """Emit the full ordered Gram/RHS, not a dense BLAS or mirrored substitute."""
    require_history_schedule(program, schedule)
    b = bindings
    b.validate(schedule)
    cpu = schedule.name == "compact-cpu"
    integer, zero, ns = schedule.index_type, schedule.zero, schedule.math_namespace
    lines = [] if cpu else list(_first_iteration())

    def failure(effect: FailureEffect, indent: int) -> list[str]:
        return [
            " " * indent + line
            for line in _failure_lines(schedule, "checked-value", effect)
        ]

    def dot(
        role: str, accumulator: str, right: str, effect: FailureEffect, indent: int
    ) -> list[str]:
        pad = " " * indent
        weight = f"{role}_omega"
        if cpu:
            return [
                f"{pad}if (!std::isfinite({weight}) ||",
                f"{pad}    !dot_product(df_vector(row_slot), {right}, dimension, {accumulator})) {{",
                *failure(effect, indent + 2),
                f"{pad}}}",
            ]
        return [
            f"{pad}if (!isfinite({weight})) {{",
            *failure(b.weight_failure, indent + 2),
            f"{pad}}}",
            emit_dot_loop(
                program,
                schedule,
                accumulator=accumulator,
                left="row_df[component]",
                right=right,
                index="component",
                extent="dimension",
                failure=effect,
                indent=indent,
            ),
        ]

    start, step = (zero, "++row") if cpu else ("threadIdx.x", "row += blockDim.x")
    lines.append(f"    for ({integer} row = {start}; row < history_count; {step}) {{")
    lines.extend(_select_history(schedule, b, "row"))
    lines.append("      double coefficient_dot = 0.0;")
    rhs = b.residual if cpu else f"{b.residual}[vector_begin + component]"
    lines.extend(dot("row", "coefficient_dot", rhs, b.coefficient_dot_failure, 6))
    coefficient = f"{b.coefficients}[row]" if cpu else "coefficient"
    product = _statement(
        program,
        schedule,
        "weighted_rhs",
        coefficient,
        {"omega": "row_omega", "value": "coefficient_dot"},
    )
    lines.append("      " + ("" if cpu else "const double ") + product)
    if not cpu:
        lines.append(f"      {b.coefficients}[coefficient_begin + row] = coefficient;")
    lines.extend(
        (
            f"      if (!{ns}isfinite({coefficient})) {{",
            *failure(b.coefficient_product_failure, 8),
            "      }",
            f"      for ({integer} column = {zero}; column < history_count; ++column) {{",
        )
    )
    lines.extend(_select_history(schedule, b, "column"))
    lines.append("        double overlap = 0.0;")
    lines.extend(
        dot(
            "column",
            "overlap",
            "df_vector(column_slot)" if cpu else "column_df[component]",
            b.overlap_failure,
            8,
        )
    )
    product = _statement(
        program,
        schedule,
        "weighted_gram",
        "value",
        {
            "row_weight": "row_omega",
            "column_weight": "column_omega",
            "overlap": "overlap",
        },
    )
    shift = _statement(
        program,
        schedule,
        "diagonal_shift",
        "value",
        {"value": "value", "omega_zero": b.omega_zero},
        "+=",
    )
    lines.extend(
        (
            "        double " + product,
            "        if (row == column) {",
            "          " + shift,
            "        }",
        )
    )
    address = (
        "row * history_count + column"
        if cpu
        else f"beta_begin + row * {b.capacity} + column"
    )
    store = f"        {b.beta}[{address}] = value;"
    if not cpu:
        lines.append(store)
    lines.extend(
        (
            f"        if (!{ns}isfinite(value)) {{",
            *failure(b.matrix_failure, 10),
            "        }",
        )
    )
    if cpu:
        lines.append(store)
    lines.extend(("      }", "    }"))
    return "\n".join(lines)
