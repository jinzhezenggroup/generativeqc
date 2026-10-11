"""Dependency-light TensorIR stationary integral weight specialization.

The production CUDA and portable CPU generators share this exact lowering.
Importing source generation must not load NumPy-backed molecular grid execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from generativeqc_compiler.tensor.cuda_inline import (
    InlineCudaOutput,
    lower_inline_cuda_output,
)

if TYPE_CHECKING:
    from .stationary_gradient import StationaryGradientPlan


def _weight_expression(
    plan: StationaryGradientPlan, source: str, *, dimension: str = "n"
) -> tuple[InlineCudaOutput, int]:
    """Specialize one generated weight through the shared inline-consumer path."""
    program = plan.integral_block(source, terms=1).weights
    locations = {
        "density_left": ("density", 0, 1),
        "density_right": ("density", 2, 3),
        "weighted_density": ("weighted_density", 0, 1),
    }
    if source == "exact_exchange":
        # Ordered (ab|cd) derivatives contract D[a,c] D[b,d], independently
        # in each spin channel. Coulomb instead uses D[a,b] D[c,d].
        locations["density_left"] = ("density", 0, 2)
        locations["density_right"] = ("density", 1, 3)
    bindings = {
        name: tuple(
            f"{pointer}[{spin} * {dimension} * {dimension} + size_t(ao[{left}]) * {dimension} + size_t(ao[{right}])]"
            for spin in range(plan.spin_blocks)
        )
        for name, (pointer, left, right) in locations.items()
    }
    lowered = lower_inline_cuda_output(program, output="weights", bindings=bindings)
    required = set(lowered.required_inputs)
    arity = max(
        (right + 1 for name, (_, _, right) in locations.items() if name in required),
        default=0,
    )
    return lowered, arity
