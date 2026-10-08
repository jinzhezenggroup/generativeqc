"""Resource-bounded capability for lane-local, value-only Rys quartets."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ir import IntegralIR


def task_parallel_rys_eligible(integral: IntegralIR) -> bool:
    """Bound the fully unrolled lane state without claiming a tuning winner.

    Larger Cartesian blocks retain the component decoder. This boundary owns
    source capability only; prepared K selection and performance qualification
    remain independent of angular-class coverage.
    """
    return (
        integral.derivative is None
        and integral.recurrence.startswith("rys")
        and integral.required_rys_roots in (1, 2)
        and max(integral.spec.angular) <= 2
        and integral.spec.angular[3] <= 1
        and integral.spec.component_count <= 27
    )
