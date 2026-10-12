"""Complete conventional UCCSD(T) forces sharing all native nuclear consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from .uccsd_gradient import (
    UCCSDGradientOptions,
    _PreparedUCCSDGradient,
    _UCCSDGradientResult,
)
from .uccsd_t_physical_response import (
    PreparedUCCSDTPhysicalResponse,
    UCCSDTRelaxedCotangents,
)

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class UCCSDTGradientResult(_UCCSDGradientResult):
    """Detached corrected total energy, positive gradient and negative forces."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.provenance.get("method") != "UCCSD(T)":
            raise ValueError("corrected nuclear result requires UCCSD(T) provenance")


class PreparedUCCSDTGradient(_PreparedUCCSDGradient):
    """Bind corrected physical weights to the shared shell-local derivative chain.

    CC, corrected Lambda, both fresh triples responses and all-rotation orbital
    stationarity must pass before any force record is published. This owner
    introduces no alternate nuclear algebra and closes no borrowed consumer.
    """

    def __init__(
        self,
        physical: PreparedUCCSDTPhysicalResponse,
        *,
        options: UCCSDGradientOptions | None = None,
    ) -> None:
        super().__init__(physical, options=options)

    def _validate_physical(self, physical: Any) -> None:
        if not isinstance(physical, PreparedUCCSDTPhysicalResponse):
            raise TypeError("corrected gradient requires a corrected physical owner")

    def _make_result(
        self, relaxed: UCCSDTRelaxedCotangents, gradient: np.ndarray, provenance: dict
    ) -> UCCSDTGradientResult:
        return UCCSDTGradientResult(
            relaxed.total_energy,
            gradient,
            -gradient,
            relaxed.response_identity,
            {**provenance, "method": "UCCSD(T)"},
        )

    def solve(self) -> UCCSDTGradientResult:
        """Publish corrected energies/forces only after every fresh native gate."""
        return cast("UCCSDTGradientResult", super().solve())
