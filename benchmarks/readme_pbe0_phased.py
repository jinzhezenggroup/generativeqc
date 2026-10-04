"""Qualification-only switch for the shared compiler Becke phase schedule.

Everything else uses the ordinary README PBE0 protocol. The public default is
unchanged. Report native phase metrics to prove execution: an admitted optional
reservation can still take the explicit capability/allocation fallback.
"""

from unittest.mock import patch

from generativeqc import _stationary_cuda
from generativeqc_compiler.method.stationary_resources import (
    StationaryCudaResources,
    plan_stationary_cuda_resources,
)

from benchmarks.readme_pbe0 import main as readme_pbe0


def main() -> None:
    """Apply the same plan choice to complete-owner admission and allocation."""

    def phased_plan(**arguments: object) -> StationaryCudaResources:
        arguments["phased_becke"] = True
        return plan_stationary_cuda_resources(**arguments)

    with patch.object(_stationary_cuda, "plan_stationary_cuda_resources", phased_plan):
        readme_pbe0()


if __name__ == "__main__":
    main()
