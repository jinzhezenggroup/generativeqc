"""Explicit AOT packaging of existing scalar-AD integral leaves.

This is a source/metadata adapter, not another integral recurrence or molecular
Hessian implementation. Unit-charge attraction and fixed WB97M-V range omega
keep physical charges and contraction scales in native primitive records.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

from generativeqc_compiler.common.provenance import canonical_hash

from .ir import CoulombKernel, four_center_eri_operator
from .ir_serialization import integral_to_payload
from .resident_second_cuda import emit_resident_df_second, emit_resident_second
from .second_derivatives import build_eri_second_ir, build_one_electron_second_ir
from .shell_spec import cartesian_components

FAMILIES = (
    "overlap",
    "kinetic",
    "nuclear_attraction",
    "eri",
    "coulomb_metric",
    "three_center_eri",
)
RADIALS = ("full_range", "long_range", "short_range")
RANGE_OMEGA = 0.3


@dataclass(frozen=True)
class IntegralRequest:
    """An exact raw Cartesian component tile, never an approximate wildcard.

    All mathematical centers survive until physical-atom scatter. A missing
    class/component is an explicit native package miss, not a JIT fallback.
    """

    family: str
    angular: tuple[int, ...]
    components: tuple[int, ...]
    radial: str = "full_range"

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError("unknown Hessian AOT integral family")
        shells = (
            4 if self.family == "eri" else 3 if self.family == "three_center_eri" else 2
        )
        if (
            type(self.angular) is not tuple
            or len(self.angular) != shells
            or any(
                type(order) is not int or not 0 <= order <= 3 for order in self.angular
            )
        ):
            raise ValueError(
                "Hessian AOT integrals require explicit s/p/d/f shell orders"
            )
        size = prod(len(cartesian_components(order)) for order in self.angular)
        if (
            type(self.components) is not tuple
            or not 1 <= len(self.components) <= 16
            or any(
                type(index) is not int or not 0 <= index < size
                for index in self.components
            )
            or tuple(sorted(set(self.components))) != self.components
        ):
            raise ValueError(
                "Hessian AOT requires one to sixteen ordered distinct raw components"
            )
        if self.radial not in RADIALS or (
            self.family != "eri" and self.radial != "full_range"
        ):
            raise ValueError(
                "only ERI Hessian AOT programs carry range-separated operators"
            )

    @property
    def name(self) -> str:
        orders = "_".join(str(value) for value in self.angular)
        components = "_".join(str(value) for value in self.components)
        return f"{self.family}_{self.radial}_{orders}_c{components}"

    @property
    def centers(self) -> int:
        return len(self.angular) + (self.family == "nuclear_attraction")

    @property
    def record_stride(self) -> int:
        return len(self.angular) + 3 * self.centers + 1

    def payload(self) -> dict:
        return {
            "family": self.family,
            "angular": self.angular,
            "components": self.components,
            "radial": self.radial,
            "omega": RANGE_OMEGA if self.radial != "full_range" else 0,
            "charge": 1,
            "packing": "value-gradient-gradient-jvp",
        }


def integral_request(payload: Any) -> IntegralRequest:
    """Reject ignored scientific controls rather than silently changing a model."""
    if not isinstance(payload, dict) or set(payload) - {
        "family",
        "angular",
        "components",
        "radial",
    }:
        raise ValueError("invalid Hessian AOT integral inventory fields")
    if not {"family", "angular", "components"} <= payload.keys():
        raise ValueError("incomplete Hessian AOT integral inventory")
    if not isinstance(payload["angular"], list) or not isinstance(
        payload["components"], list
    ):
        raise TypeError("Hessian AOT angular/components must be arrays")
    return IntegralRequest(
        payload["family"],
        tuple(payload["angular"]),
        tuple(payload["components"]),
        payload.get("radial", "full_range"),
    )


def emit_integral(request: IntegralRequest) -> tuple[str, str]:
    """Return unchanged compiler-AD arithmetic and scientific package identity."""
    if request.family in ("coulomb_metric", "three_center_eri"):
        from .df_derivatives import build_df_derivative_ir

        integral = build_df_derivative_ir(request.family, request.angular)
        source = emit_resident_df_second(
            request.family, request.angular, request.components
        )
    else:
        integral = (
            build_eri_second_ir(
                request.angular,
                operator=four_center_eri_operator(
                    CoulombKernel(
                        request.radial,
                        RANGE_OMEGA if request.radial != "full_range" else 0,
                    )
                ),
            )
            if request.family == "eri"
            else build_one_electron_second_ir(
                request.family, request.angular, charge=1.0, output="weighted_hvp"
            )
        )
        source = emit_resident_second(integral, request.components)
    identity = canonical_hash(
        {
            "schema": "generativeqc.hessian-integral-aot/v1",
            "request": request.payload(),
            "integral": integral_to_payload(integral),
        }
    )
    return source, identity
