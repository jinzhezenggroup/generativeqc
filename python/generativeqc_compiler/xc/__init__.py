"""Audited semilocal XC kernels; this does not register a public DFT method."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .spec import FunctionalSpec, UnsupportedXC, functional

__all__ = [
    "FixedDensityXC",
    "FunctionalSpec",
    "UnsupportedXC",
    "XCIntegral",
    "build_bulk_runtime_program",
    "build_program",
    "functional",
    "pack_grid_features",
    "validate_features",
]


def __getattr__(name: str) -> Any:
    if name in {"FunctionalSpec", "UnsupportedXC", "functional"}:
        # The source importer must be usable before its generated catalog is
        # current. Resolve catalog-backed public types only when requested.
        from .spec import FunctionalSpec, UnsupportedXC, functional

        return {
            "FunctionalSpec": FunctionalSpec,
            "UnsupportedXC": UnsupportedXC,
            "functional": functional,
        }[name]
    if name in {"FixedDensityXC", "XCIntegral"}:
        from .integration import FixedDensityXC, XCIntegral

        return {"FixedDensityXC": FixedDensityXC, "XCIntegral": XCIntegral}[name]
    if name == "build_bulk_runtime_program":
        from .bulk_runtime import build_bulk_runtime_program

        return build_bulk_runtime_program
    if name in {"build_program", "pack_grid_features", "validate_features"}:
        from .program import build_program, pack_grid_features, validate_features

        return {
            "build_program": build_program,
            "pack_grid_features": pack_grid_features,
            "validate_features": validate_features,
        }[name]
    raise AttributeError(name)
