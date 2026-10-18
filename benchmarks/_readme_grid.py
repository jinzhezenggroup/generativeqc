"""Explicit quadrature identities and pre-screening work for README endpoints."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from generativeqc import GridSpec

if TYPE_CHECKING:
    from typing import Any


FP64_GRID_ENERGY_TOLERANCE = 1e-8
FP64_GRID_FORCE_TOLERANCE = 1e-7


def parse_grid(value: str) -> tuple[int, int, int]:
    """Parse radial, polar and azimuthal counts using the production size limits."""
    try:
        dimensions = tuple(int(part) for part in value.split(","))
        if len(dimensions) != 3:
            raise ValueError("expected three dimensions")
        GridSpec(*dimensions)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid grid {value!r}: {error}") from error
    return dimensions


def grid_directory(dimensions: tuple[int, int, int]) -> str:
    """Keep each explicit grid in a separate timing and accuracy population."""
    GridSpec(*dimensions)
    return "grid-" + "x".join(str(count) for count in dimensions)


def grid_work(atoms: int, spec: GridSpec, *, has_vv10: bool) -> dict[str, Any]:
    """Describe the unpruned grid, not measured active or executed VV10 pairs.

    Native KS currently integrates semilocal XC and VV10 on the same grid.
    Density screening can reduce the pair domain; its square before screening
    is useful for sizing a sweep but must never be reported as executed work.
    """
    dimensions = (spec.radial_points, spec.angular_polar, spec.angular_azimuth)
    per_atom = math.prod(dimensions)
    points = atoms * per_atom
    return {
        "dimensions": list(dimensions),
        "points_per_atom": per_atom,
        "semilocal_points": points,
        "vv10_points": points if has_vv10 else 0,
        "vv10_grid_policy": "same_as_semilocal" if has_vv10 else "not_applicable",
        "vv10_unscreened_ordered_pair_upper_bound": points**2 if has_vv10 else 0,
    }
