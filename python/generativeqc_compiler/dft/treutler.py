"""Pinned Treutler/Ahlrichs radial, Lebedev angular and NWChem pruning rules.

Only immutable element-local quadratures are prepared here. Molecular points,
partition weights and their derivatives remain owned by the grid consumers.
Tables originate in PySCF 2.14.0; see THIRD_PARTY_NOTICES.md and the source
hashes in data/treutler-lebedev.json. No PySCF installation is needed at runtime
or during native source generation.
"""

from __future__ import annotations

import itertools
import json
import math
import typing
from functools import lru_cache
from pathlib import Path

from generativeqc_compiler.common.provenance import canonical_hash


@lru_cache(maxsize=1)
def tables() -> dict:
    """Load the versioned numeric input; callers must not mutate its tables."""
    return json.loads(
        Path(__file__).with_name("data").joinpath("treutler-lebedev.json").read_text()
    )


@lru_cache(maxsize=1)
def table_identity() -> str:
    """Include retained numeric inputs in scientific identities, not just a label.

    Tables are immutable for the lifetime of an installed compiler. Updating
    their contents must invalidate prepared identities even if a future import
    accidentally retains the old upstream/version strings.
    """
    return canonical_hash(tables())


def validate_element_level(atomic_number: int, level: int) -> None:
    """Reject unavailable radii instead of inventing heavy-element parameters."""
    if type(level) is not int or not 0 <= level <= 9:
        raise ValueError("Treutler grid level must be an integer in [0, 9]")
    if type(atomic_number) is not int or not 1 <= atomic_number < len(
        tables()["treutler_xi"]
    ):
        raise ValueError("Treutler atomic radii are available only for H through Lr")


def default_counts(atomic_number: int, level: int) -> tuple[int, int]:
    """Resolve the upstream element-period-dependent radial/angular orders."""
    validate_element_level(atomic_number, level)
    period = sum(atomic_number > edge for edge in (2, 10, 18, 36, 54, 86, 118))
    data = tables()
    order = data["angular_orders"][level][period]
    return data["radial_counts"][level][period], data["lebedev_orders"][str(order)]


@lru_cache(maxsize=40)
def angular_rule(count: int) -> tuple[tuple[float, float, float, float], ...]:
    """Expand octahedral symmetry orbits in deterministic canonical order.

    Lebedev weights sum to one, including signed weights in several low-order
    rules. The radial consumer supplies 4*pi; weights are never renormalized.
    """
    try:
        orbits = tables()["orbits"][str(count)]
    except KeyError as error:
        raise ValueError(f"unsupported Lebedev point count: {count}") from error
    result = []
    for code, a, b, weight in orbits:
        base = {
            0: lambda: (1.0, 0.0, 0.0),
            1: lambda: (math.sqrt(0.5), math.sqrt(0.5), 0.0),
            2: lambda: (math.sqrt(1 / 3),) * 3,
            3: lambda a=a: (a, a, math.sqrt(1 - 2 * a * a)),
            4: lambda a=a: (a, math.sqrt(1 - a * a), 0.0),
            5: lambda a=a, b=b: (a, b, math.sqrt(1 - a * a - b * b)),
        }[int(code)]()
        points = {
            tuple(
                0.0 if x == 0 else sign * x for x, sign in zip(perm, signs, strict=True)
            )
            for perm in itertools.permutations(base)
            for signs in itertools.product((-1, 1), repeat=3)
        }
        result.extend((*point, weight) for point in sorted(points))
    if len(result) != count:
        raise ValueError("Lebedev orbit inventory has an inconsistent point count")
    return tuple(result)


def radial_rule(atomic_number: int, count: int) -> tuple[tuple[float, float], ...]:
    """Return increasing radii and dr weights for the Treutler M4 transform."""
    validate_element_level(atomic_number, 3)
    if type(count) is not int or not 1 <= count <= 512:
        raise ValueError("Treutler radial count must be an integer in [1, 512]")
    scale = tables()["treutler_xi"][atomic_number] / math.log(2)
    step = math.pi / (count + 1)
    result = []
    for index in range(count, 0, -1):
        angle = index * step
        x = math.cos(angle)
        logarithm = math.log((1 - x) / 2)
        factor = scale * (1 + x) ** 0.6
        result.append(
            (
                -factor * logarithm,
                step
                * math.sin(angle)
                * factor
                * (-0.6 * logarithm / (1 + x) + 1 / (1 - x)),
            )
        )
    return tuple(result)


def angular_count(atomic_number: int, radius: float, maximum: int) -> int:
    """Apply NWChem's fixed element/radius pruning, independent of density."""
    if maximum < 50:
        return maximum
    counts = sorted(int(key) for key in tables()["orbits"] if int(key) >= 38)
    if maximum == 50:
        indices = (1, 2, 2, 2, 1)
    else:
        index = counts.index(maximum)
        indices = (1, 3, index - 1, index, index - 1)
    edges = (
        (0.25, 0.5, 1.0, 4.5)
        if atomic_number <= 2
        else (0.1667, 0.5, 0.9, 3.5)
        if atomic_number <= 10
        else (0.1, 0.4, 0.8, 2.5)
    )
    scaled = radius / tables()["bragg_radii_bohr"][atomic_number]
    return counts[indices[sum(scaled > edge for edge in edges)]]


@lru_cache(maxsize=1040, typed=True)
def atomic_point_count(atomic_number: int, level: int) -> int:
    """Count shells without expanding or retaining angular quadrature arrays."""
    count, maximum = default_counts(atomic_number, level)
    return sum(
        angular_count(atomic_number, r, maximum)
        for r, _ in radial_rule(atomic_number, count)
    )


@lru_cache(maxsize=128, typed=True)
def atomic_rule(atomic_number: int, level: int) -> typing.Any:
    """Return immutable relative xyz/raw weights, with no molecular partition."""
    import numpy as np

    radial_count, maximum = default_counts(atomic_number, level)
    chunks = []
    for radius, dr in radial_rule(atomic_number, radial_count):
        directions = np.asarray(
            angular_rule(angular_count(atomic_number, radius, maximum))
        )
        chunks.append(
            np.column_stack(
                (
                    radius * directions[:, :3],
                    4 * math.pi * radius * radius * dr * directions[:, 3],
                )
            )
        )
    combined = np.concatenate(chunks)
    return np.frombuffer(combined.tobytes(), dtype=np.float64).reshape(-1, 4)


def radii_adjustment(first: float, second: float) -> float:
    """Treutler's bounded heteronuclear correction for positive Bragg radii."""
    if not (
        math.isfinite(first) and math.isfinite(second) and first > 0 and second > 0
    ):
        raise ValueError("partition radii must be finite and positive")
    ratio = math.sqrt(float(second) / float(first))
    # A positive ratio can underflow before sqrt. Its clipped limit is exact.
    if ratio == 0.0:
        return -0.5
    return max(-0.5, min(0.5, 0.25 * (ratio - 1 / ratio)))
