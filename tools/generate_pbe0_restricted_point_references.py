"""Reproduce the frozen restricted PBE0 oracle without native/scaled algebra.

Run with the reference-test dependencies:
``python -m tools.generate_pbe0_restricted_point_references --check``.
The physical inputs remain frozen so libm differences cannot change the tested
population. This is an offline reference tool, never a production consumer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import mpmath as mp
import numpy as np

from tools import generate_xc_scf_references as original_formula

if TYPE_CHECKING:
    from typing import Any


def pbe0_energy(coordinates: list[Any]) -> Any:
    """Subtract one quarter of original spin-scaled PBE exchange from PBE."""
    coefficient = mp.mpf(3) / 8 * (3 / mp.pi) ** (mp.mpf(1) / 3) * 4 ** (mp.mpf(2) / 3)
    exchange = mp.mpf(0)
    for spin, density in enumerate(coordinates[:2]):
        gradient = coordinates[2 + 3 * spin : 5 + 3 * spin]
        reduced_square = sum(value * value for value in gradient) / (
            4 * (6 * mp.pi**2) ** (mp.mpf(2) / 3) * density ** (mp.mpf(8) / 3)
        )
        kappa = mp.mpf("0.804")
        enhancement = 1 + kappa * (
            1
            - kappa
            / (kappa + mp.mpf("0.06672455060314922") * mp.pi**2 / 3 * reduced_square)
        )
        exchange -= coefficient * density ** (mp.mpf(4) / 3) * enhancement
    return original_formula.energy(True, coordinates) - mp.mpf("0.25") * exchange


def reference(values: np.ndarray, digits: int) -> np.ndarray:
    """Differentiate eight independent physical coordinates, even at spin equality."""
    with mp.workdps(digits):
        coordinates = [mp.mpf(float(value)) for value in values]
        scale = coordinates[0] + coordinates[1]
        output = [pbe0_energy(coordinates)]
        for index in range(8):

            def displaced(parameter: Any, coordinate: int = index) -> Any:
                shifted = coordinates.copy()
                shifted[coordinate] += parameter * scale
                return pbe0_energy(shifted) / scale

            output.append(mp.diff(displaced, 0))
        return np.asarray([float(value) for value in output])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tests/data/xc/pbe0_restricted_point.tsv",
    )
    args = parser.parse_args()
    fixture = np.loadtxt(args.fixture)
    if fixture.shape != (87, 17):
        raise ValueError("restricted PBE0 oracle requires all 87 frozen cases")
    for index, row in enumerate(fixture):
        if row[0] != row[1] or not np.array_equal(row[2:5], row[5:8]):
            raise ValueError(f"fixture point {index} is not restricted")
        lower, higher = reference(row[:8], 450), reference(row[:8], 550)
        if not np.array_equal(lower.view(np.uint64), higher.view(np.uint64)):
            raise ValueError(f"reference precision has not converged at point {index}")
        if not np.array_equal(higher.view(np.uint64), row[8:].view(np.uint64)):
            raise ValueError(f"frozen reference differs at point {index}")
    print(
        json.dumps(
            {
                "cases": len(fixture),
                "digits": [450, 550],
                "status": "pass",
                "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                "formula_sha256": hashlib.sha256(
                    Path(original_formula.__file__).read_bytes()
                ).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
