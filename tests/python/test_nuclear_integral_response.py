"""Build-only exact closure and weighted integral AD, with independent values."""

import json
import os
import subprocess
import sys
from collections import defaultdict
from itertools import product
from math import prod
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.method.nuclear_integral_response import (
    projected_integral_program,
)

from tools.complete_hessian_aot_inventory import complete_inventory, inventory_payload
from tools.generate_hessian_aot import parse_inventory

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("tile", [1, 7, 16])
def test_complete_ordered_one_electron_closure(tile: int) -> None:
    original = parse_inventory(
        json.loads((ROOT / "manifests/hessian_aot_development.json").read_text())
    )
    families = ("overlap", "kinetic", "nuclear_attraction")
    result = complete_inventory(original, families, 3, components_per_tile=tile)
    assert result.ao == original.ao and result.arrays == original.arrays
    assert tuple(
        item for item in result.integrals if item.family not in families
    ) == tuple(item for item in original.integrals if item.family not in families)
    assert parse_inventory(inventory_payload(result)) == result
    covered = defaultdict(list)
    for request in result.integrals:
        if request.family in families:
            assert 0 < len(request.components) <= tile
            covered[request.family, request.angular].extend(request.components)
    for family, angular in product(families, product(range(4), repeat=2)):
        count = prod((order + 1) * (order + 2) // 2 for order in angular)
        assert sorted(covered[family, angular]) == list(range(count))
    assert len(covered) == 48


def test_complete_eri_closure_retains_explicit_radial_controls() -> None:
    empty = parse_inventory(
        {
            "schema": "generativeqc.hessian-aot-inventory/v1",
            "ao": True,
            "arrays": [],
            "integrals": [],
        }
    )
    result = complete_inventory(empty, ("eri",), 1, radials=("long_range",))
    assert {item.angular for item in result.integrals} == set(
        product(range(2), repeat=4)
    )
    assert {item.radial for item in result.integrals} == {"long_range"}


def test_df_auxiliary_g_closure_does_not_authorize_orbital_g() -> None:
    original = parse_inventory(
        json.loads((ROOT / "manifests/hessian_aot_development.json").read_text())
    )
    result = complete_inventory(
        original, ("coulomb_metric", "three_center_eri"), 1, max_auxiliary_angular=4
    )
    assert {
        item.angular for item in result.integrals if item.family == "coulomb_metric"
    } == set(product(range(5), repeat=2))
    assert {
        item.angular for item in result.integrals if item.family == "three_center_eri"
    } == set(product(range(2), range(2), range(5)))
    assert [item for item in result.integrals if item.family == "eri"] == [
        item for item in original.integrals if item.family == "eri"
    ]
    assert parse_inventory(inventory_payload(result)) == result


@pytest.mark.parametrize(
    "change",
    [
        {"max_angular": -1},
        {"max_angular": 4},
        {"max_angular": True},
        {"max_auxiliary_angular": 5},
        {"max_auxiliary_angular": True},
        {"components_per_tile": 0},
        {"components_per_tile": 17},
        {"families": ()},
        {"families": ("overlap", "overlap")},
        {"families": ("unknown",)},
        {"radials": ()},
        {"radials": ("unknown",)},
        {"radials": ("full_range", "full_range")},
    ],
)
def test_invalid_or_ambiguous_closure_fails(change: dict) -> None:
    original = parse_inventory(
        json.loads((ROOT / "manifests/hessian_aot_development.json").read_text())
    )
    with pytest.raises(ValueError, match="finite Hessian AOT closure"):
        complete_inventory(
            original, **{"families": ("overlap",), "max_angular": 1, **change}
        )


def test_projected_scalar_response_has_independent_bilinear_value_oracle() -> None:
    graph, roots = projected_integral_program()
    random = np.random.default_rng(6619)
    for _ in range(12):
        value, first, right, mixed, weight, delta_weight, projection, direction = (
            random.normal(size=8)
        )
        feeds = dict(
            zip(
                (
                    "value",
                    "first",
                    "mixed",
                    "weight",
                    "delta_weight",
                    "projection",
                    "direction",
                ),
                (value, first, mixed, weight, delta_weight, projection, direction),
                strict=True,
            )
        )
        actual = np.array([graph.evaluate(root, feeds) for root in roots])

        def integral(
            left: float,
            motion: float,
            projection: float = projection,
            value: float = value,
            first: float = first,
            right: float = right,
            mixed: float = mixed,
        ) -> float:
            return projection * (
                value + first * left + right * motion + mixed * left * motion
            )

        def energy(
            left: float,
            motion: float,
            weight: float = weight,
            delta_weight: float = delta_weight,
        ) -> float:
            return (weight + delta_weight * motion) * integral(left, motion)

        step = 0.002
        expected = [
            integral(0, 0),
            (integral(step, 0) - integral(-step, 0)) / (2 * step) * direction,
            (energy(step, 0) - energy(-step, 0)) / (2 * step),
            (
                energy(step, step)
                - energy(step, -step)
                - energy(-step, step)
                + energy(-step, -step)
            )
            / (4 * step**2),
        ]
        np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=2e-9)


def test_closure_and_response_generation_are_build_only_and_deterministic(
    tmp_path: Path,
) -> None:
    script = """
import importlib.abc, json, sys
from pathlib import Path
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'generativeqc', 'pyscf', 'cupy', 'torch'}:
            raise RuntimeError('forbidden runtime import: ' + fullname)
sys.meta_path.insert(0, Deny())
from tools.complete_hessian_aot_inventory import complete_inventory, inventory_payload
from tools.generate_hessian_aot import parse_inventory
from tools.generate_nuclear_integral_response import generate
original = parse_inventory(json.loads(Path('manifests/hessian_aot_development.json').read_text()))
result = complete_inventory(original, ('overlap', 'kinetic', 'nuclear_attraction'), 3)
Path(sys.argv[1]).write_text(json.dumps(inventory_payload(result), sort_keys=True) + generate())
"""
    files = [tmp_path / name for name in ("first", "second")]
    for path in files:
        subprocess.run(
            [sys.executable, "-c", script, str(path)],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": f"{ROOT / 'python'}:{ROOT}"},
            check=True,
            timeout=60,
            capture_output=True,
            text=True,
        )
    assert files[0].read_bytes() == files[1].read_bytes()
