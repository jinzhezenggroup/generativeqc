"""AOT inventory admission and generation without the public runtime."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from generativeqc_compiler.integral.hessian_aot import (
    IntegralRequest,
    integral_request,
)

from tools.generate_hessian_aot import parse_inventory

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "manifests/hessian_aot_development.json"


def test_development_inventory_is_explicit_not_a_complete_endpoint_claim() -> None:
    inventory = parse_inventory(json.loads(INVENTORY.read_text()))
    assert inventory.ao
    assert {"xc_polarized", "xc_unpolarized", "vv10_pair"} <= set(inventory.names)
    assert {item.family for item in inventory.integrals} == {
        "overlap",
        "kinetic",
        "nuclear_attraction",
        "eri",
        "coulomb_metric",
        "three_center_eri",
    }
    assert {item.angular[0] for item in inventory.integrals} == {0, 1, 2, 3}
    assert "eri_full_range_3_3_3_3_c0" not in inventory.names
    assert inventory.names == tuple(sorted(set(inventory.names)))


@pytest.mark.parametrize(
    "change",
    [
        {"runtime_python": True},
        {"ao": 1},
        {"arrays": ["unknown"]},
        {"arrays": ["xc_polarized", "xc_polarized"]},
        {
            "integrals": [
                {
                    "family": "overlap",
                    "angular": [0, 0],
                    "components": [0],
                    "omega": 0.4,
                }
            ]
        },
        {"integrals": [{"family": "overlap", "angular": [True, 0], "components": [0]}]},
    ],
)
def test_inventory_rejects_ignored_or_ambiguous_controls(change: dict) -> None:
    payload = {
        "schema": "generativeqc.hessian-aot-inventory/v1",
        "ao": True,
        "arrays": [],
        "integrals": [],
    }
    with pytest.raises(ValueError):
        parse_inventory({**payload, **change})


@pytest.mark.parametrize(
    "specification",
    [
        ("eri", (0, 0), (0,), "full_range"),
        ("overlap", (4, 0), (0,), "full_range"),
        ("overlap", (0, 0), (0, 0), "full_range"),
        ("overlap", (1, 0), (1, 0), "full_range"),
        ("overlap", (0, 0), (1,), "full_range"),
        ("coulomb_metric", (0, 0), (0,), "long_range"),
    ],
)
def test_raw_component_requests_have_no_wildcard_fallback(specification: tuple) -> None:
    with pytest.raises(ValueError):
        IntegralRequest(*specification)


def test_attraction_retains_external_mathematical_center_and_unit_charge() -> None:
    request = integral_request(
        {"family": "nuclear_attraction", "angular": [1, 0], "components": [0, 2]}
    )
    assert request.centers == 3
    assert request.record_stride == 12
    assert request.payload()["charge"] == 1
    assert request.components == (0, 2)


def test_build_generation_never_imports_runtime_or_gpu_packages(tmp_path: Path) -> None:
    output = tmp_path / "aot"
    script = """
import importlib.abc
import runpy
import sys
class ForbiddenRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'generativeqc', 'pyscf', 'cupy', 'torch'}:
            raise AssertionError('build-time generation imported ' + fullname)
sys.meta_path.insert(0, ForbiddenRuntime())
sys.argv = ['generate_hessian_aot.py', '--inventory', sys.argv[1], '--all', '--output-dir', sys.argv[2]]
runpy.run_path('tools/generate_hessian_aot.py', run_name='__main__')
"""
    subprocess.run(
        [sys.executable, "-c", script, str(INVENTORY), str(output)],
        check=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    inventory = parse_inventory(json.loads(INVENTORY.read_text()))
    assert {item.stem for item in output.glob("*.cu")} == set(inventory.names)
    assert (output / "registry.cpp").is_file()
    assert all(item.stat().st_size for item in output.iterdir())
