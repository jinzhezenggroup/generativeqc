"""Offline checks for the matched-basis water Hessian qualification gate."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc._model_resolution import _named_basis_record

from benchmarks.water_hessian_gpu4pyscf import (
    _reference_basis,
    compare,
    explicit_primitives,
)


def test_reference_basis_distinguishes_catalogs_with_the_same_name() -> None:
    expected = explicit_primitives(_named_basis_record("sto-3g", "cartesian"))
    actual = json.loads(json.dumps(expected))
    molecule = SimpleNamespace(_basis=actual, cart=True)
    assert _reference_basis(molecule, expected)
    actual["O"][0][1][0] = 130.70932
    with pytest.raises(ValueError, match="Cartesian primitives"):
        _reference_basis(molecule, expected)


@pytest.fixture
def pair(tmp_path: Path) -> list[Path]:
    result = []
    for engine, seconds in (("native", 0.5), ("gpu4pyscf", 1.0)):
        path = tmp_path / f"{engine}.json"
        matrix_path = path.with_suffix(".npy")
        np.save(matrix_path, np.zeros((9, 9), dtype=np.float64), allow_pickle=False)
        receipt = {
            "engine": engine,
            "job": "1",
            "gpu_uuid": "allocated",
            "capacity_mib": 32,
            "peak_mib": 10,
            "method": "r2scan",
            "provider": "df",
            "waters": 1,
            "geometry_hash": "geometry",
            "grid_hash": "quadrature",
            "grid_points": 2592,
            "fp64": True,
            "orbital_hash": "basis",
            "auxiliary_hash": "auxbasis",
            "auxbasis_response": 2,
            "endpoint_seconds": seconds,
            "matrix_hash": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            "diagnostics": {
                "completed_directions": 9,
                "cpks_physical_checks": 9,
                "numerical_host_fallbacks": 0,
                "posthoc_symmetrization": False,
            },
        }
        path.write_text(json.dumps(receipt))
        result.append(path)
    return result


@pytest.mark.parametrize(
    "key", ["orbital_hash", "auxiliary_hash", "gpu_uuid", "grid_hash"]
)
def test_compare_rejects_unmatched_inputs(pair: list[Path], key: str) -> None:
    native, oracle = pair
    receipt = json.loads(oracle.read_text())
    receipt[key] = "different"
    oracle.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=f"mismatched {key}"):
        compare(native, oracle)


def test_speed_does_not_override_raw_accuracy_or_untracked_artifact(
    pair: list[Path],
) -> None:
    native, oracle = pair
    good = compare(native, oracle)
    assert good["speed_pass"] and good["accuracy_pass"]
    raw = native.with_suffix(".npy")
    matrix = np.zeros((9, 9), dtype=np.float64)
    matrix[0, 0] = 3e-8
    np.save(raw, matrix, allow_pickle=False)
    with pytest.raises(ValueError, match="differs from its receipt"):
        compare(native, oracle)
    receipt = json.loads(native.read_text())
    receipt["matrix_hash"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    native.write_text(json.dumps(receipt))
    bad = compare(native, oracle)
    assert bad["speed_pass"] and not bad["accuracy_pass"]


def test_compare_rejects_a_repaired_hessian(pair: list[Path]) -> None:
    native, oracle = pair
    receipt = json.loads(native.read_text())
    receipt["diagnostics"]["posthoc_symmetrization"] = True
    native.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="complete raw native endpoint"):
        compare(native, oracle)


def test_compare_rejects_matching_non_fp64_runs(pair: list[Path]) -> None:
    for path in pair:
        receipt = json.loads(path.read_text())
        receipt["fp64"] = False
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="FP64"):
        compare(*pair)


def test_direct_accuracy_oracle_does_not_replace_stock_timing(pair: list[Path]) -> None:
    native, stock = pair
    oracle = stock.with_name("audited.json")
    oracle.with_suffix(".npy").write_bytes(stock.with_suffix(".npy").read_bytes())
    for path in pair:
        receipt = json.loads(path.read_text())
        receipt.update(provider="direct", auxiliary_hash=None, auxbasis_response=None)
        receipt["diagnostics"].update(
            version="1.9.0", scf_energy_tolerance=1e-13, scf_gradient_tolerance=1e-9
        )
        path.write_text(json.dumps(receipt))
    receipt.update(job="2", gpu_uuid="different", endpoint_seconds=4.0)
    receipt["diagnostics"].update(
        diagnostic_residual_correction=True,
        diagnostic_max_preconditioned_residual=7e-13,
    )
    oracle.write_text(json.dumps(receipt))
    stock_receipt = json.loads(stock.read_text())
    np.save(stock.with_suffix(".npy"), np.eye(9) * 3e-8, allow_pickle=False)
    stock_receipt["matrix_hash"] = hashlib.sha256(
        stock.with_suffix(".npy").read_bytes()
    ).hexdigest()
    stock.write_text(json.dumps(stock_receipt))
    report = compare(native, stock, oracle)
    assert report["accuracy_pass"] and report["speed_pass"]
    assert report["stock_raw_max_error"] == 3e-8
    assert report["reference_seconds"] == 1.0
    assert report["accuracy_reference_job"] == "2"
    receipt["diagnostics"]["diagnostic_max_preconditioned_residual"] = 2e-12
    oracle.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="invalid residual-audited"):
        compare(native, stock, oracle)
