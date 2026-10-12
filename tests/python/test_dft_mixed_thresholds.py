"""The DFT screen retains failed endpoints and never relaxes physical targets."""

import copy
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks.dft_mixed_thresholds import (
    grid_spec,
    inputs,
    mixed_environment,
    paired_errors,
    reference_grid_arrays,
    run_sample,
    summarize_attempts,
)


def test_bounded_warmup_controls_are_explicit() -> None:
    assert mixed_environment(1e-4, 24) == {
        "GENERATIVEQC_KS_MIXED_SWITCH_RESIDUAL": "0.0001",
        "GENERATIVEQC_KS_MIXED_MAX_ITERATIONS": "24",
    }
    assert mixed_environment(0.0, 0)["GENERATIVEQC_KS_MIXED_MAX_ITERATIONS"] == "0"


@pytest.mark.parametrize(
    "residual,iterations",
    [(float("nan"), 24), (float("inf"), 24), (-1.0, 24), (1e-4, -1)],
)
def test_invalid_warmup_controls_are_not_silently_ignored(
    residual: float, iterations: int
) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        mixed_environment(residual, iterations)


def test_fixed_and_changed_geometries_keep_the_contract_identity() -> None:
    cold, cold_path = inputs("water8")
    changed, changed_path = inputs("water8", True)
    assert len(cold["atoms"]) == len(changed["atoms"]) == 24
    assert cold_path != changed_path
    assert cold["atoms"] != changed["atoms"]


@pytest.mark.parametrize("contract_grid", [False, True])
def test_larger_reference_tiles_preserve_exact_quadrature(contract_grid: bool) -> None:
    import numpy as np
    from generativeqc import Atom, GridSpec
    from generativeqc_compiler.dft.grid import MolecularGrid

    geometry, _ = inputs("water")
    grid = MolecularGrid(
        tuple(Atom.from_value(atom) for atom in geometry["atoms"]),
        spec=grid_spec()
        if contract_grid
        else GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8),
    )
    expected = grid.explicit(max_points=20_000_000)
    points, weights = reference_grid_arrays(
        grid, tile_points=8192 if contract_grid else 317
    )
    np.testing.assert_array_equal(points, expected.points)
    np.testing.assert_array_equal(weights, expected.weights)


def test_reference_export_remains_bounded() -> None:
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="bounded export"):
        reference_grid_arrays(SimpleNamespace(npoint=20_000_001))
    with pytest.raises(ValueError, match="bounded export"):
        reference_grid_arrays(SimpleNamespace(npoint=1), tile_points=0)


def endpoint(stage: str = "cold") -> dict[str, Any]:
    return {
        "stage": stage,
        "status": 0,
        "converged": True,
        "diagnostic": {"physical_residual_max": 1e-10},
        "energy": -76.0,
        "forces": [[0.0, 0.0, 0.1]],
        "complete_seconds": 2.0,
    }


@pytest.mark.parametrize("invalid_baseline", [False, True])
@pytest.mark.parametrize(
    "changes",
    [
        {"status": 8},
        {"converged": False},
        {"diagnostic": {"physical_residual_max": 1.1e-9}},
        {"diagnostic": {"physical_residual_max": -1e-10}},
        {"diagnostic": {"physical_residual_max": float("nan")}},
        {"energy": float("nan")},
        {"forces": None},
        {"forces": [[0.0, 0.0]]},
        {"forces": [[0.0, 0.0, float("inf")]]},
        {"complete_seconds": 0.0},
        {"complete_seconds": float("inf")},
    ],
)
def test_invalid_candidate_or_baseline_cannot_qualify(
    invalid_baseline: bool, changes: dict[str, Any]
) -> None:
    actual, expected = endpoint(), endpoint()
    (expected if invalid_baseline else actual).update(changes)
    with pytest.raises(ValueError):
        paired_errors(actual, expected, forces=True)


def test_energy_and_force_pair_gates_remain_strict() -> None:
    actual, expected = endpoint(), endpoint()
    actual["energy"] += 2e-8
    with pytest.raises(ValueError, match="acceptance gate"):
        paired_errors(actual, expected, forces=True)
    actual = endpoint()
    actual["forces"][0][2] += 2e-7
    with pytest.raises(ValueError, match="acceptance gate"):
        paired_errors(actual, expected, forces=True)
    assert paired_errors(expected, expected, forces=True)["energy_error"] == 0.0


def attempt(policy: str, repeat: int = 0) -> dict[str, Any]:
    return {
        "policy": policy,
        "repeat": repeat,
        "exit_code": 0,
        "result": {"records": [endpoint()]},
    }


def test_a_failed_repeat_is_not_filtered_out_of_a_favorable_median() -> None:
    samples = [
        attempt("fp64"),
        attempt("auto"),
        attempt("fp64", 1),
        attempt("auto", 1),
    ]
    valid = summarize_attempts(samples, ["fp64", "auto"], forces=True, lifecycle=False)
    assert valid["auto"]["seconds_median"] == {"cold": 2.0}
    failed = copy.deepcopy(samples)
    failed[-1]["exit_code"] = "timeout"
    invalid = summarize_attempts(failed, ["fp64", "auto"], forces=True, lifecycle=False)
    assert invalid["auto"]["seconds_median"] is None
    assert invalid["auto"]["errors"]
    assert valid["auto"]["errors"] == []


def test_empty_or_partial_lifetimes_cannot_qualify() -> None:
    baseline, candidate = attempt("fp64"), attempt("auto")
    for records in ([], [endpoint()], [endpoint(), endpoint("warm")]):
        candidate["result"]["records"] = records
        results = summarize_attempts(
            [baseline, candidate], ["fp64", "auto"], forces=True, lifecycle=True
        )
        assert results["auto"]["seconds_median"] is None
    results = summarize_attempts([], ["auto"], forces=True, lifecycle=False)
    assert results["auto"]["seconds_median"] is None


@pytest.mark.skipif(os.name != "posix", reason="the runner targets Slurm/POSIX")
def test_sample_timeout_is_retained(tmp_path: Path) -> None:
    result = run_sample(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        tmp_path / "sample.log",
        0.1,
    )
    assert result == "timeout"
