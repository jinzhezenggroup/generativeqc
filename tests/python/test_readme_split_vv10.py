"""Split-grid feasibility must isolate NLC accuracy from XC and timing claims."""

from __future__ import annotations

import copy
import json
import lzma
import math
from pathlib import Path

import numpy as np
import pytest
from generativeqc import GridSpec

from benchmarks.readme_split_vv10 import compare_geometries, split_grid_work


def _population(nlc_grid: tuple[int, int, int]) -> dict:
    return {
        "atoms": 3,
        "basis": "spherical def2-SVP",
        "density_fitting": "none",
        "xc_grid": [48, 16, 32],
        "nlc_grid": list(nlc_grid),
        "geometries_bohr": [[[0, 0, 0]], [[0, 0, 0.001]]],
        "records": [
            {
                "geometry": geometry,
                "phase": phase,
                "energy": -76.0,
                "forces": np.zeros((3, 3)).tolist(),
                "converged": True,
            }
            for geometry, phase in ((0, "cold"), (1, "moved"))
        ],
    }


def test_split_grid_work_keeps_both_domains_and_labels_pair_bound() -> None:
    work = split_grid_work(3, GridSpec(48, 16, 32), GridSpec(32, 12, 24))
    assert work["xc_points"] == 3 * 24576
    assert work["vv10_points"] == 3 * 9216
    assert work["vv10_unscreened_ordered_pair_upper_bound"] == (3 * 9216) ** 2
    assert work["pair_bound_is_executed_work"] is False


@pytest.mark.parametrize("atoms", (0, -1, True, 1.5))
def test_split_grid_work_rejects_invalid_system_size(atoms: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        split_grid_work(atoms, GridSpec(), GridSpec())


def test_split_grid_convergence_requires_both_original_and_moved_forces() -> None:
    candidate = _population((32, 12, 24))
    baseline = _population((64, 16, 32))
    assert compare_geometries(candidate, baseline)["gate"] is True
    candidate["records"][1]["forces"][1][2] = 2e-4
    compared = compare_geometries(candidate, baseline)
    assert compared["gate"] is False
    assert compared["geometries"][0]["gate"] is True
    assert compared["geometries"][1]["gate"] is False


@pytest.mark.parametrize(
    "field", ("atoms", "basis", "density_fitting", "xc_grid", "geometries_bohr")
)
def test_split_grid_comparison_rejects_other_model_changes(field: str) -> None:
    candidate = _population((32, 12, 24))
    baseline = _population((64, 16, 32))
    candidate[field] = "different"
    with pytest.raises(ValueError, match=field):
        compare_geometries(candidate, baseline)


def test_split_grid_baseline_self_comparison_is_not_a_convergence_pass() -> None:
    baseline = _population((64, 16, 32))
    with pytest.raises(ValueError, match="self-comparison"):
        compare_geometries(copy.deepcopy(baseline), baseline)


def test_split_grid_missing_geometry_is_unknown() -> None:
    candidate = _population((32, 12, 24))
    candidate["records"].pop()
    assert compare_geometries(candidate, _population((64, 16, 32)))["gate"] is None


@pytest.mark.parametrize("failure", ("nonfinite", "shape", "unconverged"))
def test_split_grid_invalid_endpoint_cannot_pass(failure: str) -> None:
    candidate = _population((32, 12, 24))
    row = candidate["records"][0]
    if failure == "nonfinite":
        row["forces"][0][0] = float("nan")
    elif failure == "shape":
        row["forces"] = [[0.0, 0.0, 0.0]]
    else:
        row["converged"] = False
    assert compare_geometries(candidate, _population((64, 16, 32)))["gate"] is False


@pytest.mark.parametrize("tolerance", (0, -1, float("nan"), float("inf")))
def test_split_grid_tolerances_are_explicit_and_finite(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerances"):
        compare_geometries(
            _population((32, 12, 24)),
            _population((64, 16, 32)),
            force_tolerance=tolerance,
        )


def test_retained_feasibility_separates_reference_work_and_nlc_error() -> None:
    """Protect measured gates and work, without pinning source spelling or speed."""
    archive = Path("benchmarks/results/readme-vv10-grids-20261018/grid-timings.json.xz")
    summary = json.loads(lzma.decompress(archive.read_bytes()))[
        "split_grid_feasibility"
    ]
    assert summary["native_split_grid_supported"] is False
    assert len(summary["phases"]) == len(summary["phase_sample_counts"])
    populations = set()
    for values in summary["rows"]:
        row = dict(zip(summary["columns"], values, strict=True))
        identity = (row["mode"], row["atoms"], row["nlc"])
        assert identity not in populations
        populations.add(identity)
        assert len(row["raw_sha256"]) == 64
        assert len(row["phase_medians_s"]) == len(summary["phases"])
        assert all(
            math.isfinite(seconds) and seconds > 0 for seconds in row["phase_medians_s"]
        )
        assert len(row["phase_scf_work"]) == len(summary["phases"])
        for phase_work in row["phase_scf_work"]:
            assert phase_work
            assert all(
                iterations > 0 and builds >= iterations
                for iterations, builds in phase_work
            )
        errors = row["geometry_energy_force_errors"]
        if row["nlc"] == 0:
            assert errors is None
        else:
            assert len(errors) == 2
            assert all(
                0 <= energy <= summary["gates"]["energy_Eh"]
                and 0 <= force <= summary["gates"]["force_Eh_per_bohr"]
                for energy, force in errors
            )
    for diagnostic in summary["finite_difference"]:
        assert len(diagnostic["steps"]) == len(diagnostic["errors"]) == 2
        assert all(0 <= error < 1e-5 for error in diagnostic["errors"])
