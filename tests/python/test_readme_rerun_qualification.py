"""A full README refresh requires fresh, independently qualified measurements."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import pytest
from generativeqc import GridSpec

from benchmarks.readme_dft import METHODS, SCHEMA
from tools.qualify_readme_rerun import (
    README_SIZES,
    audit,
    expected_points,
    physical_protocol,
    qualify_pair,
)

if TYPE_CHECKING:
    from pathlib import Path


def _journal(engine: str) -> dict:
    return {
        "schema": SCHEMA,
        "engine": engine,
        "stage": "complete",
        "status": "measured" if engine == "reference" else "completed_unchecked",
        "scheduler": {"SLURM_JOB_ID": "fresh-job"},
        "protocol": {
            "method": "WB97M-V/RKS",
            "basis": "def2-SVP",
            "basis_identity": "independent-same-basis",
            "atoms": 3,
            "density_fitting": False,
            "grid": asdict(GridSpec()),
            "geometries_bohr": [[[0, 0, 0]], [[0, 0, 0.001]]],
            "repeats": 5,
            "force_return": "host_array_in_timed_endpoint",
        },
        "records": [
            {
                "phase": phase,
                "geometry": geometry,
                "repeat": repeat,
                "energy": -76.0,
                "forces": [[0.0, 0.0, 0.0]] * 3,
                "status": 0,
                "converged": True,
                "complete_seconds": 1.0,
                "iterations": 1,
                "fock_builds": 2,
                "gate": True if engine == "reference" else None,
                "execution_gate": True,
            }
            for phase, geometry, count in (
                ("cold", 0, 1),
                ("warm", 0, 5),
                ("moved", 1, 1),
                ("moved-warm", 1, 5),
            )
            for repeat in range(count)
        ],
    }


def test_original_homepage_matrix_includes_every_method_size_mode_and_engine() -> None:
    expected = set(expected_points(candidate_grid=None))
    assert expected == {
        (mode, (48, 16, 32), method, atoms)
        for mode in ("none", "cuda")
        for method in METHODS
        for atoms in README_SIZES
    }
    assert len(expected) * 2 == 128
    assert set(expected_points()) - expected == {
        (mode, (48, 12, 24), "wb97m-v", atoms)
        for mode in ("none", "cuda")
        for atoms in README_SIZES
    }


def test_late_reference_qualifies_saved_arrays_without_rewriting_measurement() -> None:
    native, reference = _journal("native"), _journal("reference")
    original = copy.deepcopy(native)
    point = qualify_pair(native, reference, 3)
    assert native == original
    assert point["engines"]["native"]["status"] == "measured"
    assert all(
        row["measurement_gate"] is None and row["gate"] is True
        for row in point["engines"]["native"]["records"]
    )
    assert (
        point["engines"]["native"]["records"][0]["accuracy_qualification"]
        == "post-run-independent-reference"
    )


def test_missing_moved_oracle_is_never_a_complete_accuracy_pass() -> None:
    reference = _journal("reference")
    reference["records"] = [row for row in reference["records"] if row["geometry"] == 0]
    point = qualify_pair(_journal("native"), reference, 3)
    assert point["engines"]["native"]["phase_status"]["cold"] == "measured"
    assert point["engines"]["native"]["phase_status"]["moved"] == "completed_unchecked"
    assert point["engines"]["native"]["status"] != "measured"


@pytest.mark.parametrize(
    "failure",
    ("bad-force", "nonfinite", "known-failed-gate", "missing-repeat", "unconverged"),
)
def test_every_saved_endpoint_must_pass(failure: str) -> None:
    native = _journal("native")
    row = native["records"][-1]
    if failure == "bad-force":
        row["forces"] = [[1.0, 0.0, 0.0]] * 3
    elif failure == "nonfinite":
        row["energy"] = float("nan")
    elif failure == "known-failed-gate":
        row["gate"] = False
    elif failure == "missing-repeat":
        native["records"].pop()
    else:
        row["converged"] = False
    assert (
        qualify_pair(native, _journal("reference"), 3)["engines"]["native"]["status"]
        != "measured"
    )


@pytest.mark.parametrize(
    "field", ("grid", "geometries_bohr", "basis_identity", "density_fitting", "method")
)
def test_postrun_oracle_still_requires_identical_physics(field: str) -> None:
    native = _journal("native")
    native["protocol"][field] = "different"
    with pytest.raises(ValueError, match="physical protocols differ"):
        qualify_pair(native, _journal("reference"), 3)


def test_native_resource_allowance_is_retained_not_applied_to_reference() -> None:
    native, reference = _journal("native"), _journal("reference")
    for raw in (native, reference):
        raw["protocol"]["density_fitting"] = True
        raw["protocol"]["density_fitting_memory_budget_bytes"] = 0
    native["protocol"]["density_fitting_memory_budget_bytes"] = 12 << 30
    point = qualify_pair(native, reference, 3)
    assert point["engines"]["native"]["status"] == "measured"
    assert (
        point["engines"]["native"]["resources"]["density_fitting_memory_budget_bytes"]
        == 12 << 30
    )
    assert (
        point["engines"]["reference"]["resources"][
            "density_fitting_memory_budget_bytes"
        ]
        == 0
    )
    assert native["protocol"] != reference["protocol"]


@pytest.mark.parametrize("allowance", (-1, 1 << 41, True, 1.5))
def test_resource_normalization_never_admits_invalid_capacities(
    allowance: object,
) -> None:
    protocol = _journal("native")["protocol"]
    protocol["density_fitting_memory_budget_bytes"] = allowance
    with pytest.raises(ValueError, match="invalid native DF workspace"):
        physical_protocol(protocol)


def test_audit_rejects_reused_historical_jobs_and_missing_matrix_points(
    tmp_path: Path,
) -> None:
    path = tmp_path / "measurements/none/grid-48x16x32/wb97m-v/3"
    path.mkdir(parents=True)
    for engine in ("native", "reference"):
        (path / f"{engine}.json").write_text(json.dumps(_journal(engine)))
        (path / f"{engine}.outcome").write_text(json.dumps({"exit_code": 0}))
    plan = {"allowed_job_ids": ["fresh-job"]}
    report = audit(tmp_path, plan)
    assert report["complete"] is False
    assert report["expected_engine_populations"] == 144
    assert report["expected_endpoint_calls"] == 1728
    assert report["statuses"]["measured"] == 2
    old = audit(tmp_path, {"allowed_job_ids": ["another-job"]})
    assert old["complete"] is False
    assert not old["statuses"].get("measured")
    assert any(
        "fresh campaign" in error
        for point in old["points"]
        for error in point["errors"]
    )


def test_failed_native_process_does_not_discard_completed_reference(
    tmp_path: Path,
) -> None:
    path = tmp_path / "measurements/none/grid-48x16x32/wb97m-v/3"
    path.mkdir(parents=True)
    for engine in ("native", "reference"):
        (path / f"{engine}.json").write_text(json.dumps(_journal(engine)))
        (path / f"{engine}.outcome").write_text(
            json.dumps({"exit_code": 1 if engine == "native" else 0})
        )
    report = audit(tmp_path, {"allowed_job_ids": ["fresh-job"]})
    point = next(point for point in report["points"] if point["raw_receipts"])
    assert point["engines"]["native"]["status"] == "failed"
    assert point["engines"]["reference"]["status"] == "measured"


def test_complete_journal_without_process_exit_is_not_promoted(tmp_path: Path) -> None:
    path = tmp_path / "measurements/none/grid-48x16x32/wb97m-v/3"
    path.mkdir(parents=True)
    for engine in ("native", "reference"):
        (path / f"{engine}.json").write_text(json.dumps(_journal(engine)))
    report = audit(tmp_path, {"allowed_job_ids": ["fresh-job"]})
    assert report["complete"] is False
    assert not report["statuses"].get("measured")


def test_same_grid_rerun_does_not_claim_grid_or_fast_qualification(
    tmp_path: Path,
) -> None:
    report = audit(tmp_path, {"allowed_job_ids": ["fresh-job"]})
    assert report["qualification_scope"] == "same-grid-engine-parity-only"
    assert report["grid_convergence"] == "not_evaluated"
    assert report["endpoint_precision_qualification"] == "not_evaluated"
