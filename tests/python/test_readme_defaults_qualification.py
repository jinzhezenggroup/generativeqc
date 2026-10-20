"""Qualification must reject incomplete, mismatched and inaccurate endpoints."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from benchmarks.readme_defaults import COMMON_GATES, SCHEMA
from benchmarks.readme_defaults_qualification import ORACLE_SCHEMA, identity, qualify
from benchmarks.readme_defaults_report import render


def campaign() -> tuple[dict, list[dict]]:
    """Small independently controlled receipts with a complete warm population."""
    geometry = [["O", [0, 0, 0]], ["H", [0, 1, 1]], ["H", [0, -1, 1]]]
    sample = {
        "energy": -75.0,
        "forces": [[0.0, 0.0, 0.0]] * 3,
        "seconds": 2.0,
        "iterations": 2,
        "converged": True,
    }
    record = {
        "schema": SCHEMA,
        "method": "pbe",
        "mode": "direct",
        "arm": "standard",
        "atoms": 3,
        "status": "complete",
        "geometry_bohr": geometry,
        "warm_repeats": 3,
        "source_revision": "a" * 40,
        "runner_sha256": "b" * 64,
        "protocol": {"common_observable_gates": COMMON_GATES},
        "environment": {"packages": {"pyscf": "test", "gpu4pyscf": "test"}},
        "samples": [dict(deepcopy(sample), phase="cold", repeat=0)]
        + [dict(deepcopy(sample), phase="warm", repeat=i) for i in range(3)],
    }
    oracles = []
    for backend in ("cuda", "cpu"):
        samples = []
        for level in (4, 5) if backend == "cuda" else (5,):
            samples.append(
                dict(
                    deepcopy(sample),
                    level=level,
                    xc_grid={"level": level},
                    controls={
                        "energy": 1e-11,
                        "orbital_gradient": 1e-7,
                        "screening": 1e-14,
                        "full_density_fock": True,
                        "force_grid_response": True,
                    },
                )
            )
        oracles.append(
            {
                "schema": ORACLE_SCHEMA,
                "method": "pbe",
                "atoms": 3,
                "backend": backend,
                "geometry_bohr": deepcopy(geometry),
                "status": "complete",
                "samples": samples,
                "driver_sha256": "c" * 64,
                "slurm_job_id": "1",
                "common_observable_gates": COMMON_GATES,
                "grid_stability_gate_fraction": 0.25,
                "source_revision": record["source_revision"],
                "environment": deepcopy(record["environment"]),
            }
        )
    return record, oracles


def test_all_repeats_pass_without_mutating_receipts() -> None:
    record, oracles = campaign()
    original = deepcopy((record, oracles))
    q = qualify([record], oracles)[0]
    assert q["status"] == "qualified"
    assert len(q["warm"]["errors"]) == 3
    assert q["receipt_sha256"] == identity(record)
    assert (record, oracles) == original


def test_one_inaccurate_warm_repeat_rejects_the_warm_population() -> None:
    record, oracles = campaign()
    record["samples"][2]["forces"][0][0] = 1e-3
    q = qualify([record], oracles)[0]
    assert q["cold"]["status"] == "qualified"
    assert q["warm"]["status"] == q["status"] == "failed"


@pytest.mark.parametrize("failure", ["missing_cpu", "grid_unstable", "cpu_disagrees"])
def test_untrusted_reference_cannot_qualify_production(failure: str) -> None:
    record, oracles = campaign()
    if failure == "missing_cpu":
        oracles.pop()
    elif failure == "grid_unstable":
        oracles[0]["samples"][0]["energy"] += 1e-3
        oracles[0]["grid_stability"] = {"passed": True}
    else:
        oracles[1]["samples"][0]["forces"][0][0] = 1e-3
    q = qualify([record], oracles)[0]
    assert q["status"] == q["cold"]["status"] == q["warm"]["status"] == "unverified"
    assert q["reasons"]


@pytest.mark.parametrize(
    "failure", ["geometry", "shape", "controls", "driver", "gates"]
)
def test_malformed_or_mismatched_reference_is_rejected(failure: str) -> None:
    record, oracles = campaign()
    oracle = oracles[0]
    if failure == "geometry":
        oracle["geometry_bohr"][0][1][0] += 0.01
    elif failure == "shape":
        oracle["samples"][0]["forces"] = [[0, 0, 0]]
    elif failure == "controls":
        oracle["samples"][0]["controls"]["energy"] = 1e-4
    elif failure == "driver":
        oracle.pop("driver_sha256")
    else:
        oracle["common_observable_gates"] = {}
    with pytest.raises(ValueError):
        qualify([record], oracles)


def test_plot_qualification_cannot_be_reused_for_different_timings(
    tmp_path: Path,
) -> None:
    record, oracles = campaign()
    qualifications = qualify([record], oracles)
    record["samples"][0]["seconds"] = 1
    with pytest.raises(ValueError, match="bind exactly"):
        render([record], tmp_path, qualifications=qualifications)
