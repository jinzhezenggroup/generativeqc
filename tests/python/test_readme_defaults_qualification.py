"""Qualification must reject incomplete, mismatched and inaccurate endpoints."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any

    from matplotlib.container import ErrorbarContainer

import pytest

from benchmarks.readme_defaults import COMMON_GATES, SCHEMA
from benchmarks.readme_defaults_qualification import (
    ORACLE_SCHEMA,
    REUSE_SCHEMA,
    identity,
    qualify,
)
from benchmarks.readme_defaults_report import reduce_record, render


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
        "orbital_basis": "def2-SVP",
        "basis_representation": "spherical",
        "charge": 0,
        "spin": 0,
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


def reuse_campaign() -> tuple[dict, list[dict], dict]:
    """Explicitly review the original reference producer for a new native build."""
    record, oracles = campaign()
    record["source_revision"] = "d" * 40
    manifest = {"schema": REUSE_SCHEMA, "entries": {}}
    for oracle in oracles:
        manifest["entries"][identity(oracle)] = {
            "source_revision": oracle["source_revision"],
            "driver_sha256": oracle["driver_sha256"],
            "scientific_inputs": {
                "method": "pbe",
                "orbital_basis": "def2-SVP",
                "basis_representation": "spherical",
                "charge": 0,
                "spin": 0,
            },
        }
    return record, oracles, manifest


def test_reference_reuse_preserves_producer_identities_and_all_error_gates() -> None:
    record, oracles, manifest = reuse_campaign()
    original = deepcopy((record, oracles, manifest))
    with pytest.raises(ValueError, match="reuse contract"):
        qualify([record], oracles)
    q = qualify([record], oracles, reference_reuse=manifest)[0]
    assert q["status"] == "qualified" and q["reference_reuse"] == manifest["entries"]
    assert (record, oracles, manifest) == original
    record["samples"][2]["forces"][0][0] = 1e-3
    assert qualify([record], oracles, reference_reuse=manifest)[0]["status"] == "failed"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("orbital_basis", "sto-3g"),
        ("basis_representation", "cartesian"),
        ("charge", 1),
        ("spin", 2),
    ],
)
def test_reused_reference_cannot_certify_different_scientific_inputs(
    key: str, value: str | int
) -> None:
    record, oracles, manifest = reuse_campaign()
    record[key] = value
    with pytest.raises(ValueError, match="reuse contract"):
        qualify([record], oracles, reference_reuse=manifest)


@pytest.mark.parametrize("failure", ["driver", "receipt", "revision", "missing_input"])
def test_reference_reuse_requires_exact_receipt_and_producer_provenance(
    failure: str,
) -> None:
    record, oracles, manifest = reuse_campaign()
    if failure == "driver":
        oracles[0]["driver_sha256"] = "e" * 64
    elif failure == "receipt":
        oracles[0]["samples"][0]["seconds"] += 1
    elif failure == "revision":
        oracles[0]["source_revision"] = "unknown"
    else:
        record.pop("spin")
    with pytest.raises(ValueError):
        qualify([record], oracles, reference_reuse=manifest)


def test_reviewed_reference_reuse_still_requires_grid_stability() -> None:
    record, oracles, manifest = reuse_campaign()
    contract = manifest["entries"].pop(identity(oracles[0]))
    oracles[0]["samples"][0]["forces"][0][0] = 1e-3
    manifest["entries"][identity(oracles[0])] = contract
    result = qualify([record], oracles, reference_reuse=manifest)[0]
    assert result["status"] == "unverified" and result["reasons"]


def test_one_inaccurate_warm_repeat_rejects_the_warm_population() -> None:
    record, oracles = campaign()
    record["samples"][2]["forces"][0][0] = 1e-3
    q = qualify([record], oracles)[0]
    assert q["cold"]["status"] == "qualified"
    assert q["warm"]["status"] == q["status"] == "failed"


def test_denser_reference_uses_matching_cpu_grid_and_unchanged_gates() -> None:
    """Finer references may resolve instability but never relax acceptance."""
    record, oracles = campaign()
    for sample, level in zip(oracles[0]["samples"], (5, 6), strict=True):
        sample["level"] = sample["xc_grid"]["level"] = level
    assert qualify([record], oracles)[0]["status"] == "qualified"
    oracles[0]["samples"][1]["forces"][0][0] = 1e-3
    assert qualify([record], oracles)[0]["status"] == "unverified"


def test_mismatched_cpu_spot_grid_stays_unverified() -> None:
    """Different quadratures cannot establish CPU/GPU implementation parity."""
    record, oracles = campaign()
    sample = oracles[1]["samples"][0]
    sample["level"] = sample["xc_grid"]["level"] = 6
    assert qualify([record], oracles)[0]["status"] == "unverified"


@pytest.mark.parametrize("levels", [(3, 4), (5, 5), (6, 5), (5, 7)])
def test_incomplete_or_coarse_reference_pair_is_rejected(levels: tuple) -> None:
    """Reference refinement still requires two consecutive dense grids."""
    record, oracles = campaign()
    for sample, level in zip(oracles[0]["samples"], levels, strict=True):
        sample["level"] = sample["xc_grid"]["level"] = level
    with pytest.raises(ValueError, match="dense-grid"):
        qualify([record], oracles)


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


@pytest.mark.parametrize("include_unqualified", [False, True])
def test_performance_plot_can_show_failed_accuracy_without_relabelling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, include_unqualified: bool
) -> None:
    """Accuracy controls visibility only when explicitly requested by the plot."""
    import numpy as np
    from matplotlib.axes import Axes

    record, oracles = campaign()
    for sample in record["samples"]:
        sample["forces"][0][0] = 1.0
    qualifications = qualify([record], oracles)
    assert qualifications[0]["status"] == "failed"
    visible = []
    original = Axes.errorbar

    def observe(
        self: Axes,
        x: Sequence[float],
        y: Sequence[float],
        *args: Any,
        **kwargs: Any,
    ) -> ErrorbarContainer:
        visible.extend(value for value in y if np.isfinite(value))
        return original(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", observe)
    rows = render(
        [record],
        tmp_path,
        qualifications=qualifications,
        include_unqualified=include_unqualified,
    )
    assert bool(visible) is include_unqualified
    assert rows[0]["qualification"]["status"] == "failed"
    assert rows[0]["cold"]["median"] == 2.0
    assert rows[0]["warm"]["median"] == 2.0
    assert (tmp_path / "direct.svg").is_file()


def test_historical_four_profile_reduction_does_not_promote_to_v2(
    tmp_path: Path,
) -> None:
    """Old timings remain readable, but cannot inherit new accuracy or plots."""
    record, oracles = campaign()
    record.update(schema="generativeqc.readme-defaults.v1", arm="gpu4pyscf-fast")
    original = deepcopy(record)
    reduced = reduce_record(record)
    assert reduced["arm"] == "gpu4pyscf-fast"
    assert reduced["cold"]["median"] == 2
    assert reduced["warm"]["seconds"] == [2, 2, 2]
    assert record == original
    with pytest.raises(ValueError, match="current three-profile"):
        qualify([record], oracles)
    with pytest.raises(ValueError, match="pinned v1 renderer"):
        render([record], tmp_path)
