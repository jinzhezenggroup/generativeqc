"""Protect the source-frozen restricted-point evidence and its claim boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from generativeqc_compiler.common.evidence import block_error
from generativeqc_compiler.common.performance import assess_comparison

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/pbe0-restricted-point-default-20261010"
SOURCE = "99ebd196df529bdd45107af44b86b7879d3bb69f"


def test_publication_does_not_promote_unmeasured_resource_or_source_claims() -> None:
    """Scoped timing success cannot manufacture global peak or build duration."""
    manifest = json.loads((BUNDLE / "publication.json").read_text())
    files = {
        entry["path"]: (BUNDLE / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    assert manifest["source"] == {"revision": SOURCE, "dirty": False}
    assert manifest["decision"]["status"] == "accepted"
    assert manifest["decision"]["scope"] == "numerical"
    assert manifest["archives"] == []
    assert any(
        argument.startswith("--time=")
        for argument in manifest["reproduction"]["command"]
    )
    evidence = load_publication_record(BUNDLE)
    assert evidence["revision"] == SOURCE
    assert evidence["performance"]["status"] == "not-run"
    assert evidence["stages"]["production"]["status"] == "not-run"
    assert evidence["memory"]["peak_bytes"] is None
    assert evidence["compilation"]["seconds"] is None
    assert evidence["default_route_execution"]["status"] == "pass"
    assert evidence["settings"]["fast_compile"] is False
    assert set(evidence["settings"]["timing_claim_excludes"]) == {
        "cold",
        "moving-geometry-reconvergence",
        "latest-master-4b330ff3d",
        "GPU4PySCF-relative-speedup",
    }
    assert (
        evidence["hashes"]["source"]
        == "f3a7b0e1f54d37cf970ca119c6ac73ae43c6c088882d34d8d4b41257835347da"
    )
    assert evidence["cpu_qualification"]["independent_point_references"] == 87


@pytest.mark.parametrize(
    "atoms,expected", [(48, (2304, 1179648)), (96, (4608, 2359296))]
)
@pytest.mark.parametrize("geometry,phase", [(0, "warm"), (1, "moved-warm")])
def test_default_arms_preserve_work_and_pass_recomputed_timing_gate(
    atoms: int,
    expected: tuple[int, int],
    geometry: int,
    phase: str,
) -> None:
    """Validate actual dispatch and whole endpoint work, not the request flag."""
    record = load_publication_record(BUNDLE, role="samples")["cases"][str(atoms)]
    assert record["status"] == "measured" and record["stage"] == "complete"
    assert record["source_snapshot"] == SOURCE
    assert record["selectors"] == {"baseline": "off", "candidate": None}
    rows = [row for row in record["samples"] if row["geometry"] == geometry]
    assert len(rows) == 10
    assert assess_comparison(rows) == record["assessments"][phase]
    assert record["assessments"][phase]["status"] == "pass"
    semantic_work = []
    for row in rows:
        diagnostic = row["diagnostics"]
        assert row["synchronized"]
        assert diagnostic["converged"] and diagnostic["status"] == 0
        assert diagnostic["iterations"] == diagnostic["fock_builds"] == 1
        assert diagnostic["warm_start_used"] and not diagnostic["warm_start_fallback"]
        selected = "restricted" if row["selection"] == "candidate" else "general"
        other = "general" if selected == "restricted" else "restricted"
        actual = diagnostic["actual_point_selection"]
        assert (
            actual[f"{selected}_point_batches"],
            actual[f"{selected}_point_count"],
        ) == expected
        assert actual[f"{other}_point_count"] == 0
        semantic_work.append(
            (
                {
                    name: value
                    for name, value in diagnostic["native_scf_ao_work"].items()
                    if name != "discovery_seconds"
                },
                diagnostic["native_force_components"]["work_counts"],
            )
        )
    assert all(work == semantic_work[0] for work in semantic_work)
    checkpoints = [row for row in record["checkpoints"] if row["phase"] == phase]
    for side in ("baseline", "candidate"):
        selected = [row for row in checkpoints if row["selection"] == side]
        assert len(selected) == 2
        assert selected[0]["blob_sha256"] == selected[1]["blob_sha256"]
    qualification = load_publication_record(BUNDLE)["endpoint_qualification"]["cases"][
        str(atoms)
    ]["timings"][phase]
    assert qualification["maximum_cross_arm_seed_coordinate_difference"] == 0.0
    assert qualification["maximum_cross_arm_seed_density_difference"] < 1.27e-11
    assert qualification["same_seed_between_sides"] is False


@pytest.mark.parametrize("atoms", [48, 96])
def test_all_endpoint_values_pass_the_retained_independent_reference(
    atoms: int,
) -> None:
    """Recheck setup, priming and every measured force without a GPU or oracle call."""
    samples = load_publication_record(BUNDLE, role="samples")
    record = samples["cases"][str(atoms)]
    reference = samples["references"][str(atoms)]
    assert record["protocol"] == reference["protocol"]
    actual_energy, expected_energy, actual_force, expected_force = [], [], [], []
    for row in record["setup"] + record["priming"] + record["samples"]:
        oracle = next(
            entry
            for entry in reference["records"]
            if entry["geometry"] == row["geometry"]
            and entry["phase"] in ("cold", "moved")
        )
        actual_energy.append(row["diagnostics"]["energy"])
        expected_energy.append(oracle["energy"])
        actual_force.append(row["diagnostics"]["forces"])
        expected_force.append(oracle["forces"])
    evidence = load_publication_record(BUNDLE)
    for label, actual, expected, tolerance in (
        ("energy", actual_energy, expected_energy, 1e-8),
        ("force", actual_force, expected_force, 1e-7),
    ):
        error = block_error(actual, expected, atol=tolerance, rtol=0.0)
        assert error["passed"]
        assert error == evidence["block_errors"][f"{atoms}/{label}"]


def test_reproduction_retains_real_build_and_scheduler_receipts() -> None:
    """Source reconstruction and compiler caching stay auditable without binaries."""
    recipes = load_publication_record(BUNDLE, role="reproduction")
    assert (
        "--time=" in recipes["scripts"]["run-master-endpoints.sh"]
        or "timeout --kill-after" in recipes["scripts"]["run-master-endpoints.sh"]
    )
    assert "CUDA_VISIBLE_DEVICES" in recipes["scripts"]["run-master-endpoints.sh"]
    assert (
        "-DGENERATIVEQC_CUDA_FAST_COMPILE=OFF"
        in recipes["scripts"]["build-master-official.sh"]
    )
    assert (
        "CMAKE_CUDA_COMPILER_LAUNCHER" in recipes["scripts"]["build-master-official.sh"]
    )
    evidence = load_publication_record(BUNDLE)
    assert "job=7106 visibility=1" in evidence["endpoint_receipts"]["job.txt"]
    assert evidence["endpoint_receipts"]["exit-status.txt"].strip() == "0"
    assert evidence["build_receipts"]["exit-status.txt"].strip() == "0"
    assert "ccache" in evidence["build_receipts"]["cache-version.txt"]
