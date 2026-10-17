"""Audit observed endpoints and phase costs without requiring a speed win."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from generativeqc_compiler.common.evidence import block_error

from benchmarks.compare_wb97mv_hessian_gpu4pyscf import _summarize
from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_json, load_publication_record

DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/wb97mv-hessian-dispatch-20261017"
)


def test_dispatch_publication_integrity_and_scoped_decision() -> None:
    """A numerical decision must not become an unsupported performance pass."""
    manifest = load_json(DIRECTORY / "publication.json")
    files = {
        entry["path"]: (DIRECTORY / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    evidence = load_publication_record(DIRECTORY)
    assert manifest["decision"]["scope"] == "numerical"
    assert evidence["stages"]["numerical"]["status"] == "pass"
    assert evidence["performance"]["status"] != "pass"
    assert evidence["hardware"]["exclusive_node"] is False
    assert not evidence["settings"]["scientific_grid_convergence_claim"]


def test_complete_cross_engine_matrices_work_and_phase_records() -> None:
    """Recompute all gates and medians, retaining unfavorable Hessian phases."""
    samples = load_publication_record(DIRECTORY, role="samples")
    evidence = load_publication_record(DIRECTORY)
    final = samples["runs"]["final"]
    assert final["status"] == "completed"
    for case in final["cases"].values():
        assert _summarize(case["samples"], 3e-6) == case["summary"]
        native = [row for row in case["samples"] if row["engine"] == "generativeqc"]
        reference = [row for row in case["samples"] if row["engine"] == "gpu4pyscf"]
        assert {row["repeat"] for row in native} == {0, 1, 2, 3}
        assert {row["repeat"] for row in reference} == {0, 1, 2, 3}
        for candidate, baseline in itertools.product(native, reference):
            assert block_error(
                candidate["result"]["matrix"],
                baseline["result"]["matrix"],
                atol=3e-6,
                rtol=0,
            )["passed"]
        for row in native:
            result = row["result"]
            assert result["native_source_identity"] == evidence["hashes"]["source"]
            diagnostics = result["diagnostics"]
            assert not diagnostics["runtime_python"]
            assert not diagnostics["runtime_compilation"]
            assert diagnostics["numerical_host_fallbacks"] == 0
            assert diagnostics["displaced_scf_evaluations"] == 0
            assert diagnostics["intermediate_panel_downloads"] == 0
            for phase in (
                "scf_seconds",
                "response_preparation_seconds",
                "hessian_seconds",
            ):
                assert diagnostics[phase] > 0
            assert diagnostics["induced_resident_source_evaluations"] > 0
        for observation in case["gpu_memory_observations"].values():
            assert observation["sampled_peak_bytes"] <= observation["limit_bytes"]


def test_native_old_new_comparison_keeps_semantic_work_and_matrices() -> None:
    """Observed dispatch improvements must not arise from fewer solver actions."""
    runs = load_publication_record(DIRECTORY, role="samples")["runs"]
    baseline = runs["baseline"]["cases"]["water-dimer"]["samples"]
    candidates = runs["final"]["cases"]["water-dimer"]["samples"]
    for original, candidate in itertools.product(baseline, candidates):
        if candidate["engine"] != "generativeqc":
            continue
        np.testing.assert_allclose(
            candidate["result"]["matrix"],
            original["result"]["matrix"],
            atol=3e-12,
            rtol=3e-12,
        )
        for field in (
            "completed_directions",
            "induced_fock_actions",
            "exact_primitive_components",
            "grid_mixed_contractions",
            "scf_iterations",
        ):
            assert (
                candidate["result"]["diagnostics"][field]
                == original["result"]["diagnostics"][field]
            )


def test_coupled_parity_and_canceled_intermediate_are_not_lost() -> None:
    """Preserve both backend coverage and the incomplete earlier campaign."""
    samples = load_publication_record(DIRECTORY, role="samples")
    receipts = samples["coupled_stationary"]
    assert len(receipts) == 8
    for name, batched in receipts.items():
        if "batch-chunked" in name:
            continue
        tiled = receipts[name.replace("-full.json", "-batch-chunked-full.json")]
        assert batched["batched_workspace_bytes"] > 0
        assert 0 < tiled["batched_workspace_bytes"] < batched["batched_workspace_bytes"]
        for field in ("energy", "hvp", "hessian", "response_residual"):
            np.testing.assert_allclose(
                batched[field], tiled[field], atol=3e-12, rtol=3e-12
            )
    legacy = samples["coupled_ordered_reference"]
    for name, batched in legacy.items():
        if "unbatched" in name:
            continue
        ordered = legacy[name.replace("-full.json", "-unbatched-full.json")]
        for field in ("energy", "hvp", "hessian", "response_residual"):
            np.testing.assert_allclose(
                batched[field], ordered[field], atol=3e-12, rtol=3e-12
            )
    intermediate = samples["runs"]["compact-record-intermediate"]
    assert intermediate["status"] == "running"
    assert samples["intermediate_termination"]["compact-record-intermediate"]
    assert samples["qualification_harness_failure"]
    attempts = samples["coupled_projection_attempt"]
    for spins in (1, 2):
        baseline = attempts[f"stationary-{spins}-df-spherical-ecp-full.json"]
        fallback = attempts[
            f"stationary-{spins}-df-spherical-ecp-batch-chunked-full.json"
        ]
        assert fallback["batched_workspace_bytes"] == 0
        for field in ("energy", "hvp", "hessian", "response_residual"):
            np.testing.assert_allclose(
                baseline[field], fallback[field], atol=3e-12, rtol=3e-12
            )
    for name in ("methanol", "ethane"):
        case = samples["runs"][name]["cases"][name]
        assert all(row["engine"] == "generativeqc" for row in case["samples"])


def test_ethane_projected_storage_control_preserves_work_and_full_matrix() -> None:
    """The larger native-only result is not an independent engine speed ratio."""
    runs = load_publication_record(DIRECTORY, role="samples")["runs"]
    original = runs["full-projection-ethane"]["cases"]["ethane"]["samples"][0]["result"]
    candidate = runs["ethane"]["cases"]["ethane"]["samples"][0]["result"]
    np.testing.assert_allclose(
        candidate["matrix"], original["matrix"], atol=3e-12, rtol=3e-12
    )
    assert original["diagnostics"]["exact_batched_workspace_bytes"] == 0
    assert candidate["diagnostics"]["exact_batched_workspace_bytes"] > 0
    for field in (
        "completed_directions",
        "induced_fock_actions",
        "exact_primitive_components",
        "grid_mixed_contractions",
        "scf_iterations",
    ):
        assert candidate["diagnostics"][field] == original["diagnostics"][field]
