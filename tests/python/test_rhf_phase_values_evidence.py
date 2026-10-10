"""Protect source-matched auto routing, complete work and independent numerical gates."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from generativeqc_compiler.common.evidence import block_error

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/rhf-phase-values-auto-20261010"


def test_rhf_auto_publication_keeps_measured_source_and_scoped_comparison() -> None:
    """The default heuristic must not become a universal or formal speedup claim."""
    publication = json.loads((BUNDLE / "publication.json").read_text())
    files = {
        entry["path"]: (BUNDLE / entry["path"]).read_bytes()
        for entry in publication["files"]
    }
    validate_publication(publication, files)
    evidence = load_publication_record(BUNDLE)
    comparison = evidence["retained_comparison"]
    assert publication["source"]["dirty"] is False
    assert publication["decision"]["scope"] == "numerical"
    assert evidence["performance"]["status"] == "not-run"
    assert evidence["stages"]["production"]["status"] == "not-run"
    assert (
        comparison["candidate_environment"] == "GENERATIVEQC_RHF_RESIDENT_VALUES unset"
    )
    assert (
        comparison["scoped_auto_default"] and not comparison["broad_default_promoted"]
    )
    assert len(comparison["strata"]) == 3
    assert len({row["hardware"]["uuid"] for row in comparison["strata"]}) == 3
    assert len(comparison["rows"]) == 12
    for stratum in comparison["strata"]:
        rows = [row for row in comparison["rows"] if row["cohort"] == stratum["cohort"]]
        assert [row["selection"] for row in rows] == [
            "baseline",
            "candidate",
            "candidate",
            "baseline",
        ]
        assert stratum["reduction_fraction"] > 0
    for selection in ("baseline", "candidate"):
        samples = [
            row["native_seconds"]
            for row in comparison["rows"]
            if row["selection"] == selection
        ]
        assert len(samples) == 6
        assert statistics.median(samples) == comparison[selection]["native"]["median"]
    assert all(
        row["work"] == comparison["rows"][0]["work"] for row in comparison["rows"]
    )


def test_rhf_auto_retains_actual_numerical_outputs_and_phase_lifetime_proof() -> None:
    """Recompute gates from all outputs, keeping retained and fresh oracles distinct."""
    evidence = load_publication_record(BUNDLE)
    outputs = evidence["endpoint_outputs"]
    assert len(outputs) == 13
    energy = json.loads(
        (BUNDLE / "references/independent-pyscf-energy.json").read_text()
    )
    force = json.loads(
        (BUNDLE / "references/retained-ethane230-force.json").read_text()
    )
    differences = json.loads(
        (BUNDLE / "references/retained-independent-force-fd.json").read_text()
    )
    assert block_error(
        [result["total_energy"] for result in outputs],
        [energy["total_energy"]] * len(outputs),
        atol=1e-8,
        rtol=0,
    )["passed"]
    assert block_error(
        [result["forces"] for result in outputs],
        [force["forces"]] * len(outputs),
        atol=3e-7,
        rtol=3e-7,
    )["passed"]
    assert block_error(
        [
            [
                result["forces"][3 * row["atom"] + row["axis"]]
                for row in differences["observations"]
            ]
            for result in outputs
        ],
        [[row["finite_difference"] for row in differences["observations"]]]
        * len(outputs),
        atol=3e-7,
        rtol=3e-7,
    )["passed"]
    for result in outputs:
        assert len(result["forces"]) == 24
        assert (
            result["reference_iterations"],
            result["ccsd_iterations"],
            result["ccsd_evaluations"],
        ) == (12, 20, 38)
        assert (
            result["lambda_iterations"],
            result["lambda_actions"],
            result["lambda_batch_size"],
        ) == (21, 22, 32)
        assert (
            result["z_iterations"],
            result["z_operator_actions"],
            result["orbital_shell_derivative_passes"],
        ) == (12, 13, 2)
        assert max(result["lambda_residual"], result["z_residual"]) <= 1e-9
        assert result["stationarity"] <= 1e-8
        assert not result["resident_jk_discarded_attempt"]
        assert not result["recycling_discarded_primal_attempt"]
    profile = evidence["diagnostic_profile"]
    assert profile["source"]["selection"] == "auto"
    assert (
        profile["reference"]["physical_fock_builds"]
        == profile["reference"]["phase_resident_fock_actions"]
        == 13
    )
    assert profile["source"]["source_values_completed"] == 575639415
    families = profile["rhf_actions_after_preparation"]["families"]
    assert families["resident_canonical_jk_kernel"]["launches"] == 364
    assert "canonical_jk_kernel" not in families
    assert "bounded_direct_shell_quartet_kernel" not in families
    assert "5 passed" in evidence["acceptance"]["gpu"]
    assert "zero memcheck errors" in evidence["acceptance"]["sanitizer"]
