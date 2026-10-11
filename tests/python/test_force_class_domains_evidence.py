"""Protect frozen-source mixed-f scheduling evidence and its qualification scope."""

from __future__ import annotations

import gzip
import hashlib
import json
import statistics
from pathlib import Path

import pytest
from generativeqc_compiler.common.evidence import block_error

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/force-class-domains-20261010"
REFERENCES = ROOT / "benchmarks/results/rhf-phase-values-auto-20261010/references"


def test_force_domain_publication_pins_dirty_source_and_scoped_abba() -> None:
    """Later master changes must not silently acquire the old timing provenance."""
    publication = json.loads((BUNDLE / "publication.json").read_text())
    files = {
        entry["path"]: (BUNDLE / entry["path"]).read_bytes()
        for entry in publication["files"]
    }
    validate_publication(publication, files)
    evidence = load_publication_record(BUNDLE)
    provenance = evidence["source_provenance"]
    assert publication["source"] == {
        "revision": "e55dcd2b2ec4acd52d9f45d3f9fa2ebf709d6d7d",
        "dirty": True,
    }
    assert provenance["measured_base"] == evidence["revision"]
    assert provenance["dirty"] and not provenance["latest_master_measured"]
    assert publication["decision"]["scope"] == "numerical"
    assert evidence["performance"]["status"] == "not-run"
    assert evidence["stages"]["production"]["status"] == "not-run"
    patch = gzip.decompress(files["measured-source.patch.gz"])
    assert hashlib.sha256(patch).hexdigest() == provenance["patch_sha256"]
    # Independently reconstructed from the recorded base plus patch: every blob
    # under src/, python/, include/ and cmake/, plus CMakeLists.txt and
    # benchmarks/df_ccsdt_force_endpoint.cpp.
    # Freeze those historical identities; two editable copies are not a check.
    assert (
        provenance["production_inputs_manifest_hash"]
        == evidence["hashes"]["source"]
        == "2a15d4b8ca0d4da5b1cdc5b8135f9714a345c734857648cb5b74d9a562b6a8c8"
    )
    assert provenance["production_input_count"] == 1513
    assert provenance["changed_production_inputs"] == {
        "cmake/GenerativeQCCuda.cmake": "6758c283ee13781890f50c572b07d5f2442145b0d6d716b4acc7e36ea1366781",
        "src/scf/cuda/direct_bounded_fallback.cu": "397231b17c7b1e15d35f182a7195e43b4997e96fd2a7086f4e7b086902877526",
        "src/scf/cuda/direct_force_class_domains.cu": "3790d1c595a968a04c22328a597c68d6ad110fea4498aae807472d2ee8423737",
        "src/scf/cuda/direct_force_class_domains.hpp": "b80e8cf2de366c23ad90d94490c000328a344b7183325bb03a6000c9f22db283",
        "src/scf/cuda/direct_force_class_pages.cuh": "99134c1417672cbe94188c81269d52cce1db5656362bd2e503d8410cd9ee84ee",
    }
    for path in provenance["patch_paths"]:
        assert f"diff --git a/{path} b/{path}\n".encode() in patch
    for path, expected in evidence["reference_identities"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    assert not any(entry["role"] == "input" for entry in publication["files"])

    settings = evidence["settings"]
    assert settings["rhf_values"] == "default-auto on both sides"
    assert settings["admission"] == {
        "evaluation": "Force",
        "range": "full-range",
        "max_angular_momentum": 3,
        "topology": "same-plan class-major",
        "resident_primitive_cache": "compatible",
        "materialized_derivatives": True,
        "cooperative_derivatives": True,
        "input_launch_dimensions": [256, 1, 1],
    }
    assert settings["domain_class_counts"] == [8, 5, 3, 4, 1]
    assert settings["force_passes"] == 2 and settings["precision"] == "strict FP64"
    assert settings["rollback"] == "GENERATIVEQC_DIRECT_PAIR_COOPERATIVE_DERIVATIVES=0"
    assert "all f-containing" in settings["residual_ownership"]
    comparison = evidence["retained_comparison"]
    assert comparison["all_accuracy_passed"]
    assert comparison["all_strata_positive"] and comparison["noise_floor_passed"]
    assert len(comparison["strata"]) == 3
    assert len({row["gpu"]["uuid"] for row in comparison["strata"]}) == 3
    assert len(comparison["rows"]) == len(evidence["timings"]) == 12
    for stratum in comparison["strata"]:
        rows = [
            row
            for row in comparison["rows"]
            if row["directory"] == stratum["directory"]
        ]
        assert [row["side"] for row in rows] == [
            "baseline",
            "candidate",
            "candidate",
            "baseline",
        ]
        assert stratum["reduction_fraction"] > 0
    for selection in ("baseline", "candidate"):
        rows = [row for row in comparison["rows"] if row["side"] == selection]
        assert len(rows) == 6
        for phase, descriptive in comparison["phases"][selection].items():
            assert (
                statistics.median(row["seconds"][phase] for row in rows)
                == descriptive["median"]
            )
        for row in rows:
            assert provenance[f"{selection}_library_sha256"] in row["binary_identity"]
            assert provenance[f"{selection}_endpoint_sha256"] in row["binary_identity"]
            assert row["process"]["exit_status"] == 0
            assert row["work"] == comparison["work"]
            assert row["sampled_total_device_peak_mib"] == 31151
    baseline = comparison["phases"]["baseline"]["native_seconds"]["median"]
    candidate = comparison["phases"]["candidate"]["native_seconds"]["median"]
    assert comparison["reduction_fraction"] == 1 - candidate / baseline
    assert evidence["memory"]["peak_bytes"] is None
    assert "not allocator/owned peak" in evidence["memory"]["reason"]


def test_force_domain_outputs_recompute_independent_gates_and_exact_work() -> None:
    """Retained ethane references and fresh mixed-basis FCI gates remain distinct."""
    evidence = load_publication_record(BUNDLE)
    outputs = evidence["endpoint_outputs"]
    assert len(outputs) == 13
    energy = json.loads((REFERENCES / "independent-pyscf-energy.json").read_text())
    force = json.loads((REFERENCES / "retained-ethane230-force.json").read_text())
    differences = json.loads(
        (REFERENCES / "retained-independent-force-fd.json").read_text()
    )
    recomputed = {
        "independent_ethane_energy": block_error(
            [result["total_energy"] for result in outputs],
            [energy["total_energy"]] * len(outputs),
            atol=1e-8,
            rtol=0,
        ),
        "retained_qualified_force_vector": block_error(
            [result["forces"] for result in outputs],
            [force["forces"]] * len(outputs),
            atol=3e-7,
            rtol=3e-7,
        ),
        "independent_fd4": block_error(
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
        ),
    }
    assert recomputed == evidence["block_errors"]
    assert all(row["passed"] for row in recomputed.values())
    work = evidence["retained_comparison"]["work"]
    for result in outputs:
        assert len(result["forces"]) == 24
        assert {key: result[key] for key in work} == work
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
        assert (result["z_iterations"], result["z_operator_actions"]) == (12, 13)
        assert (
            result["orbital_shell_derivative_passes"]
            == result["orbital_derivative_passes"]
            == 2
        )
        assert result["orbital_generic_derivative_passes"] == 0
        assert max(result["lambda_residual"], result["z_residual"]) <= 1e-9
        assert result["stationarity"] <= 1e-8
        assert max(result["ccsd_replay_r1_max"], result["ccsd_replay_r2_max"]) <= 1e-10
        assert max(abs(sum(result["forces"][axis::3])) for axis in range(3)) <= 1e-7
        assert not result["resident_jk_discarded_attempt"]
        assert not result["recycling_discarded_primal_attempt"]
        assert not result["orbital_applied_screening"]
        assert (
            result["triples_w_storage_bits"]
            == result["triples_w_compute_bits"]
            == result["triples_w_accumulation_bits"]
            == 64
        )
    acceptance = evidence["acceptance"]
    assert "4 passed" in acceptance["gpu"]
    assert "PASS" in acceptance["native_default"]
    assert "PASS" in acceptance["native_fallback"]
    assert "ERROR SUMMARY: 0 errors" in acceptance["sanitizer"]
    assert "LEAK SUMMARY: 0 bytes leaked" in acceptance["sanitizer"]
    assert "not full ethane memcheck" in acceptance["sanitizer_scope"]
    assert "No attachable process found" in acceptance["failed_sanitizer_attempt"]
    assert "not counted as success" in acceptance["failed_sanitizer_reason"]
    assert (
        evidence["source_provenance"]["context_first_provider_sha256"]
        in acceptance["sanitizer_binary_identity"]
    )


def test_force_domain_profile_is_separate_and_keeps_two_physical_passes() -> None:
    """Multiple owned-domain launches do not imply extra derivative passes."""
    evidence = load_publication_record(BUNDLE)
    profile = evidence["diagnostic_profile"]
    assert "separate" in profile["scope"]
    assert profile["work_matches_clean"]
    assert profile["physical_derivative_passes"] == 2
    assert profile["boundary_crossing_kernels"] == 0
    assert (
        evidence["source_provenance"]["candidate_library_sha256"]
        in profile["source_identity"]
    )
    families = profile["families"]
    assert set(families) == {
        "low_order",
        "weighted_four",
        "weighted_five",
        "cooperative",
        "materialized",
        "f_fallback",
    }
    assert all(family["launches"] == 2 for family in families.values())
    for name, family in families.items():
        resource = family["resources"][0]
        assert resource["threads"] == (
            256 if name in {"cooperative", "materialized"} else 128
        )
        if name not in {"cooperative", "materialized"}:
            assert resource["registers_per_thread"] == 255
    assert families["f_fallback"]["kernel_sum_seconds"] == pytest.approx(55.027298404)
    assert sum(
        families[name]["kernel_sum_seconds"]
        for name in ("low_order", "weighted_four", "weighted_five")
    ) == pytest.approx(0.683180201)
    assert len(evidence["raw_profile_identities"]) == 3
