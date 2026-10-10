"""Keep scoped frontier evidence source-matched and numerically reproducible."""

import gzip
import hashlib
import json
import statistics
from pathlib import Path

from generativeqc_compiler.common.evidence import block_error

from tools.generativeqc_validation.record import load_publication_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/lambda-frontier-20261010"
REFERENCES = ROOT / "benchmarks/results/rhf-phase-values-auto-20261010/references"


def test_frontier_evidence_pins_dirty_source_and_real_device_gates() -> None:
    """The publication preserves the measured patch, not a later clean SHA."""
    evidence = load_publication_record(BUNDLE)
    publication = json.loads((BUNDLE / "publication.json").read_text())
    assert publication["source"] == {
        "revision": "7685c251ded6c986b3cb63336e3c1cd31c0cd739",
        "dirty": True,
    }
    assert evidence["source_provenance"]["dirty"]
    patch = gzip.decompress((BUNDLE / "measured-source.patch.gz").read_bytes())
    assert (
        hashlib.sha256(patch).hexdigest()
        == evidence["source_provenance"]["patch_sha256"]
    )
    assert len(evidence["source_provenance"]["changed_production_inputs"]) == 4
    assert "49 passed" in evidence["acceptance"]["gpu"]
    assert "7 passed" in evidence["acceptance"]["sanitizer"]
    assert "ERROR SUMMARY: 0 errors" in evidence["acceptance"]["sanitizer"]
    assert "LEAK SUMMARY: 0 bytes leaked" in evidence["acceptance"]["sanitizer"]
    assert "not full ethane memcheck" in evidence["acceptance"]["sanitizer_scope"]
    assert evidence["performance"]["status"] == "not-run"
    assert evidence["memory"]["peak_bytes"] is None
    assert (
        "invalid source/build qualification"
        in evidence["discarded_build_comparison"]["scope"]
    )


def test_frontier_complete_timings_preserve_semantics_and_predicted_saved_work() -> (
    None
):
    """Constant-graph preparation savings cannot hide changed solver work."""
    evidence = load_publication_record(BUNDLE)
    comparison = evidence["retained_comparison"]
    assert len(comparison["rows"]) == 12 and len(comparison["strata"]) == 3
    assert comparison["all_accuracy_passed"]
    assert comparison["all_strata_positive"] and comparison["noise_floor_passed"]
    expected = comparison["expected_saved_per_skipped_preparation"]
    assert expected == {
        "operations": 103,
        "contraction_terms": 34875793292,
        "packing_bytes": 777609024,
        "gemm_summands": 34851307725,
        "gemm_calls": 33,
    }
    for counter, per_preparation in {
        "lambda_work": "contraction_terms",
        "lambda_gemm_calls": "gemm_calls",
        "lambda_gemm_summands": "gemm_summands",
        "lambda_kernels": "operations",
        "lambda_packing_output_bytes": "packing_bytes",
    }.items():
        assert (
            comparison["baseline_work"][counter] - comparison["candidate_work"][counter]
            == 22 * expected[per_preparation]
        )
    for side in ("baseline", "candidate"):
        rows = [row for row in comparison["rows"] if row["side"] == side]
        assert len(rows) == 6
        for phase, descriptive in comparison["phases"][side].items():
            assert (
                statistics.median(row["seconds"][phase] for row in rows)
                == descriptive["median"]
            )
        for row in rows:
            assert row["invariants"] == comparison["invariants"]
            assert row["process"]["exit_status"] == 0
            assert row["reuse"]["lambda_core_reuse_preparations"] == (
                side == "candidate"
            )
            assert row["reuse"]["lambda_core_reuse_actions"] == (
                23 if side == "candidate" else 0
            )
    profile = evidence["diagnostic_profile"]
    assert profile["work_matches_clean"]
    assert profile["immutable_launches_per_node"] == 1
    assert profile["dynamic_launches_per_node"] == 23
    assert profile["physical_force_domains"] == {
        "f_fallback": 2,
        **{f"class_domain_{number}": 2 for number in range(5)},
    }
    assert evidence["force_pr_integration"]["comparison"]["all_accuracy_passed"]


def test_frontier_independent_complete_endpoint_errors_recompute() -> None:
    """Retained oracles remain external to every measured native calculation."""
    evidence = load_publication_record(BUNDLE)
    outputs = evidence["endpoint_outputs"]
    assert len(outputs) == 13
    energy = json.loads((REFERENCES / "independent-pyscf-energy.json").read_text())
    forces = json.loads((REFERENCES / "retained-ethane230-force.json").read_text())
    differences = json.loads(
        (REFERENCES / "retained-independent-force-fd.json").read_text()
    )
    errors = {
        "independent_ethane_energy": block_error(
            [result["total_energy"] for result in outputs],
            [energy["total_energy"]] * len(outputs),
            atol=1e-8,
            rtol=0,
        ),
        "retained_qualified_force_vector": block_error(
            [result["forces"] for result in outputs],
            [forces["forces"]] * len(outputs),
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
    assert errors == evidence["block_errors"]
    assert all(row["passed"] for row in errors.values())
