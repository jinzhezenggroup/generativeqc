"""Verify scoped original receipts and semantic work without another GPU run."""

import base64
import hashlib
import json
import lzma
from pathlib import Path

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/df-cc-batched-replay-20261010"
)


def test_batched_replay_receipt_bytes_and_scoped_original_gates() -> None:
    manifest = json.loads((BUNDLE / "publication.json").read_text())
    files = {
        entry["path"]: (BUNDLE / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    evidence = load_publication_record(BUNDLE)
    assert manifest["decision"]["scope"] == "numerical"
    assert (
        evidence["performance"]["status"]
        == evidence["stages"]["production"]["status"]
        == "not-run"
    )
    assert evidence["settings"]["scope"] == "cuda-df-energy-only"
    assert not evidence["settings"]["internal_option_default"]
    assert evidence["settings"]["no_device_scratch_growth"]
    assert evidence["source_reconstruction"]["production_objects_recompiled"] == 21
    assert all(gate["passed"] for gate in evidence["block_errors"].values())
    for name in ("raw-receipts.json.xz", "hf-integration-receipts.json.xz"):
        receipts = json.loads(lzma.decompress(files[name]))
        for receipt in receipts.values():
            raw = (
                base64.b64decode(receipt["data"])
                if receipt["encoding"] == "base64"
                else receipt["data"].encode()
            )
            assert len(raw) == receipt["bytes"]
            assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
    for suite in ("prototype_cuda", "production_cuda"):
        record = evidence["qualification"][suite]
        assert len(record["actions"]) == 8
        assert record["carried_overflow_caught"]
        assert record["representative_memcheck_errors"] == 0
        assert record["independent_gates_pass"]
    assert len(evidence["qualification"]["cpu"]["cases"]) == 16


def test_batched_replay_pairs_are_separate_and_all_predicted_work_matches() -> None:
    evidence = load_publication_record(BUNDLE)
    work = evidence["qualification"]["work"]
    assert work["df_auxiliary_tiles"] == 31
    assert work["df_gemm_calls"] == 31 * 31
    assert work["df_accumulation_calls"] == 31
    assert work["virtual_contraction_terms"] == 488 * work["single_contraction_terms"]
    assert work["df_packing_bytes"] == 515801545216
    for suite, order in (
        ("prototype_pair", ["base", "batch"]),
        ("production_pair", ["batch", "base"]),
    ):
        samples = evidence["retained_samples"][suite]
        assert [sample["variant"] for sample in samples] == order
        before = next(sample for sample in samples if sample["variant"] == "base")
        after = next(sample for sample in samples if sample["variant"] == "batch")
        assert after["wall_seconds"] < before["wall_seconds"]
        for sample in samples:
            record = sample["record"]
            assert record["ccsd_evaluations"] == record["ccsd_iterations"] == 19
            assert record["ccsd_device_bytes"] == 8583749632
            for name, expected in sample["predicted_work"].items():
                assert record[name] == expected
        assert (
            after["record"]["ccsd_capacity"] - before["record"]["ccsd_capacity"]
            == 113476
        )
        assert (
            after["record"]["ccsd_q_tiles"] - before["record"]["ccsd_q_tiles"]
            == 31 - 488
        )
        assert (
            after["record"]["ccsd_gemm_calls"] - before["record"]["ccsd_gemm_calls"]
            == 961 - 13176
        )
        assert (
            after["record"]["ccsd_contraction_terms"]
            == before["record"]["ccsd_contraction_terms"]
        )


def test_master_hf_integration_is_scoped_and_preserves_original_gates() -> None:
    """New HF consumers need one matched endpoint pair, not repeated CC matrices."""
    evidence = load_publication_record(BUNDLE)
    integration = evidence["qualification"]["hf_integration"]
    build = integration["build"]
    assert build["master_consumer_revision"] == integration["master_revision"]
    assert integration["hf_commit"] == "cf770754ca3f059527571ab123b29d2a34f4b788"
    assert not integration["whole_master_library_rebuilt"]
    assert not integration["cc_action_matrices_repeated"]
    assert not integration["latest_master_full_reference_qualified"]
    assert len(build["recompiled_targets"]) == 15
    assert build["borrowed_objects_unchanged"]
    assert build["baseline_sha256"] != build["previous_baseline_sha256"]
    assert build["candidate_sha256"] != build["previous_candidate_sha256"]
    assert integration["summary"]["independent_scientific_gates_pass"]
    assert integration["summary"]["exact_work_predictions_pass"]
    samples = evidence["retained_samples"]["hf_integrated_pair"]
    assert [sample["variant"] for sample in samples] == ["batch", "base"]
    for sample in samples:
        for name, tolerance in (
            ("total_energy_error", 1e-8),
            ("triples_energy_error", 1e-10),
            ("physical_r1", 1e-10),
            ("physical_r2", 1e-10),
        ):
            assert 0 <= sample["gates"][name] <= tolerance
        record = sample["record"]
        assert record["ccsd_iterations"] == record["ccsd_evaluations"] == 19
        assert record["ccsd_device_bytes"] == 8583749632
        for name, expected in sample["predicted_work"].items():
            assert record[name] == expected
    before, after = samples[1]["record"], samples[0]["record"]
    assert after["ccsd_capacity"] - before["ccsd_capacity"] == 113476
    assert after["ccsd_q_tiles"] - before["ccsd_q_tiles"] == 31 - 488
    assert after["ccsd_gemm_calls"] - before["ccsd_gemm_calls"] == 961 - 13176
    assert after["ccsd_contraction_terms"] == before["ccsd_contraction_terms"]
