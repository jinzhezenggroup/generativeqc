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
    receipts = json.loads(lzma.decompress(files["raw-receipts.json.xz"]))
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
