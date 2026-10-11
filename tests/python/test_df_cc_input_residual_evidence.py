"""Offline gates for the scoped nonlinear policy, without another GPU run."""

from __future__ import annotations

import base64
import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/df-cc-input-residual-20261010"
WORK_FIELDS = {
    "df_auxiliary_slices": "ccsd_q_slices",
    "df_virtual_operations": "ccsd_q_operations",
    "df_accumulation_calls": "ccsd_accumulation_calls",
    "df_contraction_terms": "ccsd_contraction_terms",
    "df_gemm_calls": "ccsd_gemm_calls",
    "df_gemm_summands": "ccsd_gemm_summands",
    "df_packing_bytes": "ccsd_packing_bytes",
    "df_auxiliary_tiles": "ccsd_q_tiles",
    "df_accumulation_bytes": "ccsd_accumulation_bytes",
    "df_pair_evaluations": "ccsd_pair_evaluations",
    "df_pair_refusals": "ccsd_pair_refusals",
    "df_pair_projection_calls": "ccsd_pair_projection_calls",
    "df_pair_projection_bytes": "ccsd_pair_projection_bytes",
}


def test_input_residual_publication_scope_and_original_gates() -> None:
    """Scoped timing must not silently become force/global statistical proof."""
    publication = json.loads((BUNDLE / "publication.json").read_text())
    files = {
        entry["path"]: (BUNDLE / entry["path"]).read_bytes()
        for entry in publication["files"]
    }
    validate_publication(publication, files)
    evidence = load_publication_record(BUNDLE)
    assert publication["decision"]["scope"] == "numerical"
    assert evidence["performance"]["status"] == "not-run"
    assert evidence["stages"]["production"]["status"] == "not-run"
    assert evidence["settings"]["scope"] == "cuda-df-energy-only"
    assert not evidence["settings"]["diis_input_residual_default"]
    assert evidence["settings"]["cpu_and_response_legacy"]
    assert evidence["source_reconstruction"]["generated_unchanged"] == 14
    for name, gate in evidence["block_errors"].items():
        assert gate["passed"]
        assert gate["shape"] == ([1] if name.startswith("algebraic_") else [8])
    assert evidence["provenance"]["frozen_build"]["control_relink_byte_identical"]
    assert not evidence["provenance"]["frozen_build"][
        "optional_cublaslt_provider_enabled"
    ]
    assert (
        evidence["provenance"]["integrated"]["master_observed"]
        == evidence["source_reconstruction"]["integration_base"]
    )


def test_input_residual_full_records_and_work_predictions() -> None:
    """Recompute semantic work from per-graph queries, not summary booleans."""
    evidence = load_publication_record(BUNDLE)
    retained = evidence["retained_samples"]
    matched = retained["frozen_abba"]
    assert [row["variant"] for row in matched] == ["base", "input", "input", "base"]
    assert (
        len(retained["feasibility_pilot"]) == len(retained["current_master_pair"]) == 2
    )
    work = evidence["actual_graph_work"]
    for scope in ("frozen_abba", "feasibility_pilot", "current_master_pair"):
        samples = retained[scope]
        baseline = samples[0]["record"]
        for sample in samples:
            record = sample["record"]
            evaluations = record["ccsd_evaluations"]
            paired = record["ccsd_pair_evaluations"]
            ordinary = evaluations - paired
            assert evaluations > 0 and paired >= 0 and ordinary >= 0
            for field, published in WORK_FIELDS.items():
                predicted = (
                    paired * work["paired_primal"][field]
                    + ordinary * work["ordinary_primal"][field]
                    + work["physical_replay"][field]
                )
                if field == "df_pair_refusals":
                    predicted = ordinary
                elif field == "df_pair_projection_calls":
                    predicted = evaluations
                elif field == "df_pair_projection_bytes":
                    predicted = evaluations * work["paired_primal"][field]
                assert record[published] == predicted
            assert (
                record["derived_d2_iteration_evaluations"]
                == evaluations * (9 * 221) ** 2
            )
            if scope != "current_master_pair":
                assert evaluations == (19 if sample["variant"] == "input" else 38)
            if sample["variant"] == "input":
                assert evaluations == record["ccsd_iterations"]
            for capacity in (
                "numeric_capacity_bytes",
                "ccsd_capacity",
                "ccsd_device_bytes",
                "ccsd_provider_capacity",
                "ccsd_pair_binding_host_bytes",
                "ccsd_diis_history_capacity_bytes",
                "ccsd_setup_h2d_bytes",
            ):
                assert record[capacity] == baseline[capacity]
            for gate, tolerance in (
                ("total_energy_error", 1e-8),
                ("triples_energy_error", 1e-10),
                ("physical_replay_r1", 1e-10),
                ("physical_replay_r2", 1e-10),
            ):
                error = sample["gates"][gate]
                assert math.isfinite(error) and 0 <= error <= tolerance
    for scope in ("frozen_abba", "current_master_pair"):
        samples = retained[scope]
        summary = evidence["observed_endpoint_summary"][scope]
        wall = {
            variant: statistics.median(
                row["wall_seconds"] for row in samples if row["variant"] == variant
            )
            for variant in ("base", "input")
        }
        assert summary["wall_speedup"] == wall["base"] / wall["input"]


def test_input_residual_raw_receipts_are_lossless() -> None:
    """Every accepted arm input/output and budget output remains recoverable."""
    evidence = load_publication_record(BUNDLE)
    receipts = {}
    raw_receipts = json.loads(
        lzma.decompress((BUNDLE / evidence["raw_receipts_file"]).read_bytes())
    )
    for name, entry in raw_receipts.items():
        raw = (
            base64.b64decode(entry["data"], validate=True)
            if entry["encoding"] == "base64"
            else entry["data"].encode()
        )
        assert len(raw) == entry["bytes"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        receipts[name] = raw
    for scope, count in (("cpu", 48), ("native", 32), ("strong", 48)):
        descriptor = evidence["algebraic_case_records"][scope]
        assert descriptor["raw_receipt"] == scope + "/samples.json"
        samples = json.loads(receipts[descriptor["raw_receipt"]])
        assert len(samples) == count
        assert descriptor["count"] == count
        for index, sample in enumerate(samples):
            prefix = f"{scope}/{index}-{sample['variant']}"
            assert prefix + ".input" in receipts and prefix + ".txt" in receipts
            record = sample["record"]
            assert record["status"] == 0 and record["replays_called"] == 1
            for gate, tolerance in (
                ("energy_error", 2e-12),
                ("replay_r1_error", 2e-12),
                ("replay_r2_error", 2e-12),
                ("determinant_r1", 1e-10),
                ("determinant_r2", 1e-10),
            ):
                error = sample["gates"][gate]
                assert math.isfinite(error) and 0 <= error <= tolerance
            if sample["variant"] == "input":
                assert record["iterations_called"] == record["iterations"]
    assert "0 errors" in receipts["native/memcheck.log"].decode()
    for variant in ("base", "input"):
        fields = (
            receipts[f"strong/budget-{variant}.txt"].decode().splitlines()[0].split()
        )
        assert int(fields[0]) == 1
        assert int(fields[2]) == int(fields[10]) == 2
        assert int(fields[11]) == 0
    assert "5 passed" in receipts["integration/focused-tests.log"].decode()
    assert "pilot-measured.py" in evidence["reproduction_recipes"]
    assert "integration-gpu.sh" in evidence["reproduction_recipes"]
