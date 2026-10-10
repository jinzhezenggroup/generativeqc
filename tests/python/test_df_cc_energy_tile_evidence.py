"""Protect scoped automatic-tile observations and lossless Git recovery."""

import json
import statistics
from pathlib import Path

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record
from tools.restore_retained_evidence import _records

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/df-cc-energy-q32-default-20261010"
PREVIOUS = ROOT / "benchmarks/results/df-cc-ladder-dressing-factorization-20261010"


def test_auto_tile_publication_keeps_all_independent_gates() -> None:
    """Numerical acceptance must not quietly become global performance proof."""
    publication = json.loads((BUNDLE / "publication.json").read_text())
    files = {
        entry["path"]: (BUNDLE / entry["path"]).read_bytes()
        for entry in publication["files"]
    }
    validate_publication(publication, files)
    assert publication["decision"]["scope"] == "numerical"
    evidence = load_publication_record(BUNDLE, role="evidence")
    assert evidence["performance"]["status"] == "not-run"
    assert evidence["stages"]["production"]["status"] == "not-run"
    assert (
        len(
            evidence["qualification_receipts"][
                "unchanged-generated.sha256"
            ].splitlines()
        )
        == 14
    )
    for gate in evidence["block_errors"].values():
        assert gate["passed"] and gate["shape"] == [8]


def test_auto_tile_matched_samples_keep_semantic_work_and_energy_bits() -> None:
    """All original records survive compaction; pilots never enter ABBA medians."""
    samples = load_publication_record(BUNDLE, role="samples")
    matched = samples["matched"]
    assert [sample["variant"] for sample in matched] == [
        "base",
        "candidate",
        "candidate",
        "base",
    ]
    assert [len(campaign["samples"]) for campaign in samples["feasibility_only"]] == [
        2,
        1,
        1,
    ]
    summary = load_publication_record(BUNDLE, role="evidence")[
        "observed_endpoint_summary"
    ]
    baseline = matched[0]["record"]
    for sample in matched:
        record = sample["record"]
        assert record["ccsd_evaluations"] == record["ccsd_pair_evaluations"] == 38
        assert record["ccsd_pair_refusals"] == 0
        assert record["ccsd_q_batch_size"] == (8 if sample["variant"] == "base" else 32)
        for key in (
            "ccsd_iterations",
            "ccsd_q_slices",
            "ccsd_setup_h2d_bytes",
            "ccsd_gemm_summands",
            "ccsd_contraction_terms",
            "ccsd_pair_projection_calls",
            "ccsd_pair_projection_bytes",
        ):
            assert record[key] == baseline[key]
        for key in ("total_energy", "triples_energy"):
            assert record[key].hex() == baseline[key].hex()
    for variant in ("base", "candidate"):
        selected = [sample for sample in matched if sample["variant"] == variant]
        assert summary["medians"][variant][
            "complete_process_wall_seconds"
        ] == statistics.median(sample["process_wall_seconds"] for sample in selected)
    deltas = summary["candidate_minus_baseline_deltas"]
    assert deltas["ccsd_gemm_calls"] == -29070
    assert deltas["ccsd_contraction_terms"] == deltas["ccsd_gemm_summands"] == 0
    assert deltas["ccsd_capacity"] == 4368386504


def test_previous_campaign_has_a_complete_existing_git_recovery_inventory() -> None:
    """Avoid requiring historical Git objects in source distributions or CI."""
    manifest = PREVIOUS / "snapshot.manifest.json"
    entries = _records(manifest)
    assert len(entries) == 8 and sum(entry["bytes"] for entry in entries) == 21139
    assert {entry["revision"] for entry in entries} == {
        "f81640c8cd7531be60c1d9a4d2323e90934e1d1c"
    }
    assert {Path(entry["path"]).name for entry in entries} == {
        "README.md",
        "build.sh.gz",
        "measure-gpu.sh.gz",
        "measure.py.gz",
        "measured-source.patch.gz",
        "publication.json",
        "samples.json.gz",
        "validation.json.gz",
    }
