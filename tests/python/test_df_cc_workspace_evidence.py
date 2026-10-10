"""Protect scoped workspace timing, unchanged work, and all accepted records."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/df-cc-blas-workspace-20261010"


def test_workspace_publication_preserves_scope_and_original_gates() -> None:
    """A scoped observation must not become force/global statistical proof."""
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
    assert evidence["settings"]["candidate_workspace_bytes"] == 4194304
    assert evidence["settings"]["provider_allowance_bytes"] == 96 << 20
    for gate in evidence["block_errors"].values():
        assert gate["passed"] and gate["shape"] == [7]
    assert (
        evidence["qualification_receipts"]["generated_identity"]["unchanged_artifacts"]
        == 14
    )
    assert (
        "0 errors" in evidence["qualification_receipts"]["final/workspace-memcheck.log"]
    )


def test_workspace_records_keep_work_and_separate_integration_from_abba() -> None:
    """Keep every raw record; neither pilots nor final checks enter medians."""
    evidence = load_publication_record(BUNDLE)
    retained = evidence["retained_samples"]
    matched = retained["matched"]
    assert [sample["variant"] for sample in matched] == [
        "base",
        "workspace",
        "workspace",
        "base",
    ]
    assert len(retained["feasibility_only"]["samples"]) == 2
    all_samples = (
        matched
        + retained["feasibility_only"]["samples"]
        + [retained["integration_only"]]
    )
    baseline = matched[0]["record"]
    audits = {
        "ccsd_replay_r1_max",
        "ccsd_replay_r2_max",
        "ccsd_diis_maximum_pair_asymmetry",
    }
    counters = [
        key
        for key in baseline
        if key.startswith("ccsd_")
        and not key.endswith("_seconds")
        and key not in audits
    ]
    for sample in all_samples:
        record = sample["record"]
        assert all(record[key] == baseline[key] for key in counters)
        assert record["ccsd_q_batch_size"] == 32
        assert record["ccsd_evaluations"] == record["ccsd_pair_evaluations"] == 38
        assert record["ccsd_pair_refusals"] == 0
        assert record["total_energy"].hex() == baseline["total_energy"].hex()
        assert abs(record["triples_energy"] - baseline["triples_energy"]) <= 1e-11
    summary = evidence["observed_endpoint_summary"]
    for variant in ("base", "workspace"):
        selected = [sample for sample in matched if sample["variant"] == variant]
        assert summary["medians"][variant][
            "complete_process_wall_seconds"
        ] == statistics.median(sample["wall_seconds"] for sample in selected)
    assert summary["maximum_triples_energy_difference"] > 0
    assert retained["integration_only"]["slurm_job"] == "2835"


def test_workspace_source_and_current_journal_boundaries_are_explicit() -> None:
    """Current wrapper ownership is qualified separately from frozen RHF timing."""
    evidence = load_publication_record(BUNDLE)
    reconstruction = evidence["source_reconstruction"]
    assert reconstruction["total_overlay_files_verified"] == 22
    assert len(reconstruction["overlay_files"]) == 5
    assert reconstruction["q32_patch_git_revision"] == (
        "9f67e7806e3151454530baf0ee66ae8808d826f0"
    )
    assert (
        "standalone journal test only" in reconstruction["current_header_qualification"]
    )
    assert (
        "1 passed"
        in evidence["qualification_receipts"]["final/current-journal-test.log"]
    )
    assert (
        "reproduction_recipes" in evidence
        and "measure.py" in evidence["reproduction_recipes"]
    )
