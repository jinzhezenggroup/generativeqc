"""Recompute retained local E+F claims without promoting default/global policy."""

import json
from pathlib import Path

from benchmarks.selective_eri_rhf_acceptance import compare
from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/selective-eri-rhf-partial-20261011"


def test_retained_complete_pairs_reproduce_the_scoped_acceptance() -> None:
    """All actual forces, work counts and device/settings identities remain live."""
    manifest = json.loads((BUNDLE / "publication.json").read_text())
    files = {
        entry["path"]: (BUNDLE / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    evidence = load_publication_record(BUNDLE)
    samples = load_publication_record(BUNDLE, role="samples", name="samples.json")
    oracle = load_publication_record(BUNDLE, role="samples", name="oracle.json")
    assessment = compare(samples["baseline"], samples["candidate"], oracle)
    assert assessment["qualified"]
    assert assessment == evidence["retained_comparison"]
    assert len(assessment["pairs"]) == 12
    assert manifest["decision"]["scope"] == "numerical"
    assert evidence["performance"]["status"] == "not-run"
    assert evidence["stages"]["production"]["status"] == "not-run"


def test_real_memory_constrained_source_partition_is_retained() -> None:
    """The benefit is partial production replay, not a disabled fitting baseline."""
    observers = load_publication_record(
        BUNDLE, role="summary", name="source-observers.json"
    )
    for index in range(3):
        old = observers["phases"][f"baseline-{index}"][0]
        new = observers["phases"][f"candidate-{index}"][0]
        assert old["phase_resident_value_bytes"] == 0
        assert new["phase_resident_value_bytes"] == 2146754256
        assert new["phase_resident_values_built"] == 268344282
        assert new["phase_canonical_source_values"] == 541089856
        assert new["phase_direct_values_per_action"] == 272745574
        assert new["phase_resident_fock_actions"] == 20
        assert new["reference_phase_peak_bytes"] <= 4294967296
    assert observers["chronological_process_order"] == [
        "candidate-0",
        "baseline-0",
        "baseline-1",
        "candidate-1",
        "candidate-2",
        "baseline-2",
    ]
