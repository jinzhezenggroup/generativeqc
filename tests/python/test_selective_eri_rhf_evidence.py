"""Recompute retained local E+F claims without promoting default/global policy."""

import gzip
import hashlib
import json
import lzma
from pathlib import Path

from benchmarks.selective_eri_rhf_acceptance import compare
from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_publication_record

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/selective-eri-rhf-partial-20261011"


def test_compressed_input_preserves_the_measured_scientific_identity() -> None:
    """Storage compaction cannot change the exact molecular input digest."""
    raw = gzip.decompress((BUNDLE / "basis.input.gz").read_bytes())
    record = load_publication_record(BUNDLE)
    assert (
        hashlib.sha256(raw).hexdigest()
        == record["retained_comparison"]["identity"]["input_sha256"]
    )


def test_compact_patch_retains_all_native_and_endpoint_reconstruction_paths() -> None:
    """The smaller patch remains self-contained for the actual E+F endpoint."""
    manifest = json.loads((BUNDLE / "publication.json").read_text())
    patch = lzma.decompress((BUNDLE / "measured-endpoint-source.patch.xz").read_bytes())
    paths = {
        section.split(b"\n", 1)[0].split(b" b/", 1)[1].decode()
        for section in patch.split(b"diff --git ")[1:]
    }
    assert paths == set(manifest["source_reconstruction"]["included_paths"])
    assert {
        "src/hf/rhf_frame_response.cu",
        "src/scf/cuda/direct_jk.cpp",
        "src/scf/cuda/direct_jk_plan.hpp",
        "src/scf/cuda/reference_eri_policy.hpp",
        "src/scf/cuda/rhf_resident_values.cpp",
        "src/scf/cuda/rhf_resident_values.hpp",
        "src/scf/cuda_direct_jk_device.hpp",
        "benchmarks/selective_eri_rhf_endpoint.cpp",
        "benchmarks/selective_eri_rhf_endpoint.py",
        "benchmarks/selective_eri_rhf_acceptance.py",
        "tests/native/rhf_frame_response_probe.cpp",
        "tests/python/test_rhf_frame_response_cuda.py",
    } <= paths


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
