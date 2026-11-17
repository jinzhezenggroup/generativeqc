"""Replay source-bound R1 observations and reject scientifically invalid rows."""

from __future__ import annotations

import copy
import hashlib
import json
import lzma
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from benchmarks._native_dft_force_evidence import load
from benchmarks.native_dft_force_science_audit import (
    replay,
    validate_automatic,
    validate_r2scan,
)

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "benchmarks/results/r1-native-dft-forces-20261115/endpoint-observations"


@pytest.fixture(scope="module")
def scientific_documents() -> tuple[dict, dict[str, dict]]:
    documents = load(BUNDLE)[1]
    index = json.loads(documents["scientific/index.json"])
    return index, {name: json.loads(documents[name]) for name in index["files"]}


@pytest.mark.parametrize(
    ("cohort", "backend", "status", "comparisons"),
    (
        ("cpu", "cpu", "PASS", 240),
        ("cuda", "cuda", "PASS", 408),
        ("cpu-incomplete", "cpu", "INCOMPLETE", 120),
        ("cuda-incomplete", "cuda", "INCOMPLETE", None),
        ("cuda-rejected", "cuda", "FAIL", 408),
        ("cuda-head-rejected", "cuda", "FAIL", 408),
        ("cuda-bound-rejected", "cuda", "FAIL", 408),
    ),
)
def test_every_paired_population_retains_its_actual_gate(
    tmp_path: Path, cohort: str, backend: str, status: str, comparisons: int | None
) -> None:
    output = tmp_path / "audit.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.native_dft_force_audit",
            str(BUNDLE),
            "--backend",
            backend,
            "--cohort",
            cohort,
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == (0 if status == "PASS" else 1), completed.stderr
    result = json.loads(output.read_text())
    assert result["status"] == status
    if comparisons is not None:
        assert len(result["comparisons"]) == comparisons
    if cohort in {"cuda-rejected", "cuda-head-rejected", "cuda-bound-rejected"}:
        assert len(result["failures"]) == (1 if cohort == "cuda-rejected" else 2)
        assert result["failures"][0]["ratio"] > 1.05


def test_accepted_cuda_population_uses_only_its_assigned_cpu() -> None:
    documents = load(BUNDLE)[1]
    cohort = json.loads(documents["cuda/cohort.json"])
    binding = cohort["resource_binding"]
    assert binding["effective_affinity"] == [min(binding["slurm_assigned_cpus"])]
    assert binding["selected_cpu"] == binding["effective_affinity"][0]
    assert set(binding["slurm_assigned_cpus"]) <= set(binding["original_affinity"])
    assert binding["actual_cuda_uuid"] in binding["nvml_devices"]
    assert not any(
        line.startswith(binding["actual_cuda_uuid"] + ",")
        for line in binding["nvml_applications_before"].splitlines()[1:]
    )
    assert len(cohort["affinity_receipts"]) == 8
    for name, affinity in cohort["affinity_receipts"].items():
        assert affinity["affinity"] == binding["effective_affinity"], name
        assert affinity["slurm_job_id"] == binding["slurm_job_id"]
        assert affinity["cuda_visible_devices"] == binding["cuda_visible_devices"]
    for name in cohort["blocks"].values():
        metadata = json.loads(documents[name])["records"][0]["metadata"]
        assert metadata["slurm_job_id"] == binding["slurm_job_id"]
        assert metadata["cuda_visible_devices"] == binding["cuda_visible_devices"]


def test_every_nonacceptance_diagnostic_timing_row_is_retained() -> None:
    """Keep all pauses without duplicating qualification E/F in diagnostic storage."""
    index = json.loads(
        lzma.decompress((BUNDLE.parent / "diagnosis.json.xz").read_bytes())
    )
    assert index["schema"] == "generativeqc.native-dft-force-diagnosis.v1"
    assert index["qualification"] is False
    assert len(index["streams"]) == 16
    assert sum(len(stream["records"]) for stream in index["streams"].values()) == 7512
    for name, stream in index["streams"].items():
        for field in ("original_jsonl_sha256", "original_diagnostic_sha256"):
            assert len(bytes.fromhex(stream[field])) == hashlib.sha256().digest_size
        rows = stream["records"]
        assert len(rows) == stream["original_rows"]
        assert "metadata" in stream["header"]
        samples = defaultdict(list)
        for row in rows:
            assert set(row) == set(stream["columns"])
            samples[(row["case"], row["phase"])].append(row["sample"])
            assert math.isfinite(row["seconds"]) and row["seconds"] > 0
            for field in ("outer_wall_ns", "outer_thread_ns"):
                assert row[field] > 0
            for field in ("voluntary_switches", "involuntary_switches", "gc_events"):
                assert row[field] >= 0
            if name.startswith("ffi-"):
                assert len(row["ffi"]) == 1
                call = row["ffi"][0]
                assert call["status"] == 0
                assert call["function"] in {
                    "generativeqc_batch_execute",
                    "generativeqc_batch_execute_selective",
                }
                assert call["wall_ns"] > 0 and call["thread_ns"] > 0
            else:
                assert row["ffi"] == []
        expected_cases = 3 if name.startswith("ffi-") else 2
        expected_samples = 21 if "/nsys-" in name or "/strace-" in name else 100
        assert len(samples) == expected_cases * 3
        assert {phase for _, phase in samples} == {"cold", "warm", "moved"}
        for sample_ids in samples.values():
            assert sorted(sample_ids) == list(range(expected_samples))


def test_original_scientific_observations_replay_offline() -> None:
    result = replay(BUNDLE)
    assert result["status"] == "PASS"
    assert result["automatic_records"] == 34
    assert result["automatic_rows"] == 58
    assert result["automatic_functionals"] == 11
    assert result["r2scan3c_records"] == 11


def test_two_channel_singlet_is_not_confused_with_spin_multiplicity(
    scientific_documents: tuple[dict, dict[str, dict]],
) -> None:
    index, documents = scientific_documents
    record = next(
        record
        for name, record in documents.items()
        if "/automatic-forces/" in name
        and record["context"]["fixture"] == "water"
        and record["spins"] == 2
    )
    assert record["multiplicity"] == 1
    assert validate_automatic(record, index) == 3


@pytest.mark.parametrize("mutation", ("energy", "force", "owner", "sources"))
def test_validly_encoded_bad_automatic_observations_still_fail(
    scientific_documents: tuple[dict, dict[str, dict]], mutation: str
) -> None:
    index, documents = scientific_documents
    record = copy.deepcopy(
        next(
            record
            for name, record in documents.items()
            if "/automatic-forces/" in name and record["context"]["full_vector"]
        )
    )
    row = record["rows"][0]
    if mutation == "energy":
        row["reference_energy"] += 1e-3
    elif mutation == "force":
        row["independent_fd"][0] += 1e-2
    elif mutation == "owner":
        row["artifact"]["force_artifact"]["sha256"] = "0" * 64
    else:
        row["artifact"]["source_count"] = 6
    with pytest.raises(ValueError):
        validate_automatic(record, index)


@pytest.mark.parametrize(
    "name",
    (
        "canonical-3atom-r2scan3c-df-independent-force.json",
        "water-spd-resource-work.json",
        "rks-3-charge-0-production-grid-independent-oracle.json",
        "charged-ragged-failure-isolation.json",
        "ragged-force-changed-geometry.json",
    ),
)
def test_validly_encoded_bad_r2scan_observations_still_fail(
    scientific_documents: tuple[dict, dict[str, dict]], name: str
) -> None:
    index, documents = scientific_documents
    record = copy.deepcopy(documents["scientific/r2scan3c-forces/" + name])
    if name.startswith("canonical-"):
        record["max_force_error"] = 1e-3
    elif name.startswith("water-spd-"):
        record["additional_device_peak_bound"] = record["additional_device_budget"] + 1
    elif name.startswith("rks-"):
        record["gcp_energy_error"] = 1e-3
    elif name.startswith("charged-"):
        record["isolated_statuses"] = [0, 0]
    else:
        record["force_max_errors"].pop()
    with pytest.raises(ValueError):
        validate_r2scan(name, record, index)
