"""Replay complete endpoints and independent static observables without a GPU."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
GATES = (1e-9, 1e-7, 3e-8)
REFERENCE_GATES = (1e-5, 1e-4, 3e-5)
TASK_KEYS = (
    "method",
    "mode",
    "atoms",
    "geometry_bohr",
    "orbital_basis",
    "basis_representation",
    "charge",
    "spin",
    "protocol",
    "thread_controls",
    "runner_sha256",
    "occupied_xc_env",
)
WORK_KEYS = ("iterations", "fock_builds", "warm_start_used", "warm_start_fallback")


def errors(actual: dict[str, Any], expected: dict[str, Any]) -> list[float]:
    """Compare all 288 force components and energy per atom, not rounded output."""
    delta = [
        left - right
        for left_row, right_row in zip(
            actual["forces"], expected["forces"], strict=True
        )
        for left, right in zip(left_row, right_row, strict=True)
    ]
    return [
        abs(actual["energy"] - expected["energy"]) / 96,
        max(map(abs, delta)),
        math.sqrt(statistics.mean(value * value for value in delta)),
    ]


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Require matched scientific settings/work and keep every timing sample."""
    for key in TASK_KEYS:
        assert before[key] == after[key], key
    worst = [0.0, 0.0, 0.0]
    work = []
    for baseline, candidate in zip(before["samples"], after["samples"], strict=True):
        assert baseline["settings"] == candidate["settings"]
        for key in WORK_KEYS:
            assert baseline[key] == candidate[key], key
        for key in ("coulomb_exchange_work", "nonlocal_work"):
            assert baseline["ks_diagnostic"][key] == candidate["ks_diagnostic"][key]
        assert candidate["ks_diagnostic"]["coulomb_exchange_work"]["complete"]
        current = errors(candidate, baseline)
        assert all(value <= gate for value, gate in zip(current, GATES, strict=True))
        worst = [max(left, right) for left, right in zip(worst, current, strict=True)]
        work.append({key: candidate[key] for key in WORK_KEYS})
    cold = [record["samples"][0]["seconds"] for record in (before, after)]
    warm = [
        [sample["seconds"] for sample in record["samples"][1:]]
        for record in (before, after)
    ]
    medians = list(map(statistics.median, warm))
    return {
        "cold_single_observations": cold,
        "cold_ratio": cold[1] / cold[0],
        "warm_samples": warm,
        "warm_medians": medians,
        "warm_ratio": medians[1] / medians[0],
        "maximum_errors": worst,
        "gates": list(GATES),
        "work": work,
    }


def replay(publication: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on incomplete jobs, provenance drift or missing references."""
    assert publication["schema"] == "generativeqc.vv10-packet-fp64-fma.v1"
    assert publication["status"] in ("qualified", "rejected")
    assert publication["production_changed"] is False
    for key, record in publication["records"].items():
        assert record["status"] == "complete"
        assert record["atoms"] == 96 and record["mode"] == "direct"
        assert record["warm_repeats"] == 3 and len(record["samples"]) == 4
        identity = publication["identities"][key]
        assert identity["scheduler"] == record["scheduler"]
        assert record["scheduler"]["CUDA_VISIBLE_DEVICES"]
        for field in ("source_identity", "library_sha256"):
            assert identity[field] == record["native_build"][field]
            assert identity[field] == publication["image"][field]
        assert record["source_revision"] == publication["image"]["source_identity"]
        for index, sample in enumerate(record["samples"]):
            assert (sample["phase"], sample["repeat"]) == (
                ("cold", 0) if index == 0 else ("warm", index - 1)
            )
            assert sample["converged"] and sample["warm_start_used"] == bool(index)
            assert sample["iterations"] > 0 and sample["fock_builds"] > 0
            assert math.isfinite(sample["seconds"]) and sample["seconds"] > 0
            assert math.isfinite(sample["energy"]) and len(sample["forces"]) == 96
            assert all(len(row) == 3 for row in sample["forces"])
            assert all(
                math.isfinite(value) for row in sample["forces"] for value in row
            )
    for job in publication["jobs"].values():
        assert job["exit_code"] == 0 and job["completed_at"]
    for receipt in publication["device_policy_receipts"].values():
        assert receipt["status"] == 0 and receipt["policy_version"] == 2
    results = {}
    for device, method in (
        ("A100", "wb97m-v"),
        ("RTX5090", "wb97m-v"),
        ("RTX5090", "r2scan-3c"),
    ):
        prefix = f"{device}/{method}"
        results[prefix] = compare(
            publication["records"][prefix + "-incumbent"],
            publication["records"][prefix + "-candidate"],
        )
        assert results[prefix]["warm_ratio"] <= 1.05, prefix
        if device == "RTX5090":
            previous, current = results[prefix]["warm_samples"]
            assert all(
                after <= 1.05 * before
                for before, after in zip(previous, current, strict=True)
            ), prefix
        if method == "wb97m-v" and publication["status"] == "qualified":
            assert results[prefix]["cold_ratio"] < 1, prefix
    receipt = publication["reference"]
    reference_path = ROOT.parents[2] / receipt["path"]
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == receipt["sha256"]
    reference = json.loads(lzma.decompress(reference_path.read_bytes()))
    worst = [0.0, 0.0, 0.0]
    pairs = 0
    for key, record in publication["records"].items():
        if "/wb97m-v-" not in key:
            continue
        for field, expected in reference["cohort"].items():
            assert record[field] == expected, field
        for sample in record["samples"]:
            for target in reference["records"]["A100/reference"]["samples"]:
                current = errors(sample, target)
                assert all(
                    value <= gate
                    for value, gate in zip(current, REFERENCE_GATES, strict=True)
                )
                worst = [
                    max(left, right) for left, right in zip(worst, current, strict=True)
                ]
                pairs += 1
    assert pairs == 32 and receipt["timing_reused"] is False
    return {
        "status": publication["status"],
        "comparisons": results,
        "reference_pairs": pairs,
        "reference_maximum_errors": worst,
    }


def main() -> None:
    """Authenticate retained bytes before deriving the published summary."""
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    publication = json.loads(
        lzma.decompress((ROOT / "measurements.json.xz").read_bytes())
    )
    summary = replay(publication)
    assert summary == json.loads((ROOT / "summary.json").read_text())
    compact = json.loads(lzma.decompress((ROOT / "compact-probe.json.xz").read_bytes()))
    assert compact["status"] == "rejected"
    result = compact["result"]
    assert result["scheduler"]["SLURM_JOB_ID"] == "16022"
    assert result["count"] == 320000
    assert (
        hashlib.sha256(compact["files"]["probe.cu"].encode()).hexdigest()
        == result["source_sha256"]
    )
    assert [(sample["geometry"], sample["rows"]) for sample in result["samples"]] == [
        (False, 4),
        (False, 2),
        (True, 4),
        (True, 2),
    ]
    assert all(sample["seconds"] > 0 for sample in result["samples"])
    assert all(
        sample["max_difference"] < 2e-11
        for sample in result["samples"]
        if sample["rows"] == 2
    )
    operations = json.loads(lzma.decompress((ROOT / "operations.json.xz").read_bytes()))
    assert operations["restore_exit"] == 0
    assert operations["restored_policy"]["policy_version"] == 2
    assert operations["cleanup_transport"]["remote_returncode"] == 0
    assert (
        "DONE" in operations["cleanup_log"]
        and "OWNED_TRANSFER_STOPPED" in operations["cleanup_log"]
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
