"""Audit retained promotion evidence, without substituting for fresh GPU tests."""

from __future__ import annotations

import hashlib
import json
import lzma
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "benchmarks/results/native-memory-arena-20261011/evidence.json.xz"


@pytest.fixture(scope="module")
def evidence() -> dict:
    stored = RECORD.read_bytes()
    raw = lzma.decompress(stored)
    metadata = json.loads(RECORD.with_name("storage.json").read_text())
    assert len(stored) == metadata["stored_bytes"]
    assert hashlib.sha256(stored).hexdigest() == metadata["stored_sha256"]
    assert len(raw) == metadata["decoded_bytes"] == 6_352_916
    assert hashlib.sha256(raw).hexdigest() == metadata["decoded_sha256"]
    assert (
        metadata["decoded_sha256"]
        == "e8b1b1541e3df8364c72a3f7254b3c73a9ce110d7014cbfaa8241aae13acbd6e"
    )
    return json.loads(raw)


def test_cross_size_default_has_a_separate_serial_scratch_boundary(
    evidence: dict,
) -> None:
    assert evidence["base_commit"] == "a70ebfc61644910a35916790febd51aaaffc293e"
    assert evidence["selection"] == {
        "baseline": {"cross_size_arena": False, "serial_scratch": False},
        "candidate": {"cross_size_arena": True, "serial_scratch": False},
    }
    assert re.search(
        r"option\(GENERATIVEQC_NATIVE_CROSS_SIZE_ARENA\s+\"[^\"]+\"\s+ON\)",
        (ROOT / "CMakeLists.txt").read_text(),
    )
    assert (
        "bool serial_scratch_reuse{false}"
        in (ROOT / "src/cc/lambda_response.hpp").read_text()
    )


def test_cold_profile_changes_storage_without_adding_kernel_or_copy_work(
    evidence: dict,
) -> None:
    baseline = evidence["profile"]["master-baseline"]
    candidate = evidence["profile"]["master-candidate"]
    assert (
        candidate["peak_requested_live_bytes"] < baseline["peak_requested_live_bytes"]
    )
    for key in (
        "allocations",
        "kernel_launches",
        "memcpy",
        "kernel_resources",
        "normalized_static_resource_dump_sha256",
    ):
        assert candidate[key] == baseline[key], key
    assert candidate["actual_hbm_traffic"] == "not measured"


@pytest.mark.parametrize("node", ("n1", "n5"))
@pytest.mark.parametrize("endpoint", ("water24", "ethane230"))
def test_complete_lifecycle_keeps_caches_work_and_scientific_gates(
    evidence: dict, node: str, endpoint: str
) -> None:
    files = evidence["lifecycles"][node][endpoint]
    summary = files["summary.json"]
    assert summary["screen_passed"]
    assert summary["repeats"] == (1 if node == "n5" and endpoint == "ethane230" else 3)
    assert summary["provenance"]["serial_scratch"] == 0
    assert summary["provenance"]["library_sha256"] == evidence["libraries"]
    assert set(summary["stages"]) == {"cold", "warm", "moved-warm"}
    for stage in summary["stages"].values():
        assert stage["epsilon"] == 0.02
        assert stage["screen_noninferiority_gate"]
        assert stage["median_ratio"] <= 1.02
    samples = files["partial.json"]["samples"]
    for baseline, candidate in zip(
        samples["baseline"], samples["candidate"], strict=True
    ):
        assert (
            candidate["process_residency"]["maximum_mib"]
            <= baseline["process_residency"]["maximum_mib"]
        )
        for old, new in zip(baseline["stages"], candidate["stages"], strict=True):
            for key in (
                "lambda_core_reuse",
                "lambda_core_reuse_bytes",
                "lambda_audit_matrix",
                "lambda_primal_matrix",
            ):
                assert old[key] == new[key], key
            for key in ("lambda_device_bytes", "lambda_capacity"):
                assert new[key] <= old[key], key
            for key in ("lambda_residual", "z_residual", "stationarity"):
                assert new[key] < 1e-9, key
