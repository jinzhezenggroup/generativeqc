"""Replay scientific/whole-endpoint gates and reject incomplete admission evidence."""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools import validate_direct_jk_aot_resources as resource_validator

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "benchmarks/results/dft-component-queue-20261109"
REPLAY = runpy.run_path(str(FAMILY / "replay.py"))


@pytest.mark.parametrize("corrupt_resource", (False, True))
def test_component_aot_requalification_preserves_all_resources(
    corrupt_resource: bool,
) -> None:
    """Replay the actual failed report; identity refresh cannot mask REG drift."""
    folder = FAMILY / "resources"
    receipt = json.loads((folder / "receipt.json").read_text())
    records = {}
    for name, hashes in receipt["files"].items():
        packed = (folder / name).read_bytes()
        assert hashlib.sha256(packed).hexdigest() == hashes["compressed_sha256"]
        raw = lzma.decompress(packed)
        assert hashlib.sha256(raw).hexdigest() == hashes["raw_sha256"]
        records[name] = json.loads(raw)
    previous = records["previous.json.xz"]
    report = records["failed-ci-report.json.xz"]
    measurement = report["measurement"]
    assert report["status"] == "fail"
    assert resource_validator.compare(previous, measurement) == report["failures"]
    updated = resource_validator.measured_ledger(measurement)
    assert set(previous) == set(updated)
    assert {key for key in previous if previous[key] != updated[key]} == set(
        receipt["changed_identity_fields"]
    )
    assert len(updated["kernels"]) == receipt["kernel_count"]
    assert previous["kernels"] == updated["kernels"]
    if corrupt_resource:
        measurement["kernels"][next(iter(measurement["kernels"]))]["REG"] += 1
        assert any(
            "kernel resource drift" in failure
            for failure in resource_validator.compare(previous, measurement)
        )
    else:
        assert not resource_validator.compare(updated, measurement)


def test_normal_component_admission_qualifies_both_card_guards(tmp_path: Path) -> None:
    # A clean checkout has only the compact streams, never ignored GPU archives.
    for name in (
        "cohort.json",
        "fp64.xz",
        "receipts.json.xz",
        "sha256.txt",
        "qualification.py",
    ):
        shutil.copyfile(FAMILY / name, tmp_path / name)
    result = REPLAY["replay"](family=tmp_path)
    assert result["default_promotion_qualified"]
    assert not result["all_readme_leadership_established"]
    assert not result["whole_endpoint_quartet_counters_available"]
    for card, report in result["cards"].items():
        assert not report["performance_failures"]
        for name, population in report["populations"].items():
            assert population["cold_ratio"] <= (
                0.98 if card == "a100" and name.endswith("96") else 1.03
            )
            assert max(population["paired_warm_ratios"]) <= 1.03
        for guard in report["other_hybrid_guards"].values():
            assert guard["single_pair_not_abba_promotion"]
            assert max(guard["candidate_over_control_ratios"]) <= 1.03
    assert "raw_k_actual_queue_fatbin_not_normal_elf" in result["cards"]["5090"]
    assert "raw_k_candidate" not in result["cards"]["5090"]


@pytest.mark.parametrize(
    "change",
    ("forces", "focks", "warm", "image", "collector", "raw-duplicate", "prototype"),
)
def test_changed_physics_work_identity_or_timing_cannot_qualify(
    tmp_path: Path,
    change: str,
) -> None:
    REPLAY["materialize"](tmp_path)
    directory = tmp_path / "native-a100"
    if change == "prototype":
        path = tmp_path / "stream-5090-16921/raw-k-stream.log"
        path.write_text(path.read_text().replace("PASS ", "FAIL ", 1))
    elif change == "collector":
        path = directory / "b3lyp-candidate-cold-kernels.tsv"
        path.write_text(path.read_text().replace("dropped=0", "dropped=1", 1))
    elif change == "raw-duplicate":
        path = directory / "raw-k-candidate.log"
        lines = path.read_text().splitlines()
        indices = [
            index for index, line in enumerate(lines) if line.startswith("PASS ")
        ]
        lines[indices[1]] = lines[indices[0]]
        path.write_text("\n".join(lines) + "\n")
    else:
        path = directory / "b3lyp-96-2-candidate.json"
        value = json.loads(path.read_text())
        if change == "forces":
            value["samples"][0]["forces"][0][0] += 0.1
        elif change == "focks":
            value["samples"][0]["fock_builds"] += 1
        elif change == "warm":
            value["samples"][1]["seconds"] *= 2
        elif change == "image":
            value["native_build"]["library_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="gate failed|qualification failed"):
        REPLAY["qualify_materialized"](tmp_path)


@pytest.mark.parametrize("optimized", (False, True))
def test_component_evidence_standalone_without_pythonpath(
    tmp_path: Path, optimized: bool
) -> None:
    """Standalone replay must find its repo; -O must reject before imports."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONOPTIMIZE", None)
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), str(FAMILY / "replay.py")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if optimized:
        assert result.returncode != 0
        assert "Evidence validation requires assertions" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["default_promotion_qualified"]
