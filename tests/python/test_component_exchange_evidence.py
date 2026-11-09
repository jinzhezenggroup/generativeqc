"""Replay scientific/whole-endpoint gates and reject incomplete admission evidence."""

from __future__ import annotations

import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FAMILY = ROOT / "benchmarks/results/dft-component-queue-20261109"
REPLAY = runpy.run_path(str(FAMILY / "replay.py"))


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


def test_component_evidence_refuses_optimized_python() -> None:
    result = subprocess.run(
        [sys.executable, "-O", str(FAMILY / "replay.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Evidence validation requires assertions" in result.stderr
