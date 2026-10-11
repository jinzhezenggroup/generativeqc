"""The historical incremental capsule fails closed without requiring CUDA."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[2]
CAPSULE = ROOT / "benchmarks/results/issue1598-incremental-direct-jk-20261011"
POINT = "old/rhf/direct-48-incremental-1.json.gz"


def refresh_manifest(root: Path) -> None:
    """Re-sign storage mutations to reach semantic, rather than hash, gates."""
    path = root / "capsule.json"
    manifest = json.loads(path.read_text())
    for entry in manifest["files"]:
        raw = (root / entry["path"]).read_bytes()
        entry.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    path.write_text(json.dumps(manifest))


def mutate_record(
    root: Path, relative: str, mutation: Callable[[dict], object]
) -> None:
    path = root / relative
    record = json.loads(gzip.decompress(path.read_bytes()))
    mutation(record)
    raw = gzip.compress(json.dumps(record).encode(), mtime=0)
    path.write_bytes(raw)
    # Preserve selected/successful-attempt equality unless testing that gate.
    if relative.endswith(".json.gz") and "-attempt" not in relative:
        path.with_name(path.name.replace(".json.gz", "-attempt1.json.gz")).write_bytes(
            raw
        )
    refresh_manifest(root)


def shift_all_results(record: dict) -> None:
    """Keep reference gates passing while breaking the round-2 on/off gate."""
    for engine in ("generativeqc", "gpu4pyscf"):
        result = record[engine]
        for row in (result, *result["warm_samples"]):
            row["energies_hartree"][0] += 1.0


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        "intact",
        "missing_manifest",
        "missing_success",
        "missing_attempt",
        "missing_manifest_member",
        "checksum",
        "source",
        "source_receipt",
        "native_source",
        "library",
        "device",
        "uuid",
        "mode",
        "inactive",
        "iterations",
        "work_count",
        "retry",
        "false_gate",
        "nan",
        "infinity",
        "round2_pair",
        "warm_reference",
        "warm_sample",
        "negative_time",
        "cold_direction",
        "median",
        "missing_sample",
        "attempt_mismatch",
        "weakened_tolerance",
        "promotion_status",
    ],
)
def test_historical_capsule_gates(
    tmp_path: Path, optimized: bool, mutation: str
) -> None:
    shutil.copytree(CAPSULE, tmp_path, dirs_exist_ok=True)
    if mutation == "missing_manifest":
        (tmp_path / "capsule.json").unlink()
    elif mutation in {"missing_success", "missing_attempt"}:
        suffix = ".json.gz" if mutation == "missing_success" else "-attempt1.json.gz"
        (tmp_path / POINT.replace(".json.gz", suffix)).unlink()
    elif mutation in {"missing_manifest_member", "promotion_status"}:
        path = tmp_path / "capsule.json"
        manifest = json.loads(path.read_text())
        if mutation == "missing_manifest_member":
            manifest["files"] = [
                entry for entry in manifest["files"] if entry["path"] != POINT
            ]
        else:
            manifest["decision"]["status"] = "accepted"
        path.write_text(json.dumps(manifest))
    elif mutation == "checksum":
        path = tmp_path / POINT
        path.write_bytes(path.read_bytes() + b"unexpected")
    elif mutation == "source_receipt":
        (tmp_path / "old/source-sha.txt.gz").write_bytes(
            gzip.compress(("0" * 40 + "\n").encode(), mtime=0)
        )
        refresh_manifest(tmp_path)
    elif mutation == "round2_pair":
        mutate_record(tmp_path, POINT.replace("-1.json", "-2.json"), shift_all_results)
    elif mutation == "retry":
        mutate_record(
            tmp_path,
            "old/rhf/direct-24-incremental-1.outcome.gz",
            lambda d: d.update(exit_code=0, attempt=1, retried=False),
        )
    elif mutation == "attempt_mismatch":
        mutate_record(
            tmp_path,
            POINT.replace(".json.gz", "-attempt1.json.gz"),
            lambda d: d["generativeqc"].update(cold_seconds=123),
        )
    elif mutation != "intact":
        mutations = {
            "source": lambda d: d["environment"]["git"].update(commit="0" * 40),
            "native_source": lambda d: d["native_build"]["probe"].update(
                source_identity="0" * 64
            ),
            "library": lambda d: d["native_build"].update(library_sha256="0" * 64),
            "device": lambda d: d["environment"]["accelerator"].update(name="CPU"),
            "uuid": lambda d: d["generativeqc"]["eigensolver_diagnostics"][0].update(
                device_uuid="another-device"
            ),
            "mode": lambda d: d["generativeqc"]["cold_convergence"][0][
                "incremental_direct_jk"
            ].update(requested=False),
            "inactive": lambda d: d["generativeqc"]["cold_convergence"][0][
                "incremental_direct_jk"
            ].update(active=False),
            "iterations": lambda d: d["generativeqc"]["cold_convergence"][0].update(
                iterations=0
            ),
            "work_count": lambda d: d["generativeqc"]["cold_convergence"][0][
                "incremental_direct_jk"
            ].update(delta_builds=-1),
            "false_gate": lambda d: d["gate"].update(passed="false"),
            "nan": lambda d: d["generativeqc"]["forces_hartree_per_bohr"][0][
                0
            ].__setitem__(1, float("nan")),
            "infinity": lambda d: d["generativeqc"]["warm_samples"][1][
                "energies_hartree"
            ].__setitem__(0, float("inf")),
            "warm_reference": lambda d: d["gpu4pyscf"]["warm_samples"][0][
                "energies_hartree"
            ].__setitem__(0, 0),
            "warm_sample": lambda d: d["generativeqc"]["warm_samples"][0][
                "energies_hartree"
            ].__setitem__(0, 0),
            "negative_time": lambda d: d["generativeqc"].update(cold_seconds=-1),
            "cold_direction": lambda d: d["generativeqc"].update(cold_seconds=0.1),
            "median": lambda d: d["generativeqc"].update(warm_median_seconds=0.1),
            "missing_sample": lambda d: d["generativeqc"]["warm_samples"].pop(),
            "weakened_tolerance": lambda d: d["settings"]["gates"].update(
                maximum_energy_error_hartree=1
            ),
        }
        mutate_record(tmp_path, POINT, mutations[mutation])
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), str(tmp_path / "verify.py")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if mutation == "intact":
        assert result.returncode == 0, result.stderr
        summary = json.loads(result.stdout)
        assert len(summary["campaigns"]["old"]["points"]) == 13
        assert len(summary["campaigns"]["new"]["points"]) == 8
        assert len(summary["campaigns"]["old"]["pairs"]) == 5
        assert len(summary["campaigns"]["new"]["pairs"]) == 2
    else:
        assert result.returncode != 0, f"{mutation} was accepted: {result.stdout}"
