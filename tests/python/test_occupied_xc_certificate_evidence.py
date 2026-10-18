"""The offline certificate packet reader must not silently disable its gates."""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

VERIFY = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/occupied-xc-certificate-20261018/verify.py"
)
REFUSAL = "Evidence verification requires assertions"


def _fixture(tmp_path: Path, damage: str = "none") -> tuple[Path, Path]:
    """Use tiny synthetic records, without optional Git history or GPU work."""
    script = tmp_path / "verify.py"
    shutil.copyfile(VERIFY, script)
    force = struct.pack("<3d", 0.0, 0.0, 0.0)
    if damage == "force":
        force = struct.pack("<3d", 1.0, 0.0, 0.0)
    elif damage == "nonfinite":
        force = struct.pack("<3d", float("nan"), 0.0, 0.0)
    planes = b"".join(force[byte::8] for byte in range(8))
    phases = ["cold", *["warm"] * 5, "moved", *["moved-warm"] * 5]
    rows = [
        {
            "phase": phase,
            "geometry": int(phase.startswith("moved")),
            "forces_xor_f64le_byteplanes": planes.hex(),
            "forces_f64le_sha256": hashlib.sha256(force).hexdigest(),
            "converged": True,
            "energy": 1.0 if damage == "energy" else 0.0,
            "complete_seconds": 1.0,
            "iterations": 1,
            "fock_builds": 1,
        }
        for phase in phases
    ]
    source = {
        "native_identity": "synthetic",
        "runtime_files": {"libgenerativeqc.so": "synthetic"},
    }
    packet = {
        "candidate_source": source,
        "default_enabled": False,
        "incomplete": {},
        "oracles": {
            "synthetic": {
                "protocol": {},
                "geometries": {
                    str(i): {
                        "forces": [[0.0, 0.0, 0.0]],
                        "energy": 0.0,
                        "converged": True,
                    }
                    for i in range(2)
                },
            }
        },
        "populations": {
            "candidate/synthetic": {
                "case": "synthetic",
                "receipts": {"selector.txt": "1"},
                "native": {
                    "native_build": {
                        "probe": {
                            "source_identity": "wrong"
                            if damage == "source"
                            else "synthetic",
                            "policy": {"precision": "fp64"},
                        },
                        "library_sha256": "synthetic",
                    },
                    "stage": "complete",
                    "status": "measured",
                    "protocol": {},
                    "records": rows,
                },
            }
        },
    }
    raw = lzma.compress(json.dumps(packet).encode())
    if damage == "malformed":
        raw = b"not an xz packet"
    expected = hashlib.sha256(raw).hexdigest()
    if damage == "checksum":
        expected = "0" * 64
    evidence = tmp_path / "explicit.xz"
    evidence.write_bytes(raw)
    (tmp_path / "SHA256SUMS").write_text(
        f"{expected}  evidence.json.xz\n"
        f"{hashlib.sha256(script.read_bytes()).hexdigest()}  verify.py\n"
    )
    return script, evidence


def _run(
    script: Path, evidence: Path, mode: str = "normal"
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONOPTIMIZE", None)
    flags = []
    if mode == "environment":
        env["PYTHONOPTIMIZE"] = "1"
    elif mode != "normal":
        flags = [mode]
    return subprocess.run(
        [sys.executable, *flags, str(script), "--evidence", str(evidence)],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_synthetic_packet_replays_without_git_or_gpu(tmp_path: Path) -> None:
    result = _run(*_fixture(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "completed calls: 12" in result.stdout


@pytest.mark.parametrize(
    "damage", ["checksum", "malformed", "energy", "force", "nonfinite", "source"]
)
def test_normal_reader_rejects_invalid_packet(tmp_path: Path, damage: str) -> None:
    result = _run(*_fixture(tmp_path, damage))
    assert result.returncode != 0
    assert "completed calls:" not in result.stdout


@pytest.mark.parametrize("mode", ["-O", "-OO", "environment"])
@pytest.mark.parametrize("damage", ["none", "checksum", "malformed", "energy"])
def test_optimized_reader_refuses_before_accepting_packet(
    tmp_path: Path, mode: str, damage: str
) -> None:
    result = _run(*_fixture(tmp_path, damage), mode)
    assert result.returncode != 0
    assert REFUSAL in result.stderr
    assert "measured native identity:" not in result.stdout


@pytest.mark.parametrize("mode", ["-O", "environment"])
def test_optimized_reader_refuses_before_reading_missing_packet(
    tmp_path: Path, mode: str
) -> None:
    script, _ = _fixture(tmp_path)
    result = _run(script, tmp_path / "missing.xz", mode)
    assert result.returncode != 0
    assert REFUSAL in result.stderr
    assert "FileNotFoundError" not in result.stderr
