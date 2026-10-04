"""Cold-source evidence must reject inaccurate or incompletely charged results."""

from __future__ import annotations

import hashlib
import json
import lzma
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EVIDENCE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/wb97mv-cold-admission-20261004"
)


@pytest.mark.parametrize(
    "corruption",
    (
        "valid",
        "phase_sum",
        "force",
        "seed_fallback",
        "source",
        "qualification",
        "stored_hash",
    ),
)
def test_cold_source_evidence_rejects_corruption(
    tmp_path: Path, corruption: str
) -> None:
    """Rebind compressed hashes so scientific failures reach gates under -O."""
    shutil.copytree(EVIDENCE, tmp_path, dirs_exist_ok=True)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    name = (
        "support.json.xz" if corruption == "qualification" else "matched3-lda16.json.xz"
    )
    data = json.loads(lzma.decompress((tmp_path / name).read_bytes()))
    if corruption == "qualification":
        data["qualification"]["full_native_dft"] = False
    elif corruption == "phase_sum":
        # Change every copy of the source receipt, preserving their agreement.
        for sample in [
            data["native_cold"],
            data["native_priming"],
            *data["native_samples"],
        ]:
            sample["semilocal_seed"]["source_solve_seconds"] += 1
    elif corruption == "force":
        data["native_samples"][1]["forces_hartree_per_bohr"][0][0][0] += 1e-3
    elif corruption == "seed_fallback":
        data["native_cold"]["convergence"][0]["warm_start_fallback"] = True
    elif corruption == "source":
        data["native_build"]["probe"]["source_identity"] = "invalid"
    payload = (json.dumps(data) + "\n").encode()
    stored = lzma.compress(payload)
    (tmp_path / name).write_bytes(stored)
    manifest["files"][name] = {
        "bytes": len(stored),
        "sha256": hashlib.sha256(stored).hexdigest(),
        "decoded_bytes": len(payload),
        "decoded_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if corruption == "stored_hash":
        manifest["files"][name]["sha256"] = "invalid"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, "-O", str(tmp_path / "verify.py"), "3"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if corruption == "valid":
        assert result.returncode == 0, result.stderr
        assert "gate passed." in result.stdout
    else:
        assert result.returncode != 0
        assert "ValueError" in result.stderr
        assert "gate passed." not in result.stdout
