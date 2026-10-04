"""Keep live TZVPD plots bound to complete, accurate and fully charged endpoints."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PUBLICATION = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/wb97mv-tzvpd-cold-20261004"
)


@pytest.mark.parametrize(
    "mutation", ["valid", "force", "lifecycle", "basis", "xc", "source"]
)
def test_live_tzvpd_publication_checks_all_calls_under_optimization(
    tmp_path: Path, mutation: str
) -> None:
    """Rebind storage hashes so scientific corruption reaches the inner gates."""
    shutil.copytree(PUBLICATION, tmp_path, dirs_exist_ok=True)
    samples_path = tmp_path / "samples.json.gz"
    samples = json.loads(gzip.decompress(samples_path.read_bytes()))
    reports = samples["points"]["6"]["reports"]
    if mutation == "force":
        reports["lda16"]["records"][-1]["forces"][0][0] += 1e-3
    elif mutation == "lifecycle":
        reports["lda16"]["preliminary_density"]["source_solve_seconds"] += 1
    elif mutation == "basis":
        reports["lda16"]["preliminary_density"]["target_basis_identity"][
            "basis_identity"
        ] = "wrong"
    elif mutation == "xc":
        reports["reference"]["records"][-1]["reference_xc_backend"]["backend"] = (
            "unknown"
        )
    elif mutation == "source":
        reports["none"]["native_build"]["probe"]["source_identity"] = "wrong"
    samples_path.write_bytes(gzip.compress(json.dumps(samples).encode(), mtime=0))
    evidence_path = tmp_path / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    for attachment in evidence["attachments"]:
        attachment["sha256"] = hashlib.sha256(
            (tmp_path / attachment["path"]).read_bytes()
        ).hexdigest()
    evidence_path.write_text(json.dumps(evidence))
    manifest_path = tmp_path / "publication.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        data = (tmp_path / entry["path"]).read_bytes()
        entry.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest_path.write_text(json.dumps(manifest))
    checked = subprocess.run(
        [sys.executable, "-O", str(tmp_path / "verify.py")],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if mutation == "valid":
        assert checked.returncode == 0, checked.stderr
        assert json.loads(checked.stdout)["accepted_endpoint_calls"] == 108
    else:
        assert checked.returncode != 0
        assert "accepted_endpoint_calls" not in checked.stdout
