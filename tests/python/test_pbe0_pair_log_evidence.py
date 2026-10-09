"""Displaced-reference coverage cannot be replaced by duplicate initial rows."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "benchmarks/results/pbe0-order5-pair-logs-20261005"


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    "mutation", ["valid", "reordered", "missing_moved", "duplicate_repeat"]
)
def test_complete_geometry_inventory(
    tmp_path: Path, mutation: str, optimized: bool
) -> None:
    """Re-sign modified storage so the regression reaches semantic validation."""
    shutil.copytree(EVIDENCE, tmp_path, dirs_exist_ok=True)
    name = "reference-96.json.gz"
    path = tmp_path / name
    record = json.loads(gzip.decompress(path.read_bytes()))
    if mutation == "reordered":
        record["records"].reverse()
    elif mutation == "missing_moved":
        # Previously, 12 initial references still produced 288 total pairings
        # because none were available to check the displaced native rows.
        initial = record["records"][:6]
        record["records"] = initial + copy.deepcopy(initial)
    elif mutation == "duplicate_repeat":
        record["records"][-1] = copy.deepcopy(record["records"][-2])
    payload = gzip.compress((json.dumps(record) + "\n").encode(), mtime=0)
    path.write_bytes(payload)
    # Update both storage envelopes: the changed samples must get past hash
    # binding before the semantic inventory check can reject the mutation.
    manifest_path = tmp_path / "publication.json"
    manifest = json.loads(manifest_path.read_text())
    member = next(
        member for member in manifest["files"] if member["role"] == "evidence"
    )
    validation_path = tmp_path / member["path"]
    raw_validation = validation_path.read_bytes()
    validation = json.loads(
        gzip.decompress(raw_validation)
        if validation_path.suffix == ".gz"
        else raw_validation
    )
    for attachment in validation["attachments"]:
        if attachment["path"] == name:
            attachment["sha256"] = hashlib.sha256(payload).hexdigest()
    validation_payload = (json.dumps(validation) + "\n").encode()
    validation_path.write_bytes(
        gzip.compress(validation_payload, mtime=0)
        if validation_path.suffix == ".gz"
        else validation_payload
    )
    for member in manifest["files"]:
        content = (tmp_path / member["path"]).read_bytes()
        member.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    manifest_path.write_text(json.dumps(manifest) + "\n")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT), str(ROOT / "python"), environment.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), str(tmp_path / "verify.py")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if mutation in ("valid", "reordered"):
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["pairings"] == 288
    else:
        assert result.returncode != 0
        assert "geometry/phase/repeat inventory" in result.stderr
