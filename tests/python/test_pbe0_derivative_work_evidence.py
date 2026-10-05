"""Retained angular timings must stay bound to the corresponding kernel events."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "benchmarks/results/pbe0-derivative-work-20261005"


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize(
    "mutation", ["valid", "reordered", "labels", "duplicate", "duration"]
)
def test_angular_event_identity_and_duration_binding(
    tmp_path: Path, mutation: str, optimized: bool
) -> None:
    """Rebind storage hashes so corrupted event pairs reach the semantic gate."""
    shutil.copytree(EVIDENCE, tmp_path, dirs_exist_ok=True)
    name = "angular/kernel-events-96.json"
    path = tmp_path / name
    events = json.loads(path.read_text())
    if mutation == "reordered":
        # Row order is not part of the kernel/event identity.
        events.reverse()
    elif mutation == "labels":
        # Preserve both separate multisets, but exchange orders 0 and 4.
        events[0]["name"], events[4]["name"] = events[4]["name"], events[0]["name"]
    elif mutation == "duplicate":
        # A name-keyed dictionary must not silently collapse an extra launch.
        events.append(events[0].copy())
    elif mutation == "duration":
        # Nsight stores integer nanoseconds, so even one ns must remain exact.
        events[0]["end_ns"] += 1
    payload = (json.dumps(events, indent=2) + "\n").encode()
    path.write_bytes(payload)
    manifest_path = tmp_path / "publication.json"
    manifest = json.loads(manifest_path.read_text())
    for member in manifest["files"]:
        if member["path"] == name:
            member["bytes"] = len(payload)
            member["sha256"] = hashlib.sha256(payload).hexdigest()
            break
    else:
        pytest.fail("event projection missing from publication")
    manifest_path.write_text(json.dumps(manifest) + "\n")
    # The copied script starts with tmp_path on sys.path. Bind the repository
    # helpers explicitly, including when CI has no root entry in PYTHONPATH.
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT), environment.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [sys.executable, *(["-O"] if optimized else []), str(tmp_path / "verify.py")],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=environment,
    )
    if mutation in ("valid", "reordered"):
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == json.loads(
            (EVIDENCE / "summary.json").read_text()
        )
    else:
        assert result.returncode != 0
        assert "ValueError" in result.stderr
        assert "CSV kernel" in result.stderr
