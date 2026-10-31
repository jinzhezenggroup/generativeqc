"""Receipt replay refuses disabled assertions and unauthenticated archives."""

import hashlib
import json
import lzma
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path("benchmarks/results/portable-df-response-20261031/report.py")
ARCHIVE = REPORT.with_name("receipts.json.xz")
MANIFEST = Path("benchmarks/evidence-reviews/portable-df-response-20261031.json")


def run_report(
    root: Path, *flags: str, optimize: str = ""
) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONOPTIMIZE=optimize, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, *flags, str(root / REPORT), "--verify"],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.fixture
def report_tree(tmp_path: Path) -> Path:
    target = tmp_path / REPORT
    target.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / REPORT, target)
    return tmp_path


@pytest.mark.parametrize(("flags", "optimize"), [("-O", ""), ("-OO", ""), ("", "1")])
def test_optimized_replay_refuses_before_loading_receipts(
    report_tree: Path, flags: str, optimize: str
) -> None:
    result = run_report(report_tree, *([flags] if flags else []), optimize=optimize)
    assert result.returncode != 0
    assert "requires assertions" in result.stderr
    assert "FileNotFoundError" not in result.stderr


@pytest.mark.parametrize("damage", [None, "size", "sha256", "missing", "duplicate"])
def test_archive_requires_unique_independent_manifest_binding(
    report_tree: Path, damage: str | None
) -> None:
    dependency = lzma.compress(b"{}")
    (report_tree / "dependency.json.xz").write_bytes(dependency)
    digest = hashlib.sha256(dependency).hexdigest()
    bundle = {
        "oracle_path": "dependency.json.xz",
        "oracle_sha256": digest,
        "reuse_path": "dependency.json.xz",
        "reuse_sha256": digest,
        "records": {},
        "comparisons": [],
    }
    archive = lzma.compress(json.dumps(bundle).encode())
    entry = {
        "path": ARCHIVE.as_posix(),
        "bytes": len(archive),
        "sha256": hashlib.sha256(archive).hexdigest(),
    }
    # Archive remains well-formed, with internally consistent embedded hashes.
    # Only its separately retained size/hash binding rejects these replacements.
    if damage == "size":
        archive = lzma.compress(json.dumps(dict(bundle, changed=True)).encode())
    elif damage == "sha256":
        archive = bytearray(archive)
        archive[-1] ^= 1
        archive = bytes(archive)
    (report_tree / ARCHIVE).write_bytes(archive)
    manifest = report_tree / MANIFEST
    manifest.parent.mkdir(parents=True)
    entries = (
        [] if damage == "missing" else [entry] * (2 if damage == "duplicate" else 1)
    )
    manifest.write_text(json.dumps({"files": entries}))
    (report_tree / REPORT.with_name("provenance.json.xz")).write_bytes(dependency)
    result = run_report(report_tree)
    if damage is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        expected = damage if damage in ("size", "sha256") else "manifest entry"
        assert f"receipts.json.xz {expected}" in result.stderr
