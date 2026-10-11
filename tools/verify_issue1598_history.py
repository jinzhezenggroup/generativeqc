"""Recover and audit the exact reviewed historical capsule; never qualify a build.

Payloads remain in existing Git history, outside the bounded results tree.
Only an explicit --fetch permits fetching the pinned public commit when absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "benchmarks/results/issue1598-incremental-direct-jk-20261011"
COMMIT = "0d712298bd59f1842911ae481e3790b9a417df21"
TREE = "4db07a60ec635f7914d39936b55c33b20b54037a"
MANIFEST_SHA256 = "b60e01f6e79dc20d76cd21302798d0d40a23ed4c4e68a27f3f2811623e7b9cab"
REMOTE = "https://github.com/jinzhezenggroup/generativeqc.git"


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], timeout=120)


def restore(destination: Path, *, fetch: bool = False) -> None:
    """Restore Git-original bytes with fixed tree and SHA256 integrity gates."""
    if fetch:
        available = subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e", COMMIT],
            capture_output=True,
            check=False,
            timeout=30,
        )
        if available.returncode:
            git("fetch", "--no-tags", "--depth=1", REMOTE, COMMIT)
    tree = git("rev-parse", f"{COMMIT}:{PREFIX}").decode().strip()
    if tree != TREE:
        raise ValueError("historical tree identity mismatch")
    raw = git("show", f"{COMMIT}:{PREFIX}/capsule.json")
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("historical manifest checksum mismatch")
    manifest = json.loads(raw)
    members = {entry["path"]: entry for entry in manifest["files"]}
    listing = git("ls-tree", "-r", "--name-only", TREE).decode().splitlines()
    if set(listing) != {*members, "capsule.json"}:
        raise ValueError("historical inventory mismatch")
    for relative in listing:
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative:
            raise ValueError("unsafe historical member")
        payload = git("show", f"{TREE}:{relative}")
        if relative in members:
            entry = members[relative]
            if (
                len(payload) != entry["bytes"]
                or hashlib.sha256(payload).hexdigest() != entry["sha256"]
            ):
                raise ValueError(f"historical member checksum mismatch: {relative}")
        target = destination / relative
        if target.exists():
            raise ValueError(f"refusing to overwrite recovered member: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    pointer = json.loads((ROOT / PREFIX / "capsule.json").read_bytes())
    if (
        pointer.get("schema") != "generativeqc.historical-benchmark-history-pointer.v1"
        or pointer.get("commit") != COMMIT
        or pointer.get("tree") != TREE
        or pointer.get("manifest_sha256") != MANIFEST_SHA256
        or pointer.get("path") != PREFIX
        or pointer.get("decision") != "negative-evidence; opt-in retained"
    ):
        raise ValueError("historical pointer identity mismatch")
    with tempfile.TemporaryDirectory(prefix="issue1598-history-") as folder:
        destination = Path(folder)
        restore(destination, fetch=args.fetch)
        result = subprocess.check_output(
            [sys.executable, str(destination / "verify.py")], timeout=60
        )
        summary = json.loads(result)
        # The historical verifier's 'cold' field times prepared-plan execution.
        # Preserve its audited numerical checks, without certifying setup cost.
        summary["cold_timing_scope"] = "prepared-plan-execute-only; setup excluded"
        summary["historical_commit"] = COMMIT
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
