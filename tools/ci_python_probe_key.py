"""Fingerprint committed probe inputs, independent of generated checkout files."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

# This is a transport snapshot identity. ccache still verifies every actual
# compiler/input/flag dependency; this must never override its artifact keys.
INPUTS = (
    ".github/workflows/ci.yml",
    ".github/scripts",
    ".github/actions/restore-python-probe-ccache",
    "cmake",
    "python/generativeqc_compiler",
    "python/generativeqc",
    "CMakeLists.txt",
    "include",
    "src",
    "tools",
    "pyproject.toml",
    "uv.lock",
    "manifests/xtbloom-d3.json",
    "data/parameters/d3_production.bin",
    "tests/python",
    "tests/native",
    "tests/test_ci_python_probe_key.py",
    "tests/test_ci_probe_snapshot.py",
    "benchmarks",
)


def fingerprint(root: Path) -> str:
    """Hash sorted committed path/mode/blob identities, not commit metadata."""
    result = subprocess.run(
        ["git", "ls-tree", "-r", "-z", "HEAD", "--", *INPUTS],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=15,
    )
    records = result.stdout.split(b"\0")
    if not records[-1] == b"" or not records[0]:
        raise ValueError("probe fingerprint requires a nonempty committed inventory")
    digest = hashlib.sha256(b"generativeqc-python-probe-inputs-v1\0")
    for record in sorted(records[:-1]):
        digest.update(record + b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    value = fingerprint(Path(__file__).resolve().parents[1])
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as handle:
        handle.write(f"fingerprint={value}\n")
    print(f"Committed Python probe inputs: {value}")


if __name__ == "__main__":
    main()
