"""Rerun the measured API experiment from a verified equivalent source checkout.

The original local Git commits are retained as measurement provenance, but are
not required as public Git objects. Current HEAD is admitted only when its full
production-source map has the exact recorded aggregate SHA-256.
"""

from __future__ import annotations

import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--build-source-revision",
        help="Verified source revision for the supplied rebuilt library; defaults to current HEAD after source-hash verification",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Without this option only verify provenance and print the command",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    original = runpy.run_path(str(HERE / "run.py"), run_name="frozen_api_runner")
    expected = json.loads((HERE / "provenance.json").read_text())
    source = args.source_root.resolve(strict=True)
    library = args.library.resolve(strict=True)
    actual = original["source_snapshot"](source)
    if actual["source_hash"] != expected["source"]["source_hash"]:
        raise ValueError(
            "production source does not match the measured API implementation"
        )
    revision = args.build_source_revision or actual["revision"]
    git = original["git"]
    revision = git(source, "rev-parse", revision).decode().strip()
    paths = ("src", "include", "python", "cmake", "CMakeLists.txt")
    if git(source, "diff", revision, "--", *paths) or actual["untracked_source_files"]:
        raise ValueError(
            "supplied library build revision differs from verified production source"
        )
    current_library_hash = original["sha"](library)
    command = [
        sys.executable,
        str(HERE / "run.py"),
        "--run",
        "--source-root",
        str(source),
        "--library",
        str(library),
        "--output",
        str(args.output),
        "--build-source-revision",
        revision,
        "--expected-source-revision",
        actual["revision"],
        "--expected-library-sha256",
        current_library_hash,
        "--timeout",
        str(args.timeout),
    ]
    print(
        json.dumps(
            {
                "verified_production_source_hash": actual["source_hash"],
                "reproduction_source_revision": actual["revision"],
                "reproduction_library_build_revision": revision,
                "reproduction_library_sha256": current_library_hash,
                "historical_library_sha256": expected["library_sha256"],
                "historical_revision_required_as_git_object": False,
                "command": command,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.run:
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
