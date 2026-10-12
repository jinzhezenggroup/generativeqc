"""Select native CLI CI depth without hiding monolithic link regressions.

GitHub pull requests and merge groups compare the checked-out SHA against the
trusted base SHA. A missing or unreadable comparison fails closed to the full
release build; doc-only changes do not spend an expensive CUDA build.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_FULL_FILES = frozenset(
    {
        "CMakeLists.txt",
        "CMakePresets.json",
        ".github/workflows/ci.yml",
        "tools/package_native_cli.py",
        "tools/native_cli_ci_scope.py",
    }
)
_FULL_PREFIXES = (
    "cmake/",
    "src/cli/",
    ".github/actions/install-native-cli-deps/",
    "tests/python/test_native_cli_",
)
_FAST_PREFIXES = (
    "src/",
    "include/",
    "python/generativeqc/",
    "python/generativeqc_compiler/",
    "tools/",
    "tests/native/",
    "manifests/",
    "data/",
    "upstream/",
)
_FAST_FILES = frozenset({"pyproject.toml", "uv.lock"})
_SHA = re.compile(r"[0-9a-f]{40}\Z")


def classify_paths(paths: Iterable[str]) -> str:
    """Return release, fast or skip; escalate a mix to its strongest tier."""
    tier = "skip"
    for path in paths:
        if path in _FULL_FILES or path.startswith(_FULL_PREFIXES):
            return "release"
        if path in _FAST_FILES or path.startswith(_FAST_PREFIXES):
            tier = "fast"
    return tier


def select_tier(event: str, base: str, head: str) -> str:
    """Use NUL-delimited Git paths; uncertainties require a full build."""
    if event not in {"pull_request", "merge_group"}:
        return "release"
    if not _SHA.fullmatch(base) or not _SHA.fullmatch(head):
        return "release"
    try:
        result = subprocess.run(
            # Include both paths of renames, including native-source removals
            # whose unchanged contents were moved into a documentation path.
            ["git", "diff", "--no-renames", "--name-only", "-z", base, head, "--"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "release"
    paths = [os.fsdecode(path) for path in result.stdout.split(b"\0") if path]
    return classify_paths(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--head-sha", required=True)
    args = parser.parse_args()
    print(select_tier(args.event, args.base_sha, args.head_sha))


if __name__ == "__main__":
    main()
