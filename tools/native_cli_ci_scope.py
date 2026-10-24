"""Select native CLI CI depth without hiding monolithic link regressions.

Pull requests compare against their base SHA. Merge groups include the entire
unmerged queue relative to the actual target branch, not just the last synthetic
queue predecessor. An uncertain comparison requires the full release build.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
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


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, timeout=60
    ).stdout


def _merge_group_base(base: str, head: str, target_ref: str) -> str:
    """Resolve one current target/queue ancestor without trusting a stale ref."""
    if not target_ref.startswith("refs/heads/"):
        raise ValueError("merge-group target must be a full branch ref")
    _git("check-ref-format", target_ref)
    if _git("rev-parse", "--verify", "HEAD").decode().strip() != head:
        raise ValueError("merge-group head does not match the checkout")
    _git("merge-base", "--is-ancestor", base, head)
    # origin is established by the pinned checkout action. Fetch only this exact
    # branch into an isolated local ref; a failed fetch must not use old contents.
    fetched_ref = "refs/ci/native-cli-target"
    _git(
        "fetch",
        "--no-tags",
        "--no-recurse-submodules",
        "--no-write-fetch-head",
        "origin",
        f"+{target_ref}:{fetched_ref}",
    )
    target = _git("rev-parse", "--verify", f"{fetched_ref}^{{commit}}").decode().strip()
    if not _SHA.fullmatch(target):
        raise ValueError("unresolved merge-group target")
    ancestors = _git("merge-base", "--all", target, head).decode().splitlines()
    if len(ancestors) != 1 or not _SHA.fullmatch(ancestors[0]):
        raise ValueError("merge-group target ancestry is ambiguous")
    ancestor = ancestors[0]
    if ancestor == head:
        raise ValueError("merge-group head is already in the target branch")
    if ancestor != target:
        # A divergent target may have reverted/resolved away a queued change.
        # Diffing only from its merge base could then hide a reintroduced change.
        raise ValueError("merge-group target has diverged from the checked queue")
    print(
        f"Native CLI queue comparison: target={target} base={ancestor} head={head}",
        file=sys.stderr,
    )
    return ancestor


def select_tier(event: str, base: str, head: str, target_ref: str = "") -> str:
    """Use NUL-delimited Git paths; uncertainties require a full build."""
    if event not in {"pull_request", "merge_group"}:
        return "release"
    if not _SHA.fullmatch(base) or not _SHA.fullmatch(head):
        return "release"
    try:
        if event == "merge_group":
            base = _merge_group_base(base, head, target_ref)
        paths_output = _git(
            # Include both paths of renames, including native-source removals
            # whose unchanged contents were moved into a documentation path.
            "diff",
            "--no-renames",
            "--name-only",
            "-z",
            base,
            head,
            "--",
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        print("Native CLI comparison unavailable; requiring release", file=sys.stderr)
        return "release"
    paths = [os.fsdecode(path) for path in paths_output.split(b"\0") if path]
    return classify_paths(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--target-ref", default="")
    args = parser.parse_args()
    print(select_tier(args.event, args.base_sha, args.head_sha, args.target_ref))


if __name__ == "__main__":
    main()
