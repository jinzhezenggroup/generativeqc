"""Bound one optional, immutable compiler-miss delta to one pull-request ref."""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
from pathlib import Path

MAX_BYTES = 256 * 1024**2


def identity() -> tuple[str, Path]:
    """Accept only a numeric PR and its actual Actions merge ref, never master."""
    number = os.environ.get("PR_NUMBER", "")
    if (
        os.environ.get("GITHUB_EVENT_NAME") != "pull_request"
        or re.fullmatch(r"[1-9][0-9]*", number) is None
        or os.environ.get("GITHUB_REF") != f"refs/pull/{number}/merge"
    ):
        raise ValueError("PR compiler delta requires a matching pull_request merge ref")
    version = os.environ.get("CCACHE_VERSION", "")
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", version) is None:
        raise ValueError("invalid pinned ccache version")
    temporary = Path(os.environ["RUNNER_TEMP"])
    if not temporary.is_absolute() or any(c in str(temporary) for c in "\r\n"):
        raise ValueError("runner temporary directory must be an absolute path")
    root = temporary.resolve(strict=True)
    path = root / f"cuda-cli-pr-ccache-{number}"
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("PR compiler delta must not redirect outside its directory")
    # No source/run/attempt identity: GitHub stores at most one entry per PR for
    # this schema/toolchain. Strict ccache still validates all compiler inputs.
    key = f"ccache-pr-cli-delta-v1-linux-cuda12.9-sm120-ccache{version}-pr-{number}"
    return key, path


def emit(**values: str) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
        stream.writelines(f"{key}={value}\n" for key, value in values.items())


def cache_files(path: Path) -> list[Path]:
    """Reject links and special files before trim can traverse secondary storage."""
    if not path.exists():
        return []
    if not path.is_dir() or path.is_symlink():
        raise ValueError("PR compiler delta must be a real directory")
    files = []
    for root, directories, names in os.walk(path, followlinks=False):
        for name in directories + names:
            entry = Path(root) / name
            mode = entry.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode) or entry.stat().st_nlink != 1:
                raise ValueError("PR compiler delta must contain only ordinary files")
            files.append(entry)
    return files


def bound(path: Path) -> None:
    """Trim only secondary storage; never clean the restored local cache."""
    if not cache_files(path):
        return
    subprocess.run(
        ["ccache", "--trim-dir", str(path), "--trim-max-size", "256MiB"],
        check=True,
        timeout=45,
    )
    files = cache_files(path)
    size = sum(entry.stat().st_size for entry in files)
    if size > MAX_BYTES:
        raise ValueError("PR compiler delta exceeds its 256 MiB upload bound")
    # A tag alone is not reusable compilation work.
    if not any(entry.name != "CACHEDIR.TAG" for entry in files):
        return
    print(f"PR CLI compiler delta: {size} bytes (maximum {MAX_BYTES})")
    emit(available="true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "bound"))
    args = parser.parse_args()
    key, path = identity()
    if args.operation == "prepare":
        emit(key=key, path=str(path))
    else:
        bound(path)


if __name__ == "__main__":
    main()
