"""Store bounded, exact-source fast CUDA compiler misses only within one PR ref."""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
from pathlib import Path

MAX_BYTES = 512 * 1024**2


def identity() -> tuple[str, Path]:
    """Never accept master, a merge group, or another PR's compiler objects."""
    number = os.environ.get("PR_NUMBER", "")
    if (
        os.environ.get("GITHUB_EVENT_NAME") != "pull_request"
        or re.fullmatch(r"[1-9][0-9]*", number) is None
        or os.environ.get("GITHUB_REF") != f"refs/pull/{number}/merge"
    ):
        raise ValueError("fast CUDA delta requires its exact pull_request merge ref")
    version = os.environ.get("CCACHE_VERSION", "")
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", version) is None:
        raise ValueError("invalid pinned ccache version")
    temporary = Path(os.environ["RUNNER_TEMP"])
    if not temporary.is_absolute() or any(c in str(temporary) for c in "\r\n"):
        raise ValueError("runner temporary directory must be absolute")
    root = temporary.resolve(strict=True)
    path = root / f"cuda-fast-pr-ccache-{number}"
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("PR CUDA cache must stay inside its runner directory")
    key = f"ccache-pr-fast-cuda-v1-linux-12.9-sm120-ccache{version}-pr-{number}"
    return key, path


def emit(**values: str) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
        stream.writelines(f"{name}={value}\n" for name, value in values.items())


def cache_files(path: Path) -> list[Path]:
    """Do not archive symlinks, hardlinks, devices or other special entries."""
    if not path.exists():
        return []
    if not path.is_dir() or path.is_symlink():
        raise ValueError("PR CUDA cache must be a real directory")
    files = []
    for root, directories, names in os.walk(path, followlinks=False):
        for name in directories + names:
            entry = Path(root) / name
            mode = entry.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode) or entry.stat().st_nlink != 1:
                raise ValueError("PR CUDA cache must contain only ordinary files")
            files.append(entry)
    return files


def bound(path: Path) -> None:
    """Trim only the secondary miss store, never the restored master cache."""
    if not cache_files(path):
        return
    subprocess.run(
        ["ccache", "--trim-dir", str(path), "--trim-max-size", "512MiB"],
        check=True,
        timeout=45,
    )
    files = cache_files(path)
    size = sum(entry.stat().st_size for entry in files)
    if size > MAX_BYTES:
        raise ValueError("PR fast CUDA cache exceeds its 512 MiB upload cap")
    if not any(entry.name != "CACHEDIR.TAG" for entry in files):
        return
    print(f"PR fast CUDA compiler misses: {size} bytes (maximum {MAX_BYTES})")
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
