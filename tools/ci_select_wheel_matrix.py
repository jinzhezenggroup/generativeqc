"""Select cheap PR or complete release wheel targets without weakening full-matrix gates."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

FAST_ARCHITECTURES = "120"
FULL_ARCHITECTURES = "80;90-real;120-real"

# Diff-based and fail-closed: source/packaging/architecture changes cannot
# silently receive a one-architecture qualification.
FULL_MATRIX_FILES = frozenset(
    {
        ".github/workflows/wheels.yml",
        "CMakeLists.txt",
        "cmake/GenerativeQCCudaImplib.cmake",
        "cmake/cuda_wheel_payload.ld",
        "pyproject.toml",
        "python/ci/prepare-wheel-build.sh",
        "python/ci/repair-wheel.sh",
        "tools/ci_cpu_ccache_snapshot.py",
        "tools/ci_select_wheel_matrix.py",
        "tools/generate_cuda_implib.py",
        "tools/generate_nvidia_host_api.py",
        "tools/link_cuda_implib.py",
    }
)
FULL_MATRIX_PREFIXES = (
    ".github/actions/restore-wheel-ccache/",
    "cmake/3rdparty/implib/",
    "src/runtime/nvidia_host_api/",
)


def requires_full_matrix(changed_paths: Iterable[str]) -> bool:
    """Flag PR edits to the wheel's target, linker, toolchain, or packaging path."""
    return any(
        path in FULL_MATRIX_FILES
        or any(path.startswith(prefix) for prefix in FULL_MATRIX_PREFIXES)
        for path in changed_paths
    )


def select_architectures(event_name: str, changed_paths: Iterable[str] = ()) -> str:
    """PRs are cheap unless their changed files require native cross-arch proof."""
    if event_name == "pull_request":
        return (
            FULL_ARCHITECTURES
            if requires_full_matrix(changed_paths)
            else FAST_ARCHITECTURES
        )
    if event_name in {"push", "schedule", "workflow_dispatch"}:
        return FULL_ARCHITECTURES
    raise ValueError(f"unsupported wheel event: {event_name}")


def pull_request_paths(base: str, head: str) -> list[str]:
    """Compare exact immutable PR commits; never downgrade on a missing diff."""
    if not all(re.fullmatch(r"[0-9a-fA-F]{40}", sha) for sha in (base, head)):
        raise ValueError("missing or invalid PR base/head SHA")
    changed = subprocess.check_output(
        ["git", "diff", "--no-ext-diff", "--name-only", "-z", base, head]
    )
    return [os.fsdecode(path) for path in changed.split(b"\0") if path]


def main() -> None:
    event_name = os.environ["GITHUB_EVENT_NAME"]
    changed_paths = (
        pull_request_paths(os.environ["GQC_PR_BASE_SHA"], os.environ["GQC_PR_HEAD_SHA"])
        if event_name == "pull_request"
        else []
    )
    architectures = select_architectures(event_name, changed_paths)
    mode = "full" if architectures == FULL_ARCHITECTURES else "fast"
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        output.write(f"architectures={architectures}\nmode={mode}\n")
    print(f"Wheel CUDA matrix: {mode} ({architectures})")


if __name__ == "__main__":
    main()
