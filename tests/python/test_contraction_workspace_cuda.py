"""Owned cuBLAS storage is bounded, journaled and optional on a real device."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_owned_workspace_cuda_lifetime(tmp_path: Path) -> None:
    """Require a scheduler-owned GPU; injected failures test policy, not speed."""
    if os.environ.get("GENERATIVEQC_DF_CC_CUDA_TEST") != "1":
        pytest.skip("requires explicitly allocated GPU validation window")
    compiler, cache = shutil.which("nvcc"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires nvcc and ccache")
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "workspace"
    build = subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-arch=sm_120",
            "-I" + str(root / "src"),
            str(root / "tests/native/test_contraction_workspace_cuda.cu"),
            "-lcublas",
            "-Xlinker=--wrap=cudaMalloc",
            "-Xlinker=--wrap=cudaMemGetInfo",
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "CCACHE_BASEDIR": str(root)},
    )
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
