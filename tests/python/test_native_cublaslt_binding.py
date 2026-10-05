"""Native cached cuBLASLt execution, opt-in under a finite GPU allocation."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_prepared_cublaslt_execution(tmp_path: Path) -> None:
    """Qualify all layouts and repeated replay against independent affine sums."""
    if os.environ.get("GENERATIVEQC_CUBLASLT_CUDA_TEST") != "1":
        pytest.skip("requires explicit finite Slurm real-device qualification")
    compiler, cache = shutil.which("nvcc"), shutil.which("ccache")
    assert compiler and cache, "cuBLASLt qualification requires nvcc and ccache"
    root = Path(__file__).resolve().parents[2]
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    obj, executable = tmp_path / "cublaslt.o", tmp_path / "cublaslt"
    for command in (
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-arch=sm_120",
            "-DGENERATIVEQC_HAS_CUBLASLT=1",
            "-DGENERATIVEQC_TEST_HOOKS=1",
            "-I" + str(root / "src"),
            "-c",
            str(root / "tests/native/test_native_cublaslt.cu"),
            "-o",
            str(obj),
        ],
        [
            compiler,
            str(obj),
            "-arch=sm_120",
            "-lcublasLt",
            "-lcublas",
            "-o",
            str(executable),
        ],
        [str(executable)],
    ):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            env={**os.environ, "CCACHE_BASEDIR": str(root)},
        )
        assert result.returncode == 0, result.stdout + result.stderr
