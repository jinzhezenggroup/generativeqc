"""Execute chronological ring bookkeeping and independent device Gram gates."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("cuda", [False, True], ids=["metadata", "cuda-gram"])
def test_history_wrap_retirement_and_restart(cuda: bool, tmp_path: Path) -> None:
    if cuda and os.environ.get("GENERATIVEQC_CC_DIIS_RING_CUDA_TEST") != "1":
        pytest.skip("requires a finite Slurm GPU allocation")
    if cuda:
        assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    cache = shutil.which("ccache")
    compiler = shutil.which("nvcc" if cuda else "c++")
    if not cache or not compiler:
        pytest.skip("ccache and the selected compiler are required")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    source = (
        ROOT
        / "tests/native"
        / ("test_cuda_cc_diis_ring.cu" if cuda else "test_cc_diis_ring.cpp")
    )
    obj, executable = tmp_path / "probe.o", tmp_path / "probe"
    flags = ["-std=c++20", "-O2", "-I" + str(ROOT / "src")]
    if cuda:
        flags += ["-arch=" + os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_120")]
    subprocess.run(
        [cache, compiler, *flags, "-c", str(source), "-o", str(obj)],
        check=True,
        capture_output=True,
        timeout=180,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    subprocess.run(
        [compiler, str(obj), "-o", str(executable), *(["-lcublas"] if cuda else [])],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=120)


def test_cc_diis_gram_submits_only_semantic_tensor_gemm() -> None:
    source = (ROOT / "src/cc/cuda_state.cuh").read_text(encoding="utf-8")
    assert "cublasDgemm" not in source
    assert (
        "gemm(context, 'N', 'T', history, history, elements, errors, errors, gram,"
        in source
    )
