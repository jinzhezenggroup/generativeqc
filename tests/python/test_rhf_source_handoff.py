"""Real CUDA detached-source lifetime, independent ERIs and admission boundaries."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_detached_rhf_source(tmp_path: Path) -> None:
    if os.environ.get("GENERATIVEQC_RHF_SOURCE_HANDOFF_TEST") != "1":
        pytest.skip("requires explicitly allocated CUDA library")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU checks require Slurm"
    cache, compiler = shutil.which("ccache"), shutil.which("c++")
    assert cache and compiler
    cuda = Path(os.environ.get("CUDA_HOME", "/group/software/cuda-12.9.1"))
    library = Path(os.environ["GENERATIVEQC_LIBRARY"]).resolve()
    obj, exe = tmp_path / "probe.o", tmp_path / "probe"
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-DGENERATIVEQC_HAS_CUDA=1",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-I" + str(cuda / "include"),
            "-c",
            str(ROOT / "tests/native/test_rhf_source_handoff.cpp"),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
    )
    subprocess.run(
        [
            compiler,
            str(obj),
            str(library),
            "-L" + str(cuda / "lib64"),
            "-lcudart",
            "-Wl,-rpath," + str(library.parent),
            "-Wl,-rpath," + str(cuda / "lib64"),
            "-o",
            str(exe),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(exe)], check=True, capture_output=True, timeout=180)
