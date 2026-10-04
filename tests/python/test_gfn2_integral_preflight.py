"""Compiler admission widths and independent native validation/publication gates."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from generativeqc_compiler.integral.gfn2_force_schedule import (
    emit_gfn2_force_preflight_schedule,
)

ROOT = Path(__file__).resolve().parents[2]


def test_integral_preflight_width_policy(tmp_path: Path) -> None:
    """Exercise emitted ceiling arithmetic without overflow at extreme extents."""
    compiler, ccache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    assert ccache is not None, "ccache required for native builds"
    subprocess.run([ccache, "--version"], check=True, capture_output=True)
    source = tmp_path / "policy.cpp"
    source.write_text(
        "#include <cstdint>\n#include <limits>\n"
        + emit_gfn2_force_preflight_schedule()
        + r"""
int main() {
  if (gfn2_force_preflight_threads(4095,1) != 64) return 1;
  if (gfn2_force_preflight_threads(4096,1) != 64) return 2;
  if (gfn2_force_preflight_threads(4097,1) != 256) return 3;
  if (gfn2_force_preflight_threads(8192,2) != 64) return 4;
  if (gfn2_force_preflight_threads(8193,2) != 256) return 5;
  const auto maximum = std::numeric_limits<std::int64_t>::max();
  if (gfn2_force_preflight_threads(maximum,1) != 256) return 6;
  if (gfn2_force_preflight_threads(maximum,maximum) != 64) return 7;
  if (gfn2_force_preflight_threads(0,0) != 64) return 8;
}
"""
    )
    obj, binary = tmp_path / "policy.o", tmp_path / "policy"
    subprocess.run(
        [ccache, compiler, "-std=c++17", "-c", str(source), "-o", str(obj)], check=True
    )
    subprocess.run([compiler, str(obj), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)


def test_integral_preflight_cuda(tmp_path: Path) -> None:
    """Compare selected and retained admission widths, including faulted Graphs."""
    if os.environ.get("GENERATIVEQC_TEST_GFN2_CUDA") != "1":
        pytest.skip("explicit GFN2 CUDA qualification is disabled")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("CUDA qualification requires Slurm")
    nvcc, ccache = shutil.which("nvcc"), shutil.which("ccache")
    if nvcc is None:
        pytest.skip("CUDA toolkit required")
    if ccache is None:
        pytest.fail("ccache required for CUDA builds")
    subprocess.run([ccache, "--version"], check=True, capture_output=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_gfn2_sdq_native.py"),
            "--cuda-output",
            str(tmp_path / "generated_gfn2_sdq_cuda.cuh"),
        ],
        check=True,
        timeout=60,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_gfn2_h0_native.py"),
            "--output",
            str(tmp_path / "generated_gfn2_h0_native.hpp"),
        ],
        check=True,
        timeout=60,
    )
    obj, binary = tmp_path / "preflight.o", tmp_path / "preflight"
    subprocess.run(
        [
            ccache,
            nvcc,
            "-std=c++20",
            "-O3",
            "-I",
            str(tmp_path),
            "-I",
            str(ROOT / "src/xtb/native/src"),
            "-c",
            str(ROOT / "tests/native/test_gfn2_integral_preflight.cu"),
            "-o",
            str(obj),
        ],
        check=True,
        timeout=180,
    )
    subprocess.run([nvcc, str(obj), "-o", str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=180)
