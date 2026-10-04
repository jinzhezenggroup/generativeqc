"""Bounded compiler policy and independent H0/Pulay force schedule gates."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from generativeqc_compiler.method.gfn2_h0_force_schedule import (
    emit_gfn2_h0_force_schedule,
)

ROOT = Path(__file__).resolve().parents[2]


def test_h0_force_tile_policy_limits(tmp_path: Path) -> None:
    """Exercise actual emitted arithmetic at tile tails, cap and int64 limits."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    source = tmp_path / "policy.cpp"
    source.write_text(
        "#include <cstdint>\n#include <limits>\n"
        + emit_gfn2_h0_force_schedule()
        + r"""
int main() {
  if (gfn2_h0_force_threads != 128) return 1;
  if (gfn2_h0_force_pair_tiles(127,1) != 1) return 2;
  if (gfn2_h0_force_pair_tiles(128,1) != 1) return 3;
  if (gfn2_h0_force_pair_tiles(129,1) != 2) return 4;
  if (gfn2_h0_force_pair_tiles(256,2) != 1) return 5;
  if (gfn2_h0_force_pair_tiles(257,2) != 2) return 6;
  if (gfn2_h0_force_pair_tiles(32640,1) != 255) return 7;
  if (gfn2_h0_force_pair_tiles(32641,1) != 256) return 8;
  if (gfn2_h0_force_pair_tiles(32768,1) != 256) return 9;
  if (gfn2_h0_force_pair_tiles(32769,1) != 256) return 10;
  const auto maximum = std::numeric_limits<std::int64_t>::max();
  if (gfn2_h0_force_pair_tiles(maximum,1) != 256) return 11;
  if (gfn2_h0_force_pair_tiles(maximum,maximum) != 1) return 12;
  if (gfn2_h0_force_pair_tiles(0,0) != 1) return 13;
}
"""
    )
    binary = tmp_path / "policy"
    subprocess.run([compiler, "-std=c++17", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)


def test_h0_force_cuda_tiled_schedule(tmp_path: Path) -> None:
    """Qualify real tiled kernels with an analytic oracle and the retained route."""
    if os.environ.get("GENERATIVEQC_TEST_GFN2_CUDA") != "1":
        pytest.skip("explicit GFN2 CUDA qualification is disabled")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("GFN2 CUDA qualification requires a Slurm allocation")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("CUDA compiler required for the isolated native harness")
    ccache = shutil.which("ccache")
    if ccache is None:
        pytest.fail("ccache is required for CUDA compilation")
    subprocess.run([ccache, "--version"], check=True, capture_output=True)
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
    native = ROOT / "src/xtb/native"
    objects = []
    for i, source in enumerate(
        (
            ROOT / "tests/native/test_gfn2_h0_force_schedule.cu",
            native / "src/backends/cuda/gfn2_h0_force.cu",
        )
    ):
        output = tmp_path / f"part{i}.o"
        subprocess.run(
            [
                ccache,
                nvcc,
                "-std=c++20",
                "-O3",
                "-I",
                str(tmp_path),
                "-I",
                str(native),
                "-I",
                str(native / "src"),
                "-c",
                str(source),
                "-o",
                str(output),
            ],
            check=True,
            timeout=180,
        )
        objects.append(str(output))
    binary = tmp_path / "h0_force"
    subprocess.run([nvcc, *objects, "-o", str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=180)
