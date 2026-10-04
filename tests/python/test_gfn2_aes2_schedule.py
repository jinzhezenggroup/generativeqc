"""Bounded compiler policy and ordered CUDA AES2 schedule equivalence."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from generativeqc_compiler.method.gfn2_aes2_schedule import emit_gfn2_aes2_schedule

ROOT = Path(__file__).resolve().parents[2]


def test_aes2_atom_policy_limits(tmp_path: Path) -> None:
    """Check the actual emitted host selector, including overflow-sized extents."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    source = tmp_path / "policy.cpp"
    source.write_text(
        "#include <cstdint>\n#include <limits>\n"
        + emit_gfn2_aes2_schedule()
        + r"""
int main() {
  if (gfn2_aes2_peer_threads != 32) return 1;
  if (gfn2_aes2_atom_tiles(31,1) != 1) return 2;
  if (gfn2_aes2_atom_tiles(32,1) != 1) return 3;
  if (gfn2_aes2_atom_tiles(33,1) != 33) return 4;
  if (gfn2_aes2_atom_tiles(64,2) != 1) return 5;
  if (gfn2_aes2_atom_tiles(65,2) != 33) return 6;
  if (gfn2_aes2_atom_tiles(127,1) != 127) return 7;
  if (gfn2_aes2_atom_tiles(128,1) != 128) return 8;
  if (gfn2_aes2_atom_tiles(129,1) != 129) return 9;
  if (gfn2_aes2_atom_tiles(255,1) != 255) return 13;
  if (gfn2_aes2_atom_tiles(256,1) != 256) return 14;
  if (gfn2_aes2_atom_tiles(257,1) != 256) return 15;
  const auto maximum = std::numeric_limits<std::int64_t>::max();
  if (gfn2_aes2_atom_tiles(maximum,1) != 256) return 10;
  if (gfn2_aes2_atom_tiles(maximum,maximum) != 1) return 11;
  if (gfn2_aes2_atom_tiles(0,0) != 1) return 12;
}
"""
    )
    binary = tmp_path / "policy"
    subprocess.run([compiler, "-std=c++17", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)


def test_aes2_cuda_ordered_schedule(tmp_path: Path) -> None:
    """Compare peer tiles to the retained fused route through real public launchers."""
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
            str(ROOT / "tools/generate_gfn2_aes2_native.py"),
            "--cuda-output",
            str(tmp_path / "generated_gfn2_aes2_native.cuh"),
        ],
        check=True,
        timeout=60,
    )
    native = ROOT / "src/xtb/native"
    objects = []
    for i, source in enumerate(
        (
            ROOT / "tests/native/test_gfn2_aes2_schedule.cu",
            native / "src/backends/cuda/gfn2_aes2.cu",
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
    binary = tmp_path / "aes2"
    subprocess.run([nvcc, *objects, "-o", str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=120)
