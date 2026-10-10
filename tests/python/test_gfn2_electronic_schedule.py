"""Compiler launch policy and actual tiled CUDA publication contracts."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from generativeqc_compiler.method.gfn2_electronic_schedule import (
    emit_gfn2_electronic_schedule,
)

ROOT = Path(__file__).resolve().parents[2]


def test_electronic_tile_policy_limits(tmp_path: Path) -> None:
    """Compile policy ceilings at boundaries, including overflow-sized inputs."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    source = tmp_path / "policy.cpp"
    source.write_text(
        "#include <cstdint>\n#include <limits>\n"
        + emit_gfn2_electronic_schedule()
        + r"""
int main() {
  if (gfn2_occupation_solve_count(1,4,4)!=1) return 11;
  if (gfn2_occupation_solve_count(1,4,3)!=2) return 12;
  if (gfn2_occupation_solve_count(2,4,4)!=2) return 13;
  if (gfn2_occupation_solve_count(1,0,0)!=1) return 14;
  if (gfn2_electronic_matrix_tiles(1,1)!=1) return 1;
  if (gfn2_electronic_matrix_tiles(256,1)!=1) return 2;
  if (gfn2_electronic_matrix_tiles(257,1)!=2) return 3;
  if (gfn2_electronic_matrix_tiles(512,2)!=1) return 4;
  if (gfn2_electronic_matrix_tiles(513,2)!=2) return 5;
  if (gfn2_electronic_matrix_tiles(32768,1)!=128) return 6;
  if (gfn2_electronic_matrix_tiles(32769,1)!=128) return 7;
  const auto maximum=std::numeric_limits<std::int64_t>::max();
  if (gfn2_electronic_matrix_tiles(maximum,1)!=128) return 8;
  if (gfn2_electronic_matrix_tiles(maximum,maximum)!=1) return 9;
  if (gfn2_electronic_matrix_tiles(0,0)!=1) return 10;
}
"""
    )
    binary = tmp_path / "policy"
    subprocess.run([compiler, "-std=c++17", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)


@pytest.mark.parametrize("family", ["matrices", "matrices_receipt", "occupations"])
def test_tiled_electronic_cuda_publication(tmp_path: Path, family: str) -> None:
    """Real kernels gate ragged matrix tiles and exact spin-task sharing."""
    if os.environ.get("GENERATIVEQC_TEST_GFN2_CUDA") != "1":
        pytest.skip("explicit GFN2 CUDA qualification is disabled")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("GFN2 CUDA qualification requires a Slurm allocation")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("CUDA compiler required for the isolated native harness")
    launcher = shutil.which("sccache") or shutil.which("ccache")
    if launcher is None:
        pytest.fail("sccache or ccache is required for CUDA compilation")
    subprocess.run([launcher, "--version"], check=True, capture_output=True)
    architecture = os.environ.get("GENERATIVEQC_TEST_CUDA_ARCH", "")
    if architecture and not re.fullmatch(r"sm_[0-9]+", architecture):
        pytest.fail("GENERATIVEQC_TEST_CUDA_ARCH must name one sm_NN target")
    arch_flags = [f"-arch={architecture}"] if architecture else []
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_gfn2_electronic_cuda.py"),
            "--output",
            str(tmp_path / "generated_gfn2_electronic_native.cuh"),
        ],
        check=True,
        timeout=60,
    )
    density_command = [
        sys.executable,
        str(ROOT / "tools/generate_gfn2_density_cuda.py"),
        "--output",
        str(tmp_path / "generated_gfn2_density_contract.inc"),
    ]
    if family == "matrices_receipt":
        density_command.extend(
            [
                "--instrumented-output",
                str(tmp_path / "generated_gfn2_density_contract_receipt.inc"),
            ]
        )
    subprocess.run(density_command, check=True, timeout=60)
    objects = []
    native = ROOT / "src/xtb/native"
    sources = (
        (
            ROOT / "tests/native/test_gfn2_electronic_schedule.cu",
            native / "src/backends/cuda/gfn2_hamiltonian.cu",
            native / "src/backends/cuda/gfn2_density.cu",
        )
        if family.startswith("matrices")
        else (
            ROOT / "tests/native/test_gfn2_occupation_sharing.cu",
            native / "src/backends/cuda/gfn2_occupations.cu",
        )
    )
    for i, source in enumerate(sources):
        output = tmp_path / f"part{i}.o"
        subprocess.run(
            [
                launcher,
                nvcc,
                "-std=c++20",
                "-O3",
                *arch_flags,
                *(
                    ["-DGENERATIVEQC_GFN2_DENSITY_WORK_DIAGNOSTICS=1"]
                    if family == "matrices_receipt"
                    else []
                ),
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
    binary = tmp_path / "electronic"
    subprocess.run(
        [nvcc, *arch_flags, *objects, "-o", str(binary)], check=True, timeout=60
    )
    subprocess.run([str(binary)], check=True, timeout=120)
