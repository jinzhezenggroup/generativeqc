"""Actual physical owner and generated maps under deferred host CUDA execution.

This is a host lifetime/error-path test. It does not qualify CUDA scheduling,
real provider identity or physical molecular integrals.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from tools.generate_wb97mv_fock_response import generate

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/native/fixtures/wb97mv_fock_queue"


def _queued_cuda(source: str) -> str:
    """Replace CUDA launch syntax, retaining complete controller/kernel bodies."""

    def launch(match: re.Match[str]) -> str:
        configuration = match[2].rsplit(",", 3)
        if len(configuration) != 4:
            raise ValueError(f"unsupported emulated launch configuration: {match[2]}")
        return (
            f"queue_kernel({configuration[0]}, {configuration[1]}, [=]() {{ "
            f"{match[1]}({match[3]}); }});"
        )

    transformed = re.sub(r"(\w+)<<<(.*?)>>>\((.*?)\);", launch, source, flags=re.DOTALL)
    if "<<<" in transformed:
        raise ValueError("untranslated CUDA launch")
    return transformed


def test_physical_fock_drains_failed_borrowed_work(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Every real generated launch boundary rejects without outstanding borrows."""
    shims = (
        "cuda_runtime.h",
        "cuda_runtime_api.h",
        "dft/cuda_ks_grid_reference.hpp",
        "dft/cuda_ks_final_state.hpp",
        "dft/cuda_wb97mv_grid.hpp",
        "dft/semilocal_family.hpp",
        "molecule/basis.hpp",
        "response/cuda_cpks.hpp",
        "scf/cuda_density_fitting_device.hpp",
        "scf/cuda_direct_jk_device.hpp",
        "scf/cuda_fock_execution.hpp",
    )
    for name in shims:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#pragma once\n#include "providers.hpp"\n')
    (tmp_path / "response/wb97mv_fock_response.cuh").write_text(
        _queued_cuda(generate())
    )
    (tmp_path / "physical_fock_host.cpp").write_text(
        _queued_cuda((ROOT / "src/dft/cuda_ks_wb97mv_response.cu").read_text())
    )
    executable = native_cxx.build_executable(
        [FIXTURE / "probe.cpp"],
        tmp_path / "probe",
        compile_args=(
            "-std=c++20",
            "-O0",
            "-ffp-contract=off",
            f"-I{tmp_path}",
            f"-I{FIXTURE}",
            f"-I{ROOT / 'src'}",
        ),
    )
    result = subprocess.run(
        [str(executable)], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS spins=") == 4
