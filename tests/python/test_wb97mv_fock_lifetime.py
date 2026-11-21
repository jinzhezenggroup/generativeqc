"""Actual physical owner and scalar affine maps under a deferred host CUDA queue.

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
    """Queue launches by value, retaining the controller and scalar kernels."""

    def launch(match: re.Match[str]) -> str:
        configuration = match[2].rsplit(",", 3)
        if len(configuration) != 4:
            raise ValueError(f"unsupported emulated launch configuration: {match[2]}")
        return f"queue_kernel({configuration[0]}, {configuration[1]}, {match[1]}, {match[3]});"

    transformed = re.sub(r"(\w+)<<<(.*?)>>>\((.*?)\);", launch, source, flags=re.DOTALL)
    if "<<<" in transformed:
        raise ValueError("untranslated CUDA launch")
    return transformed


def _host_fock_maps() -> str:
    """Preserve admission and dispatch; forbid executing block-cooperative CUDA on the host."""
    source = re.sub(
        r"(^__global__ void \w+_matrix\([^{}]+\) \{).*?^}",
        r'\1 throw std::logic_error("tiled kernel in scalar queue probe"); }',
        generate(),
        flags=re.MULTILINE | re.DOTALL,
    )
    return _queued_cuda(source)


def test_physical_fock_drains_failed_borrowed_work(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    """Every scalar-path launch boundary rejects without outstanding borrows."""
    from test_ks_point_batch_resources import _configure_stationary_domains

    from tools.generate_xc_split_hybrid_registry import emit_registry

    _configure_stationary_domains(tmp_path)
    (tmp_path / "generated_split_hybrid_registry.cuh").write_text(emit_registry())
    shims = (
        "cuda_runtime.h",
        "cuda_runtime_api.h",
        "dft/cuda_hessian_semilocal_model.hpp",
        "dft/cuda_ks_grid_reference.hpp",
        "dft/cuda_ks_final_state.hpp",
        "dft/cuda_wb97mv_grid.hpp",
        "dft/r2scan_rvv10_model.hpp",
        "dft/semilocal_family.hpp",
        "dft/split_hybrid_cuda_selection.hpp",
        "molecule/basis.hpp",
        "response/cuda_cpks.hpp",
        "runtime/resource_cuda.cuh",
        "scf/cuda_density_fitting_basis.hpp",
        "scf/cuda_density_fitting_device.hpp",
        "scf/cuda_direct_jk_device.hpp",
        "scf/cuda_fock_execution.hpp",
    )
    for name in shims:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('#pragma once\n#include "providers.hpp"\n')
    (tmp_path / "response/wb97mv_fock_response.cuh").write_text(_host_fock_maps())
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
    assert result.stdout.count("PASS spins=") == 8
