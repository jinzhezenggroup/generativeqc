"""Optional pinned-oracle execution gates for the explicit density-work domain.

The CPU gate exercises the production point ABI, including minority outputs.
The CUDA gate exercises the common prepared tile runtime in both spin layouts.
These are point-contract gates, not physical-birth or public endpoint evidence.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc_compiler.common.compiler_cache import run_cached_compiler
from generativeqc_compiler.common.compiler_process import run_compiler
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.xc.bulk_point_program import (
    bind_runtime_semilocal_point_program,
)
from generativeqc_compiler.xc.bulk_runtime import build_bulk_runtime_program
from generativeqc_compiler.xc.cuda import CudaXC, compile_cuda
from generativeqc_compiler.xc.density_work import DENSITY_WORK_DOMAIN

from tools.qualify_libxc_compiled_cpu import _translation_unit

if TYPE_CHECKING:
    from generativeqc_compiler.xc.bulk_runtime import BulkRuntimeProgram

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("LDA_C_VWN_4", "LDA_C_CHACHIYO", "LDA_C_PZ_MOD")


@pytest.fixture(scope="module")
def oracle() -> Any:
    if os.environ.get("GENERATIVEQC_DENSITY_WORK_NATIVE_TEST") != "1":
        pytest.skip("opt-in pinned density-work execution gate")
    pyscf = pytest.importorskip("pyscf")
    from pyscf.dft import libxc

    assert pyscf.__version__ == "2.14.0" and libxc.__version__ == "7.0.0"
    return libxc


def _inputs(program: BulkRuntimeProgram) -> np.ndarray:
    if program.spec.spin == "unpolarized":
        return np.array([[0.0, 0.62, 0.8, 1.1e-15, 1.9e-15, 2.1e-15, 1e-20, 1e160]])
    return np.array(
        [
            [0.0, 0.62, 0.8, 0.0, 0.0, 0.55e-15, 1.1e-15, 1e-20],
            [0.62, 0.0, 0.0, 0.8, 1.1e-15, 0.55e-15, 1.1e-15, 1e-20],
        ]
    )


def _reference(oracle: Any, name: str, features: np.ndarray) -> np.ndarray:
    polarized = features.shape[0] == 2
    exc, vxc, _, _ = oracle.eval_xc(name, features, spin=int(polarized), deriv=1)
    gradient = np.asarray(vxc[0]).T if polarized else np.asarray(vxc[0])[None, :]
    return np.vstack((features.sum(axis=0) * exc, gradient))


@pytest.mark.parametrize("name", NAMES)
def test_native_cpu_density_work_matches_complete_pinned_outputs(
    name: str, oracle: Any, tmp_path: Path
) -> None:
    compiler = shutil.which("c++")
    assert compiler is not None, "opt-in execution requires a C++ compiler"
    program = build_bulk_runtime_program(name, domain=DENSITY_WORK_DOMAIN)
    binding = bind_runtime_semilocal_point_program(program)
    features = _inputs(program)
    cases = tuple(
        (str(index), rho, np.zeros((2, 3)), np.zeros(2))
        for index, rho in enumerate(features.T)
    )
    source = tmp_path / "point.cpp"
    source.write_text(
        _translation_unit(
            binding.emit_source(), binding.identity, binding.domain_version, cases
        )
    )
    object_path = tmp_path / "point.o"
    compiled = run_cached_compiler(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
            "-c",
            str(source),
            "-o",
            str(object_path),
        ],
        180,
        label="density work CPU point",
    )
    assert compiled.returncode == 0 and not compiled.timed_out, compiled.stderr
    executable = tmp_path / "point"
    linked = run_compiler(
        [compiler, str(object_path), "-o", str(executable)],
        60,
        label="density work CPU link",
    )
    assert linked.returncode == 0 and not linked.timed_out, linked.stderr
    completed = subprocess.run(
        [str(executable)], capture_output=True, text=True, check=True, timeout=30
    )
    lines = completed.stdout.splitlines()
    assert lines[0] == f"{binding.identity} {binding.domain_version}"
    observed = np.array([[float(value) for value in row.split()] for row in lines[1:]])
    np.testing.assert_allclose(
        observed[:, :3].T, _reference(oracle, name, features), atol=2e-14, rtol=2e-12
    )
    np.testing.assert_array_equal(observed[:, 3:], 0.0)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_prepared_cuda_density_work_matches_complete_pinned_outputs(
    name: str, spin: str, oracle: Any, tmp_path: Path
) -> None:
    if os.environ.get("GENERATIVEQC_XC_CUDA_TEST") != "1":
        pytest.skip("opt-in Slurm CUDA execution gate")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU execution requires srun"
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "preserve scheduler visibility"
    compiler = shutil.which(os.environ.get("GENERATIVEQC_NVCC", "nvcc"))
    assert compiler is not None, "opt-in execution requires NVCC"
    program = build_bulk_runtime_program(name, spin=spin, domain=DENSITY_WORK_DOMAIN)
    features = _inputs(program)
    original = features.copy()
    artifact = compile_cuda(
        program,
        CudaCompilerAdapter(Path(compiler), cuda_target_info("sm_120")),
        tmp_path / "runtime",
    )
    with CudaXC(program, artifact, tile_points=16) as executor:
        observed = executor.evaluate(features)
    np.testing.assert_array_equal(features, original)
    np.testing.assert_allclose(
        observed, _reference(oracle, name, features), atol=2e-14, rtol=2e-12
    )
