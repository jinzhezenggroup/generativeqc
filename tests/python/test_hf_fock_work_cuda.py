"""Actual HF operator work and CUDA graph replay, independent of iteration totals."""

from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires a finite Slurm GPU allocation",
)


def test_assembly_counts_execution_and_resets_between_replays(
    tmp_path: Path, required_native_cxx: NativeCxx
) -> None:
    """Real stream/graph execution must count every pass, including replay tails."""
    from generativeqc_compiler.tensor.scf_cuda import emit_density_cuda

    (tmp_path / "generated_scf_density_cuda.cuh").write_text(emit_density_cuda())
    compiler = replace(
        required_native_cxx, compiler=str(Path(os.environ["CUDA_PATH"]) / "bin/nvcc")
    )
    executable = compiler.build_executable(
        [
            ROOT / "tests/native/df_fock_counter_probe.cu",
            ROOT / "src/scf/cuda/df_scf_kernels.cu",
            ROOT / "src/scf/cuda/scf_state_kernels.cu",
        ],
        tmp_path / "counter",
        compile_args=(
            "-std=c++20",
            "-O2",
            "-arch=" + os.environ.get("GENERATIVEQC_TEST_CUDA_ARCH", "sm_120"),
            "-I" + str(tmp_path),
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
        ),
        compile_timeout=600,
    )
    subprocess.run([executable], check=True, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("fitted", (False, True), ids=("direct", "df"))
def test_hf_complete_work_and_frozen_replay(fitted: bool) -> None:
    """Independent HF forces accompany cold/warm work export and reset checks."""
    from generativeqc import Calculator
    from pyscf import gto, lib, scf
    from test_scf_df_auxiliary_g_cuda import auxiliary_snapshot

    lib.num_threads(1)
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8)), ("H", (1.7, 0.0, -0.6))]
    mol = gto.M(atom=atoms, basis="def2-svp", unit="Bohr", verbose=0)
    reference = scf.RHF(mol)
    options = {}
    if fitted:
        reference = reference.density_fit()
        options = {
            "density_fitting": "cuda",
            "auxiliary_basis": auxiliary_snapshot(mol),
        }
    reference.conv_tol, reference.conv_tol_grad = 1e-12, 1e-9
    reference.kernel()
    assert reference.converged
    gradient = reference.nuc_grad_method()
    if fitted:
        gradient.auxbasis_response = True
    forces = -gradient.kernel()
    calculator = Calculator(method="rhf", basis="def2-svp", device="cuda", **options)
    try:
        with calculator.prepare_batch([atoms], warm_start=True) as owner:
            counts = []
            for repeat in range(3):
                result = owner.execute(
                    properties=("energy", "forces"), strict=True
                ).items[0]
                assert result.converged and result.executed_backend == "cuda"
                assert result.fock_builds is not None and result.fock_builds > 0
                assert abs(result.energy - reference.e_tot) < 1e-7
                np.testing.assert_allclose(result.forces, forces, atol=1e-6, rtol=0)
                if not fitted:
                    work = result.precision
                    assert work is not None and work["operator_work_counters_valid"]
                    assert result.fock_builds == sum(
                        work[key]
                        for key in (
                            "mixed_stage_fock_builds",
                            "strict_stage_fock_builds",
                            "post_scf_fock_builds",
                        )
                    )
                if repeat:
                    assert result.warm_start_used
                owner.set_warm_start_updates(False)
                counts.append(result.fock_builds)
            assert counts[1] == counts[2]
    finally:
        calculator.clear_cache()
