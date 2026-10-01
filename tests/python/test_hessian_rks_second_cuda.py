"""Real-GPU qualification for mixed RKS Hessian second-integral execution."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, Primitive, Shell
from generativeqc.rks_hessian import rks_hessian, rks_hvp
from generativeqc.rks_response import NativeRKSResponse
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.dft import NativeAO

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESPONSE_CUDA_TEST") != "1",
    reason="explicit real-GPU qualification",
)

ATOMS = [("H", (0.0, 0.0, -0.72)), ("H", (0.0, 0.0, 0.72))]
BASIS = (
    Shell(0, 0, (Primitive(1.2, 1.0),)),
    Shell(1, 0, (Primitive(1.2, 1.0),)),
)
GRID = GridSpec(radial_points=10, angular_polar=4, angular_azimuth=8)


@pytest.fixture(scope="module")
def compiler() -> CudaCompilerAdapter:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    nvcc = shutil.which("nvcc")
    assert nvcc, "selected CUDA qualification needs nvcc on PATH"
    return CudaCompilerAdapter(
        Path(nvcc),
        cuda_target_info(os.environ.get("GENERATIVEQC_TEST_CUDA_ARCH", "sm_120")),
    )


@pytest.fixture(scope="module")
def rks_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[NativeRKSResponse, np.ndarray, Path]]:
    calculator = Calculator(
        method="lda-rks",
        basis=BASIS,
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )
    cache = tmp_path_factory.mktemp("rks-hessian-cuda-second")
    direction = np.array([[0.05, -0.03, 0.21], [-0.05, 0.03, -0.21]])
    direction /= np.linalg.norm(direction)
    with (
        calculator.prepare_batch([ATOMS]) as batch,
        NativeAO(ATOMS, basis=BASIS) as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        with NativeRKSResponse.from_native(batch, basis, tile_points=257) as operator:
            yield operator, direction, cache


def test_complete_rks_hvp_can_use_cuda_second_integrals(
    rks_case: tuple[NativeRKSResponse, np.ndarray, Path],
    compiler: CudaCompilerAdapter,
) -> None:
    operator, direction, cache = rks_case
    solver = GMRESOptions(atol=1e-12, rtol=1e-11)
    expected = rks_hvp(
        operator,
        direction,
        cache=cache,
        solver_options=solver,
    )
    actual = rks_hvp(
        operator,
        direction,
        cache=cache,
        solver_options=solver,
        second_backend="cuda",
        second_compiler=compiler,
    )

    np.testing.assert_allclose(actual.value, expected.value, atol=1e-9, rtol=5e-10)
    for name in ("one_electron", "coulomb", "overlap_pulay"):
        np.testing.assert_allclose(
            actual.components[name], expected.components[name], atol=1e-9, rtol=5e-10
        )
        diagnostic = actual.diagnostics["integral_providers"][name]
        assert diagnostic["provider_backend"] == "cuda"
        assert diagnostic["record_batch_uploads"] > 0
        assert diagnostic["result_tile_downloads"] > 0
        assert diagnostic["raw_hessian_downloads"] == 0
        assert diagnostic["peak_device_bytes"] > 0

    assert actual.diagnostics["response_first_integral_backend"] == "cpu"
    assert actual.diagnostics["second_integral_backend"] == "cuda"
    assert actual.diagnostics["execution_residency"] == "mixed-host-device"


def test_full_rks_hessian_threads_cuda_second_integrals(
    rks_case: tuple[NativeRKSResponse, np.ndarray, Path],
    compiler: CudaCompilerAdapter,
) -> None:
    operator, _, cache = rks_case
    solver = GMRESOptions(atol=1e-12, rtol=1e-11)
    expected = rks_hessian(
        operator,
        block_size=3,
        cache=cache,
        solver_options=solver,
    )
    actual = rks_hessian(
        operator,
        block_size=3,
        cache=cache,
        solver_options=solver,
        second_backend="cuda",
        second_compiler=compiler,
    )

    np.testing.assert_allclose(actual.matrix, expected.matrix, atol=2e-9, rtol=8e-10)
    assert actual.diagnostics["second_integral_backend"] == "cuda"
    assert actual.diagnostics["execution_residency"] == "mixed-host-device"
    assert not actual.diagnostics["posthoc_symmetrization"]
    assert all(
        block["diagnostics"]["second_integral_backend"] == "cuda"
        for block in actual.diagnostics["blocks"]
    )
