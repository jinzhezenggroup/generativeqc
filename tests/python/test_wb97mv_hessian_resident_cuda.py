"""Resident lowering and no-host-fallback gates; run only on a Slurm GPU."""

import ctypes as ct
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, Primitive, Shell
from generativeqc._cuda_hessian_execution import CudaAO, CudaHessianExecution
from generativeqc._cuda_hessian_response import resident_gmres
from generativeqc.response_problem import ResponseSolveError
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.common.array_execution import array_execution
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.integral.ir import four_center_eri_operator
from generativeqc_compiler.integral.range_separation import CoulombKernel
from generativeqc_compiler.integral.resident_second_cuda import emit_resident_second
from generativeqc_compiler.integral.second_derivatives import build_eri_second_ir
from generativeqc_compiler.integral.weighted_eri import (
    build_weighted_eri_ir,
    build_weighted_eri_kernel,
)
from generativeqc_compiler.tensor import Program, TensorSpec, execute, input_tensor

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_WB97MV_HESSIAN_CUDA_TEST") != "1",
    reason="requires an explicitly allocated Slurm GPU",
)


@contextmanager
def device_scope(execution: CudaHessianExecution) -> Iterator[None]:
    with execution.device, execution.stream, array_execution(execution):
        yield


@pytest.fixture(scope="module")
def execution(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[CudaHessianExecution]:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    compiler = CudaCompilerAdapter(
        Path(shutil.which("nvcc")), cuda_target_info("sm_120")
    )
    owner = CudaHessianExecution(compiler, device_id=0)
    with device_scope(owner):
        owner.initialize(tmp_path_factory.mktemp("resident-lowering"))
    yield owner
    with device_scope(owner):
        owner.check()


@pytest.mark.parametrize("angular", (0, 1, 2, 3))
def test_resident_spherical_ao_jets_match_native(
    angular: int, execution: CudaHessianExecution
) -> None:
    atoms = [("H", (0.1, -0.2, 0.3)), ("H", (-0.4, 0.3, 1.4))]
    shells = (
        Shell(0, angular, (Primitive(0.7, 1),)),
        Shell(1, 0, (Primitive(0.9, 1),)),
    )
    points = np.random.default_rng(819).normal(size=(17, 3))
    with NativeAO(atoms, shells, representation="real_spherical") as basis:
        expected = basis.evaluate(points, 3)
        with device_scope(execution):
            actual = execution.publish(CudaAO(basis, execution).evaluate(points, 3))
    np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("family", ("short_range", "long_range"))
@pytest.mark.parametrize("angular", ((0, 0, 0, 0), (1, 0, 1, 0), (3, 0, 0, 0)))
def test_resident_integral_gradient_hvp_against_interval_oracle(
    family: str, angular: tuple, execution: CudaHessianExecution
) -> None:
    from test_rsh_second_derivatives import CENTERS, EXPONENTS, _gradient

    radial = CoulombKernel(family, 0.3)
    operator = four_center_eri_operator(radial)
    first = build_weighted_eri_kernel(
        build_weighted_eri_ir(angular, operator=operator), (0,)
    )
    integral = build_eri_second_ir(angular, operator=operator)
    direction = np.random.default_rng(1888).normal(size=(4, 3)) * 0.1
    expected_gradient = _gradient(first, radial, CENTERS)
    step = 1e-4
    expected_hvp = (
        _gradient(first, radial, CENTERS + step * direction)
        - _gradient(first, radial, CENTERS - step * direction)
    ) / (2 * step)
    with device_scope(execution):
        cp = execution.cp
        function = execution.function(
            emit_resident_second(integral, (0,)),
            "generativeqc_resident_integral",
            [
                ct.c_void_p,
                ct.c_size_t,
                ct.c_void_p,
                ct.c_void_p,
                ct.c_void_p,
                ct.c_void_p,
                ct.c_void_p,
            ],
            integral=True,
        )
        records = cp.asarray([(*EXPONENTS, *CENTERS.reshape(-1), 0.73)])
        atoms, vector = cp.arange(4, dtype=cp.int64), cp.asarray(direction)
        output = cp.empty((1, 25))
        execution.launch(
            function,
            records.data.ptr,
            1,
            atoms.data.ptr,
            vector.data.ptr,
            output.data.ptr,
        )
        actual = execution.publish(output)[0]
    np.testing.assert_allclose(
        actual[1:13].reshape(4, 3), expected_gradient, atol=2e-12, rtol=2e-12
    )
    np.testing.assert_allclose(
        actual[13:].reshape(4, 3), expected_hvp, atol=2e-8, rtol=2e-6
    )


def test_resident_scalar_graph_keeps_inactive_branches_lazy(
    execution: CudaHessianExecution,
) -> None:
    graph = Graph()
    value = graph.variable("value")
    root = graph.select_le(
        value, graph.constant(0), graph.constant(0), graph.stable_unary("log", value)
    )
    with device_scope(execution):
        result = execution.graph(
            graph, (root,), {"value": execution.cp.asarray([-1.0, 0.0, 2.0])}
        )[0]
        actual = execution.publish(result)
    np.testing.assert_allclose(actual, [0, 0, np.log(2)], atol=2e-15)


def test_nonfinite_tensor_is_rejected_before_publication(
    execution: CudaHessianExecution,
) -> None:
    value = input_tensor("value", TensorSpec(role="input"))
    program = Program({"value": value})
    with device_scope(execution):
        try:
            result = execute(program, {"value": execution.cp.asarray(float("nan"))})
            with pytest.raises(FloatingPointError, match="resident compiler-AD"):
                execution.publish(result.outputs["value"])
        finally:
            execution.status.fill(0)


@pytest.mark.parametrize("scale", (1.0, 1e-270))
def test_resident_gmres_true_residual_and_failure_gates(
    scale: float, execution: CudaHessianExecution
) -> None:
    matrix = np.array(
        [[2, 0.1, 0, 0], [-0.3, 1.5, 0.2, 0], [0, 0.4, 1.3, 0.1], [0.1, 0, -0.2, 2.5]]
    )
    rhs = np.array([0.1, -0.2, 0.3, 0.4]) * scale
    expected = np.linalg.solve(matrix, rhs)
    options = GMRESOptions(rtol=1e-11, restart=2, max_iterations=80)
    with device_scope(execution):
        cp = execution.cp
        resident = cp.asarray(matrix)
        result = resident_gmres(
            cp,
            lambda vector: resident @ vector,
            cp.asarray(rhs),
            cp.diag(resident),
            options,
        )
        actual = execution.publish(result.solution)
        with pytest.raises(ResponseSolveError):
            resident_gmres(
                cp,
                lambda vector: resident @ vector,
                cp.asarray(rhs),
                cp.diag(resident),
                GMRESOptions(rtol=1e-14, restart=1, max_iterations=1),
            )
    np.testing.assert_allclose(actual / scale, expected / scale, atol=3e-12, rtol=3e-11)
    assert result.residual_norm <= options.rtol * np.linalg.norm(rhs / scale) * scale


def test_complete_endpoint_forbids_host_ad_and_has_bounded_device_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.common import array_graph
    from generativeqc_compiler.integral.df_second_execute import DFSecondEvaluator

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [("H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))]
    vector = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    calculator = Calculator(
        method="wb97m-v",
        device="cuda",
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("resident endpoint called a host AD/AO evaluator")

    monkeypatch.setattr(NativeAO, "evaluate", forbidden)
    monkeypatch.setattr(DFSecondEvaluator, "contract", forbidden)
    monkeypatch.setattr(array_graph, "_operation", forbidden)
    dense = calculator.hessian_vector_product(atoms, vector, cache=tmp_path)
    budget = 8 * (64 * 2**2 + 16384) + 1024
    streamed = calculator.hessian_vector_product(
        atoms, vector, cache=tmp_path, integral_budget_bytes=budget
    )
    assert dense.diagnostics["eri_primal_storage"] == "resident-dense"
    assert streamed.diagnostics["eri_primal_storage"] == "device-streamed"
    np.testing.assert_allclose(streamed.value, dense.value, atol=3e-11, rtol=3e-11)
    assert streamed.diagnostics["numerical_host_fallbacks"] == 0
    assert streamed.diagnostics["intermediate_panel_downloads"] == 0


def test_combined_ecp_df_spin_response_with_spherical_d_auxiliary(
    tmp_path: Path,
) -> None:
    from test_wb97mv_ecp_hessian import synthetic_basis

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [
        ("Na", (0.13, -0.21, 0.17)),
        ("H", (0.43, 0.19, 2.1)),
        ("H", (-0.3, 0.4, -1.6)),
    ]
    direction = np.array(
        [[0.02, -0.03, 0.1], [-0.04, 0.01, -0.02], [0.02, 0.02, -0.08]]
    )
    auxiliary = (
        Shell(0, 0, (Primitive(0.8, 1),)),
        Shell(0, 2, (Primitive(0.6, 1),)),
        Shell(1, 0, (Primitive(0.65, 1),)),
        Shell(2, 0, (Primitive(0.75, 1),)),
    )
    common = {
        "method": "wb97m-v-uks",
        "basis": synthetic_basis(),
        "auxiliary_basis": auxiliary,
        "basis_representation": "spherical",
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "max_iterations": 240,
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    }
    results = [
        Calculator(
            device=device, density_fitting=device, **common
        ).hessian_vector_product(
            atoms,
            direction,
            multiplicity=2,
            cache=tmp_path,
            solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
        )
        for device in ("cpu", "cuda")
    ]
    np.testing.assert_allclose(results[1].value, results[0].value, atol=3e-8, rtol=2e-6)
    assert results[1].diagnostics["response_iterations"] > 0
    assert results[1].diagnostics["ecp_ad_backend"] == "cuda"
    assert results[1].diagnostics["df_integral_ad_backend"] == "cuda"
    assert results[1].diagnostics["execution_residency"] == "device-resident"
