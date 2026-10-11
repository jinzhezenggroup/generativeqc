"""CUDA planning without a device, and separately opted-in Slurm execution."""

import os
from pathlib import Path

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.tensor.cuda_emit import emit_cuda
from generativeqc_compiler.tensor.cuda_plan import plan_cuda


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_cuda_derivative_plans_and_missing_toolchain_fail_closed(
    dtype: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc import profiles
    from generativeqc_compiler.tensor import interpreter

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("CUDA fell back to a CPU oracle")

    monkeypatch.setattr(interpreter, "execute", forbidden)
    monkeypatch.setattr(profiles, "find_nvcc", lambda: None)
    compiled = xp.compile(
        lambda values: xp.sum(values * values), backend="cuda", cache=tmp_path
    )
    transformed = xp.value_and_grad(compiled)
    values = np.asarray([0.2, 0.7, 1.4], dtype=dtype)
    program = transformed.lower(values)
    plan = plan_cuda(program, cuda_target_info("sm_120"))
    assert set(plan.program.outputs) == {"output", "bar_values"}
    assert emit_cuda(plan)
    with pytest.raises(RuntimeError, match="NVCC"):
        transformed(values)
    assert transformed.compilation_info()["native_executables"] == 0
    with pytest.raises(ValueError, match="work budget"):
        xp.value_and_grad(compiled, max_work=0)(values)


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_ARRAY_API_CUDA_TEST") != "1",
    reason="requires explicit allowed-node Slurm GPU allocation",
)
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_slurm_native_cuda_primal_and_derivative(
    dtype: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.tensor import autodiff, interpreter

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests must run through srun"
    assert os.environ.get("SLURMD_NODENAME") in {"n1", "n2", "n4", "n5"}

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("native CUDA delegated to reference execution/AD")

    monkeypatch.setattr(interpreter, "execute", forbidden)
    monkeypatch.setattr(autodiff, "jvp", forbidden)
    monkeypatch.setattr(autodiff, "vjp", forbidden)
    compiled = xp.compile(
        lambda values: xp.sum(xp.exp(-values * values)),
        backend="cuda",
        cache=tmp_path,
        cuda_arch=os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_120"),
    )
    values = np.asarray([0.2, 0.7, 1.4], dtype=dtype)
    expected = np.exp(-values * values)
    transformed = xp.value_and_grad(compiled)
    try:
        value, gradient = transformed(values)
        tolerance = 2e-6 if dtype == np.float32 else 2e-13
        np.testing.assert_allclose(
            value, np.sum(expected), rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            gradient, -2 * values * expected, rtol=tolerance, atol=tolerance
        )
        assert transformed.compilation_info()["native_executables"] == 1
    finally:
        transformed.close()
        compiled.close()
