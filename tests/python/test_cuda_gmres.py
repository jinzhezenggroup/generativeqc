"""Independent native GMRES oracles; GPU execution is explicitly Slurm-only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.tensor.cuda_gmres import emit_cuda_gmres


def test_generation_without_runtime_imports(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = """
import importlib.abc
import runpy
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('generativeqc', 'cupy', 'torch', 'pyscf'):
            raise AssertionError('runtime imported during source generation: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = [sys.argv[1], '--output', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    output = tmp_path / "kernels.cuh"
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(root / "tools/generate_cuda_gmres.py"),
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert output.read_text() == emit_cuda_gmres()


@pytest.fixture
def probe() -> Path:
    executable = os.environ.get("GENERATIVEQC_CUDA_GMRES_PROBE")
    if not executable:
        pytest.skip("requires the optional native CUDA GMRES probe")
    path = Path(executable)
    assert path.is_file()
    return path


def _run(
    probe: Path, *arguments: object, values: object = (), env: dict | None = None
) -> dict:
    result = subprocess.run(
        [str(probe), *(str(value) for value in arguments)],
        input=" ".join(
            format(float(value), ".17g") for value in np.asarray(values).reshape(-1)
        ),
        text=True,
        check=True,
        capture_output=True,
        timeout=60,
        env=env,
    )
    return json.loads(result.stdout)


def test_native_admission_without_device(probe: Path) -> None:
    assert _run(probe, "contract") == {"contract": "passed"}


def _solve(
    probe: Path,
    matrix: np.ndarray,
    rhs: np.ndarray,
    diagonal: np.ndarray | None = None,
    *,
    restart: int = 16,
    max_iterations: int = 200,
    reorthogonalizations: int = 2,
    relative_tolerance: float = 1e-11,
    absolute_tolerance: float = 1e-13,
    diagonal_floor: float = 1e-12,
    mode: str = "normal",
    env: dict | None = None,
) -> dict:
    assert os.environ.get("SLURM_JOB_ID"), "all real-GPU tests must use srun"
    return _run(
        probe,
        "solve",
        len(rhs),
        restart,
        max_iterations,
        reorthogonalizations,
        relative_tolerance,
        absolute_tolerance,
        1e-14,
        diagonal_floor,
        int(diagonal is not None),
        mode,
        256 << 20,
        values=np.concatenate(
            [matrix.reshape(-1), rhs, *([diagonal] if diagonal is not None else [])]
        ),
        env=env,
    )


def _accept(result: dict, matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    assert result["cuda_error"] == 0
    assert result["converged"] and result["status"] == 1
    solution = np.array(result["solution"])
    expected = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(solution, expected, atol=2e-10, rtol=2e-10)
    residual = np.linalg.norm((matrix @ solution - rhs) / (np.max(abs(rhs)) or 1))
    target = result["tolerance"] / (np.max(abs(rhs)) or 1)
    assert residual <= target * 1.03 + 2e-14
    assert result["residual_norm"] <= result["tolerance"]
    return solution


@pytest.mark.parametrize("dimension,restart", [(1, 1), (9, 9), (33, 5), (257, 7)])
@pytest.mark.parametrize("precondition", [False, True])
def test_coupled_nonsymmetric_solve(
    probe: Path, dimension: int, restart: int, precondition: bool
) -> None:
    random = np.random.default_rng(904 + dimension)
    matrix = random.normal(size=(dimension, dimension)) / max(dimension, 1)
    matrix += np.diag(np.linspace(1, 3, dimension))
    rhs = random.normal(size=dimension)
    diagonal = np.diag(matrix).copy() if precondition else None
    result = _solve(probe, matrix, rhs, diagonal, restart=restart)
    _accept(result, matrix, rhs)
    if result["iterations"]:
        assert result["operator_actions"] >= result["iterations"]
    if precondition:
        assert result["preconditioner_actions"] > 0
    if dimension > restart and not precondition:
        assert result["restarts"] > 0


def test_signed_diagonal_and_one_pass_mgs(probe: Path) -> None:
    matrix = np.array([[-3, 0.1, 0.04], [0.2, 2, -0.03], [0.1, 0.06, -4]])
    rhs = np.array([0.4, -0.1, 0.3])
    result = _solve(probe, matrix, rhs, np.diag(matrix), reorthogonalizations=1)
    _accept(result, matrix, rhs)


@pytest.mark.parametrize("scale", [1e-250, 1e200])
def test_scaled_norm_avoids_squared_underflow_and_overflow(
    probe: Path, scale: float
) -> None:
    matrix = np.array([[2, 0.1], [-0.1, 3]]) * scale
    rhs = np.array([1, -2]) * scale
    result = _solve(
        probe, matrix, rhs, np.diag(matrix), absolute_tolerance=0, diagonal_floor=0
    )
    _accept(result, matrix, rhs)
    assert result["rhs_norm"] / scale == pytest.approx(np.sqrt(5), rel=2e-15)


def test_empty_and_zero_rhs_skip_operator(probe: Path) -> None:
    for dimension in (0, 4):
        result = _solve(
            probe, np.eye(dimension), np.zeros(dimension), mode="fail_action"
        )
        assert result["converged"]
        assert result["operator_actions"] == result["iterations"] == 0
        assert result["solution"] == [0] * dimension


def test_singular_operator_does_not_publish_projected_solution(probe: Path) -> None:
    result = _solve(probe, np.zeros((3, 3)), np.ones(3))
    assert result["cuda_error"] == 0
    assert result["status"] == 3 and not result["converged"]
    assert result["solution"] is None and result["untouched"]


def test_iteration_exhaustion_preserves_solution(probe: Path) -> None:
    matrix = np.diag(np.linspace(1, 8, 8))
    rhs = np.ones(8)
    result = _solve(probe, matrix, rhs, restart=1, max_iterations=1)
    assert result["status"] == 2 and not result["converged"]
    assert result["residual_norm"] > result["tolerance"]
    assert result["solution"] is None and result["untouched"]


def test_zero_projected_residual_never_bypasses_true_gate(probe: Path) -> None:
    result = _solve(probe, np.eye(3), np.ones(3), mode="inconsistent_action")
    assert result["status"] == 3 and not result["converged"]
    assert result["residual_norm"] == pytest.approx(np.sqrt(3))
    assert result["residual_norm"] > result["tolerance"]
    assert result["solution"] is None and result["untouched"]


@pytest.mark.parametrize(
    "mode,status",
    [("nan_rhs", 4), ("nan_diagonal", 5), ("nan_action", 6), ("nan_second", 6)],
)
def test_nonfinite_failure_preserves_solution(
    probe: Path, mode: str, status: int
) -> None:
    result = _solve(probe, np.eye(3), np.ones(3), np.ones(3), mode=mode)
    assert result["cuda_error"] == 0 and result["status"] == status
    assert (
        not result["converged"] and result["solution"] is None and result["untouched"]
    )


def test_gap_floor_refuses_unseparated_response(probe: Path) -> None:
    result = _solve(probe, np.eye(3), np.ones(3), np.array([1, 1e-13, 1]))
    assert result["status"] == 5 and not result["converged"] and result["untouched"]


def test_overflowing_preconditioner_cannot_publish(probe: Path) -> None:
    result = _solve(probe, np.eye(3), np.ones(3), np.full(3, 1e-310), diagonal_floor=0)
    assert result["status"] == 5 and not result["converged"] and result["untouched"]


@pytest.mark.parametrize(
    "mode", ["fail_action", "fail_second", "alias", "overlap", "short", "unaligned"]
)
def test_native_error_preserves_solution(probe: Path, mode: str) -> None:
    result = _solve(probe, np.eye(3), np.ones(3), mode=mode)
    assert result["cuda_error"] != 0
    assert result["solution"] is None and result["untouched"]


def test_no_python_or_compiler_at_execution(probe: Path, tmp_path: Path) -> None:
    env = {
        **os.environ,
        "PATH": str(tmp_path),
        "PYTHONHOME": str(tmp_path / "absent"),
        "PYTHONPATH": str(tmp_path / "absent"),
    }
    matrix = np.array([[2, 0.4], [-0.1, 3]])
    rhs = np.array([0.1, 0.3])
    _accept(_solve(probe, matrix, rhs, env=env), matrix, rhs)
