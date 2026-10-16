"""Independent displaced-SCF oracles for compiler/native coupled KS response.

The model is a weak, self-adjoint linear spin-coupled Fock functional. Its
displaced generalized eigensolves do not use CPKS response equations or AD.
Finite differences are test oracles only, never a production derivative path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.method.ks_cpks import build_ks_cpks_maps
from generativeqc_compiler.tensor import execute
from generativeqc_compiler.tensor.cuda_affine import emit_affine_cuda
from generativeqc_compiler.tensor.ir import constant, input_tensor, slice_tensor
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2


def _inverse_root(matrix: np.ndarray) -> np.ndarray:
    energies, vectors = np.linalg.eigh(matrix)
    return (vectors / np.sqrt(energies)) @ vectors.T


def _model(nao: int, counts: tuple[int, ...], *, degenerate: bool = False) -> dict:
    random = np.random.default_rng(1924 + sum(counts))
    spins = len(counts)
    occupation = 2 if spins == 1 else 1
    raw = random.normal(scale=0.08, size=(nao, nao))
    overlap = np.eye(nao) + raw @ raw.T
    inverse_root = _inverse_root(overlap)
    coefficients, energies = [], []
    for count in counts:
        frame = inverse_root @ np.linalg.qr(random.normal(size=(nao, nao)))[0]
        occupied = (
            np.full(count, -1.2) if degenerate else np.linspace(-1.5, -0.9, count)
        )
        coefficients.append(frame)
        energies.append(np.concatenate([occupied, np.linspace(0.4, 1.3, nao - count)]))
    coefficients, energies = np.array(coefficients), np.array(energies)
    direction = np.array(
        [_symmetric(random.normal(scale=0.2, size=(nao, nao))) for _ in counts]
    )
    if spins == 2:
        direction[1] = 0  # A source in alpha must induce a nonzero beta response.
    overlap_direction = _symmetric(random.normal(scale=0.08, size=(nao, nao)))
    basis = _symmetric(random.normal(scale=0.4, size=(nao, nao)))
    kernel = 0.025 * (
        np.eye(nao * nao) + np.outer(basis.reshape(-1), basis.reshape(-1))
    )
    spin_coupling = np.full((spins, spins), 0.7) + np.eye(spins) * 0.3
    operation = np.kron(spin_coupling, kernel)

    def induced(density: np.ndarray) -> np.ndarray:
        return (operation @ density.reshape(-1)).reshape(spins, nao, nao)

    density = np.array(
        [
            occupation * frame[:, :count] @ frame[:, :count].T
            for frame, count in zip(coefficients, counts, strict=True)
        ]
    )
    physical = np.array(
        [
            overlap @ (frame * eps) @ frame.T @ overlap
            for frame, eps in zip(coefficients, energies, strict=True)
        ]
    )
    return {
        "nao": nao,
        "counts": counts,
        "occupation": occupation,
        "coefficients": coefficients,
        "energies": energies,
        "overlap": overlap,
        "overlap_direction": overlap_direction,
        "frozen": direction,
        "operation": operation,
        "induced": induced,
        "density": density,
        "hcore": physical - induced(density),
    }


def _displaced_scf(model: dict, displacement: float) -> tuple[np.ndarray, np.ndarray]:
    root = _inverse_root(model["overlap"] + displacement * model["overlap_direction"])
    density = model["density"].copy()
    for _ in range(2000):
        fock = (
            model["hcore"] + displacement * model["frozen"] + model["induced"](density)
        )
        next_density, weights = [], []
        for matrix, count in zip(fock, model["counts"], strict=True):
            energies, vectors = np.linalg.eigh(root @ matrix @ root)
            occupied = (root @ vectors)[:, :count]
            next_density.append(model["occupation"] * occupied @ occupied.T)
            weights.append(
                model["occupation"] * (occupied * energies[:count]) @ occupied.T
            )
        next_density = np.array(next_density)
        if np.linalg.norm(next_density - density) < 2e-14:
            return next_density, np.array(weights)
        density = 0.4 * density + 0.6 * next_density
    raise AssertionError("independent displaced SCF did not converge")


def _oracle(model: dict, step: float) -> tuple[np.ndarray, np.ndarray]:
    plus, minus = _displaced_scf(model, step), _displaced_scf(model, -step)
    return tuple(
        (positive - negative) / (2 * step)
        for positive, negative in zip(plus, minus, strict=True)
    )


def _compiler_response(model: dict) -> tuple[np.ndarray, np.ndarray]:
    programs, feeds, sizes = [], [], []
    for spin, count in enumerate(model["counts"]):
        frame, energies = model["coefficients"][spin], model["energies"][spin]
        programs.append(
            build_ks_cpks_maps(model["nao"], count, occupation=model["occupation"])
        )
        feeds.append(
            {
                "coefficients": frame,
                "occupied": frame[:, :count],
                "virtuals": frame[:, count:],
                "occupied_energies": energies[:count],
                "virtual_energies": energies[count:],
                "occupied_energy_matrix": np.diag(energies[:count]),
                "overlap_direction": model["overlap_direction"],
                "frozen_fock": model["frozen"][spin],
            }
        )
        sizes.append(count * (model["nao"] - count))

    def run(stage: str, additions: list[dict]) -> list[dict]:
        return [
            execute(program[stage], {**feed, **extra}).outputs
            for program, feed, extra in zip(programs, feeds, additions, strict=True)
        ]

    def split(vector: np.ndarray) -> list[dict]:
        offsets = np.cumsum([0, *sizes])
        return [
            {
                "rotation": vector[offsets[spin] : offsets[spin + 1]].reshape(
                    count, model["nao"] - count
                )
            }
            for spin, count in enumerate(model["counts"])
        ]

    metric = run("metric", [{} for _ in sizes])
    induced = model["induced"](np.array([value["density"] for value in metric]))
    rhs = np.concatenate(
        [
            value["rhs"].reshape(-1)
            for value in run("rhs", [{"induced_fock": value} for value in induced])
        ]
    )

    def action(vector: np.ndarray) -> np.ndarray:
        rotations = split(vector)
        density = np.array([value["d_density"] for value in run("rotation", rotations)])
        induced = model["induced"](density)
        return np.concatenate(
            [
                value["action"].reshape(-1)
                for value in run(
                    "action",
                    [
                        {**rotation, "induced_fock": value}
                        for rotation, value in zip(rotations, induced, strict=True)
                    ],
                )
            ]
        )

    dimension = sum(sizes)
    matrix = (
        np.column_stack([action(column) for column in np.eye(dimension)])
        if dimension
        else np.empty((0, 0))
    )
    solution = np.linalg.solve(matrix, rhs)
    density = run(
        "density",
        [
            {**rotation, "metric_motion": value["metric_motion"]}
            for rotation, value in zip(split(solution), metric, strict=True)
        ],
    )
    density_direction = np.array([value["density"] for value in density])
    induced = model["induced"](density_direction)
    weights = run(
        "weights",
        [
            {"orbital_motion": value["orbital_motion"], "induced_fock": matrix}
            for value, matrix in zip(density, induced, strict=True)
        ],
    )
    return density_direction, np.array([value["d_weights"] for value in weights])


CASES = [
    (4, (2,)),
    (4, (2, 1)),
    (3, (0, 2)),
    (3, (3, 0)),
    (3, (0,)),
    (3, (3,)),
    (12, (5, 4)),
]


@pytest.mark.parametrize("nao,counts", CASES)
@pytest.mark.parametrize("degenerate", [False, True])
def test_compiler_maps_against_displaced_scf(
    nao: int, counts: tuple[int, ...], degenerate: bool
) -> None:
    model = _model(nao, counts, degenerate=degenerate)
    actual = _compiler_response(model)
    for step in (1e-4, 3e-4):
        for computed, expected in zip(actual, _oracle(model, step), strict=True):
            np.testing.assert_allclose(computed, expected, atol=3e-8, rtol=3e-7)


def test_compiler_preserves_cross_spin_response() -> None:
    model = _model(4, (2, 1))
    model["overlap_direction"][:] = 0
    actual = _compiler_response(model)
    assert np.linalg.norm(actual[0][1]) > 1e-6
    for computed, expected in zip(actual, _oracle(model, 1e-4), strict=True):
        np.testing.assert_allclose(computed, expected, atol=3e-8, rtol=3e-7)


def test_aot_generation_is_build_only_and_deterministic(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = """
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('generativeqc', 'cupy', 'torch', 'pyscf'):
            raise AssertionError('runtime import: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = [sys.argv[1], '--output', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    outputs = [tmp_path / name for name in ("first.cuh", "second.cuh")]
    for output in outputs:
        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(root / "tools/generate_ks_cpks.py"),
                str(output),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def test_affine_lowering_rejects_unqualified_domains() -> None:
    space = IndexSpace("example", "ao", 2)
    spec = TensorSpec((Index("axis", space),), role="input")
    value = input_tensor("values", spec)
    for program, extents in (
        (Program({"output": value}), {}),
        (Program({"output": slice_tensor(value, ((0, 1),))}), {"example": "size"}),
        (
            Program(
                {"output": constant((1, 2), TensorSpec(spec.indices, role="constant"))}
            ),
            {"example": "size"},
        ),
    ):
        with pytest.raises(ValueError):
            emit_affine_cuda({"example": program}, namespace="test", extents=extents)


@pytest.fixture
def cpks_probe() -> Path:
    value = os.environ.get("GENERATIVEQC_CUDA_CPKS_PROBE")
    if not value:
        pytest.skip("requires the optional native CUDA CPKS probe")
    path = Path(value)
    assert path.is_file()
    return path


def test_native_contract_without_device(cpks_probe: Path) -> None:
    result = subprocess.run(
        [str(cpks_probe), "contract"],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == {"contract": "passed"}


def _native(
    cpks_probe: Path,
    model: dict,
    *,
    layout: str = "row",
    mode: str = "normal",
    iterations: int = 200,
    budget: int = 512 << 20,
    env: dict | None = None,
    reference_only: bool = False,
) -> dict:
    assert os.environ.get("SLURM_JOB_ID"), "real-GPU tests must run through srun"
    coefficients = model["coefficients"]
    if layout == "column":
        coefficients = coefficients.transpose(0, 2, 1)
    counts = model["counts"]
    values = np.concatenate(
        [
            value.reshape(-1)
            for value in (
                coefficients,
                model["energies"],
                model["frozen"],
                model["overlap_direction"],
                model["operation"],
            )
        ]
    )
    result = subprocess.run(
        [
            str(cpks_probe),
            "reference" if reference_only else "solve",
            str(model["nao"]),
            str(len(counts)),
            str(counts[0]),
            str(counts[1] if len(counts) > 1 else 0),
            layout,
            mode,
            str(iterations),
            "3",
            str(budget),
            "1e-12",
        ],
        input=" ".join(format(value, ".17g") for value in values),
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
        env=env,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("nao,counts", CASES)
@pytest.mark.parametrize("layout", ["row", "column"])
@pytest.mark.parametrize("degenerate", [False, True])
def test_native_coupled_cpks_against_displaced_scf(
    cpks_probe: Path, nao: int, counts: tuple[int, ...], layout: str, degenerate: bool
) -> None:
    model = _model(nao, counts, degenerate=degenerate)
    result = _native(cpks_probe, model, layout=layout)
    assert result["cuda_error"] == 0 and result["converged"], result
    assert result["authentications"] == 2
    assert result["residual"] <= result["tolerance"]
    expected = _oracle(model, 1e-4)
    for key, value in zip(("density", "weights"), expected, strict=True):
        actual = np.array(result[key]).reshape(value.shape)
        np.testing.assert_allclose(actual, value, atol=3e-8, rtol=3e-7)
        np.testing.assert_allclose(
            actual, actual.transpose(0, 2, 1), atol=1e-12, rtol=1e-12
        )


@pytest.mark.parametrize(
    "mode",
    [
        "nan-input",
        "asymmetric-input",
        "nan-provider",
        "asymmetric-provider",
        "partial-provider",
        "capture",
        "wrong-stream",
        "fail-provider",
        "late-provider",
        "stale-entry",
        "stale-publish",
        "short-workspace",
        "misaligned-workspace",
        "alias-workspace",
        "alias-output",
        "zero-gap",
    ],
)
def test_failure_does_not_publish(cpks_probe: Path, mode: str) -> None:
    result = _native(cpks_probe, _model(4, (2, 1)), mode=mode)
    assert not result["converged"], result
    assert result["cuda_error"] != 0 or result["status"] in (2, 3), result
    assert all(value == 9817 for value in result["density"])
    assert all(value == -3277 for value in result["weights"])
    if mode == "stale-entry":
        assert result["actions"] == 0
    if mode == "stale-publish":
        assert result["authentications"] == 2 and result["actions"] > 0


def test_exhaustion_and_budget_do_not_publish(cpks_probe: Path) -> None:
    model = _model(4, (2, 1))
    for iterations, budget in ((1, 512 << 20), (200, 8)):
        result = _native(cpks_probe, model, iterations=iterations, budget=budget)
        assert not result["converged"]
        assert all(value == 9817 for value in result["density"])
        assert all(value == -3277 for value in result["weights"])


def test_native_execution_without_python_or_compiler(
    cpks_probe: Path, tmp_path: Path
) -> None:
    environment = {
        **os.environ,
        "PATH": str(tmp_path),
        "PYTHONHOME": str(tmp_path / "missing"),
        "PYTHONPATH": str(tmp_path / "missing"),
        "CUPY_CACHE_DIR": str(tmp_path / "missing"),
    }
    result = _native(cpks_probe, _model(4, (2, 1)), env=environment)
    assert result["cuda_error"] == 0 and result["converged"]


def test_native_preserves_cross_spin_response(cpks_probe: Path) -> None:
    model = _model(4, (2, 1))
    model["overlap_direction"][:] = 0
    result = _native(cpks_probe, model)
    assert result["cuda_error"] == 0 and result["converged"]
    actual = np.array(result["density"]).reshape(2, 4, 4)
    assert np.linalg.norm(actual[1]) > 1e-6
    np.testing.assert_allclose(actual, _oracle(model, 1e-4)[0], atol=3e-8, rtol=3e-7)


@pytest.mark.parametrize("nao,counts", CASES)
def test_reference_weights_reuse_cpks_primal(nao: int, counts: tuple[int, ...]) -> None:
    model = _model(nao, counts)
    for spin, count in enumerate(counts):
        occupied = model["coefficients"][spin, :, :count]
        energies = np.diag(model["energies"][spin, :count])
        actual = execute(
            build_ks_cpks_maps(nao, count, occupation=model["occupation"])[
                "reference_weights"
            ],
            {"occupied": occupied, "occupied_energy_matrix": energies},
        ).outputs["weights"]
        np.testing.assert_allclose(
            actual, model["occupation"] * occupied @ energies @ occupied.T, atol=3e-15
        )


@pytest.mark.parametrize("nao,counts", CASES)
@pytest.mark.parametrize("layout", ("row", "column"))
def test_native_reference_weights_without_python(
    cpks_probe: Path, nao: int, counts: tuple[int, ...], layout: str
) -> None:
    model = _model(nao, counts)
    result = _native(
        cpks_probe,
        model,
        layout=layout,
        reference_only=True,
        env={
            **os.environ,
            "PATH": "/nonexistent",
            "PYTHONHOME": "/nonexistent",
            "PYTHONPATH": "/nonexistent",
        },
    )
    assert result["cuda_error"] == 0 and result["actions"] == 0
    assert result["authentications"] == 2
    assert np.all(np.array(result["density"]) == 9817)
    actual = np.array(result["weights"]).reshape(len(counts), nao, nao)
    for spin, count in enumerate(counts):
        occupied = model["coefficients"][spin, :, :count]
        expected = (
            model["occupation"]
            * (occupied * model["energies"][spin, :count])
            @ occupied.T
        )
        np.testing.assert_allclose(actual[spin], expected, atol=3e-15, rtol=3e-15)


@pytest.mark.parametrize(
    "mode",
    (
        "nan-input",
        "nan-energy",
        "capture",
        "wrong-stream",
        "stale-entry",
        "stale-publish",
        "short-workspace",
        "misaligned-workspace",
        "alias-workspace",
        "alias-output",
    ),
)
def test_native_reference_weights_failure_preserves_output(
    cpks_probe: Path, mode: str
) -> None:
    result = _native(cpks_probe, _model(4, (2, 1)), mode=mode, reference_only=True)
    assert result["cuda_error"] != 0
    assert result["actions"] == 0
    assert np.all(np.array(result["density"]) == 9817)
    assert np.all(np.array(result["weights"]) == -3277)
