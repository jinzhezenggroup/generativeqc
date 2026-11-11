"""Molecular adapter admission and ownership, not mocked chemistry qualification."""

from __future__ import annotations

import typing
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import Calculator
from generativeqc.saddle import SaddleOptions, optimize_saddle
from generativeqc_compiler.method import resolve_method

ATOMS = (("O", (0, 0, 0)), ("H", (1.4, 1.0, 0)), ("H", (-1.4, 1.0, 0)))
MASSES = (15.99491461957, 1.00782503223, 1.00782503223)


@pytest.fixture
def calculator(monkeypatch: pytest.MonkeyPatch) -> Calculator:
    try:
        result = Calculator(
            method=resolve_method("B3LYP"),
            device="cpu",
            basis="sto-3g",
            basis_representation="spherical",
        )
    except (OSError, RuntimeError) as error:
        pytest.skip(f"native library unavailable: {error}")

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("admission refusal started a production SCF")

    monkeypatch.setattr(result, "prepare_batch", forbidden)
    return result


@pytest.mark.parametrize(
    "changes",
    [
        {"multiplicity": 3},
        {"charge": True},
        {"masses": (1, 1)},
        {"masses": (16, 1, 0)},
        {"minimum_occupied_overlap": 0},
        {"minimum_occupied_overlap": float("nan")},
        {"root_overlap_budget_bytes": 0},
        {"output_budget_bytes": 1},
        {"grid_budget_bytes": -1},
        {"options": SaddleOptions()},
        {"evaluation_progress": 1},
    ],
)
def test_fail_closed_before_scf(
    calculator: Calculator, changes: dict[str, typing.Any]
) -> None:
    request = {"masses": MASSES, **changes}
    with pytest.raises((ValueError, TypeError, MemoryError, NotImplementedError)):
        optimize_saddle(calculator, ATOMS, **request)


def test_unqualified_hessian_before_scf(
    calculator: Calculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    monkeypatch.setattr(
        calculator,
        "_capabilities",
        replace(calculator.capabilities, supported_second_order=frozenset()),
    )
    with pytest.raises(NotImplementedError, match="public hessian"):
        optimize_saddle(calculator, ATOMS, masses=MASSES)


def test_cancel_before_preparation(calculator: Calculator) -> None:
    result = optimize_saddle(calculator, ATOMS, masses=MASSES, cancelled=lambda: True)
    assert not result.converged
    assert result.optimizer.receipt["status"] == "cancelled"
    assert not result.optimizer.receipt["has_valid_state"]
    assert result.harmonics is None
    assert not result.diagnostics["evaluations"]
    assert result.diagnostics["scf_iterations"] == 0


def test_preparation_failure_is_a_retained_result(
    calculator: Calculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise MemoryError("prepared model capacity")

    monkeypatch.setattr(calculator, "prepare_batch", failed)
    result = optimize_saddle(calculator, ATOMS, masses=MASSES)
    assert not result.converged
    assert result.optimizer.receipt["status"] == "evaluation_failed"
    assert result.optimizer.failures == (
        ("energy_gradient", "MemoryError", "prepared model capacity"),
    )
    assert not result.diagnostics["work_counters_complete"]
    assert result.diagnostics["evaluations"][-1]["status"] == "failed"
    np.testing.assert_array_equal(
        result.optimizer.coordinates, [atom[1] for atom in ATOMS]
    )


def test_model_mutation_in_stage_observer_fails_before_scf(
    calculator: Calculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(event: typing.Mapping[str, typing.Any]) -> None:
        monkeypatch.setattr(calculator, "_model_signature", lambda: "changed model")

    result = optimize_saddle(
        calculator, ATOMS, masses=MASSES, evaluation_progress=mutate
    )
    assert not result.converged
    assert "model changed" in result.optimizer.failures[0][2]
    assert result.optimizer.receipt["energy_gradient_calls"] == 1


def test_harmonic_publication_failure_retains_native_geometry(
    calculator: Calculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc import _molecular_saddle
    from generativeqc.saddle import SaddleResult

    positions = np.array([atom[1] for atom in ATOMS])
    native = SaddleResult(
        positions,
        np.zeros_like(positions),
        np.eye(9),
        np.zeros_like(positions),
        np.ones(3),
        {
            "has_valid_state": True,
            "stationary_verified": True,
            "status": "first_order_saddle",
        },
        "synthetic publication control",
        (),
        0.1,
    )

    def failed(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        raise MemoryError("harmonic workspace")

    monkeypatch.setattr(
        _molecular_saddle, "correct_saddle", lambda *args, **kwargs: native
    )
    monkeypatch.setattr("generativeqc.vibrations.analyze_harmonic_modes", failed)
    result = optimize_saddle(calculator, ATOMS, masses=MASSES)
    assert not result.converged and result.optimizer.converged
    assert result.harmonics is None
    assert result.diagnostics["final_status"] == "harmonic_analysis_failed"
    assert result.diagnostics["harmonic_failure"] == (
        "MemoryError",
        "harmonic workspace",
    )
    np.testing.assert_array_equal(result.optimizer.coordinates, positions)


def test_closed_owners_before_d3_publication(
    calculator: Calculator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An executable lifecycle test; synthetic derivatives are not molecular evidence."""
    from contextlib import contextmanager
    from dataclasses import dataclass
    from types import MappingProxyType

    from generativeqc import _molecular_saddle
    from generativeqc.composite_rks_hessian import CompositeRKSResponse
    from generativeqc.saddle import SaddleProgress, SaddleResult

    live = set()
    initial_occupied = np.eye(3)[:, :2]
    trial_occupied = initial_occupied.copy()
    trial_occupied[:, 0] = (0.95, 0, np.sqrt(1 - 0.95**2))
    later_occupied = trial_occupied.copy()
    later_occupied[2, 0] *= -1
    orbital_sources = iter((initial_occupied, trial_occupied, later_occupied))

    @contextmanager
    def owned(name: str, value: typing.Any) -> typing.Iterator[typing.Any]:
        live.add(name)
        try:
            yield value
        finally:
            live.remove(name)

    def prepare(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        primal = SimpleNamespace(
            energy=-1.0,
            forces=np.zeros((3, 3)),
            iterations=7,
            fock_builds=8,
            converged=True,
            succeeded=True,
            physical_residual_rms=1e-12,
        )
        return owned(
            "batch",
            SimpleNamespace(execute=lambda **kwargs: SimpleNamespace(items=(primal,))),
        )

    def response(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        occupied = next(orbital_sources)
        coefficients = np.column_stack((occupied, np.eye(3)[:, 2]))
        state = SimpleNamespace(
            identity=SimpleNamespace(spin="unpolarized", geometry_identity="geometry"),
            coefficients=coefficients[None],
            occupations=np.array([[2, 2, 0]]),
            grid=SimpleNamespace(identity="grid"),
        )
        return owned("response", SimpleNamespace(state=state))

    @dataclass(frozen=True)
    class Electronic:
        matrix: np.ndarray
        _diagnostics: typing.Mapping[str, typing.Any]

        @property
        def diagnostics(self) -> typing.Mapping[str, typing.Any]:
            return self._diagnostics

    def build(*args: typing.Any, **kwargs: typing.Any) -> Electronic:
        assert live == {"batch", "ao", "response"}
        return Electronic(
            np.eye(9),
            {
                "nuclear_response_solves": 9,
                "response_residual_norms": (1e-12,) * 9,
                "raw_symmetry_error": 0.0,
            },
        )

    def compose(
        electronic: Electronic, *args: typing.Any, **kwargs: typing.Any
    ) -> Electronic:
        assert not live
        return electronic

    def drive(coordinates: typing.Any, **callbacks: typing.Any) -> SaddleResult:
        initial = np.array(coordinates)
        _energy, gradient = callbacks["energy_gradient"](initial)
        matrix = callbacks["hessian"](initial)
        receipt = MappingProxyType(
            {
                "stationary_verified": False,
                "has_valid_state": True,
                "status": "attempt_limit",
            }
        )
        callbacks["progress"](SaddleProgress(initial, receipt, 0.1))
        callbacks["energy_gradient"](initial - 0.01)
        displaced = initial + 0.01
        callbacks["energy_gradient"](displaced)
        callbacks["hessian"](displaced)
        return SaddleResult(
            initial,
            gradient,
            matrix,
            np.zeros_like(initial),
            np.ones(3),
            receipt,
            "unit control",
            (),
            0.2,
        )

    monkeypatch.setattr(calculator, "prepare_batch", prepare)
    monkeypatch.setattr(calculator, "_d3_second_order_spec", object())
    monkeypatch.setattr(
        "generativeqc_compiler.dft.NativeAO", lambda *args, **kwargs: owned("ao", None)
    )
    monkeypatch.setattr(CompositeRKSResponse, "from_native", response)
    monkeypatch.setattr(
        "generativeqc.composite_rks_hessian.composite_rks_hessian", build
    )
    monkeypatch.setattr("generativeqc.d3_hessian.compose_second_order", compose)
    monkeypatch.setattr(
        "generativeqc.overlap.cross_overlap", lambda *args, **kwargs: np.eye(3)
    )
    monkeypatch.setattr(_molecular_saddle, "correct_saddle", drive)
    result = optimize_saddle(calculator, ATOMS, masses=MASSES)
    assert not result.converged
    assert not live
    assert result.diagnostics["scf_iterations"] == 21
    assert result.diagnostics["nuclear_response_solves"] == 18
    overlaps = [
        event["minimum_occupied_overlap"]
        for event in result.diagnostics["evaluations"]
        if event["stage"] == "energy_gradient" and event["status"] == "complete"
    ]
    np.testing.assert_allclose(overlaps, [1, 0.95, 0.95])
