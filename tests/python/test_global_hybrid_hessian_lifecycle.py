"""Failure and replay contracts for the low-level full-range spherical owner."""

import typing
from dataclasses import replace
from pathlib import Path

import generativeqc.composite_rks_hessian as composite
import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc.response_problem import ResponseSolveError
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.method import resolve_method
from test_integral_autodiff_second import require_native_library  # noqa: F401

ATOMS = [("H", (0.03, -0.02, -0.72)), ("H", (0.11, 0.13, 0.72))]
DIRECTION = np.array([[0.02, -0.03, -0.1], [-0.02, 0.03, 0.1]])
GRID = GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
SOLVER = GMRESOptions(atol=1e-12, rtol=1e-11)


@pytest.fixture(params=("B3LYP", "PBE0"))
def calculator(request: pytest.FixtureRequest) -> typing.Iterator[Calculator]:
    """Keep the physical MethodIR and fresh native ownership in each case."""
    calculator = Calculator(
        method=resolve_method(request.param),
        device="cpu",
        basis_representation="spherical",
        ks_options=KsOptions(grid=GRID),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
    )
    try:
        yield calculator
    finally:
        calculator.clear_cache()


def test_mismatched_geometry_and_same_size_basis_reject_without_poisoning_batch(
    calculator: Calculator,
) -> None:
    """Neither an equal AO count nor a successful SCF authenticates another basis."""
    shifted = [
        (symbol, np.asarray(position) + offset)
        for (symbol, position), offset in zip(ATOMS, 0.1 * DIRECTION, strict=True)
    ]
    with calculator.prepare_batch([ATOMS], warm_start=False) as batch:
        batch.execute(strict=True, properties=("energy",))
        with NativeAO(ATOMS, representation="spherical") as basis:
            altered_shells = tuple(
                replace(
                    shell,
                    primitives=tuple(
                        replace(primitive, exponent=1.01 * primitive.exponent)
                        for primitive in shell.primitives
                    ),
                )
                for shell in basis.shells
            )
            with (
                NativeAO(shifted, representation="spherical") as stale_geometry,
                NativeAO(
                    ATOMS, basis=altered_shells, representation="spherical"
                ) as stale_basis,
            ):
                assert basis.nao == stale_basis.nao == stale_geometry.nao
                for invalid_basis in (stale_geometry, stale_basis):
                    with pytest.raises(ValueError):
                        composite.CompositeRKSResponse.from_native(batch, invalid_basis)
            with composite.CompositeRKSResponse.from_native(batch, basis) as operator:
                operator.validate_current()
        with pytest.raises(ValueError):
            operator.validate_current()


def test_changed_geometry_replay_and_failed_actions_leave_no_stale_result(
    calculator: Calculator, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recover on the live state, and reject old owners after actual batch replay."""
    positions = np.asarray([position for _, position in ATOMS])
    changed_positions = positions + 0.8 * DIRECTION
    changed_atoms = [
        (symbol, position)
        for (symbol, _), position in zip(ATOMS, changed_positions, strict=True)
    ]
    with calculator.prepare_batch([ATOMS], warm_start=False) as batch:
        batch.execute(strict=True, properties=("energy",))
        with (
            NativeAO(ATOMS, representation="spherical") as basis,
            composite.CompositeRKSResponse.from_native(batch, basis) as original,
        ):
            initial = composite.composite_rks_hvp(
                original, DIRECTION, cache=tmp_path, solver_options=SOLVER
            )
            batch.execute([changed_positions], strict=True, properties=("energy",))
            with pytest.raises(ValueError):
                original.validate_current()
        with NativeAO(changed_atoms, representation="spherical") as basis:
            with pytest.raises(MemoryError):
                composite.CompositeRKSResponse.from_native(
                    batch, basis, grid_budget_bytes=1
                )
            with composite.CompositeRKSResponse.from_native(batch, basis) as current:
                with pytest.raises(ResponseSolveError):
                    composite.composite_rks_hvp(
                        current,
                        DIRECTION,
                        cache=tmp_path,
                        solver_options=GMRESOptions(max_workspace_bytes=1),
                    )
                with pytest.raises(ValueError, match="output_budget_bytes"):
                    composite.composite_rks_hessian(current, output_budget_bytes=1)
                original_second = composite.generated_weighted_second_integral_hvp

                def fail_overlap(
                    topology: typing.Any,
                    source_name: str,
                    *args: typing.Any,
                    **kwargs: typing.Any,
                ) -> typing.Any:
                    """Fail a required source after response without altering its peers."""
                    if source_name == "overlap_pulay":
                        raise FloatingPointError("injected overlap assembly failure")
                    return original_second(topology, source_name, *args, **kwargs)

                with monkeypatch.context() as patch:
                    patch.setattr(
                        composite,
                        "generated_weighted_second_integral_hvp",
                        fail_overlap,
                    )
                    with pytest.raises(FloatingPointError, match="injected overlap"):
                        composite.composite_rks_hvp(
                            current, DIRECTION, cache=tmp_path, solver_options=SOLVER
                        )
                current.validate_current()
                changed = composite.composite_rks_hvp(
                    current, DIRECTION, cache=tmp_path, solver_options=SOLVER
                )
                assert np.max(abs(changed.value - initial.value)) > 1e-4
                assert changed.diagnostics["displaced_scf_evaluations"] == 0
                assert not changed.diagnostics["public_calculator_endpoint"]
                batch.execute([positions], strict=True, properties=("energy",))
                with pytest.raises(ValueError):
                    current.validate_current()
        with (
            NativeAO(ATOMS, representation="spherical") as basis,
            composite.CompositeRKSResponse.from_native(batch, basis) as restored,
        ):
            replay = composite.composite_rks_hvp(
                restored, DIRECTION, cache=tmp_path, solver_options=SOLVER
            )
            np.testing.assert_allclose(replay.value, initial.value, atol=2e-11, rtol=0)
            np.testing.assert_array_equal(initial.direction, DIRECTION)


@pytest.mark.parametrize("functional", ("B3LYP", "PBE0"))
def test_nonconverged_scf_cannot_publish_stationary_response(functional: str) -> None:
    """An explicitly exhausted native SCF is not an admissible response reference."""
    atoms = [("He", (0.03, -0.02, -0.72)), ("H", (0.11, 0.13, 0.72))]
    calculator = Calculator(
        method=resolve_method(functional),
        device="cpu",
        basis_representation="spherical",
        ks_options=KsOptions(grid=GRID),
        max_iterations=1,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )
    try:
        with (
            calculator.prepare_batch([atoms], charges=[1], warm_start=False) as batch,
            NativeAO(atoms, representation="spherical", charge=1) as basis,
        ):
            result = batch.execute(strict=False, properties=("energy",))
            assert not result.items[0].converged
            with pytest.raises(RuntimeError):
                composite.CompositeRKSResponse.from_native(batch, basis)
    finally:
        calculator.clear_cache()
