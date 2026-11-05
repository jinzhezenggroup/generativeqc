"""Low-level coupled-spin qualification, not public spherical Ni admission."""

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._dft_gradient import StationaryKsState
from generativeqc.composite_rks_hessian import (
    CompositeRKSResponse,
    CompositeUKSResponse,
    composite_rks_hessian,
    composite_rks_hvp,
)
from generativeqc.response_problem import ResponseSolveError
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.method import resolve_method
from test_dft_complete_cpu import independent_global_hybrid_gradient
from test_integral_autodiff_second import require_native_library  # noqa: F401

ATOMS = [("H", (0, 0.1, -1.6)), ("H", (0.3, -0.2, 0)), ("H", (-0.1, 0.4, 1.6))]
DIRECTION = np.array([[0.02, -0.03, 0.1], [-0.04, 0.01, -0.02], [0.02, 0.02, -0.08]])
SOLVER = GMRESOptions(atol=1e-12, rtol=1e-11)


@pytest.fixture(params=("spherical", "cartesian"))
def full_range_uks(request: pytest.FixtureRequest) -> Iterator[tuple]:
    """Keep two occupied spin spaces and a physical, label-independent MethodIR."""
    representation = request.param
    method = replace(
        resolve_method("B3LYP", spin="polarized"),
        identifier="coupled-full-range-low-level-qualification",
    )
    calculator = Calculator(
        method=method,
        device="cpu",
        basis_representation=representation,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
    )
    assert not calculator.capabilities.supported_second_order
    try:
        yield calculator, method, representation
    finally:
        calculator.clear_cache()


def _independent(basis: NativeAO, state: StationaryKsState, method: object) -> tuple:
    """Solve an independent UKS state, preserving the physical VWN-RPA graph."""
    pytest.importorskip("pyscf")
    functional = method.primitives[0].functional
    code = "+".join(f"{float(weight)}*{name}" for name, weight in functional.components)
    code += f"+{float(method.primitives[1].coefficient)}*HF"
    return independent_global_hybrid_gradient(
        basis,
        state,
        "full-range-uks",
        xc=code,
        return_mean_field=True,
    )


def test_full_range_coupled_response_matches_independent_spin_blocks(
    full_range_uks: tuple,
) -> None:
    """Match both spin densities and same/cross-spin actions, not just energy."""
    calculator, method, representation = full_range_uks
    with calculator.prepare_batch(
        [ATOMS], multiplicities=[2], warm_start=False
    ) as batch:
        point = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        with (
            NativeAO(
                ATOMS, basis="sto-3g", representation=representation, multiplicity=2
            ) as basis,
            CompositeRKSResponse.from_native(batch, basis) as operator,
        ):
            assert isinstance(operator, CompositeUKSResponse)
            assert operator.layout.occupied == (2, 1)
            assert operator.plan.method.semantic_payload() == method.semantic_payload()
            energy, gradient, reference = _independent(basis, operator.state, method)
            assert point.energy == pytest.approx(energy, abs=2e-10)
            np.testing.assert_allclose(point.forces, -gradient, atol=2e-9, rtol=0)
            np.testing.assert_allclose(
                operator.state.overlap, reference.get_ovlp(), atol=2e-12, rtol=0
            )
            np.testing.assert_allclose(
                operator.state.density, reference.make_rdm1(), atol=2e-9, rtol=0
            )
            np.testing.assert_allclose(
                np.einsum("sij,ji->s", operator.state.density, operator.state.overlap),
                [2, 1],
                atol=2e-12,
                rtol=0,
            )
            assert (
                np.linalg.norm(reference.get_grad(reference.mo_coeff, reference.mo_occ))
                < 2e-9
            )
            response = reference.gen_response(hermi=1)
            generator = np.random.default_rng(2844)
            for spin in range(2):
                delta = np.zeros_like(operator.state.density)
                trial = generator.normal(size=(basis.nao, basis.nao)) * 0.03
                delta[spin] = trial + trial.T
                actual = operator.induced_fock(delta)
                np.testing.assert_allclose(actual, response(delta), atol=2e-9, rtol=0)
                assert np.linalg.norm(actual[1 - spin]) > 1e-6


def test_full_range_uks_raw_hessian_matches_independent_moving_grid_gradients(
    full_range_uks: tuple, tmp_path: Path, record_property: object
) -> None:
    """Freeze raw 2e-8 invariants and independent 1e-5 Eh/Bohr² matrix gates.

    Each independent reference reconverges without a native density seed on
    the identical explicit moving grid. These s-shell radicals deliberately
    do not qualify external Ni d/f bases, the default grid, or public endpoints.
    """
    calculator, method, representation = full_range_uks
    with calculator.prepare_batch(
        [ATOMS], multiplicities=[2], warm_start=False
    ) as batch:
        batch.execute(strict=True, properties=("energy",))
        with (
            NativeAO(
                ATOMS, basis="sto-3g", representation=representation, multiplicity=2
            ) as basis,
            CompositeRKSResponse.from_native(batch, basis) as operator,
        ):
            result = composite_rks_hvp(
                operator, DIRECTION, cache=tmp_path, solver_options=SOLVER
            )
            hessian = composite_rks_hessian(
                operator, cache=tmp_path, solver_options=SOLVER
            )
            assert not result.diagnostics["public_calculator_endpoint"]
            assert not hessian.diagnostics["public_calculator_endpoint"]
            assert result.diagnostics["response_residual_norm"] < 2e-10
            assert result.diagnostics["nuclear_response_solves"] == 1
            assert result.diagnostics["displaced_scf_evaluations"] == 0
            assert result.diagnostics["pair_evaluations"] == 0
            assert set(result.components) == {
                "one_electron",
                "coulomb",
                "exact_exchange",
                "xc_ao",
                "xc_grid",
                "xc_weight",
                "overlap_pulay",
                "nuclear",
            }
            np.testing.assert_allclose(
                sum(result.components.values()), result.value, atol=2e-12, rtol=0
            )
            np.testing.assert_allclose(
                hessian.matrix, hessian.matrix.T, atol=2e-8, rtol=0
            )
            np.testing.assert_allclose(
                hessian.matrix @ np.tile(np.eye(3), (len(ATOMS), 1)),
                0,
                atol=2e-8,
                rtol=0,
            )
            np.testing.assert_allclose(
                hessian.matrix @ DIRECTION.reshape(-1),
                result.value.reshape(-1),
                atol=2e-8,
                rtol=0,
            )
            second = np.random.default_rng(2944).normal(size=DIRECTION.shape) * 0.03
            other = composite_rks_hvp(
                operator, second, cache=tmp_path, solver_options=SOLVER
            )
            assert np.sum(second * result.value) == pytest.approx(
                np.sum(DIRECTION * other.value), abs=2e-8
            )
            record_property("hvp_diagnostics", str(result.diagnostics))
            record_property("hessian_diagnostics", str(hessian.diagnostics))

    directions = np.eye(DIRECTION.size).reshape(DIRECTION.size, *DIRECTION.shape)
    for step in (1e-3, 5e-4, 3e-4):
        independent = np.empty_like(hessian.matrix)
        for column, direction in enumerate(directions):
            gradients = []
            for sign in (-1, 1):
                atoms = [
                    (element, np.asarray(position) + sign * step * motion)
                    for (element, position), motion in zip(
                        ATOMS, direction, strict=True
                    )
                ]
                with calculator.prepare_batch(
                    [atoms], multiplicities=[2], warm_start=False
                ) as batch:
                    batch.execute(strict=True, properties=("energy",))
                    with NativeAO(
                        atoms,
                        basis="sto-3g",
                        representation=representation,
                        multiplicity=2,
                    ) as basis:
                        state = StationaryKsState.from_native(batch, basis)
                        try:
                            _, gradient, reference = _independent(basis, state, method)
                            np.testing.assert_allclose(
                                state.density,
                                reference.make_rdm1(),
                                atol=2e-9,
                                rtol=0,
                            )
                            gradients.append(gradient)
                        finally:
                            state._source.close()
            independent[:, column] = (
                (gradients[1] - gradients[0]) / (2 * step)
            ).reshape(-1)
        error = float(np.max(abs(hessian.matrix - independent)))
        record_property(f"independent_raw_error_{step}", error)
        assert error < 1e-5


def test_full_range_uks_wrong_spin_state_and_failed_response_recover(
    full_range_uks: tuple, tmp_path: Path
) -> None:
    """Keep charge/spin identities, generation invalidation and bounded failures."""
    calculator, _, representation = full_range_uks
    changed_positions = np.asarray([position for _, position in ATOMS]) + DIRECTION
    changed_atoms = [
        (symbol, position)
        for (symbol, _), position in zip(ATOMS, changed_positions, strict=True)
    ]
    with calculator.prepare_batch(
        [ATOMS], multiplicities=[2], warm_start=False
    ) as batch:
        batch.execute(strict=True, properties=("energy",))
        for charge, multiplicity in ((2, 2), (0, 4)):
            with (
                NativeAO(
                    ATOMS,
                    representation=representation,
                    charge=charge,
                    multiplicity=multiplicity,
                ) as wrong_state,
                pytest.raises(ValueError),
            ):
                CompositeRKSResponse.from_native(batch, wrong_state)
        with (
            NativeAO(ATOMS, representation=representation, multiplicity=2) as basis,
            CompositeRKSResponse.from_native(batch, basis) as original,
        ):
            assert original.layout.occupied == (2, 1)
            batch.execute([changed_positions], strict=True, properties=("energy",))
            with pytest.raises(ValueError):
                original.validate_current()
        with NativeAO(
            changed_atoms, representation=representation, multiplicity=2
        ) as basis:
            with pytest.raises(MemoryError):
                CompositeRKSResponse.from_native(batch, basis, grid_budget_bytes=1)
            with CompositeRKSResponse.from_native(batch, basis) as current:
                assert current.layout.occupied == (2, 1)
                with pytest.raises(ResponseSolveError):
                    composite_rks_hvp(
                        current,
                        DIRECTION,
                        cache=tmp_path,
                        solver_options=GMRESOptions(max_workspace_bytes=1),
                    )
                with pytest.raises(ResponseSolveError):
                    composite_rks_hvp(
                        current,
                        DIRECTION,
                        cache=tmp_path,
                        solver_options=GMRESOptions(
                            atol=1e-14, rtol=1e-13, max_iterations=1
                        ),
                    )
                with pytest.raises(ValueError, match="output_budget_bytes"):
                    composite_rks_hessian(current, output_budget_bytes=1)
                recovered = composite_rks_hvp(
                    current, DIRECTION, cache=tmp_path, solver_options=SOLVER
                )
                assert recovered.diagnostics["response_residual_norm"] < 2e-10
                assert not recovered.diagnostics["public_calculator_endpoint"]
                current.validate_current()
            with pytest.raises(ValueError):
                current.validate_current()
