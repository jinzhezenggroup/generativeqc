"""Independent scaled PBE point/grid curvature and native-owner failure gates."""

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._dft_gradient import StationaryKsState
from generativeqc.composite_rks_hessian import (
    _CompositeGrid,
    _NativeSemilocalGridPoints,
)
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.stationary_grid_second import (
    SemilocalPointPartials,
    StationaryGridSecondOrder,
)
from test_global_hybrid_grid_second_order import _fields, _reference
from test_integral_autodiff_second import require_native_library  # noqa: F401

ATOMS = [("H", (0, 0, -0.72)), ("H", (0, 0, 0.72))]
GRID = GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("scaled", (False, True))
def test_native_pbe_grid_curvature_matches_independent_libxc(
    spins: int, scaled: bool
) -> None:
    """Freeze 2e-11 primal/potential and 2e-8 JVP gates for both spin layouts."""
    method = resolve_method("PBE0", spin="polarized" if spins == 2 else "unpolarized")
    if scaled:
        primitive = method.primitives[0]
        functional = replace(
            primitive.functional,
            identifier="scaled-native-pbe",
            components=(
                ("GGA_X_PBE", Fraction(63, 100)),
                ("GGA_C_PBE", Fraction(81, 100)),
            ),
        )
        method = replace(
            method,
            identifier="scaled-native-full-range-pbe",
            primitives=(
                replace(primitive, functional=functional),
                replace(method.primitives[1], coefficient=Fraction(37, 100)),
            ),
        )
    calculator = Calculator(
        method=method, device="cpu", ks_options=KsOptions(grid=GRID)
    )
    with (
        calculator.prepare_batch([ATOMS], warm_start=False) as batch,
        NativeAO(ATOMS, basis="sto-3g") as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        state = StationaryKsState.from_native(batch, basis)
        try:
            functional = method.primitives[0].functional
            provider = _NativeSemilocalGridPoints(state, functional)
            model = StationaryGridSecondOrder(functional, point_provider=provider)
            fields = _fields(spins)
            generator = np.random.default_rng(2224)
            left, right = (generator.normal(size=fields.shape) * 0.03 for _ in range(2))
            actual = model.evaluate(fields, right)
            energy, coefficients = _reference(functional, fields)
            assert actual.energy == pytest.approx(energy, abs=2e-11)
            np.testing.assert_allclose(
                actual.coefficients, coefficients, atol=2e-11, rtol=2e-10
            )
            for step in (3e-3, 1e-3, 3e-4):
                plus = _reference(functional, fields + step * right)[1]
                minus = _reference(functional, fields - step * right)[1]
                np.testing.assert_allclose(
                    actual.directional_coefficients,
                    (plus - minus) / (2 * step),
                    atol=2e-8,
                    rtol=2e-6,
                )
            transpose = model.evaluate(fields, left)
            assert np.sum(left * actual.directional_coefficients) == pytest.approx(
                np.sum(right * transpose.directional_coefficients), abs=2e-12
            )
            assert actual.pair_evaluations == 0
            for value in actual.nonlocal_correlation:
                np.testing.assert_array_equal(value, 0)
        finally:
            state._source.close()


@pytest.mark.parametrize("defect", ("shape", "nonfinite", "overflow"))
def test_point_provider_failure_never_publishes_grid_partials(defect: str) -> None:
    """Reject malformed native outputs and overflow in the weighted contraction."""
    functional = resolve_method("PBE0").primitives[0].functional
    fields = np.ones((9, 1))
    potential = np.zeros((5, 1))
    energy = np.ones(2 if defect == "shape" else 1)
    expected = ValueError if defect == "shape" else FloatingPointError
    if defect == "nonfinite":
        energy[0] = np.nan
    if defect == "overflow":
        fields[5] = 1e308
        potential[0] = 1e308
    point = SemilocalPointPartials(energy, potential, np.zeros_like(potential))
    provider = SimpleNamespace(
        functional_identity=functional.identity,
        identity="failure-injection-point-provider",
        workspace_bytes_per_point=32768,
        evaluate=lambda values, moving: point,
    )
    model = StationaryGridSecondOrder(functional, point_provider=provider)
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(expected):
        model.evaluate(fields, np.zeros_like(fields))


def test_composite_grid_lends_its_reserved_capacity_to_native_ao() -> None:
    """A sufficiently funded grid must not inherit an unrelated 64 MiB AO cap.

    Exercise the native point/grid result with a conservative synthetic basis
    capacity, without allocating a large fixture just to test the byte lease.
    """
    calculator = Calculator(
        method=resolve_method("PBE0"), device="cpu", ks_options=KsOptions(grid=GRID)
    )
    with (
        calculator.prepare_batch([ATOMS], warm_start=False) as batch,
        NativeAO(ATOMS) as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        state = StationaryKsState.from_native(batch, basis)
        try:

            class BudgetedBasis:
                numeric_bytes = 70 << 20

                def __getattr__(self, name: str) -> object:
                    return getattr(basis, name)

                def evaluate(
                    self,
                    points: np.ndarray,
                    order: int,
                    *,
                    budget_bytes: int = 64 << 20,
                ) -> np.ndarray:
                    if budget_bytes < self.numeric_bytes:
                        raise ValueError("AO tile exceeds its numeric capacity")
                    return basis.evaluate(points, order, budget_bytes=budget_bytes)

            reference = _CompositeGrid(
                state, basis, grid_budget_bytes=128 << 20, max_nonlocal_pairs=1
            )
            funded = _CompositeGrid(
                state,
                BudgetedBasis(),
                grid_budget_bytes=128 << 20,
                max_nonlocal_pairs=1,
            )
            assert funded.reference.energy == reference.reference.energy
            np.testing.assert_array_equal(
                funded.reference.coefficients, reference.reference.coefficients
            )
            with pytest.raises(MemoryError, match="AO jet/motion"):
                _CompositeGrid(
                    state,
                    BudgetedBasis(),
                    grid_budget_bytes=64 << 20,
                    max_nonlocal_pairs=1,
                )
        finally:
            state._source.close()


def test_native_pbe_grid_rejects_identity_budget_and_closed_owner() -> None:
    """No invalid provider, insufficient native tile or expired state publishes a result."""
    method = resolve_method("PBE0")
    functional = method.primitives[0].functional
    calculator = Calculator(
        method=method, device="cpu", ks_options=KsOptions(grid=GRID)
    )
    with (
        calculator.prepare_batch([ATOMS], warm_start=False) as batch,
        NativeAO(ATOMS, basis="sto-3g") as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        state = StationaryKsState.from_native(batch, basis)
        try:
            provider = _NativeSemilocalGridPoints(state, functional)
            fields = _fields(1)
            with pytest.raises(ValueError, match="identity mismatch"):
                StationaryGridSecondOrder(
                    replace(functional, identifier="different-identity"),
                    point_provider=provider,
                )
            with pytest.raises(MemoryError, match="native point tile"):
                StationaryGridSecondOrder(
                    functional,
                    point_provider=provider,
                    budget_bytes=fields.nbytes * 32 + 1,
                ).evaluate(fields, fields)
            model = StationaryGridSecondOrder(functional, point_provider=provider)
            vacuum = np.zeros_like(fields)
            vacuum[5] = 1
            result = model.evaluate(vacuum, np.zeros_like(vacuum))
            np.testing.assert_array_equal(result.coefficients, 0)
            np.testing.assert_array_equal(result.directional_coefficients, 0)
            forbidden = np.zeros_like(vacuum)
            forbidden[0] = 1
            with pytest.raises(RuntimeError):
                model.evaluate(vacuum, forbidden)
            batch.close()
            with pytest.raises(RuntimeError):
                model.evaluate(fields, fields)
        finally:
            state._source.close()
