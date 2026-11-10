"""Fixed-state semilocal response reuse, bounded fallback and atomic publication."""

from types import SimpleNamespace

import generativeqc.composite_rks_hessian as composite
import numpy as np
import pytest
from generativeqc.composite_rks_hessian import _ao_coefficients, _CompositeGrid
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.stationary_grid_second import (
    SemilocalPointPartials,
    StationaryGridSecondOrder,
)
from generativeqc_compiler.xc.potential import assemble_coefficients


def _grid(spins: int, *, nonlocal_correlation: bool = False) -> _CompositeGrid:
    """Bind small physical AO fields without an SCF or a compiled integral owner."""
    method = resolve_method(
        "WB97M-V" if nonlocal_correlation else "B3LYP",
        spin="polarized" if spins == 2 else "unpolarized",
    )
    generator = np.random.default_rng(2844)
    grid = object.__new__(_CompositeGrid)
    grid.model = StationaryGridSecondOrder(
        method.primitives[0].functional,
        method.primitives[-1] if nonlocal_correlation else None,
        budget_bytes=64 << 20,
    )
    grid.jets = generator.normal(size=(4, 7, 3)) * 0.3
    grid.zero_jets = np.zeros_like(grid.jets)
    grid.state = SimpleNamespace(
        density=np.array([np.eye(3) * (0.4 + spin * 0.2) for spin in range(spins)])
    )
    grid.grid = SimpleNamespace(
        points=generator.normal(size=(7, 3)), weights=np.linspace(0.2, 0.9, 7)
    )
    grid.feature_budget_bytes = 32 << 20
    grid.fields = np.vstack(
        (grid.feature_fields(), grid.grid.weights, grid.grid.points.T)
    )
    grid._point_kernel = None
    grid.statistics = {
        "actions": 0,
        "seconds": 0.0,
        "pair_evaluations": 0,
        "point_kernel_bytes": 0,
        "point_kernel_reserved_bytes": 0,
        "point_kernel_builds": 0,
        "point_kernel_reuses": 0,
        "point_kernel_fallbacks": 0,
    }
    return grid


def _direct(grid: _CompositeGrid, density: np.ndarray) -> np.ndarray:
    moving = np.zeros_like(grid.fields)
    moving[: grid.model.feature_count] = grid.feature_fields(
        left=grid.zero_jets, delta_density=density
    )
    coefficients = grid.model.evaluate(grid.fields, moving).directional_coefficients
    return grid.public_fock(
        assemble_coefficients(
            grid.jets, _ao_coefficients(coefficients), np.ones(len(grid.grid.points))
        )
    )


@pytest.mark.parametrize("spins", (1, 2))
def test_cached_actions_match_raw_point_ad_and_multistep_primal_differences(
    spins: int,
) -> None:
    grid = _grid(spins)
    generator = np.random.default_rng(2845)
    reserve = 2 * grid.model.feature_count**2 * len(grid.grid.points) * 8
    budgets = (grid.feature_budget_bytes, grid.model.budget_bytes)
    for spin in range(spins):
        density = np.zeros_like(grid.state.density)
        trial = generator.normal(size=(3, 3)) * 0.02
        density[spin] = trial + trial.T
        expected = _direct(grid, density)
        actual = grid.apply(density)
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-12)
        np.testing.assert_allclose(
            grid.apply_transpose(density), actual, atol=0, rtol=0
        )
        moving = np.zeros_like(grid.fields)
        moving[: grid.model.feature_count] = grid.feature_fields(
            left=grid.zero_jets, delta_density=density
        )
        for step in (1e-3, 3e-4):
            plus, minus = (
                grid.model.evaluate(
                    grid.fields + sign * step * moving, np.zeros_like(moving)
                ).coefficients
                for sign in (1, -1)
            )
            finite = grid.public_fock(
                assemble_coefficients(
                    grid.jets,
                    _ao_coefficients((plus - minus) / (2 * step)),
                    np.ones(len(grid.grid.points)),
                )
            )
            np.testing.assert_allclose(actual, finite, atol=2e-10, rtol=2e-6)
        if spins == 2:
            assert np.linalg.norm(actual[1 - spin]) > 1e-7
    assert grid.statistics["point_kernel_builds"] == 1
    assert grid.statistics["point_kernel_reuses"] >= spins
    assert (grid.feature_budget_bytes, grid.model.budget_bytes) == tuple(
        budget - reserve for budget in budgets
    )
    assert not grid._point_kernel.flags.writeable


def test_cache_capacity_declines_without_changing_original_response() -> None:
    grid = _grid(2)
    grid.feature_budget_bytes = grid.model.feature_count**2 * len(grid.grid.points) * 8
    budgets = (grid.feature_budget_bytes, grid.model.budget_bytes)
    density = np.full_like(grid.state.density, 0.01)
    expected = _direct(grid, density)
    np.testing.assert_allclose(grid.apply(density), expected, atol=0, rtol=0)
    assert grid._point_kernel is None
    assert (grid.feature_budget_bytes, grid.model.budget_bytes) == budgets
    assert grid.statistics["point_kernel_fallbacks"] > 0


@pytest.mark.parametrize("failure", (MemoryError, RuntimeError))
def test_late_candidate_failure_restores_both_budgets_and_allows_recovery(
    monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    grid = _grid(2)
    density = np.full_like(grid.state.density, 0.01)
    expected = _direct(grid, density)
    budgets = (grid.feature_budget_bytes, grid.model.budget_bytes)
    original = grid.evaluate
    pending = True

    def fail_once(direction: np.ndarray) -> object:
        nonlocal pending
        if pending and np.all(direction[2] == 1):
            pending = False
            raise failure("injected point-kernel candidate failure")
        return original(direction)

    monkeypatch.setattr(grid, "evaluate", fail_once)
    if failure is MemoryError:
        np.testing.assert_allclose(grid.apply(density), expected, atol=0, rtol=0)
    else:
        with pytest.raises(RuntimeError, match="injected"):
            grid.apply(density)
    assert grid._point_kernel is None
    assert (grid.feature_budget_bytes, grid.model.budget_bytes) == budgets
    np.testing.assert_allclose(grid.apply(density), expected, atol=2e-13, rtol=2e-12)
    assert grid._point_kernel is not None


def test_nonlocal_pair_response_keeps_its_original_operator() -> None:
    grid = _grid(2, nonlocal_correlation=True)
    density = np.full_like(grid.state.density, 0.01)
    expected = _direct(grid, density)
    np.testing.assert_allclose(grid.apply(density), expected, atol=0, rtol=0)
    assert grid._point_kernel is None
    assert grid.statistics["pair_evaluations"] > 0


def test_retained_kernel_does_not_displace_required_ao_feature_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _grid(2)
    density = np.full_like(grid.state.density, 0.01)
    expected = _direct(grid, density)
    budgets = (grid.feature_budget_bytes, grid.model.budget_bytes)
    original = composite.features
    pending = True

    def constrained_features(*arguments: object, **options: object) -> np.ndarray:
        nonlocal pending
        if pending and grid._point_kernel is not None:
            pending = False
            raise MemoryError("AO feature workspace unavailable with retained panel")
        return original(*arguments, **options)

    monkeypatch.setattr(composite, "features", constrained_features)
    np.testing.assert_allclose(grid.apply(density), expected, atol=0, rtol=0)
    assert grid._point_kernel is None
    assert (grid.feature_budget_bytes, grid.model.budget_bytes) == budgets
    assert grid.statistics["point_kernel_bytes"] == 0
    assert grid.statistics["point_kernel_reserved_bytes"] == 0
    np.testing.assert_allclose(grid.apply(density), expected, atol=2e-13, rtol=2e-12)


def test_retained_panel_does_not_displace_required_geometry_ad_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _grid(2)
    budgets = (grid.feature_budget_bytes, grid.model.budget_bytes)
    direction = np.zeros_like(grid.fields)
    direction[-4:] = 0.01
    expected = grid.evaluate(direction)
    grid.apply(np.full_like(grid.state.density, 0.01))
    original = grid.model.evaluate

    def constrained_partials(fields: np.ndarray, moving: np.ndarray) -> object:
        if grid._point_kernel is not None:
            raise MemoryError("geometry AD tile unavailable with retained panel")
        return original(fields, moving)

    monkeypatch.setattr(grid.model, "evaluate", constrained_partials)
    actual = grid.evaluate(direction)
    np.testing.assert_array_equal(actual.coefficients, expected.coefficients)
    np.testing.assert_array_equal(
        actual.directional_coefficients, expected.directional_coefficients
    )
    assert grid._point_kernel is None
    assert (grid.feature_budget_bytes, grid.model.budget_bytes) == budgets


def test_native_point_provider_keeps_its_physical_direction_contract() -> None:
    grid = _grid(1)

    def point_partials(
        fields: np.ndarray, direction: np.ndarray
    ) -> SemilocalPointPartials:
        if np.max(abs(direction)) > 0.5:
            raise FloatingPointError(
                "outside the provider's admitted directional domain"
            )
        return SemilocalPointPartials(
            np.sum(fields**2, axis=0) * 0.5, fields.copy(), direction.copy()
        )

    provider = SimpleNamespace(
        functional_identity=grid.model.functional.identity,
        identity="bounded-physical-direction-test-provider",
        workspace_bytes_per_point=1024,
        evaluate=point_partials,
    )
    grid.model = StationaryGridSecondOrder(
        grid.model.functional, point_provider=provider
    )
    density = np.full_like(grid.state.density, 0.01)
    np.testing.assert_allclose(
        grid.apply(density), _direct(grid, density), atol=0, rtol=0
    )
    assert grid._point_kernel is None


def test_kernel_is_not_shared_between_distinct_fixed_states() -> None:
    first, second = _grid(2), _grid(2)
    second.fields[0] *= 1.1
    density = np.full_like(first.state.density, 0.01)
    first_result, second_result = first.apply(density), second.apply(density)
    assert not np.shares_memory(first._point_kernel, second._point_kernel)
    assert np.max(abs(first_result - second_result)) > 1e-7
    density[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        first.apply(density)


def test_overflow_in_a_cached_response_is_not_published() -> None:
    grid = _grid(2)
    grid.apply(np.full_like(grid.state.density, 0.01))
    grid._point_kernel = np.full_like(grid._point_kernel, np.finfo(float).max)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(FloatingPointError, match="nonfinite.*response"),
    ):
        grid.apply(np.full_like(grid.state.density, 10.0))
