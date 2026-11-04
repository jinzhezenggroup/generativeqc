"""Independent gates for the native full-range-hybrid grid second partials.

These are compiler/source gates, not evidence of a complete molecular Hessian.
The independent Libxc owner is used only by the test oracle.
"""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.stationary_grid_second import (
    StationaryGridSecondOrder,
)


def _fields(spins: int) -> np.ndarray:
    """Keep the independent comparisons away from all density-tail branches."""
    generator = np.random.default_rng(2843)
    fields = generator.normal(size=(5 * spins + 4, 7)) * 0.04
    for channel in range(spins):
        fields[5 * channel] = generator.uniform(0.2, 0.8, 7)
        fields[5 * channel + 4] = generator.uniform(0.1, 0.3, 7)
    fields[5 * spins] = generator.uniform(-0.2, 0.6, 7)
    return fields


def _reference(functional: object, fields: np.ndarray) -> tuple[float, np.ndarray]:
    """Use Libxc's analytic first derivatives, independently pulled to Cartesian fields."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    spins = 2 if functional.spin == "polarized" else 1
    density = np.stack(
        [fields[5 * channel : 5 * channel + 4] for channel in range(spins)]
    )
    code = "+".join(f"{float(weight)}*{name}" for name, weight in functional.components)
    exc, potential, _, _ = libxc.eval_xc(
        code, density if spins == 2 else density[0], spin=spins - 1, deriv=1
    )
    partials = np.zeros_like(fields)
    weights = fields[5 * spins]
    if spins == 1:
        partials[0] = weights * potential[0]
        partials[1:4] = weights * 2 * potential[1] * density[0, 1:4]
    else:
        partials[0] = weights * potential[0][:, 0]
        partials[5] = weights * potential[0][:, 1]
        same_alpha, cross, same_beta = potential[1].T
        partials[1:4] = weights * (
            2 * same_alpha * density[0, 1:4] + cross * density[1, 1:4]
        )
        partials[6:9] = weights * (
            2 * same_beta * density[1, 1:4] + cross * density[0, 1:4]
        )
    partials[5 * spins] = density[:, 0].sum(axis=0) * exc
    return float(np.dot(weights, partials[5 * spins])), partials


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("scaled", (False, True))
def test_grid_second_partials_match_independent_b3lyp_definition(
    spins: int, scaled: bool
) -> None:
    """Freeze 2e-11 primal/gradient and 2e-8 HVP gates before molecular promotion."""
    method = resolve_method("B3LYP", spin="polarized" if spins == 2 else "unpolarized")
    functional = method.primitives[0].functional
    if scaled:
        functional = replace(
            functional,
            identifier="custom-native-full-range-grid",
            components=tuple(
                (name, weight * Fraction(3, 4))
                for name, weight in functional.components
            ),
        )
    fields = _fields(spins)
    generator = np.random.default_rng(2844)
    left, right = (generator.normal(size=fields.shape) * 0.03 for _ in range(2))
    model = StationaryGridSecondOrder(functional, max_pair_evaluations=1)
    result = model.evaluate(fields, right)
    energy, partials = _reference(functional, fields)
    assert result.energy == pytest.approx(energy, abs=2e-11)
    np.testing.assert_allclose(result.coefficients, partials, atol=2e-11, rtol=2e-10)
    for step in (3e-3, 1e-3, 3e-4):
        minus = _reference(functional, fields - step * right)[1]
        plus = _reference(functional, fields + step * right)[1]
        np.testing.assert_allclose(
            result.directional_coefficients,
            (plus - minus) / (2 * step),
            atol=2e-8,
            rtol=2e-6,
        )
    transpose = model.evaluate(fields, left)
    assert np.sum(left * result.directional_coefficients) == pytest.approx(
        np.sum(right * transpose.directional_coefficients), abs=2e-12
    )
    assert result.pair_evaluations == 0
    for value in result.nonlocal_correlation:
        np.testing.assert_array_equal(value, np.zeros_like(fields))
    np.testing.assert_array_equal(result.coefficients[-3:], 0)
    np.testing.assert_array_equal(result.directional_coefficients[-3:], 0)


def test_semilocal_grid_retains_native_vacuum_and_resource_failures() -> None:
    functional = resolve_method("B3LYP").primitives[0].functional
    model = StationaryGridSecondOrder(functional)
    fields = np.zeros((9, 3))
    fields[5] = 1
    fields[0] = (0, 1e-19, 1e-18)
    result = model.evaluate(fields, np.ones_like(fields))
    assert result.energy == 0
    np.testing.assert_array_equal(result.coefficients, 0)
    np.testing.assert_array_equal(result.directional_coefficients, 0)
    with pytest.raises(MemoryError, match="retained grid"):
        StationaryGridSecondOrder(functional, budget_bytes=1).evaluate(fields, fields)
    fields[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.evaluate(fields, fields)
