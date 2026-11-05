"""Independent fixed/JVP cotangents for prepared shared stationary consumers."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc.composite_rks_hessian import _PreparedIntegralWeights
from generativeqc_compiler.method import (
    StationaryHVPPlan,
    StationaryMeanField,
    resolve_method,
)
from generativeqc_compiler.method.spec import RangeSeparatedExchangePrimitive
from generativeqc_compiler.method.stationary_gradient import SCF_POINT_MODEL


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize("functional", ("B3LYP", "PBE0", "CAM-B3LYP", "WB97M-V"))
@pytest.mark.parametrize("sizes", ((1, 3, 6), (91, 3, 10)))
def test_prepared_integral_weights_match_independent_density_products(
    functional: str, spin: str, sizes: tuple[int, ...]
) -> None:
    """Check signed pullbacks and density products below/above one block's cap.

    The larger case crosses the ordinary compiler budget for both all-AO pair
    weights and a Cartesian ffff quartet, without changing any admission limit.
    """
    method = resolve_method(functional, spin=spin)
    plan = StationaryHVPPlan(method, StationaryMeanField(SCF_POINT_MODEL))
    generator = np.random.default_rng(2843)
    spins = 2 if spin == "polarized" else 1
    offsets = np.cumsum((0, *sizes))
    transform = generator.normal(size=(sum(sizes), 9)) * 0.2
    inputs = generator.normal(size=(4, spins, 9, 9))
    inputs += inputs.transpose(0, 1, 3, 2)

    def pullback(density: np.ndarray) -> np.ndarray:
        return transform @ density @ transform.T

    density, moving, weighted, dweighted = map(pullback, inputs)
    operator = SimpleNamespace(
        topology=SimpleNamespace(shell_sizes=sizes, nbf=sum(sizes)),
        integral_basis=SimpleNamespace(density=pullback),
        state=SimpleNamespace(density=inputs[0], weighted_density=inputs[2]),
        plan=plan,
    )
    response = SimpleNamespace(
        density_derivative=inputs[1][0] if spins == 1 else inputs[1],
        energy_weighted_density_derivative=inputs[3][0] if spins == 1 else inputs[3],
    )
    sources = ("one_electron", "overlap_pulay", "coulomb") + tuple(
        name for name in plan.source_names if "exchange" in name
    )
    for source in sources:
        prepared = _PreparedIntegralWeights(operator, response, source)
        if source in ("one_electron", "overlap_pulay"):
            expected, delta = (
                (-weighted.sum(axis=0), -dweighted.sum(axis=0))
                if source == "overlap_pulay"
                else (density.sum(axis=0), moving.sum(axis=0))
            )
            np.testing.assert_allclose(
                prepared.fixed(), expected, atol=2e-14, rtol=2e-14
            )
            np.testing.assert_allclose(
                prepared.moving_weights(), delta, atol=2e-14, rtol=2e-14
            )
            continue
        for slots in ((2, 0, 1, 0), (1, 1, 0, 2), (2, 2, 2, 2)):
            shape = tuple(sizes[slot] for slot in slots)
            expected = np.empty(shape)
            delta = np.empty(shape)
            for component in product(*(range(size) for size in shape)):
                first, second, third, fourth = (
                    int(offsets[shell]) + index
                    for shell, index in zip(slots, component, strict=True)
                )
                if source == "coulomb":
                    left, right = (
                        density[:, first, second].sum(),
                        density[:, third, fourth].sum(),
                    )
                    dleft, dright = (
                        moving[:, first, second].sum(),
                        moving[:, third, fourth].sum(),
                    )
                    expected[component] = 0.5 * left * right
                    delta[component] = 0.5 * (dleft * right + left * dright)
                else:
                    fraction = (
                        method.full_range_exact_exchange
                        if source == "exact_exchange"
                        else next(
                            primitive.coefficient
                            for primitive in method.primitives
                            if isinstance(primitive, RangeSeparatedExchangePrimitive)
                            and source
                            == "exchange_" + primitive.operator.replace("-", "_")
                        )
                    )
                    factor = -float(fraction) * (0.5 if spins == 2 else 0.25)
                    left, right = density[:, first, third], density[:, second, fourth]
                    dleft, dright = moving[:, first, third], moving[:, second, fourth]
                    expected[component] = factor * np.dot(left, right)
                    delta[component] = factor * (
                        np.dot(dleft, right) + np.dot(left, dright)
                    )
            np.testing.assert_allclose(
                prepared.fixed(slots), expected, atol=2e-14, rtol=2e-14
            )
            np.testing.assert_allclose(
                prepared.moving_weights(slots), delta, atol=2e-14, rtol=2e-14
            )
            np.testing.assert_allclose(
                prepared.fixed(slots), expected, atol=2e-14, rtol=2e-14
            )
