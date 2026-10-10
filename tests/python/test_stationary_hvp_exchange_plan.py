"""Independent compiler-only global-hybrid exchange Hessian-vector source gates."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from generativeqc_compiler.method import (
    MethodSpec,
    StationaryHVPPlan,
    StationaryMeanField,
    UnsupportedMethod,
    resolve_method,
)
from generativeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
)
from generativeqc_compiler.tensor import Program, execute


@pytest.mark.parametrize(
    ("name", "fraction"),
    [("PBE0", Fraction(1, 4)), ("B3LYP", Fraction(1, 5))],
)
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_full_range_exchange_hvp_matches_independent_displaced_gradient(
    name: str, fraction: Fraction, spin: str
) -> None:
    method = resolve_method(name, spin=spin)
    envelope = StationaryMeanField(SCF_POINT_MODEL)
    plan = StationaryHVPPlan(method, envelope)
    gradient = StationaryGradientPlan(method, envelope)
    quartets = tuple(product(range(2), repeat=4))
    terms, coords = len(quartets), 4
    block = plan.integral_block("exact_exchange", terms=terms, coordinates=coords)
    gradient_block = gradient.integral_block(
        "exact_exchange",
        terms=terms,
        coordinates=coords,
        differentiate_densities=True,
    )
    assert block.objective.logical_hash == gradient_block.objective.logical_hash
    assert block.weights.logical_hash == gradient_block.weights.logical_hash
    assert block.response_inputs == ("density_left", "density_right")

    spins = 2 if spin == "polarized" else 1
    rng = np.random.default_rng(18011)
    density = rng.normal(size=(spins, 2, 2))
    density = 0.5 * (density + density.transpose(0, 2, 1))
    direction = rng.normal(size=(spins, 2, 2))
    direction = 0.5 * (direction + direction.transpose(0, 2, 1))
    left = np.array([[d[i, j] for i, k, j, l in quartets] for d in density])
    right = np.array([[d[k, l] for i, k, j, l in quartets] for d in density])
    dl = np.array([[d[i, j] for i, k, j, l in quartets] for d in direction])
    dr = np.array([[d[k, l] for i, k, j, l in quartets] for d in direction])
    first = rng.normal(size=(terms, coords))
    second = rng.normal(size=(terms, coords))
    feeds = {
        "density_left": left,
        "density_right": right,
        "d_density_left": dl,
        "d_density_right": dr,
    }

    # RKS occupation-weighted D: -a/4. UKS same-spin only: -a/2.
    factor = -float(fraction) / (4 if spins == 1 else 2)
    expected_w = factor * np.sum(left * right, axis=0)
    expected_dw = factor * np.sum(dl * right + left * dr, axis=0)
    weights = execute(block.weights, feeds).outputs["weights"]
    response = execute(block.response_weights, feeds).outputs["response_weights"]
    np.testing.assert_allclose(weights, expected_w, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(response, expected_dw, atol=2e-14, rtol=2e-14)
    if spins == 2:
        cross_spin = factor * (left[0] * right[1] + left[1] * right[0])
        assert not np.allclose(weights, expected_w + cross_spin)

    input_hvp = {
        "response_weights": response,
        "integral_derivatives": first,
        "weighted_second_hvp": expected_w @ second,
    }
    outputs = execute(block.contraction, input_hvp).outputs
    expected_response = expected_dw @ first
    expected_second = expected_w @ second
    full = expected_response + expected_second
    np.testing.assert_allclose(outputs["response"], expected_response, atol=2e-13)
    np.testing.assert_allclose(outputs["second"], expected_second, atol=2e-13)
    np.testing.assert_allclose(outputs["hvp"], full, atol=2e-13)

    # Independent ordered-quartet gradient at displaced density/integral.
    # This checks both density directions and the second-integral term.
    def independent_gradient(step: float) -> np.ndarray:
        result = np.zeros(coords)
        for t in range(terms):
            for s in range(spins):
                lhs = float(left[s, t] + step * dl[s, t])
                rhs = float(right[s, t] + step * dr[s, t])
                for q in range(coords):
                    result[q] += (
                        factor * lhs * rhs * float(first[t, q] + step * second[t, q])
                    )
        return result

    errors = []
    for step in (1e-3, 2e-4, 4e-5):
        ref = (independent_gradient(step) - independent_gradient(-step)) / (2 * step)
        errors.append(float(np.max(np.abs(outputs["hvp"] - ref))))
    assert errors[-1] < 2e-7
    assert errors[-1] < errors[0] / 20
    assert np.max(np.abs(outputs["response"])) > 1e-3
    assert np.max(np.abs(outputs["second"])) > 1e-3
    assert not np.allclose(outputs["response"], full)
    assert not np.allclose(outputs["second"], full)

    replay = Program.loads(block.contraction.dumps())
    np.testing.assert_array_equal(
        execute(replay, input_hvp).outputs["hvp"], outputs["hvp"]
    )
    for backend in ("cpu", "cuda"):
        with pytest.raises(NotImplementedError, match="qualification"):
            plan.require_native_endpoint(backend)


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_custom_fraction_is_composed_and_zero_exchange_is_absent(
    spin: str,
) -> None:
    for fraction in (Fraction(0), Fraction(1, 4), Fraction(1, 2)):
        method = resolve_method(
            MethodSpec(
                "custom-hybrid-hvp",
                (
                    ("GGA_X_PBE", Fraction(1) - fraction),
                    ("GGA_C_PBE", Fraction(1)),
                ),
                exact_exchange=fraction,
            ),
            spin=spin,
        )
        plan = StationaryHVPPlan(method, StationaryMeanField(SCF_POINT_MODEL))
        assert ("exact_exchange" in plan.source_names) == bool(fraction)
        if fraction:
            block = plan.integral_block("exact_exchange", terms=3)
            spins = 2 if spin == "polarized" else 1
            densities = np.ones((spins, 3))
            weights = execute(
                block.weights,
                {"density_left": densities, "density_right": densities},
            ).outputs["weights"]
            expected = -float(fraction) * spins / (4 if spins == 1 else 2)
            np.testing.assert_allclose(weights, expected, atol=2e-15)
        else:
            with pytest.raises(ValueError, match="integral primitive"):
                plan.integral_block("exact_exchange", terms=3)


@pytest.mark.parametrize("name", ("CAM-B3LYP", "R2SCAN", "PBE0-D3(BJ)"))
def test_unsupported_second_order_primitives_remain_fail_closed(name: str) -> None:
    with pytest.raises(UnsupportedMethod, match="second-order rule|ingredients"):
        StationaryHVPPlan(resolve_method(name), StationaryMeanField(SCF_POINT_MODEL))
