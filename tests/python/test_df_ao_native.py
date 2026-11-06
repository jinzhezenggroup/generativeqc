"""Independent Gram and directional-adjoint oracles for the fitted AO map."""

import numpy as np
from generativeqc_compiler.method.df_ao_source import (
    factor_block_program,
    factor_programs,
    programs,
)
from generativeqc_compiler.tensor import execute


def test_fitted_ao_value_and_raw_source_adjoint() -> None:
    """Keep both Gram legs and both raw-source/metric-root derivative paths."""
    rng = np.random.default_rng(88831)
    raw = rng.normal(scale=0.2, size=(6, 6, 3))
    root = rng.normal(scale=0.2, size=(3, 3))
    seed = rng.normal(size=(6, 6, 6, 6))
    primal, pullback = programs()
    inputs = {"three_center": raw, "inverse_root": root}
    actual = execute(primal, inputs).outputs["eri"]
    factors = np.einsum("mnP,PQ->mnQ", raw, root)
    np.testing.assert_allclose(
        actual, np.einsum("mnQ,klQ->mnkl", factors, factors), atol=1e-13, rtol=0
    )
    bars = execute(pullback, {**inputs, "bar_eri": seed}).outputs
    directions = {name: rng.normal(size=value.shape) for name, value in inputs.items()}
    step = 1e-5
    values = [
        execute(
            primal,
            {
                name: value + sign * step * directions[name]
                for name, value in inputs.items()
            },
        ).outputs["eri"]
        for sign in (-1, 1)
    ]
    finite_difference = np.sum(seed * (values[1] - values[0])) / (2 * step)
    adjoint = sum(
        np.sum(bars[f"bar_{name}"] * direction)
        for name, direction in directions.items()
    )
    np.testing.assert_allclose(adjoint, finite_difference, atol=3e-7, rtol=0)


def test_three_index_spin_transform_and_complete_pullback() -> None:
    """Independently cover both orbital legs and a nonsymmetric metric root."""
    rng = np.random.default_rng(2980)
    raw = rng.normal(scale=0.2, size=(6, 6, 3))
    root = rng.normal(scale=0.2, size=(3, 3))
    primal, reverse = factor_programs()
    transformed = []
    for spin in range(2):
        coefficients = rng.normal(scale=0.3, size=(6, 6))
        inputs = {
            "three_center": raw,
            "inverse_root": root,
            "coefficients": coefficients,
        }
        factors = execute(primal, inputs).outputs["factors"]
        expected = np.einsum("mp,nq,mnP,PQ->pqQ", coefficients, coefficients, raw, root)
        np.testing.assert_allclose(factors, expected, atol=2e-14, rtol=0)
        transformed.append(factors)
        seed = rng.normal(size=factors.shape)
        bars = execute(reverse, {**inputs, "bar_factors": seed}).outputs
        for name, value in inputs.items():
            direction = rng.normal(size=value.shape)
            step = 1e-5
            energies = [
                np.vdot(
                    seed,
                    execute(
                        primal, {**inputs, name: value + sign * step * direction}
                    ).outputs["factors"],
                )
                for sign in (-1, 1)
            ]
            np.testing.assert_allclose(
                np.vdot(bars[f"bar_{name}"], direction),
                (energies[1] - energies[0]) / (2 * step),
                atol=2e-8,
                rtol=0,
                err_msg=f"spin {spin}, input {name}",
            )
    block = execute(
        factor_block_program(),
        {
            "left_factors": transformed[0].reshape(36, 3)[:6],
            "right_factors": transformed[1].reshape(36, 3)[:6],
        },
    ).outputs["block"]
    np.testing.assert_allclose(
        block,
        transformed[0].reshape(36, 3)[:6] @ transformed[1].reshape(36, 3)[:6].T,
        atol=2e-14,
        rtol=0,
    )
