"""Independent Gram and directional-adjoint oracles for the fitted AO map."""

import numpy as np
from generativeqc_compiler.method.df_ao_source import programs
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
