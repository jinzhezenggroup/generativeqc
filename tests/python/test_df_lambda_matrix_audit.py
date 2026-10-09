"""Original expanded independent Lambda equations through optional FP64 GEMM."""

from __future__ import annotations

import typing

import numpy as np
import pytest
from generativeqc_compiler.cc.df_lambda import retained_response_programs
from generativeqc_compiler.cc.df_lambda_matrix import matrix_program
from generativeqc_compiler.tensor import execute
from test_df_cc_lambda import case, feeds, run
from test_df_cc_lambda import probe as _native_probe

probe = _native_probe


@pytest.mark.parametrize("o,v", [(1, 3), (2, 3), (3, 2)])
def test_matrix_audit_keeps_expanded_core_and_original_virtual_equations(
    o: int, v: int
) -> None:
    """No solver cuts or cached primal intermediates enter either audit program."""
    _, _, feeds = case(o, v, q=5)
    rng = np.random.default_rng(2136)
    one = rng.normal(size=(o, v))
    two = rng.normal(size=(o, o, v, v))
    core = retained_response_programs(o, v)["independent_transpose"]
    # The native representative virtual graph is symbolic, so use the shared
    # builder with the requested small dimensions for NumPy equation comparison.
    from generativeqc_compiler.cc.df_equations import (
        build_df_virtual_response_programs,
    )

    virtual = build_df_virtual_response_programs(o, v).amplitude_vjp.program
    frame = {**feeds, "bar_singles_residual": one, "bar_doubles_residual": two}
    expected = execute(core, frame).outputs
    actual = execute(matrix_program(core), frame).outputs
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=2e-11, rtol=0)
    for start in (0, 3):
        stop = min(5, start + 3)
        frame = {
            **feeds,
            "bov": feeds["bov"][start:stop],
            "bvv": feeds["bvv"][start:stop],
            "bar_df_virtual_singles": one,
            "bar_df_virtual_doubles": two,
        }
        actual = execute(
            matrix_program(virtual, batch_size=stop - start), frame
        ).outputs
        rows = [
            execute(
                virtual,
                {**frame, "bov": feeds["bov"][index], "bvv": feeds["bvv"][index]},
            ).outputs
            for index in range(start, stop)
        ]
        for name in actual:
            np.testing.assert_allclose(
                actual[name], np.stack([row[name] for row in rows]), atol=2e-11, rtol=0
            )


@pytest.mark.parametrize("o,v", [(1, 3), (2, 3), (3, 2)])
@pytest.mark.parametrize("source", [False, True])
def test_native_matrix_audit_preserves_outputs_and_scalar_audit_fallback(
    probe: typing.Any, o: int, v: int, source: bool
) -> None:
    _, _, arrays = case(o, v, q=5)
    status, control, _, old_counts, error = run(
        probe,
        arrays,
        source=source,
        core_reuse=False,
        audit_matrix=False,
        batch_two=True,
    )
    assert status == 0, error
    status, actual, values, counts, error = run(
        probe, arrays, source=source, batch_two=True
    )
    assert status == 0, error
    assert counts[24] == 1 and counts[25] > 0
    assert counts[1] == old_counts[1]
    assert max(values[1:]) < 1e-9
    for value, expected in zip(actual, control, strict=True):
        np.testing.assert_array_equal(value, expected)
    status, fallback, _, bounded, error = run(
        probe,
        arrays,
        source=source,
        core_reuse=False,
        budget=int(old_counts[2]),
        batch_two=True,
    )
    assert status == 0, error
    assert bounded[24] == 0 and bounded[13] == 1
    for value, expected in zip(fallback, control, strict=True):
        np.testing.assert_array_equal(value, expected)


@pytest.mark.parametrize("matrix", [False, True])
def test_nonfinite_independent_audit_never_publishes(
    probe: typing.Any, matrix: bool
) -> None:
    _, _, arrays = case(2, 3, q=5)
    status, output, _, _, error = run(
        probe, arrays, matrix=matrix, overflow=True, overflow_audit=True
    )
    assert status != 0 and "nonfinite native DF Lambda" in error
    assert all(np.isnan(values).all() for values in output)


@pytest.mark.parametrize("scale,shift", [(1e-3, 0.0), (1e-3, 1.0), (1.0, 1e3)])
def test_small_denominators_and_common_fock_shift_keep_strict_audit(
    probe: typing.Any, scale: float, shift: float
) -> None:
    """Scale one Hamiltonian consistently; a common shift must cancel physically."""
    factors, fock, _ = case(2, 3, q=5)
    shifted_fock = scale * fock + shift * np.eye(fock.shape[0])
    arrays = feeds(np.sqrt(scale) * factors, shifted_fock, 2)
    status, control, old_values, old_counts, error = run(
        probe,
        arrays,
        source=True,
        core_reuse=False,
        audit_matrix=False,
        batch_two=True,
    )
    assert status == 0, error
    status, actual, values, counts, error = run(
        probe, arrays, source=True, batch_two=True
    )
    assert status == 0, error
    assert counts[20] == 1 and counts[24] == 1
    assert counts[1] == old_counts[1]
    assert max(values[1:]) < 1e-9 and max(old_values[1:]) < 1e-9
    for value, expected in zip(actual, control, strict=True):
        np.testing.assert_array_equal(value, expected)
