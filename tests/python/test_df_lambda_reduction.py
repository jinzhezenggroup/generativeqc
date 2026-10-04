"""Staged DF response against expanded AD and separate NumPy finite differences."""

from __future__ import annotations

import typing

import numpy as np
import pytest
from generativeqc_compiler.cc.df_lambda_reduction import (
    build_df_lambda_reduction_programs,
)
from generativeqc_compiler.cc.doubles import build_ccsd_program
from generativeqc_compiler.cc.lambda_equations import (
    AMPLITUDES,
    RESIDUALS,
    build_parameter_vjp,
)
from generativeqc_compiler.tensor import execute, transpose_program
from test_df_cc_auxiliary_reduction import _contraction_terms
from test_df_cc_native_solver import _case

from tools.generate_df_ccsd_native import programs as old_virtual
from tools.generativeqc_cc.df_factorized import virtual_corrections


def prepare(pipeline: typing.Any, feeds: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """Materialize the immutable cuts once, independently of response seeds."""
    prepared = execute(pipeline.primal.prepare, feeds).outputs
    sums = {
        name: np.zeros(node.spec.shape)
        for name, node in pipeline.primal.auxiliary.outputs.items()
    }
    for bov, bvv in zip(feeds["bov"], feeds["bvv"], strict=True):
        row = execute(
            pipeline.primal.auxiliary, {**feeds, **prepared, "bov": bov, "bvv": bvv}
        ).outputs
        for name in sums:
            sums[name] += row[name]
    return prepared, sums


def evaluate(pipeline: typing.Any, feeds: dict, seeds: dict) -> tuple[dict, dict]:
    """Validation-only composition; native owners must retain each live cut."""
    prepared, sums = prepare(pipeline, feeds)
    core = execute(pipeline.core, {**feeds, **sums, **seeds}).outputs
    tau_seed = np.zeros_like(prepared["df_tau"])
    t1 = core["bar_t1"].copy()
    t2 = core["bar_t2"].copy()
    factors = {name: np.zeros_like(feeds[name]) for name in ("bov", "bvv")}
    for q, (bov, bvv) in enumerate(zip(feeds["bov"], feeds["bvv"], strict=True)):
        frame = {**feeds, **prepared, **core, "bov": bov, "bvv": bvv}
        adjoint = execute(pipeline.auxiliary, frame).outputs
        t1 += adjoint["bar_t1"]
        t2 += adjoint["bar_t2"]
        tau_seed += adjoint["bar_df_tau"]
        row = execute(pipeline.factors, frame).outputs
        for name, values in factors.items():
            values[q] = row["bar_" + name]
    detached = execute(pipeline.prepare, {**feeds, "bar_df_tau": tau_seed}).outputs
    return {
        "bar_t1": t1 + detached["bar_t1"],
        "bar_t2": t2 + detached["bar_t2"],
    }, factors


@pytest.mark.parametrize("o,v,q", [(1, 3, 2), (2, 3, 4), (3, 2, 3)])
def test_staged_amplitude_and_factor_response(o: int, v: int, q: int) -> None:
    _, _, feeds = _case(o, v, q)
    rng = np.random.default_rng(1810 + o)
    feeds["t1"] = rng.normal(scale=0.06, size=(o, v))
    two = rng.normal(scale=0.04, size=(o, o, v, v))
    feeds["t2"] = (two + two.transpose(1, 0, 3, 2)) / 2
    seeds = {
        "bar_singles_residual": rng.normal(size=(o, v)),
        "bar_doubles_residual": rng.normal(size=(o, o, v, v)),
    }
    pipeline = build_df_lambda_reduction_programs(o, v)
    actual, factors = evaluate(pipeline, feeds, seeds)
    expanded = transpose_program(
        build_ccsd_program(o, v, form="expanded", diagnostics=False),
        RESIDUALS,
        inputs=AMPLITUDES,
    ).program
    expected = execute(expanded, {**feeds, **seeds}).outputs
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=3e-12, rtol=0)
    directions = {name: rng.normal(size=feeds[name].shape) for name in factors}
    directions["bvv"] = (directions["bvv"] + directions["bvv"].transpose(0, 2, 1)) / 2
    analytic = sum(float(np.sum(factors[name] * directions[name])) for name in factors)
    # Hold retained integral blocks fixed: this owner differentiates only the
    # virtual interaction. Independent NumPy equations have no staging cuts.
    for step in (1e-5, 3e-6):
        energies = []
        for sign in (-1, 1):
            r1, r2 = virtual_corrections(
                feeds["bov"] + sign * step * directions["bov"],
                feeds["bvv"] + sign * step * directions["bvv"],
                feeds["t1"],
                feeds["t2"],
            )
            energies.append(
                float(
                    np.sum(r1 * seeds["bar_singles_residual"])
                    + np.sum(r2 * seeds["bar_doubles_residual"])
                )
            )
        np.testing.assert_allclose(
            (energies[1] - energies[0]) / (2 * step), analytic, atol=2e-8, rtol=0
        )


def test_staged_retained_parameters_use_the_same_physical_cuts() -> None:
    """Arbitrary T and seeds expose errors hidden by a converged residual."""
    o, v, q = 2, 3, 4
    _, _, feeds = _case(o, v, q)
    rng = np.random.default_rng(1809)
    feeds["t1"] = rng.normal(scale=0.06, size=(o, v))
    two = rng.normal(scale=0.04, size=(o, o, v, v))
    feeds["t2"] = (two + two.transpose(1, 0, 3, 2)) / 2
    seeds = {
        "bar_correlation_energy": np.array(1.0),
        "bar_singles_residual": rng.normal(size=(o, v)),
        "bar_doubles_residual": rng.normal(size=(o, o, v, v)),
    }
    pipeline = build_df_lambda_reduction_programs(o, v)
    prepared, sums = prepare(pipeline, feeds)
    expanded = build_ccsd_program(o, v, form="expanded", diagnostics=False)
    for name, program in pipeline.parameters.items():
        actual = execute(program, {**feeds, **prepared, **sums, **seeds}).outputs
        expected = execute(
            build_parameter_vjp(expanded, name).program, {**feeds, **seeds}
        ).outputs
        np.testing.assert_allclose(
            actual["bar_" + name], expected["bar_" + name], atol=3e-12, rtol=0
        )


def test_runtime_shape_response_work_and_virtual_storage() -> None:
    from generativeqc_compiler.cc.df_lambda import retained_response_programs

    pipeline = build_df_lambda_reduction_programs(2, 3)
    for program in (
        pipeline.core,
        pipeline.auxiliary,
        pipeline.prepare,
        pipeline.factors,
        *pipeline.parameters.values(),
    ):
        assert all(
            sum(i.space.kind == "virtual" for i in node.spec.indices) <= 2
            for node in program.live_nodes
        )
    old = retained_response_programs(2, 3)["transpose"]
    old_q = old_virtual("cuda")["amplitude_vjp"]
    for o, v, q in ((9, 221, 488), (21, 243, 666)):
        before = _contraction_terms(old, o, v) + q * _contraction_terms(old_q, o, v)
        after = (
            _contraction_terms(pipeline.core, o, v)
            + q * _contraction_terms(pipeline.auxiliary, o, v)
            + _contraction_terms(pipeline.prepare, o, v)
        )
        # This excludes amortized immutable primal preparation, not repeated
        # preparation per Q or per Krylov action; native accounting must report it.
        assert 4 * after < before
