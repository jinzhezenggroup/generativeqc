"""Bounded FP32 Lambda proposals with original FP64 solve/publication gates."""

from __future__ import annotations

import typing

import numpy as np
import pytest
from generativeqc_compiler.cc.df_lambda_matrix import matrix_program
from generativeqc_compiler.cc.df_lambda_reduction import (
    build_df_lambda_reduction_programs,
)
from generativeqc_compiler.cc.df_mixed_precision import (
    DF_LAMBDA_WARMUP_QUALIFICATION,
    lambda_auxiliary_warmup_program,
)
from generativeqc_compiler.tensor import execute
from test_df_cc_lambda import case, run
from test_df_cc_lambda import probe as _native_probe
from test_df_cc_native_solver import _case
from test_df_lambda_reduction import prepare

probe = _native_probe


@pytest.mark.parametrize(
    "occupied,virtuals,auxiliaries,batch", [(1, 3, 5, 2), (2, 3, 5, 3), (3, 2, 3, 8)]
)
def test_typed_auxiliary_adjoint_retains_Q_rows_and_fp64_source(
    occupied: int, virtuals: int, auxiliaries: int, batch: int
) -> None:
    """Input casts are explicit, and no approximate result certifies convergence."""
    _, _, feeds = _case(occupied, virtuals, auxiliaries)
    pipeline = build_df_lambda_reduction_programs(occupied, virtuals)
    prepared, cuts = prepare(pipeline, feeds)
    rng = np.random.default_rng(2278)
    seeds = {
        "bar_singles_residual": rng.normal(size=(occupied, virtuals)),
        "bar_doubles_residual": rng.normal(
            size=(occupied, occupied, virtuals, virtuals)
        ),
    }
    core = execute(matrix_program(pipeline.core), {**feeds, **cuts, **seeds}).outputs
    for start in range(0, auxiliaries, batch):
        stop = min(auxiliaries, start + batch)
        source = matrix_program(pipeline.auxiliary, batch_size=stop - start)
        identity = source.logical_hash
        candidate = lambda_auxiliary_warmup_program(source)
        assert source.logical_hash == identity
        assert candidate.provenance["fp64_source_equation"] == identity
        assert (
            candidate.provenance["native_precision_boundary"]
            == DF_LAMBDA_WARMUP_QUALIFICATION
        )
        assert all(node.spec.dtype == "float32" for node in candidate.live_nodes)
        frame = {
            **feeds,
            **prepared,
            **core,
            "bov": feeds["bov"][start:stop],
            "bvv": feeds["bvv"][start:stop],
        }
        actual = execute(
            candidate,
            {
                name: np.asarray(values, dtype=np.float32)
                for name, values in frame.items()
            },
        ).outputs
        strict = execute(source, frame).outputs
        assert actual.keys() == strict.keys()
        for name, values in actual.items():
            assert values.dtype == np.float32
            assert values.shape == strict[name].shape
            np.testing.assert_allclose(values, strict[name], atol=2e-6, rtol=2e-6)


def test_generated_auxiliary_has_distinct_precision_and_ordered_widening() -> None:
    """The same emitter supplies typed scratch, RN arithmetic and Q consumption."""
    from tools.generate_df_lambda import (
        cuda_header,
        cuda_source,
        mixed_auxiliary_accumulation,
        mixed_auxiliary_program,
    )

    program = mixed_auxiliary_program()
    assert (
        program.provenance["native_precision_boundary"]
        == DF_LAMBDA_WARMUP_QUALIFICATION
    )
    header = cuda_header()
    assert program.logical_hash in header
    assert "struct MixedAuxiliaryCudaState" in header
    generated = cuda_source()
    body = generated.split("MixedAuxiliaryOutputs run_mixed_auxiliary(", 1)[1]
    body = body.split("MixedAuxiliaryOutputs run_mixed_auxiliary_cuda", 1)[0]
    assert "float* response_arena{}" in header
    assert "auto* arena=s.response_arena;" in body
    assert "s.contractions.execute(" in body
    assert "__fmul_rn" in generated and "__fadd_rn" in generated
    accumulation = mixed_auxiliary_accumulation()[1]
    assert "double value=target_" in accumulation
    assert "for(std::size_t Q=0;Q<q;++Q)" in accumulation
    assert "value+values." in accumulation


@pytest.mark.parametrize("occupied,virtuals", [(1, 3), (2, 3), (2, 7)])
@pytest.mark.parametrize("source", [False, True])
def test_native_mixed_proposal_finishes_with_fresh_fp64_gates(
    probe: typing.Any, occupied: int, virtuals: int, source: bool
) -> None:
    """Compare all Lambda/parameter/factor outputs with the original strict solve."""
    _, _, arrays = case(occupied, virtuals, q=5)
    status, strict, _, old_counts, error = run(
        probe, arrays, source=source, primal_matrix=True
    )
    assert status == 0, error
    status, actual, values, counts, error = run(
        probe, arrays, source=source, primal_matrix=True, mixed_precision=True
    )
    assert status == 0, error
    assert counts[29] == 1 and counts[30] == 0
    assert counts[33] > 0 and counts[34] > 0 and counts[35] == 1
    assert counts[36] > 0 and counts[37] > 0 and counts[38] > 0
    assert 0 < counts[39] <= 16
    assert counts[1] == counts[33] + counts[34]
    assert counts[3] <= old_counts[3]
    assert max(values[1:]) < 1e-9
    for candidate, expected in zip(actual, strict, strict=True):
        np.testing.assert_allclose(candidate, expected, atol=5e-11, rtol=5e-11)


def test_finite_mixed_exhaustion_is_only_a_warm_start(probe: typing.Any) -> None:
    """Even a carried two-step proposal receives a new FP64 physical action."""
    _, _, arrays = case(2, 7, q=5)
    status, strict, _, _, error = run(probe, arrays, source=True, primal_matrix=True)
    assert status == 0, error
    status, actual, values, counts, error = run(
        probe,
        arrays,
        source=True,
        primal_matrix=True,
        mixed_precision=True,
        mixed_limit_two=True,
    )
    assert status == 0, error
    assert counts[29] == counts[35] == 1
    assert 0 < counts[39] <= 2 and counts[34] > 0
    assert max(values[1:]) < 1e-9
    for candidate, expected in zip(actual, strict, strict=True):
        np.testing.assert_allclose(candidate, expected, atol=5e-11, rtol=5e-11)


def test_optional_matrix_refusal_keeps_original_fp64_actions(probe: typing.Any) -> None:
    """No precision conversion or extra device lease occurs on unsupported admission."""
    _, _, arrays = case(2, 3, q=5)
    status, strict, _, old_counts, error = run(probe, arrays, matrix=False)
    assert status == 0, error
    status, actual, values, counts, error = run(
        probe, arrays, matrix=False, mixed_precision=True
    )
    assert status == 0, error
    assert counts[29] == 0 and counts[30] == 1
    assert counts[33] == counts[35] == counts[36] == counts[38] == 0
    assert counts[3] == old_counts[3]
    assert max(values[1:]) < 1e-9
    for candidate, expected in zip(actual, strict, strict=True):
        np.testing.assert_array_equal(candidate, expected)


def test_nonfinite_mixed_action_cannot_publish_or_retry_strict(
    probe: typing.Any,
) -> None:
    """Sticky numerical failure propagates; it is not an optional-storage fallback."""
    _, _, arrays = case(2, 7, q=5)
    status, result, _, _, error = run(
        probe, arrays, mixed_precision=True, overflow=True, primal_matrix=True
    )
    assert status != 0 and "nonfinite" in error
    assert all(np.isnan(values).all() for values in result)
