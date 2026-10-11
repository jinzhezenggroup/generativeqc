"""FP32 DF auxiliary warmup must finish with independent FP64 certificates."""

from __future__ import annotations

import os
import subprocess
import typing
from collections import Counter

import numpy as np
import pytest
from generativeqc_compiler.cc.df_hoist import build_df_auxiliary_reduction_programs
from generativeqc_compiler.cc.df_lambda_matrix import matrix_program
from generativeqc_compiler.cc.df_mixed_precision import (
    DF_CC_WARMUP_QUALIFICATION,
    auxiliary_warmup_program,
)
from generativeqc_compiler.tensor import describe_precision, execute
from test_df_cc_native_solver import _case, _run, _stream
from test_df_cc_native_solver import solver_probe as _solver_probe

from tools.generate_df_ccsd_hoisted import (
    batched_auxiliary_program,
    mixed_auxiliary_program,
)
from tools.generate_rccsd_native import (
    _arena_plan,
    _cuda_program,
    cross_size_arena_candidate,
)

solver_probe = _solver_probe

if typing.TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("occupied,virtual,batch", [(1, 3, 1), (2, 3, 3), (3, 2, 2)])
def test_auxiliary_warmup_preserves_equations_and_bounded_rounding(
    occupied: int, virtual: int, batch: int
) -> None:
    _, _, arrays = _case(occupied, virtual, batch)
    pipeline = build_df_auxiliary_reduction_programs(occupied, virtual)
    values = {**arrays, **execute(pipeline.prepare, arrays).outputs}
    strict = matrix_program(pipeline.auxiliary, batch_size=batch)
    mixed = auxiliary_warmup_program(strict)
    actual = execute(
        mixed, {name: value.astype(np.float32) for name, value in values.items()}
    ).outputs
    expected = execute(strict, values).outputs
    assert mixed.provenance["fp64_source_equation"] == strict.logical_hash
    for name, reference in expected.items():
        assert actual[name].dtype == np.float32
        np.testing.assert_allclose(actual[name], reference, atol=2e-8, rtol=2e-5)
    assert Counter(node.op for node in mixed.nodes) == Counter(
        node.op for node in strict.nodes
    )
    schedule = describe_precision(mixed)
    assert {qualification for _, qualification in schedule.qualification_scope} == {
        DF_CC_WARMUP_QUALIFICATION
    }


@pytest.mark.parametrize("cross_size", [False, True])
def test_mixed_arena_halves_bytes_without_changing_slot_lifetimes(
    cross_size: bool,
) -> None:
    with cross_size_arena_candidate(cross_size):
        strict = _arena_plan(batched_auxiliary_program())
        mixed = _arena_plan(mixed_auxiliary_program())
    assert strict.slots == mixed.slots
    assert dict(strict.node_slots) == dict(mixed.node_slots)
    assert strict.identity != mixed.identity
    assert mixed.storage_dtype == "float32"


def test_native_mixed_emission_is_typed_and_pedantic() -> None:
    program = mixed_auxiliary_program()
    source = _cuda_program(
        program,
        "mixed_auxiliary",
        "MixedAuxiliaryOutputs",
        input_overrides={
            node.attrs["name"]: "s." + node.attrs["name"]
            for node in program.live_nodes
            if node.op == "input"
        },
        state_type="MixedAuxiliaryCudaState",
        arena_field="auxiliary_arena",
        batch_dim=True,
        output_fields=tuple(program.outputs),
        prepared_contractions="s.contractions",
    )
    assert "->float*" in source
    assert "runtime::PrecisionDtype::Fp32" in source
    assert "const double*" not in source
    assert "__dmul_rn" not in source


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_CC_CUDA_TEST") != "1",
    reason="requires finite Slurm GPU allocation",
)
@pytest.mark.parametrize(
    "occupied,virtual,batch", [(2, 3, 5), (3, 7, 9), (3, 7, 14), (3, 10, 33)]
)
def test_native_warmup_has_strict_final_energy_amplitudes_and_residuals(
    solver_probe: tuple[Path, bool], occupied: int, virtual: int, batch: int
) -> None:
    if not solver_probe[1]:
        pytest.skip("the mixed auxiliary schedule is CUDA-only")
    _, _, arrays = _case(occupied, virtual, batch)
    strict, strict_one, strict_two = _run(solver_probe, arrays, occupied_pairs=False)
    mixed, mixed_one, mixed_two = _run(solver_probe, arrays, mixed_precision=True)
    assert strict["status"] == mixed["status"] == 0
    assert mixed["mixed_precision"] == 1 and mixed["mixed_precision_refused"] == 0
    assert mixed["mixed_evaluations"] > 0 and mixed["fp32_gemm_calls"] > 0
    assert mixed["precision_switches"] == 1
    assert mixed["strict_batch_size"] == max(1, min(batch, 8) // 2)
    assert max(mixed["r1"], mixed["r2"]) <= 1e-10
    np.testing.assert_allclose(mixed["energy"], strict["energy"], atol=2e-11, rtol=0)
    np.testing.assert_allclose(mixed_one, strict_one, atol=2e-10, rtol=0)
    np.testing.assert_allclose(mixed_two, strict_two, atol=2e-10, rtol=0)


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_CC_CUDA_TEST") != "1",
    reason="requires finite Slurm GPU allocation",
)
def test_mixed_resource_refusal_keeps_strict_execution(
    solver_probe: tuple[Path, bool],
) -> None:
    if not solver_probe[1]:
        pytest.skip("CUDA-only candidate")
    _, _, arrays = _case(2, 3, 5)
    strict, one, two = _run(solver_probe, arrays, matrix=False)
    refused, actual_one, actual_two = _run(
        solver_probe, arrays, matrix=False, mixed_precision=True
    )
    assert strict["status"] == refused["status"] == 0
    assert refused["mixed_precision_refused"] == 1
    assert refused["mixed_evaluations"] == refused["fp32_gemm_calls"] == 0
    np.testing.assert_array_equal(actual_one, one)
    np.testing.assert_array_equal(actual_two, two)


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_CC_CUDA_TEST") != "1",
    reason="requires finite Slurm GPU allocation",
)
@pytest.mark.parametrize("symmetric", [True, False])
def test_strict_pair_boundary_preserves_supplied_state_contract(
    solver_probe: tuple[Path, bool], symmetric: bool
) -> None:
    """Warm-start projection must not repair an asymmetric supplied guess."""
    if not solver_probe[1]:
        pytest.skip("CUDA-only candidate")
    _, _, arrays = _case(3, 7, 14)
    if not symmetric:
        arrays["t2"][0, 1, 0, 1] = np.nextafter(arrays["t2"][0, 1, 0, 1], 1.0)
    strict, one, two = _run(
        solver_probe, arrays, packed_diis=True, occupied_pairs=False
    )
    mixed, actual_one, actual_two = _run(
        solver_probe, arrays, packed_diis=True, mixed_precision=True
    )
    assert strict["status"] == mixed["status"] == 0
    assert mixed["mixed_pair_projection_calls"] == int(symmetric)
    if symmetric:
        assert mixed["packed_diis"] == 1
        assert mixed["pair_evaluations"] > 0
        assert mixed["mixed_pair_projection_bytes"] == 2 * arrays["t2"].nbytes
    else:
        assert mixed["pair_initial_symmetry_refused"] == 1
    np.testing.assert_allclose(mixed["energy"], strict["energy"], atol=2e-11, rtol=0)
    np.testing.assert_allclose(actual_one, one, atol=2e-10, rtol=0)
    np.testing.assert_allclose(actual_two, two, atol=2e-10, rtol=0)


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_CC_CUDA_TEST") != "1",
    reason="requires finite Slurm GPU allocation",
)
def test_warmup_limit_discards_the_carried_mixed_trial(
    solver_probe: tuple[Path, bool],
) -> None:
    """Exhaustion in a trial must reevaluate its accepted state in FP64."""
    if not solver_probe[1]:
        pytest.skip("CUDA-only candidate")
    _, _, arrays = _case(3, 7, 14)
    strict, one, two = _run(solver_probe, arrays)
    mixed, actual_one, actual_two = _run(
        solver_probe,
        arrays,
        mixed_precision=True,
        mixed_two_evaluation_limit=True,
    )
    assert mixed["status"] == strict["status"] == 0
    assert mixed["mixed_evaluations"] == 2
    assert mixed["precision_switches"] == 1
    assert max(mixed["r1"], mixed["r2"]) <= 1e-10
    np.testing.assert_allclose(mixed["energy"], strict["energy"], atol=2e-11, rtol=0)
    np.testing.assert_allclose(actual_one, one, atol=2e-10, rtol=0)
    np.testing.assert_allclose(actual_two, two, atol=2e-10, rtol=0)


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_CC_CUDA_TEST") != "1",
    reason="requires finite Slurm GPU allocation",
)
def test_fp32_overflow_cannot_publish_a_strict_result(
    solver_probe: tuple[Path, bool],
) -> None:
    if not solver_probe[1]:
        pytest.skip("CUDA-only candidate")
    _, _, arrays = _case(2, 3, 5)
    arrays["bvv"].fill(1e40)
    process = subprocess.run(
        [str(solver_probe[0])],
        input=_stream(arrays, True, mixed_precision=True),
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert process.returncode != 0
    assert b"nonfinite RCCSD generated CUDA tensor" in process.stderr
    assert not process.stdout
