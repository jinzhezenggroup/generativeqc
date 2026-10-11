"""The optional physical Q schedule never consumes hoisted primal cuts."""

from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.cc.df_equations import build_df_virtual_correction_program
from generativeqc_compiler.cc.df_lambda_matrix import matrix_program
from generativeqc_compiler.tensor import execute
from test_df_cc_virtual_equations import _case, _dense_correction

from tools.generate_df_ccsd_native import batched_replay_program, packed_replay_program

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "occupied,virtuals,auxiliary,batch", [(1, 1, 3, 2), (2, 3, 5, 4), (3, 2, 7, 4)]
)
def test_batched_expanded_actions_preserve_each_original_q(
    occupied: int, virtuals: int, auxiliary: int, batch: int
) -> None:
    """Nonzero amplitudes and tails must match the independent dense inventory."""
    feeds = _case(occupied, virtuals, auxiliary)
    original = build_df_virtual_correction_program(occupied, virtuals)
    sums = {name: np.zeros(node.spec.shape) for name, node in original.outputs.items()}
    for start in range(0, auxiliary, batch):
        stop = min(auxiliary, start + batch)
        frame = {
            **feeds,
            "bov": feeds["bov"][start:stop],
            "bvv": feeds["bvv"][start:stop],
        }
        program = matrix_program(original, batch_size=stop - start)
        actual = execute(program, frame).outputs
        for lane in range(stop - start):
            expected = execute(
                original,
                {**feeds, "bov": frame["bov"][lane], "bvv": frame["bvv"][lane]},
            ).outputs
            for name, values in actual.items():
                np.testing.assert_allclose(
                    values[lane], expected[name], atol=2e-12, rtol=0
                )
                sums[name] += values[lane]
    for name, expected in _dense_correction(feeds).items():
        np.testing.assert_allclose(sums[name], expected, atol=2e-12, rtol=0)


def test_batched_replay_has_only_original_inputs_and_bounded_virtual_axes() -> None:
    program = batched_replay_program()
    assert {
        node.attrs["name"] for node in program.live_nodes if node.op == "input"
    } == {"bov", "bvv", "t1", "t2"}
    assert all(
        sum(index.space.kind == "virtual" for index in node.spec.indices) <= 2
        for node in program.live_nodes
    )
    assert (
        packed_replay_program().logical_hash
        == "83897491af50dd86ff4b054efdb67f9f960cea064278030e20f143bcf153f6d6"
    )
    assert (
        program.logical_hash
        == "d54486f16ed5bca0aa33a35a7d0faf837a45d801560e840d1e8de4e1993a32c3"
    )


def test_automatic_batching_is_physical_cuda_df_energy_only() -> None:
    """Response and internal callers keep one Q and all old resource fallbacks."""
    header = (ROOT / "src/cc/solver.hpp").read_text()
    owner = (ROOT / "src/cc/cuda_solver.cu").read_text()
    dispatch = (ROOT / "src/methods/rccsd_method.cpp").read_text()
    assert "bool df_replay_auxiliary_batch{false};" in header
    assert (
        "solver_options.df_replay_auxiliary_batch =\n      execution.cuda_requested() && correlation_auxiliary && !retain_df_response;"
        in dispatch
    )
    assert "void virtual_corrections() { virtual_corrections(false); }" in owner
    assert "virtual_corrections(true);" in owner
    assert "layout.df_arena_elements" in owner
    assert owner.index(
        "if (allocation == cudaErrorMemoryAllocation && replay_batch)"
    ) < owner.index(
        "if (allocation == cudaErrorMemoryAllocation && contractions.workspace_bytes())"
    )
