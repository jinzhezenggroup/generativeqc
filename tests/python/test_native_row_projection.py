"""Leading-row views retain canonical science and fail closed on interior slices."""

from __future__ import annotations

import json

import pytest
from generativeqc_compiler.method.df_mo_source import build_df_mo_source_program
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter
from generativeqc_compiler.tensor.native_lowering import projected_contraction_request


def test_projection_preserves_parent_science_and_physical_input_order() -> None:
    program = build_df_mo_source_program(5, 7)
    first = program.outputs["transformed"].inputs[1]
    adapter = TensorLoweringAdapter(program)
    parent = adapter.request(first, backend="cuda")
    assert projected_contraction_request(adapter, first) == parent
    fixed = (first.attrs["labels"][0][0],)
    sliced = projected_contraction_request(adapter, first, fixed_modes=fixed)
    assert sliced.scientific_identity == parent.scientific_identity
    assert sliced.semantic_identity != parent.semantic_identity
    assert (
        dict(sliced.semantics)["parent_semantic_identity"] == parent.semantic_identity
    )
    fixed_json = dict(sliced.semantics)["fixed_modes"]
    assert isinstance(fixed_json, str)
    assert json.loads(fixed_json) == list(fixed)
    assert sliced.operands[0].shape == (5, 7)
    assert sliced.operands[0].strides == (7, 1)
    assert sliced.operands[0].modes == parent.operands[0].modes[1:]
    assert sliced.operands[1] == parent.operands[1]
    assert sliced.shape == (5, 7)
    assert dict(sliced.semantics)["output_elements"] == 35
    assert dict(sliced.semantics)["reduction_extent"] == 5
    swapped = projected_contraction_request(
        adapter, first, fixed_modes=fixed, operand_order=(1, 0)
    )
    assert swapped.operands == (
        sliced.operands[1],
        sliced.operands[0],
        sliced.operands[2],
    )
    assert swapped.scientific_identity == parent.scientific_identity
    assert swapped.semantic_identity != sliced.semantic_identity
    assert swapped.precisions == sliced.precisions  # Both inputs remain strict FP64.


@pytest.mark.parametrize("kind", ["interior", "duplicate", "unknown", "order"])
def test_projection_rejects_nonleading_or_invalid_views(kind: str) -> None:
    program = build_df_mo_source_program(5, 7)
    first = program.outputs["transformed"].inputs[1]
    labels = first.attrs["labels"][0]
    fixed = {
        "interior": (labels[1],),
        "duplicate": (labels[0], labels[0]),
        "unknown": (999,),
        "order": (),
    }[kind]
    with pytest.raises(ValueError, match="leading fixed|invalid fixed|ordered einsum"):
        projected_contraction_request(
            TensorLoweringAdapter(program),
            first,
            fixed_modes=fixed,
            operand_order=(0, 0) if kind == "order" else (0, 1),
        )
