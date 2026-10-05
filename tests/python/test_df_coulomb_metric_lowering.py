"""Streamed metric algebra, explicit updates, and provider-neutral region identity."""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc_compiler.tensor.contraction_update import contraction_update_request
from generativeqc_compiler.tensor.df_coulomb_metric import metric_program
from generativeqc_compiler.tensor.interpreter import execute
from generativeqc_compiler.tensor.ir import add, einsum, input_tensor
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter
from generativeqc_compiler.tensor.metric_lowering import OPERATIONS, metric_request
from generativeqc_compiler.tensor.program import Program
from generativeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec
from generativeqc_compiler.tensor.vector_lowering import vector_portfolio


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("naux,panel", [(1, 1), (7, 3), (13, 8)])
def test_metric_views_against_scalar_oracle(
    operation: str, naux: int, panel: int
) -> None:
    program = metric_program(naux, panel, operation)
    root = program.outputs["result"]
    product = root.inputs[1] if operation == "metric_charge" else root
    rng = np.random.default_rng(1886)
    matrix = rng.normal(size=product.inputs[0].spec.shape)
    vector = rng.normal(size=product.inputs[1].spec.shape)
    seed = rng.normal(size=root.spec.shape)
    expected = np.zeros(root.spec.shape)
    for out in range(expected.size):
        for inner in range(vector.size):
            entry = (
                matrix[out, inner]
                if operation == "metric_project"
                else matrix[inner, out]
            )
            expected[out] += float(entry) * float(vector[inner])
    inputs = {"matrix": matrix, "vector": vector}
    if operation == "metric_charge":
        inputs["seed"] = seed
        expected += seed
    actual = execute(program, inputs).outputs["result"]
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_metric_candidate_contract_includes_padding_and_seed(operation: str) -> None:
    adapter, root, product, cuda = metric_request(operation, backend="cuda")
    _, _, _, cpu = metric_request(operation, backend="cpu")
    assert cpu.semantic_identity == cuda.semantic_identity
    assert cpu.scientific_identity == cuda.scientific_identity
    candidates, _, _ = vector_portfolio(cuda, "a" * 64)
    assert len(candidates) == 2
    for candidate in candidates:
        assert candidate.request is cuda and candidate.execution is not None
        assert candidate.execution.precision == cuda.precisions[0]
        assert candidate.execution.layouts == cuda.operands
    if operation == "metric_potential":
        assert cuda.operands[0].shape == (5, 3)
        assert cuda.operands[0].strides == (5, 1)
    if operation == "metric_charge":
        assert len(cuda.input_dtypes) == 3
        assert cuda.precisions[0].input_dtypes == ("float64",) * 3
        assert (
            cuda.operands[2].alias_group
            == cuda.operands[3].alias_group
            == "donated-seed"
        )
        assert dict(cuda.effects) == {"output": "donated", "donated_input": "input:2"}
        assert (
            cuda.semantic_identity
            != adapter.request(product, backend="cuda").semantic_identity
        )
        assert dict(cuda.semantics)["node_hash"] == adapter.hashes[root]
    else:
        assert len(cuda.input_dtypes) == 2


def test_update_rejects_changed_add_coefficients_and_escaping_seed() -> None:
    original = metric_program(5, 3, "metric_charge")
    seed, product = original.outputs["result"].inputs
    root = add(seed, product, coefficients=(2, 1))
    with pytest.raises(ValueError, match="unit add coefficients"):
        contraction_update_request(
            TensorLoweringAdapter(Program({"result": root})), root, backend="cuda"
        )
    root = original.outputs["result"]
    for escape in (seed, add(seed, seed)):
        with pytest.raises(ValueError, match="exclusive last use"):
            contraction_update_request(
                TensorLoweringAdapter(Program({"result": root, "old": escape})),
                root,
                backend="cuda",
            )


def test_seed_cannot_alias_a_product_input() -> None:
    space = IndexSpace("shared", "auxiliary", 5)
    i, j = Index("i", space), Index("j", space)
    matrix = input_tensor("matrix", TensorSpec((i, j), role="input"))
    vector = input_tensor("shared", TensorSpec((j,), role="input"))
    product = einsum("ij,j->i", matrix, vector)
    root = add(vector, product)
    with pytest.raises(ValueError, match="exclusive last use"):
        contraction_update_request(
            TensorLoweringAdapter(Program({"result": root})), root, backend="cuda"
        )
