"""Complete weighted regions retain the original SSA and physical batch views."""

import numpy as np
import pytest
from generativeqc_compiler.dft.cosx_contraction import cosx_esp_program
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    einsum,
    input_tensor,
)
from generativeqc_compiler.tensor.batch_scaled_contraction import (
    batch_scaled_contraction_request,
)
from generativeqc_compiler.tensor.interpreter import execute
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter


@pytest.mark.parametrize("points,columns", [(1, 1), (5, 17), (13, 7)])
def test_weighted_esp_matches_long_double(points: int, columns: int) -> None:
    """Asymmetric ESP, signed/zero weights and unequal dimensions expose bad cuts."""
    program = cosx_esp_program(points, columns)
    generator = np.random.default_rng(1884)
    esp = generator.normal(size=(points, columns, columns))
    projected = generator.normal(size=(points, columns))
    weight = np.linspace(-0.7, 0.9, points)
    weight[0] = 0
    expected = np.zeros((points, columns))
    for point in range(points):
        for row in range(columns):
            value = np.longdouble(0)
            for column in range(columns):
                value += (
                    np.longdouble(esp[point, row, column]) * projected[point, column]
                )
            expected[point, row] = value * np.longdouble(weight[point])
    actual = execute(program, {"esp": esp, "projected": projected, "weight": weight})
    np.testing.assert_allclose(
        actual.outputs["result"], expected, atol=3e-13, rtol=3e-13
    )
    root = program.outputs["result"]
    adapter = TensorLoweringAdapter(program)
    cpu, cuda = [
        batch_scaled_contraction_request(adapter, root, backend=backend)
        for backend in ("cpu", "cuda")
    ]
    assert cpu.scientific_identity == cuda.scientific_identity
    assert cpu.semantic_identity == cuda.semantic_identity
    assert dict(cpu.semantics)["node_hash"] == adapter.hashes[root]
    assert dict(cpu.semantics)["reduction_extent"] == columns
    assert dict(cpu.semantics)["batch_scale_mode"] == 0
    assert cpu.input_dtypes == ("float64",) * 3
    assert [value.operand for value in cpu.operands] == [
        "input:0",
        "input:1",
        "input:2",
        "output",
    ]
    assert cpu.operands[2].shape == (points,)
    assert cpu.operands[2].modes == (0,)
    assert dict(cpu.effects)["nonfinite_publication"] == "sticky-error-and-zero"


def test_published_intermediate_rejects_fusion() -> None:
    program = cosx_esp_program(5, 7)
    root = program.outputs["result"]
    product = next(value for value in root.inputs if value.op == "einsum")
    published = Program({"result": root, "unweighted": product})
    with pytest.raises(ValueError, match="exclusive"):
        batch_scaled_contraction_request(
            TensorLoweringAdapter(published), root, backend="cuda"
        )


def test_weight_axis_cannot_be_a_matrix_row() -> None:
    point = Index("p", IndexSpace("points", "batch", 5))
    ao = IndexSpace("columns", "ao", 7)
    row, column = Index("m", ao), Index("n", ao)
    left = input_tensor("left", TensorSpec((point, row), role="input"))
    right = input_tensor("right", TensorSpec((row, column), role="input"))
    weight = input_tensor("weight", TensorSpec((point,), role="input"))
    product = einsum("pm,mn->pn", left, right)
    root = einsum("p,pn->pn", weight, product)
    with pytest.raises(ValueError, match="leading shared"):
        batch_scaled_contraction_request(
            TensorLoweringAdapter(Program({"result": root})), root, backend="cuda"
        )


def test_outer_scale_cannot_reorder_publication() -> None:
    program = cosx_esp_program(5, 7)
    product = next(
        value for value in program.outputs["result"].inputs if value.op == "einsum"
    )
    weight = next(
        value for value in program.outputs["result"].inputs if value.op == "input"
    )
    reordered = einsum("p,pm->mp", weight, product)
    with pytest.raises(ValueError, match="reorder"):
        batch_scaled_contraction_request(
            TensorLoweringAdapter(Program({"result": reordered})),
            reordered,
            backend="cuda",
        )
