"""Ordered derivative regions retain the existing scalar program contract."""

from dataclasses import replace

import numpy as np
import pytest
from generativeqc_compiler.dft.cosx_contraction import (
    cosx_esp_program,
    cosx_jet_projection_program,
    cosx_matrix_program,
    emit_cosx_derivative_contractions,
)
from generativeqc_compiler.method.cosx_derivative_runtime import (
    build_cosx_projection_update_program,
    build_cosx_scale_program,
)
from generativeqc_compiler.tensor import Program, add, einsum, execute, input_tensor
from generativeqc_compiler.tensor.batch_scaled_contraction import (
    batch_scaled_contraction_request,
)
from generativeqc_compiler.tensor.checked_contraction import checked_contraction_request
from generativeqc_compiler.tensor.lowering import TensorLoweringAdapter


@pytest.mark.parametrize("weighted", (False, True))
def test_complete_checked_identity(weighted: bool) -> None:
    program = (
        cosx_esp_program(5, 3) if weighted else cosx_matrix_program(5, 3, update=False)
    )
    adapter = TensorLoweringAdapter(program)
    root = program.outputs["result"]
    update, scale = build_cosx_projection_update_program(), build_cosx_scale_program()
    requests = []
    for backend in ("cpu", "cuda"):
        base = (
            batch_scaled_contraction_request(adapter, root, backend=backend)
            if weighted
            else adapter.request(root, backend=backend)
        )
        product = next(n for n in root.inputs if n.op == "einsum") if weighted else root
        request = checked_contraction_request(
            base, update, scale if weighted else None, contraction=product
        )
        assert request.scientific_identity == base.scientific_identity
        assert dict(request.semantics)["scalar_update_hash"] == update.logical_hash
        assert (
            dict(request.effects)["reduction_step"]
            == "finite-inputs-and-output-or-zero"
        )
        assert request.constraints.determinism == "exact-order"
        requests.append(request)
    assert requests[0].semantic_identity == requests[1].semantic_identity


def test_checked_recognition_rejects_changed_program_or_region() -> None:
    program = cosx_matrix_program(5, 3, update=False)
    base = TensorLoweringAdapter(program).request(
        program.outputs["result"], backend="cuda"
    )
    update = build_cosx_projection_update_program()
    bad = Program(
        {"updated": add(update.outputs["updated"], update.outputs["updated"])}
    )
    with pytest.raises(ValueError, match="update graph"):
        checked_contraction_request(base, bad, contraction=program.outputs["result"])
    factors = [
        input_tensor(n.attrs["name"], replace(n.spec, dtype="float32"))
        for n in program.outputs["result"].inputs
    ]
    fp32 = Program({"result": einsum("pm,mn->pn", *factors)})
    fp32_request = TensorLoweringAdapter(fp32).request(
        fp32.outputs["result"], backend="cuda"
    )
    with pytest.raises(ValueError, match="complete FP64"):
        checked_contraction_request(
            fp32_request, update, contraction=fp32.outputs["result"]
        )
    with pytest.raises(ValueError, match="original node"):
        checked_contraction_request(
            base,
            update,
            build_cosx_scale_program(),
            contraction=program.outputs["result"],
        )


def test_asymmetric_jet_projection_against_independent_long_double() -> None:
    points, columns = 5, 4
    jets = (np.arange(3 * points * columns).reshape(3, points, columns) - 23) / 41
    density = (np.arange(columns * columns).reshape(columns, columns) - 5) / 29
    expected = np.zeros((3, points, columns), dtype=np.longdouble)
    for axis in range(3):
        for point in range(points):
            for column in range(columns):
                for row in range(columns):
                    expected[axis, point, column] += np.longdouble(
                        jets[axis, point, row]
                    ) * np.longdouble(density[row, column])
    actual = execute(
        cosx_jet_projection_program(points, columns), {"jets": jets, "density": density}
    ).outputs["result"]
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=3e-14)


def test_derivative_emission_keeps_checked_scalar_owner_and_axis_schedule() -> None:
    update, scale = build_cosx_projection_update_program(), build_cosx_scale_program()
    source = emit_cosx_derivative_contractions(update, scale)
    assert source == emit_cosx_derivative_contractions(update, scale)
    assert update.logical_hash in source and scale.logical_hash in source
    assert "ordered-checked-scalar" in source
    assert "generated_cosx_derivative::accumulate_projection" in source
    assert "execute_checked<ScalarStep<false>>" in source
    assert "axis < 3" in source and "jets+axis*stride" in source
    assert "cublasDgemm(" not in source


@pytest.mark.parametrize("coefficient", (-1, 2))
def test_checked_native_binding_rejects_nonunit_original_coefficient(
    coefficient: int,
) -> None:
    from generativeqc_compiler.tensor.native_lowering import contraction_initializer

    original = cosx_matrix_program(5, 3, update=False)
    product = einsum(
        "pm,mn->pn", *original.outputs["result"].inputs, coefficient=coefficient
    )
    adapter = TensorLoweringAdapter(Program({"result": product}))
    with pytest.raises(ValueError, match="unit original contraction"):
        contraction_initializer(
            adapter,
            product,
            lambda index: index.space.name,
            transpose=("N", "N"),
            extents=("1", "points", "columns", "columns"),
            coefficient="1.0",
            checked_update=build_cosx_projection_update_program(),
        )


@pytest.mark.parametrize("weighted", (False, True))
def test_checked_request_binds_original_node_proof(weighted: bool) -> None:
    program = (
        cosx_esp_program(5, 3) if weighted else cosx_matrix_program(5, 3, update=False)
    )
    adapter = TensorLoweringAdapter(program)
    root = program.outputs["result"]
    request = (
        batch_scaled_contraction_request(adapter, root, backend="cuda")
        if weighted
        else adapter.request(root, backend="cuda")
    )
    unrelated = cosx_matrix_program(7, 3, update=False).outputs["result"]
    with pytest.raises(ValueError, match="original node"):
        checked_contraction_request(
            request,
            build_cosx_projection_update_program(),
            build_cosx_scale_program() if weighted else None,
            contraction=unrelated,
        )


def test_checked_request_rejects_unit_proof_for_scaled_request() -> None:
    original = cosx_matrix_program(5, 3, update=False).outputs["result"]
    product = einsum("pm,mn->pn", *original.inputs, coefficient=2)
    adapter = TensorLoweringAdapter(Program({"result": product}))
    request = adapter.request(product, backend="cuda")
    update = build_cosx_projection_update_program()
    with pytest.raises(ValueError, match="unit original contraction"):
        checked_contraction_request(request, update, contraction=product)
    with pytest.raises(ValueError, match="original node"):
        checked_contraction_request(request, update, contraction=original)


@pytest.mark.parametrize("outer", (False, True))
def test_checked_weighted_region_keeps_inner_and_outer_unit_guards(outer: bool) -> None:
    original = cosx_esp_program(5, 3).outputs["result"]
    weight, product = original.inputs
    if not outer:
        product = einsum("pmn,pn->pm", *product.inputs, coefficient=2)
    root = einsum("p,pm->pm", weight, product, coefficient=2 if outer else 1)
    adapter = TensorLoweringAdapter(Program({"result": root}))
    with pytest.raises(ValueError, match="cannot reduce, reorder, broadcast or cast"):
        batch_scaled_contraction_request(adapter, root, backend="cuda")
