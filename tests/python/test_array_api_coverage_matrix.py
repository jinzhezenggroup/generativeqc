"""Machine-readable operation coverage must not imply native Array API support."""

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.array_api.capabilities import (
    SUPPORTED_FUNCTIONS,
    capabilities,
)


def test_coverage_matrix_matches_the_compiler_preview_inventory() -> None:
    report = capabilities()
    matrix = report["operation_matrix"]
    assert set(matrix) == SUPPORTED_FUNCTIONS == set(report["functions"])
    assert report["target_standard_for_gap_analysis"] == "2025.12"
    assert report["array_api_version"] is None
    assert report["array_namespace_protocol"] is False
    assert report["operation_matrix_scope"] == (
        "compiler-preview-inventory-excludes-public-helpers"
    )
    assert xp.capabilities()["operation_matrix"] == matrix
    for operation in matrix.values():
        assert callable(getattr(xp, operation["function"]))
    assert matrix["einsum_extension"]["function"] == "einsum"
    assert matrix["slice_static"]["function"] == "slice"
    assert matrix["take_static"]["function"] == "take"
    assert "asarray" in xp.capabilities()["functions"] and "asarray" not in matrix


def test_coverage_distinguishes_boolean_data_and_reference_execution() -> None:
    matrix = capabilities()["operation_matrix"]
    assert matrix["equal"]["dtypes"] == "float32|float64 -> bool"
    assert matrix["equal"]["ad"] == "non-differentiable"
    assert matrix["sum"]["audited_keywords"]["keepdims"] == "generic-arrays-only"
    assert matrix["astype"]["audited_keywords"]["device"] == "None-only"
    assert "where" not in matrix
    for name, operation in matrix.items():
        if name not in {
            "reshape_explicit_indices",
            "broadcast_to_explicit_indices",
            "broadcast_shapes",
        }:
            assert operation["eager"] == "numpy-reference"
            assert operation["capture"] == "tensorir-reference"
        assert operation["native_cpu"] is False
        assert operation["native_cuda"] is False


def test_coverage_report_is_detached_from_internal_keyword_metadata() -> None:
    first = capabilities()["operation_matrix"]
    first["sum"]["audited_keywords"]["keepdims"] = "changed-by-caller"
    first["broadcast_shapes"]["capture"] = "changed-by-caller"
    second = capabilities()["operation_matrix"]
    assert second["sum"]["audited_keywords"]["keepdims"] == "generic-arrays-only"
    assert second["broadcast_shapes"]["capture"] == "python-static-shape"


def test_astype_coverage_matches_eager_and_captured_admission() -> None:
    operation = capabilities()["operation_matrix"]["astype"]
    assert operation["dtypes"] == "float32|float64"
    assert operation["audited_keywords"]["dtype"] == (
        "real float32/float64 only; bool identity unsupported"
    )
    values = np.asarray([1.0, 2.0], dtype=np.float32)
    assert xp.astype(values, xp.float32, copy=False) is values
    assert not np.shares_memory(xp.astype(values, xp.float32), values)
    np.testing.assert_array_equal(
        xp.compile(lambda x: xp.astype(x, xp.float64), backend="reference")(values),
        values.astype(np.float64),
    )
    mask = np.asarray([True, False])
    np.testing.assert_array_equal(xp.astype(mask, xp.bool), mask)
    with pytest.raises(TypeError, match="float32/float64"):
        xp.compile(lambda x: xp.astype(x, xp.bool), backend="reference")(mask)
    np.testing.assert_array_equal(xp.astype(values, xp.float32, device="cpu"), values)
    with pytest.raises(ValueError, match="device=None"):
        xp.compile(
            lambda x: xp.astype(x, xp.float32, device="cpu"), backend="reference"
        )(values)


@pytest.mark.parametrize("name", ("reshape", "broadcast_to"))
def test_explicit_index_coverage_requires_symbolic_arrays(name: str) -> None:
    operation = capabilities()["operation_matrix"][f"{name}_explicit_indices"]
    assert operation["function"] == name
    assert operation["eager"] == "unsupported-symbolic-metadata-only"
    assert operation["capture"] == "tensorir-reference"
    indices = (xp.Index("p", xp.IndexSpace("ao", "ao", 2)),)
    spec = xp.TensorSpec(indices, role="input")
    values = np.asarray([1.0, 2.0])

    def expression(x: object) -> object:
        if name == "reshape":
            return xp.reshape(x, (2,), indices=indices)
        return xp.broadcast_to(x, (2,), indices=indices, axes=(0,))

    with pytest.raises(ValueError, match="symbolic"):
        expression(values)
    program = xp.trace(expression, {"x": spec})
    assert program.outputs["output"].spec.indices == indices


def test_broadcast_shapes_coverage_is_static_in_both_modes() -> None:
    operation = capabilities()["operation_matrix"]["broadcast_shapes"]
    assert operation["eager"] == operation["capture"] == "python-static-shape"
    assert (
        operation["dtypes"]
        == operation["device"]
        == operation["ad"]
        == ("not-applicable-static-shape-tuples")
    )
    assert xp.broadcast_shapes((2,), (1, 2)) == (1, 2)

    @xp.compile(backend="reference")
    def expression(x: object) -> object:
        return xp.reshape(x, xp.broadcast_shapes(x.shape, (1, 2)))

    values = np.asarray([1.0, 2.0])
    np.testing.assert_array_equal(expression(values), values.reshape(1, 2))
    assert {node.op for node in expression.lower(values).nodes} == {"input", "reshape"}
