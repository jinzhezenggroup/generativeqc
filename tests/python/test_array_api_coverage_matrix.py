"""Machine-readable operation coverage must not imply native Array API support."""

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc_compiler.array_api.capabilities import (
    SUPPORTED_FUNCTIONS,
    capabilities,
)
from generativeqc_compiler.tensor.cpu import CPU_PRIMITIVES, describe_cpu_compilation

# Independent examples exercise the actual frontend-to-TensorIR lowering, rather
# than assuming that public operation names match native primitive names.
_NATIVE_CPU_CASES = {
    "add": lambda x: xp.add(x, 2),
    "subtract": lambda x: xp.subtract(x, 2),
    "multiply": lambda x: xp.multiply(x, x),
    "divide": lambda x: xp.divide(x, 2),
    "negative": xp.negative,
    "pow": lambda x: xp.pow(x, 2),
    "exp": xp.exp,
    "log": xp.log,
    "sqrt": xp.sqrt,
    "square": xp.square,
    "reciprocal": xp.reciprocal,
    "zeros": lambda x: xp.zeros(x.shape, dtype=x.dtype),
    "ones": lambda x: xp.ones(x.shape, dtype=x.dtype),
    "full": lambda x: xp.full(x.shape, 2, dtype=x.dtype),
    "zeros_like": lambda x: xp.zeros_like(x),
    "ones_like": lambda x: xp.ones_like(x),
    "full_like": lambda x: xp.full_like(x, 2),
    "sum": lambda x: xp.sum(x, axis=0, keepdims=True),
    "mean": lambda x: xp.mean(x, axis=0, keepdims=True),
    "permute_dims": lambda x: xp.permute_dims(x, (1, 0)),
    "reshape": lambda x: xp.reshape(x, (3, 2)),
    "broadcast_to": lambda x: xp.broadcast_to(x, (2, 2, 3)),
    "broadcast_arrays": lambda x: {
        name: value
        for name, value in zip(
            ("left", "right"), xp.broadcast_arrays(x, x[:1, :]), strict=True
        )
    },
    "expand_dims": lambda x: xp.expand_dims(x, axis=0),
    "squeeze": lambda x: xp.squeeze(x[:1, :], axis=0),
    "moveaxis": lambda x: xp.moveaxis(x, 0, 1),
    "flip": lambda x: xp.flip(x),
    "matrix_transpose": xp.matrix_transpose,
    "slice": lambda x: xp.slice(x, ((0, 1), (0, 3))),
    "take": lambda x: xp.take(x, (1, 0), axis=0),
    "slice_static": lambda x: xp.slice(x, ((0, 1), (0, 3))),
    "take_static": lambda x: xp.take(x, (1, 0), axis=0),
    "matmul": lambda x: xp.matmul(x, xp.matrix_transpose(x)),
    "einsum_extension": lambda x: xp.einsum("ij,ij->", x, x),
}


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
            assert operation["capture"] == "tensorir"
        assert operation["native_cuda"] is False


def test_native_cpu_coverage_is_explicit_and_float64_only() -> None:
    report = capabilities()
    matrix = report["operation_matrix"]
    supported = set(_NATIVE_CPU_CASES) | {
        "reshape_explicit_indices",
        "broadcast_to_explicit_indices",
    }
    assert {name for name, entry in matrix.items() if entry["native_cpu"]} == supported
    assert report["native_cpu_scope"] == (
        "bounded-tensorir-subset-not-array-api-conformance"
    )
    assert "all-live-nodes-float64" in report["native_cpu_requires"]
    assert (
        "static-shapes-within-byte-work-node-budgets" in report["native_cpu_requires"]
    )
    for name, entry in matrix.items():
        assert entry["native_cpu_dtypes"] == (("float64",) if name in supported else ())
    for name in (
        "equal",
        "not_equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
    ):
        assert matrix[name]["native_cpu_notes"] == (
            "unsupported-boolean-results-and-comparison-primitives"
        )


@pytest.mark.parametrize("name", _NATIVE_CPU_CASES)
@pytest.mark.parametrize("dtype", ("float64", "float32"))
def test_native_cpu_coverage_matches_source_emission(name: str, dtype: str) -> None:
    values = np.arange(1, 7, dtype=dtype).reshape(2, 3)
    program = xp.compile(_NATIVE_CPU_CASES[name]).lower(values)
    if dtype == "float32":
        with pytest.raises(ValueError, match="float64"):
            describe_cpu_compilation(program)
        return
    prepared, source, resources, _ = describe_cpu_compilation(program)
    assert {node.op for node in prepared.live_nodes} <= CPU_PRIMITIVES
    assert {node.spec.dtype for node in prepared.live_nodes} == {"float64"}
    assert "generativeqc_tensor_cpu::run" in source
    assert resources["required_bytes"] > 0


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
    assert operation["native_cpu"] is False
    assert operation["native_cpu_notes"] == (
        "dtype-conversion-unsupported; float64-identity-only-after-cast-elision"
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


@pytest.mark.parametrize(
    "source_dtype,target_dtype", (("float32", "float64"), ("float64", "float32"))
)
def test_native_cpu_dtype_conversion_is_not_qualified(
    source_dtype: str, target_dtype: str
) -> None:
    values = np.asarray([1, 2], dtype=source_dtype)
    program = xp.compile(lambda x: xp.astype(x, target_dtype)).lower(values)
    assert "cast" in {node.op for node in program.live_nodes}
    with pytest.raises(ValueError, match="float64|unsupported CPU primitive: cast"):
        describe_cpu_compilation(program)


@pytest.mark.parametrize("copy", (True, False))
def test_native_cpu_identity_astype_only_elides_the_cast(copy: bool) -> None:
    values = np.asarray([1.0, 2.0])
    program = xp.compile(lambda x: xp.astype(x, xp.float64, copy=copy)).lower(values)
    prepared, _, _, _ = describe_cpu_compilation(program)
    assert {node.op for node in prepared.live_nodes} == {"input"}


@pytest.mark.parametrize(
    "name", ("equal", "not_equal", "greater", "greater_equal", "less", "less_equal")
)
def test_native_cpu_boolean_comparisons_are_not_qualified(name: str) -> None:
    values = np.asarray([1.0, 2.0])
    program = xp.compile(lambda x: getattr(xp, name)(x, 1)).lower(values)
    assert program.outputs["output"].spec.dtype == "bool"
    with pytest.raises(ValueError, match="bool data or comparisons"):
        describe_cpu_compilation(program)


def test_native_cpu_boolean_creation_is_not_qualified() -> None:
    program = xp.compile(lambda x: xp.ones_like(x)).lower(np.asarray([True, False]))
    with pytest.raises(ValueError, match="bool data or comparisons"):
        describe_cpu_compilation(program)


@pytest.mark.parametrize("name", ("reshape", "broadcast_to"))
def test_explicit_index_coverage_requires_symbolic_arrays(name: str) -> None:
    operation = capabilities()["operation_matrix"][f"{name}_explicit_indices"]
    assert operation["function"] == name
    assert operation["eager"] == "unsupported-symbolic-metadata-only"
    assert operation["capture"] == "tensorir"
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
    assert operation["native_cpu"] is True
    prepared, source, _, _ = describe_cpu_compilation(program)
    assert {node.op for node in prepared.live_nodes} <= CPU_PRIMITIVES
    assert "generativeqc_tensor_cpu::run" in source


def test_broadcast_shapes_coverage_is_static_in_both_modes() -> None:
    operation = capabilities()["operation_matrix"]["broadcast_shapes"]
    assert operation["eager"] == operation["capture"] == "python-static-shape"
    assert operation["native_cpu"] is False
    assert operation["native_cpu_notes"] == "python-static-shape; no-native-operation"
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
    describe_cpu_compilation(expression.lower(values))

    with pytest.raises(TypeError, match="VibeArray"):
        xp.compile(lambda x: xp.broadcast_shapes(x.shape, (1, 2))).lower(values)
