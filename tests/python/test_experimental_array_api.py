"""Experimental public Array API facade over the canonical TensorIR frontend."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from fractions import Fraction

import numpy as np
from generativeqc.experimental import API_VERSION as EXPERIMENTAL_API_VERSION
from generativeqc.experimental import array_api as xp
from generativeqc.extensions import tensor
from generativeqc_compiler.array_api import namespace as compiler_xp
from generativeqc_compiler.array_api import trace as compiler_trace


def _vector_spec() -> xp.TensorSpec:
    ao = xp.IndexSpace("ao", "ao", 3)
    return xp.TensorSpec((xp.Index("p", ao),), role="input")


def test_experimental_package_keeps_array_surface_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                import generativeqc.experimental as experimental

                assert experimental.API_VERSION == 1
                assert experimental.__all__ == ["API_VERSION", "array_api"]
                assert "array_api" in dir(experimental)
                assert "generativeqc.experimental.array_api" not in sys.modules
                assert not any(
                    name.startswith("generativeqc_compiler.array_api")
                    for name in sys.modules
                )
                """
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_public_preview_reuses_canonical_tensorir_identity_and_execution() -> None:
    spec = _vector_spec()
    public = xp.trace(
        lambda x: {"out": xp.sum(Fraction(1, 2) * x * x)},
        {"x": spec},
    )
    internal = compiler_trace(
        lambda x: {"out": compiler_xp.sum(Fraction(1, 2) * x * x)},
        {"x": spec},
    )

    assert isinstance(public, xp.Program)
    assert public.logical_hash == internal.logical_hash
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    execution = tensor.execute(public, {"x": values})
    assert execution.outputs["out"] == 7.0


def test_compiled_observable_uses_array_syntax_without_tensor_specs() -> None:
    @xp.compile
    def observable(C: object, occupation: object, O: object) -> object:
        density = (C * occupation) @ C.T
        return xp.sum(density * O)

    coefficients = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64
    )
    occupation = xp.asarray([2.0, 1.0], dtype=xp.float64)
    operator = np.eye(3, dtype=np.float64)

    actual = observable(coefficients, occupation, operator)
    weighted = coefficients * occupation
    expected = np.sum((weighted @ coefficients.T) * operator)
    np.testing.assert_allclose(actual, expected)

    program = observable.lower(coefficients, occupation, operator)
    assert isinstance(program, xp.Program)
    assert {node.attrs["name"] for node in program.nodes if node.op == "input"} == {
        "C",
        "occupation",
        "O",
    }


def test_generic_indexing_reshape_and_batched_matmul_match_numpy() -> None:
    @xp.compile
    def transform(x: object, y: object) -> object:
        column = x[::-1, None]
        flat = xp.reshape(column, (1, -1))
        return flat @ y

    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    actual = transform(x, y)
    expected = x[::-1, None].reshape(1, -1) @ y
    np.testing.assert_allclose(actual, expected)

    @xp.compile
    def batched(left: object, right: object) -> object:
        return left @ right

    left = np.arange(12.0, dtype=np.float64).reshape(2, 2, 3)
    right = np.arange(24.0, dtype=np.float64).reshape(2, 3, 4)
    np.testing.assert_allclose(batched(left, right), left @ right)


def test_generic_broadcasting_and_exact_scalar_operators_match_numpy() -> None:
    @xp.compile
    def expression(matrix: object, vector: object) -> object:
        return (matrix + 1) * vector - Fraction(1, 2)

    matrix = np.arange(6.0, dtype=np.float64).reshape(2, 3)
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    expected = (matrix + 1.0) * vector - 0.5
    np.testing.assert_allclose(expression(matrix, vector), expected)


def test_public_preview_declares_experimental_nonconformance() -> None:
    assert EXPERIMENTAL_API_VERSION == 1
    assert xp.API_VERSION == 1
    report = xp.capabilities()
    assert report["public_api_version"] == 1
    assert report["surface"] == "array-api-shaped-experimental-public-preview"
    assert report["stability"] == "experimental"
    assert report["import_path"] == "generativeqc.experimental.array_api"
    assert report["array_api_version"] is None
    assert report["array_namespace_protocol"] is False
    assert report["implicit_broadcast"] is True
    assert report["reshape_requires_explicit_indices"] is False
    assert report["scientific_metadata_requires_explicit_indices"] is True
    assert report["compiled_call"] == "shape-dtype-specialized-tensorir-reference"

    value = xp.input_array("x", _vector_spec())
    assert isinstance(value, xp.VibeArray)
    assert value.size == 3
    assert not hasattr(value, "__array_namespace__")


def test_public_preview_exports_array_style_operations() -> None:
    for name in (
        "add",
        "asarray",
        "broadcast_to",
        "compile",
        "divide",
        "einsum",
        "exp",
        "log",
        "matmul",
        "matrix_transpose",
        "multiply",
        "negative",
        "permute_dims",
        "pow",
        "reshape",
        "slice",
        "sqrt",
        "subtract",
        "sum",
        "take",
    ):
        assert callable(getattr(xp, name))


def test_asarray_rejects_implicit_external_device_transfer() -> None:
    class External:
        def __dlpack_device__(self) -> tuple[int, int]:
            return (2, 0)

    with np.testing.assert_raises_regex(TypeError, "explicit handoff"):
        xp.asarray(External())
