"""Array API 2025.12 real-floating dtype helpers and explicit promotion gates."""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp
from generativeqc.extensions import tensor


@pytest.mark.parametrize(
    ("operation", "oracle"),
    (
        (xp.add, np.add),
        (xp.subtract, np.subtract),
        (xp.multiply, np.multiply),
        (xp.divide, np.divide),
        (xp.matmul, np.matmul),
        (
            lambda x, y: xp.einsum("ij,jk->ik", x, y),
            lambda x, y: np.einsum("ij,jk->ik", x, y),
        ),
    ),
)
@pytest.mark.parametrize(
    ("first", "second"),
    (
        (np.float32, np.float64),
        (np.float64, np.float32),
        (np.float32, np.float32),
        (np.float64, np.float64),
    ),
)
def test_eager_and_compiled_binary_promote_like_array_api(
    operation: object, oracle: object, first: object, second: object
) -> None:
    def expression(x: object, y: object) -> object:
        return operation(x, y)

    x = np.asarray([[2.0, 3.0], [4.0, 6.0]], dtype=first)
    y = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=second)
    expected = oracle(x, y)
    eager = expression(x, y)
    compiled = xp.compile(expression)(x, y)
    expected_dtype = np.result_type(x.dtype, y.dtype)
    assert eager.dtype == compiled.dtype == expected_dtype
    np.testing.assert_allclose(eager, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(compiled, expected, rtol=2e-6, atol=2e-6)


def test_mixed_dtype_broadcasting_inserts_explicit_cast_nodes() -> None:
    @xp.compile
    def expression(x: object, y: object) -> object:
        return x + y

    x = np.asarray([[1.0], [2.0]], dtype=np.float32)
    y = np.asarray([[3.0, 4.0, 5.0]], dtype=np.float64)
    program = expression.lower(x, y)
    casts = [node for node in program.nodes if node.op == "cast"]
    assert casts and all(node.attrs["dtype"] == "float64" for node in casts)
    assert program.outputs["output"].spec.dtype == "float64"
    np.testing.assert_array_equal(expression(x, y), x + y)


def test_mixed_dtype_scalar_arrays_follow_promotion_rules() -> None:
    f32 = np.asarray(2.0, dtype=np.float32)
    f64 = np.asarray([1.0, 3.0], dtype=np.float64)

    @xp.compile
    def expression(x: object, y: object) -> object:
        return x * y + 0.5

    expected = f32 * f64 + 0.5
    assert expected.dtype == np.dtype("float64")
    np.testing.assert_array_equal(expression(f32, f64), expected)


@pytest.mark.parametrize("dtype", (xp.float32, xp.float64))
def test_astype_copy_contract_and_explicit_tensorir_cast(dtype: object) -> None:
    values = np.asarray([1.0, 2.0], dtype=dtype)
    assert xp.astype(values, dtype, copy=False) is values
    copied = xp.astype(values, dtype, copy=True)
    assert copied is not values and not np.shares_memory(copied, values)

    other = xp.float64 if dtype == xp.float32 else xp.float32
    converted = xp.astype(values, other)
    assert converted.dtype == other
    np.testing.assert_array_equal(converted, values.astype(other))

    @xp.compile
    def expression(x: object) -> object:
        return xp.astype(x, other)

    program = expression.lower(values)
    assert any(node.op == "cast" for node in program.nodes)
    np.testing.assert_array_equal(expression(values), values.astype(other))


def test_scientific_float_precision_remains_explicit() -> None:
    ao = xp.IndexSpace("ao", "ao", 2)
    indices = (xp.Index("p", ao),)
    spec32 = xp.TensorSpec(indices, dtype="float32", role="input")
    spec64 = xp.TensorSpec(indices, dtype="float64", role="input")
    with pytest.raises(ValueError, match="dtype"):
        xp.trace(lambda x, y: x + y, {"x": spec32, "y": spec64})

    converted = xp.trace(
        lambda x, y: xp.astype(x, xp.float64) + y,
        {"x": spec32, "y": spec64},
    )
    values32 = np.asarray([1.0, 2.0], dtype=np.float32)
    values64 = np.asarray([3.0, 4.0], dtype=np.float64)
    np.testing.assert_array_equal(
        tensor.execute(converted, {"x": values32, "y": values64}).outputs["output"],
        values32.astype(np.float64) + values64,
    )


def test_mixed_precision_capture_reuses_tensorir_ad() -> None:
    @xp.compile(differentiable=("x",))
    def expression(x: object, y: object) -> object:
        return xp.sum(x * y)

    x = np.asarray([1.0, 2.0], dtype=np.float32)
    y = np.asarray([3.0, 4.0], dtype=np.float64)
    direction = np.asarray([0.5, -1.0], dtype=np.float32)
    program = expression.lower(x, y)
    forward = tensor.jvp(program, {"x": x, "y": y}, {"x": direction})
    np.testing.assert_allclose(
        forward.output_tangents["output"],
        np.sum(direction.astype(np.float64) * y),
    )


def test_real_floating_dtype_introspection_and_promotion() -> None:
    assert xp.can_cast(xp.float32, xp.float64)
    assert not xp.can_cast(xp.float64, xp.float32)
    assert xp.can_cast(xp.float64, xp.float64)
    assert xp.can_cast(np.ones(2, dtype=np.float32), xp.float64)

    assert xp.result_type(xp.float32, xp.float64) == xp.float64
    assert xp.result_type(xp.float32, 0.5) == xp.float32
    assert xp.result_type(np.ones(2, dtype=np.float64), 1.0) == xp.float64
    assert xp.isdtype(xp.float32, "real floating")
    assert xp.isdtype(xp.float64, ("integral", "real floating"))
    assert xp.isdtype(xp.float32, "numeric")
    assert xp.isdtype(xp.float32, xp.float32)
    assert not xp.isdtype(xp.float32, "integral")
    assert not xp.isdtype(xp.float32, xp.float64)


@pytest.mark.parametrize("dtype", (xp.float32, xp.float64))
def test_finfo_preserves_machine_limits(dtype: object) -> None:
    for entry in (dtype, np.ones(2, dtype=dtype)):
        actual = xp.finfo(entry)
        expected = np.finfo(dtype)
        assert actual.dtype == dtype
        assert actual.bits == expected.bits
        assert actual.eps == expected.eps
        assert actual.max == expected.max
        assert actual.min == expected.min
        assert actual.smallest_normal == expected.smallest_normal


def test_dtype_admission_fails_closed() -> None:
    values = np.ones(2, dtype=np.float64)
    with pytest.raises(TypeError, match="float32/float64"):
        xp.astype(values, np.int32)
    with pytest.raises(TypeError, match="bool"):
        xp.astype(values, xp.float32, copy=0)
    with pytest.raises(ValueError, match="device=None"):
        xp.astype(values, xp.float64, device="cuda:0")
    with pytest.raises(TypeError, match="array or dtype"):
        xp.result_type(1.0, 2.0)
    with pytest.raises(ValueError, match="unsupported dtype kind"):
        xp.isdtype(xp.float32, "floating")
    with pytest.raises(TypeError, match="float32/float64"):
        xp.can_cast(np.dtype("int32"), xp.float64)

    class Foreign:
        def __dlpack_device__(self) -> tuple[int, int]:
            return (2, 0)

        def __array__(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("foreign conversion is forbidden")

    with pytest.raises(TypeError, match="explicit handoff"):
        xp.finfo(Foreign())


def test_capability_report_keeps_nonconformance_explicit() -> None:
    report = xp.capabilities()
    assert report["array_api_version"] is None
    assert report["array_namespace_protocol"] is False
    assert report["dtype_promotion"] == "generic-float32-float64-explicit-cast"
    assert report["scientific_dtype_promotion"] is False
    for operation in ("astype", "can_cast", "finfo", "isdtype", "result_type"):
        assert operation in report["functions"]
