"""Captured asarray copy policy, explicit casts and detached output ownership."""

from typing import Any

import array_api_strict as strict
import numpy as np
import pytest
from generativeqc.experimental import array_api as xp


@pytest.mark.parametrize(
    "source,target", [(np.float32, np.float64), (np.float64, np.float32)]
)
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
@pytest.mark.parametrize("backend", ["reference", "cpu"])
def test_asarray_copy_false_rejects_dtype_conversion(
    source: Any, target: Any, shape: Any, backend: str
) -> None:
    values = np.ones(shape, dtype=source)
    with pytest.raises(ValueError):
        strict.asarray(
            strict.asarray(values),
            dtype=getattr(strict, np.dtype(target).name),
            copy=False,
        )
    with pytest.raises(ValueError):
        xp.asarray(values, dtype=target, copy=False)
    compiled = xp.compile(
        lambda x: xp.asarray(x, dtype=target, copy=False), backend=backend
    )
    try:
        with pytest.raises(ValueError, match="requires a copy"):
            compiled(values)
    finally:
        compiled.close()


@pytest.mark.parametrize(
    "source,target", [(np.float32, np.float64), (np.float64, np.float32)]
)
@pytest.mark.parametrize("copy", [None, True])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_asarray_explicit_casts_remain_supported(
    source: Any, target: Any, copy: Any, shape: Any
) -> None:
    values = np.resize(np.array([-0.0, 0.0, 1.1, -2.5], dtype=source), shape)
    expected = np.asarray(values, dtype=target)
    compiled = xp.compile(
        lambda x: xp.asarray(x, dtype=target, copy=copy), backend="reference"
    )
    try:
        for result in (xp.asarray(values, dtype=target, copy=copy), compiled(values)):
            assert result.dtype == expected.dtype
            assert result.shape == expected.shape
            np.testing.assert_array_equal(result, expected)
            np.testing.assert_array_equal(np.signbit(result), np.signbit(expected))
    finally:
        compiled.close()


@pytest.mark.parametrize("dtype", [np.bool_, np.float32, np.float64])
@pytest.mark.parametrize("explicit_dtype", [False, True])
@pytest.mark.parametrize("shape", [(), (0,), (2, 3)])
def test_asarray_same_dtype_copy_false(
    dtype: Any, explicit_dtype: bool, shape: Any
) -> None:
    values = np.resize(np.array([-0.0, 0.0, 1.0, -2.0], dtype=dtype), shape)
    target = dtype if explicit_dtype else None
    eager = xp.asarray(values, dtype=target, copy=False)
    if values.size:
        assert np.shares_memory(eager, values)
    compiled = xp.compile(
        lambda x: xp.asarray(x, dtype=target, copy=False), backend="reference"
    )
    try:
        result = compiled(values)
        assert result.dtype == values.dtype
        np.testing.assert_array_equal(result, values)
        if dtype is not np.bool_:
            np.testing.assert_array_equal(np.signbit(result), np.signbit(values))
        assert not np.shares_memory(result, values)
    finally:
        compiled.close()


def test_asarray_scientific_metadata_and_copy_admission() -> None:
    ao = xp.IndexSpace("active_ao", "ao", 5, spin="alpha")
    axes = (xp.Index("p", ao, 1, 4, (3, 1)),)
    spec = xp.TensorSpec(axes, dtype="float32", role="input", differentiable=True)
    symbolic = xp.input_array("x", spec)
    with pytest.raises(ValueError, match="requires a copy"):
        xp.asarray(symbolic, dtype=xp.float64, copy=False)
    for copy in (None, True):
        result = xp.asarray(symbolic, dtype=xp.float64, copy=copy)
        assert result.dtype == "float64"
        assert result.node.spec.indices == axes
        assert result.node.spec.differentiable
    result = xp.asarray(symbolic, dtype=xp.float32, copy=False)
    assert result.node.spec == spec


@pytest.mark.parametrize(
    "source,target", [(np.bool_, np.float64), (np.float64, np.bool_)]
)
def test_asarray_copy_flag_preserves_bool_real_boundary(
    source: Any, target: Any
) -> None:
    compiled = xp.compile(
        lambda x: xp.asarray(x, dtype=target, copy=False), backend="reference"
    )
    try:
        with pytest.raises(TypeError, match="cross-kind"):
            compiled.lower(np.ones((2,), dtype=source))
    finally:
        compiled.close()


@pytest.mark.parametrize("dtype", [np.bool_, np.float64])
def test_asarray_native_identity_keeps_detached_storage(
    dtype: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    from generativeqc_compiler.tensor import interpreter

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("native asarray used reference execution")

    monkeypatch.setattr(interpreter, "execute", forbidden)
    values = np.array([[0.0, -0.0, 2.0], [3.0, -4.0, 0.0]], dtype=dtype)[:, ::2]
    compiled = xp.compile(
        lambda x: xp.asarray(x, dtype=dtype, copy=False), cache=tmp_path
    )
    try:
        result = compiled(values)
        np.testing.assert_array_equal(result, values)
        assert result.dtype == values.dtype
        assert not np.shares_memory(result, values)
        expected = values.copy()
        values[...] = 0
        np.testing.assert_array_equal(result, expected)
    finally:
        compiled.close()
