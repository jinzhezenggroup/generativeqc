"""Public contraction semantics, independent NumPy oracles and derivative gates."""

from typing import Any

import numpy as np
import pytest
from generativeqc.experimental import array_api as xp


@pytest.mark.parametrize(
    "dtypes",
    [(np.float32, np.float32), (np.float64, np.float64), (np.float32, np.float64)],
)
@pytest.mark.parametrize(
    ("left_shape", "right_shape", "axes"),
    [
        ((2, 3, 4), (4, 3, 5), ((-1, 1), (0, -2))),
        ((2, 3, 4), (3, 4, 5), 2),
        ((2, 3), (3, 2), ((1, 0), (0, 1))),
        ((2,), (3,), 0),
        ((), (2, 3), ((), ())),
        ((), (), 0),
        ((2, 0), (0, 3), 1),
        ((0,), (0,), 1),
    ],
)
def test_tensordot_eager_capture_numpy(
    left_shape: Any, right_shape: Any, axes: Any, dtypes: Any
) -> None:
    rng = np.random.default_rng(2111)
    left = np.asarray(rng.normal(size=left_shape), dtype=dtypes[0])
    right = np.asarray(rng.normal(size=right_shape), dtype=dtypes[1])
    compiled = xp.compile(
        lambda a, b: xp.tensordot(a, b, axes=axes), backend="reference"
    )
    expected = np.tensordot(left, right, axes=axes)
    for result in (xp.tensordot(left, right, axes=axes), compiled(left, right)):
        assert result.dtype == expected.dtype
        assert result.shape == expected.shape
        assert not np.shares_memory(result, left)
        assert not np.shares_memory(result, right)
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    "dtypes",
    [(np.float32, np.float32), (np.float64, np.float64), (np.float32, np.float64)],
)
@pytest.mark.parametrize(
    ("left_shape", "right_shape", "axis"),
    [
        ((2, 1, 4), (3, 4), -1),
        ((2, 4, 1), (4, 3), -2),
        ((4, 2), (4, 1), 0),
        ((4,), (4,), -1),
        ((2, 0), (1, 0), -1),
        ((0, 4), (1, 4), -1),
    ],
)
def test_vecdot_eager_capture_numpy(
    left_shape: Any, right_shape: Any, axis: int, dtypes: Any
) -> None:
    rng = np.random.default_rng(2112)
    left = np.asarray(rng.normal(size=left_shape), dtype=dtypes[0])
    right = np.asarray(rng.normal(size=right_shape), dtype=dtypes[1])
    compiled = xp.compile(lambda a, b: xp.vecdot(a, b, axis=axis), backend="reference")
    expected = np.vecdot(left, right, axis=axis)
    for result in (xp.vecdot(left, right, axis=axis), compiled(left, right)):
        assert result.dtype == expected.dtype
        assert result.shape == expected.shape
        assert not np.shares_memory(result, left)
        assert not np.shares_memory(result, right)
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    ("axes", "error"),
    [
        (-1, ValueError),
        (3, ValueError),
        (True, TypeError),
        (1.0, TypeError),
        (((0, -2), (0, 1)), ValueError),
        (((0,), ()), ValueError),
        (((2,), (0,)), ValueError),
        (((True,), (0,)), TypeError),
        ((0, 0), TypeError),
        (((0.0,), (0,)), TypeError),
    ],
)
def test_tensordot_rejects_invalid_axes(axes: Any, error: type[Exception]) -> None:
    compiled = xp.compile(
        lambda a, b: xp.tensordot(a, b, axes=axes), backend="reference"
    )
    with pytest.raises(error):
        compiled.lower(np.ones((2, 2)), np.ones((2, 2)))


@pytest.mark.parametrize("axis", [True, 1.0, (), 2, -3])
def test_vecdot_rejects_invalid_axis(axis: Any) -> None:
    compiled = xp.compile(lambda a, b: xp.vecdot(a, b, axis=axis), backend="reference")
    with pytest.raises((TypeError, ValueError)):
        compiled.lower(np.ones((2, 2)), np.ones((2, 2)))


@pytest.mark.parametrize(
    ("operation", "left", "right"),
    [
        (lambda a, b: xp.tensordot(a, b, axes=1), (2, 1), (3, 2)),
        (lambda a, b: xp.vecdot(a, b), (2, 1), (2, 3)),
        (lambda a, b: xp.vecdot(a, b), (2, 4), (3, 4)),
        (lambda a, b: xp.vecdot(a, b), (), ()),
    ],
)
def test_invalid_contraction_shapes_fail_in_eager_and_capture(
    operation: Any, left: Any, right: Any
) -> None:
    a, b = np.ones(left), np.ones(right)
    with pytest.raises((ValueError, IndexError)):
        operation(a, b)
    with pytest.raises(ValueError):
        xp.compile(lambda a, b: operation(a, b), backend="reference").lower(a, b)


@pytest.mark.parametrize(
    "operation", [lambda a, b: xp.tensordot(a, b, axes=1), xp.vecdot]
)
def test_contractions_reject_boolean_and_scientific_domains(operation: Any) -> None:
    with pytest.raises(TypeError, match="float32 or float64"):
        xp.compile(lambda a, b: operation(a, b), backend="reference").lower(
            np.ones(2, dtype=bool), np.ones(2, dtype=bool)
        )
    space = xp.IndexSpace("ao", "ao", 2)
    spec = xp.TensorSpec((xp.Index("p", space),), role="input")
    a = xp.input_array("a", spec)
    b = xp.input_array("b", spec)
    with pytest.raises(TypeError, match="scientific domains"):
        operation(a, b)


@pytest.mark.parametrize(
    "operation", [lambda a, b: xp.tensordot(a, b, axes=0), xp.vecdot]
)
def test_contraction_label_budget(operation: Any) -> None:
    # Metadata only: no large allocation or compiler invocation.
    rank = 27 if operation is not xp.vecdot else 53
    a = np.ones((1,) * rank)
    with pytest.raises(ValueError, match="52-label"):
        xp.compile(lambda a, b: operation(a, b), backend="reference").lower(a, a)


def _derivative_gate(kind: str, backend: str, **options: Any) -> None:
    rng = np.random.default_rng(2113)
    if kind == "tensordot":
        left = rng.normal(size=(2, 3, 4))
        right = rng.normal(size=(4, 3, 2))
        function = lambda a, b: xp.tensordot(a, b, axes=((2, 1), (0, 1)))
        oracle = lambda a, b: np.tensordot(a, b, axes=((2, 1), (0, 1)))
    else:
        left = rng.normal(size=(2, 1, 4))
        right = rng.normal(size=(3, 4))
        function = lambda a, b: xp.vecdot(a, b)
        oracle = lambda a, b: np.sum(a * b, axis=-1)
    dl, dr = rng.normal(size=left.shape), rng.normal(size=right.shape)
    value, tangent = xp.jvp(
        function, (left, right), (dl, dr), backend=backend, **options
    )
    np.testing.assert_allclose(value, oracle(left, right), rtol=2e-13, atol=2e-13)
    step = 1e-5
    finite = (
        oracle(left + step * dl, right + step * dr)
        - oracle(left - step * dl, right - step * dr)
    ) / (2 * step)
    np.testing.assert_allclose(tangent, finite, rtol=2e-9, atol=2e-9)
    primal, pullback = xp.vjp(function, left, right, backend=backend, **options)
    np.testing.assert_allclose(primal, value, rtol=2e-13, atol=2e-13)
    weight = rng.normal(size=value.shape)
    gl, gr = pullback(weight)
    # Independent coordinate finite differences check each adjoint, including
    # reduction over vecdot's broadcast dimensions.
    for source, gradient in ((left, gl), (right, gr)):
        expected = np.empty_like(source)
        for index in np.ndindex(source.shape):
            saved = source[index]
            source[index] = saved + step
            plus = np.sum(weight * oracle(left, right))
            source[index] = saved - step
            minus = np.sum(weight * oracle(left, right))
            source[index] = saved
            expected[index] = (plus - minus) / (2 * step)
        np.testing.assert_allclose(gradient, expected, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(
        np.sum(weight * tangent),
        np.sum(gl * dl) + np.sum(gr * dr),
        rtol=2e-13,
        atol=2e-13,
    )
    pullback.close()


@pytest.mark.parametrize("kind", ["tensordot", "vecdot"])
def test_contraction_reference_derivatives(kind: str) -> None:
    _derivative_gate(kind, "reference")


@pytest.mark.parametrize("kind", ["tensordot", "vecdot"])
def test_contraction_native_derivatives(
    kind: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.tensor import autodiff, interpreter

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("native contraction delegated to a reference evaluator")

    monkeypatch.setattr(interpreter, "execute", forbidden)
    monkeypatch.setattr(autodiff, "jvp", forbidden)
    monkeypatch.setattr(autodiff, "vjp", forbidden)
    _derivative_gate(kind, "cpu", cache=tmp_path)


@pytest.mark.parametrize(
    ("left_shape", "right_shape", "axes"),
    [((), (), 0), ((2, 0), (0, 3), 1), ((2,), (3,), 0)],
)
def test_tensordot_scalar_outer_and_empty_derivatives(
    left_shape: Any, right_shape: Any, axes: Any
) -> None:
    left = np.full(left_shape, 2.0)
    right = np.full(right_shape, 3.0)

    def expression(a: Any, b: Any) -> Any:
        return xp.tensordot(a, b, axes=axes)

    value, tangent = xp.jvp(
        expression,
        (left, right),
        (np.ones_like(left), np.ones_like(right)),
        backend="reference",
    )
    expected = np.tensordot(left, right, axes=axes)
    expected_tangent = np.tensordot(
        np.ones_like(left), right, axes=axes
    ) + np.tensordot(left, np.ones_like(right), axes=axes)
    np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(tangent, expected_tangent)
    _, pullback = xp.vjp(expression, left, right, backend="reference")
    gl, gr = pullback(np.ones_like(value))
    np.testing.assert_array_equal(gl, np.full(left.shape, right.sum()))
    np.testing.assert_array_equal(gr, np.full(right.shape, left.sum()))
    pullback.close()


@pytest.mark.parametrize(
    ("left_shape", "right_shape", "axes"),
    [
        ((), (), 0),
        ((2,), (3,), 0),
        ((2, 0), (0, 3), 1),
        ((2, 3), (3, 2), ((1, 0), (0, 1))),
    ],
)
def test_native_tensordot_boundary_values(
    left_shape: Any, right_shape: Any, axes: Any, tmp_path: Any
) -> None:
    left, right = np.full(left_shape, 2.0), np.full(right_shape, 3.0)
    function = xp.compile(lambda a, b: xp.tensordot(a, b, axes=axes), cache=tmp_path)
    result = function(left, right)
    np.testing.assert_array_equal(result, np.tensordot(left, right, axes=axes))
    assert not np.shares_memory(result, left)
    function.close()


@pytest.mark.parametrize(
    ("left_shape", "right_shape", "axis"),
    [
        ((2, 4, 1), (4, 3), -2),
        ((0, 4), (1, 4), -1),
        ((2, 0), (1, 0), -1),
        ((4,), (4,), -1),
    ],
)
def test_native_vecdot_boundary_values(
    left_shape: Any, right_shape: Any, axis: int, tmp_path: Any
) -> None:
    left, right = np.full(left_shape, 2.0), np.full(right_shape, 3.0)
    function = xp.compile(lambda a, b: xp.vecdot(a, b, axis=axis), cache=tmp_path)
    np.testing.assert_array_equal(
        function(left, right), np.vecdot(left, right, axis=axis)
    )
    function.close()
