"""Experimental public Array-API-shaped frontend over canonical TensorIR.

The ordinary path is shape/dtype based and deliberately hides TensorIR metadata.
Advanced users may still call ``trace`` with explicit scientific index spaces.
This preview does not implement ``__array_namespace__`` and does not claim Array
API conformance.
"""

from __future__ import annotations

import builtins
import functools
import inspect
import typing
from fractions import Fraction

import numpy as np
from generativeqc_compiler.array_api import (
    DLPACK_INTEROP_VERSION,
    FRONTEND_VERSION,
    SUPPORTED_FUNCTIONS,
    DLPackDevice,
    DLPackImport,
    DLPackInteropError,
    ExactScalar,
    VibeArray,
    dlpack_device,
    import_dlpack,
    input_array,
    trace,
)
from generativeqc_compiler.array_api import (
    capabilities as _compiler_capabilities,
)
from generativeqc_compiler.array_api import namespace as _namespace
from generativeqc_compiler.array_api.trace import active_capture as _active_capture
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
)
from generativeqc_compiler.tensor import (
    cast as _cast,
)
from generativeqc_compiler.tensor import (
    execute as _execute,
)

API_VERSION = 1

float32 = np.dtype("float32")
float64 = np.dtype("float64")


def _symbolic(*values: object) -> bool:
    return any(isinstance(value, VibeArray) for value in values)


def _check_host_values(value: object) -> None:
    pending = [value]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if isinstance(item, np.ndarray):
            if not item.dtype.hasobject:
                continue
            children = item.flat
        else:
            if callable(getattr(item, "__dlpack_device__", None)) or (
                inspect.getattr_static(item, "__cuda_array_interface__", None)
                is not None
            ):
                raise TypeError(
                    "asarray does not perform implicit external-device transfer; "
                    "use import_dlpack for an explicit handoff"
                )
            if not isinstance(item, (list, tuple)):
                continue
            children = item
        identity = id(item)
        if identity not in seen:
            seen.add(identity)
            pending.extend(children)


def _eager_array(value: object) -> np.ndarray:
    # Reuse the public host boundary before NumPy can invoke foreign array hooks.
    array = asarray(value)
    assert isinstance(array, np.ndarray)
    return _finite_eager(array)


def _finite_eager(value: typing.Any) -> typing.Any:
    if not np.all(np.isfinite(value)):
        raise ValueError("eager TensorIR reference values must be finite")
    return value


def _eager_compute(
    operation: typing.Callable[..., typing.Any],
    *operands: object,
    **kwargs: typing.Any,
) -> typing.Any:
    with np.errstate(all="ignore"):
        result = operation(*operands, **kwargs)
    return _finite_eager(result)


def _eager_scalar(value: object, dtype: np.dtype, name: str) -> np.ndarray:
    factor = _namespace._generic_scalar(value, name)
    with np.errstate(all="ignore"):
        return _finite_eager(np.asarray(float(factor), dtype=dtype))


def _eager_operands(*values: object, scalars: bool = False) -> tuple[np.ndarray, ...]:
    arrays = tuple(
        None
        if scalars and type(value) in (int, float, str, Fraction)
        else _eager_array(value)
        for value in values
    )
    dtype = next((array.dtype for array in arrays if array is not None), float64)
    if any(array is not None and array.dtype != dtype for array in arrays):
        raise ValueError("operand dtypes must agree; implicit promotion is unsupported")
    return tuple(
        _eager_scalar(value, dtype, "arithmetic scalar") if array is None else array
        for value, array in zip(values, arrays, strict=True)
    )


def add(x1: object, x2: object) -> typing.Any:
    if _symbolic(x1, x2):
        return _namespace.add(x1, x2)
    return _eager_compute(np.add, *_eager_operands(x1, x2, scalars=True))


def subtract(x1: object, x2: object) -> typing.Any:
    if _symbolic(x1, x2):
        return _namespace.subtract(x1, x2)
    return _eager_compute(np.subtract, *_eager_operands(x1, x2, scalars=True))


def multiply(x1: object, x2: object) -> typing.Any:
    if _symbolic(x1, x2):
        return _namespace.multiply(x1, x2)
    return _eager_compute(np.multiply, *_eager_operands(x1, x2, scalars=True))


def divide(x1: object, x2: object) -> typing.Any:
    if _symbolic(x1, x2):
        return _namespace.divide(x1, x2)
    return _eager_compute(np.divide, *_eager_operands(x1, x2, scalars=True))


def negative(x: object) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.negative(x)
    return _eager_compute(np.negative, _eager_array(x))


def pow(x: object, exponent: object) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.pow(x, exponent)
    array = _eager_array(x)
    if np.any(array <= 0):
        raise ValueError("tensor power domain requires strictly positive input")
    factor = _namespace._generic_scalar(exponent, "exponent")
    rounded = _eager_scalar(factor, array.dtype, "exponent")
    if factor and rounded == 0:
        raise ValueError("power exponent is not representable")
    return _eager_compute(np.power, array, rounded)


def exp(x: object) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.exp(x)
    return _eager_compute(np.exp, _eager_array(x))


def log(x: object) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.log(x)
    array = _eager_array(x)
    if np.any(array <= 0):
        raise ValueError("tensor log domain requires strictly positive input")
    return _eager_compute(np.log, array)


def sqrt(x: object) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.sqrt(x)
    array = _eager_array(x)
    if np.any(array < 0):
        raise ValueError("tensor sqrt domain requires nonnegative input")
    return _eager_compute(np.sqrt, array)


def square(x: object) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.square(x)
    return _eager_compute(np.square, _eager_array(x))


def reciprocal(x: object) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.reciprocal(x)
    return _eager_compute(np.reciprocal, _eager_array(x))


def reshape(
    x: object,
    shape: tuple[int, ...],
    *,
    indices: tuple[Index, ...] | None = None,
) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.reshape(x, shape, indices=indices)
    if indices is not None:
        raise ValueError("explicit TensorIR indices require a symbolic array")
    array = _eager_array(x)
    return np.reshape(array, _namespace._reshape_shape(shape, array.size))


def broadcast_to(
    x: object,
    shape: tuple[int, ...],
    *,
    indices: tuple[Index, ...] | None = None,
    axes: tuple[int, ...] | None = None,
) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.broadcast_to(x, shape, indices=indices, axes=axes)
    if indices is not None or axes is not None:
        raise ValueError(
            "explicit TensorIR broadcast metadata requires a symbolic array"
        )
    return np.broadcast_to(_eager_array(x), _namespace._shape(shape, "broadcast_to"))


def broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    return _namespace.broadcast_shapes(*shapes)


def broadcast_arrays(*arrays: object) -> tuple[typing.Any, ...]:
    if not arrays:
        return ()
    if any(isinstance(value, VibeArray) for value in arrays):
        return _namespace.broadcast_arrays(*arrays)
    values = tuple(_eager_array(value) for value in arrays)
    return tuple(np.broadcast_arrays(*values))


def expand_dims(x: object, axis: int | tuple[int, ...]) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.expand_dims(x, axis)
    array = _eager_array(x)
    _namespace._expand_shape(array.shape, axis)
    return np.expand_dims(array, axis)


def squeeze(x: object, axis: int | tuple[int, ...]) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.squeeze(x, axis)
    array = _eager_array(x)
    _namespace._squeezed_shape(array.shape, axis)
    return np.squeeze(array, axis=axis)


def moveaxis(
    x: object, source: int | tuple[int, ...], destination: int | tuple[int, ...]
) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.moveaxis(x, source, destination)
    array = _eager_array(x)
    order = _namespace._moveaxis_order(array.ndim, source, destination)
    return np.transpose(array, order)


def flip(x: object, *, axis: int | tuple[int, ...] | None = None) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.flip(x, axis=axis)
    array = _eager_array(x)
    axes = (
        tuple(range(array.ndim))
        if axis is None
        else _namespace._normalized_axes(axis, array.ndim, "flip")
    )
    return np.flip(array, axis=axes)


def slice(x: object, ranges: tuple[tuple[int, int], ...]) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.slice(x, ranges)
    array = _eager_array(x)
    if not isinstance(ranges, tuple) or any(
        not isinstance(bounds, tuple)
        or len(bounds) != 2
        or any(type(bound) is not int for bound in bounds)
        for bounds in ranges
    ):
        raise TypeError("slice ranges must be a static tuple of (start, stop) pairs")
    if len(ranges) != array.ndim:
        raise ValueError("slice requires one half-open range per axis")
    if any(
        not 0 <= start <= stop <= extent
        for (start, stop), extent in zip(ranges, array.shape, strict=True)
    ):
        raise ValueError("slice range is outside its input axis")
    return array[tuple(builtins.slice(start, stop) for start, stop in ranges)]


def take(x: object, indices: tuple[int, ...], *, axis: int) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.take(x, indices, axis=axis)
    array = _eager_array(x)
    normalized_axis = _namespace._axis(axis, array.ndim, "take")
    if not isinstance(indices, tuple) or any(
        type(index) is not int for index in indices
    ):
        raise TypeError("take indices must be a static tuple of integers")
    if any(not 0 <= index < array.shape[normalized_axis] for index in indices):
        raise ValueError("gather position is outside its input axis")
    return np.take(array, indices, axis=normalized_axis)


def sum(
    x: object,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: object = None,
    keepdims: bool = False,
) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)
    array = _eager_array(x)
    if dtype is not None:
        raise ValueError("frontend sum does not insert dtype conversions")
    if type(keepdims) is not bool:
        raise TypeError("keepdims must be a bool")
    axes = _namespace._reduction_axes(axis, array.ndim, "sum")
    return _eager_compute(np.sum, array, axis=axes, keepdims=keepdims)


def mean(
    x: object,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.mean(x, axis=axis, keepdims=keepdims)
    array = _eager_array(x)
    if type(keepdims) is not bool:
        raise TypeError("keepdims must be a bool")
    axes = _namespace._reduction_axes(axis, array.ndim, "mean")
    count = 1
    for position in axes:
        count *= array.shape[position]
    if count == 0:
        raise ValueError("mean of an empty reduction is unsupported")
    return _eager_compute(np.mean, array, axis=axes, keepdims=keepdims)


def permute_dims(x: object, axes: tuple[int, ...]) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.permute_dims(x, axes)
    array = _eager_array(x)
    return np.transpose(array, _namespace._permutation(axes, array.ndim))


def matrix_transpose(x: object) -> typing.Any:
    if isinstance(x, VibeArray):
        return _namespace.matrix_transpose(x)
    array = _eager_array(x)
    if array.ndim < 2:
        raise ValueError(
            "matrix_transpose requires an array with at least two dimensions"
        )
    return np.swapaxes(array, -1, -2)


def matmul(x1: object, x2: object) -> typing.Any:
    if _symbolic(x1, x2):
        return _namespace.matmul(x1, x2)
    return _eager_compute(np.matmul, *_eager_operands(x1, x2))


def einsum(
    equation: str,
    *operands: object,
    coefficient: ExactScalar = 1,
) -> typing.Any:
    if _symbolic(*operands):
        return _namespace.einsum(equation, *operands, coefficient=coefficient)
    arrays = _eager_operands(*operands)
    # Validate notation and repeated-label extents through the canonical frontend.
    # These inputs carry metadata only; eager numerical work still runs in NumPy.
    _namespace.einsum(
        equation,
        *(
            input_array(
                f"operand_{position}", _generic_spec(array, differentiable=False)
            )
            for position, array in enumerate(arrays)
        ),
        coefficient=coefficient,
    )
    factor = _namespace._exact(coefficient, "einsum coefficient")
    result = _eager_compute(np.einsum, equation, *arrays, optimize=False)
    return _eager_compute(
        np.multiply, result, _eager_scalar(factor, result.dtype, "einsum coefficient")
    )


def _dtype_name(dtype: object) -> str:
    try:
        name = np.dtype(dtype).name
    except TypeError as exc:
        raise TypeError("dtype must describe float32 or float64") from exc
    if name not in ("float32", "float64"):
        raise TypeError("experimental Array API currently supports float32/float64")
    return name


def asarray(
    value: object,
    *,
    dtype: object = None,
    copy: bool | None = None,
) -> typing.Any:
    """Convert a host value or preserve/cast one symbolic array.

    Concrete conversion is intentionally CPU/NumPy-only in this preview. Objects
    advertising DLPack are never silently copied from another array runtime; use
    ``import_dlpack`` for that explicit handoff.
    """
    if copy not in (None, True, False):
        raise TypeError("copy must be True, False, or None")
    if isinstance(value, VibeArray):
        if dtype is None:
            return value
        name = _dtype_name(dtype)
        if name == value.dtype:
            return value
        return VibeArray(_cast(value.node, name))
    _check_host_values(value)
    target = None if dtype is None else _dtype_name(dtype)
    if copy is True:
        array = np.array(value, dtype=target, copy=True)
    elif copy is False:
        array = np.array(value, dtype=target, copy=False)
    else:
        array = np.asarray(value, dtype=target)
    if array.dtype.name not in ("float32", "float64"):
        raise TypeError(
            "experimental Array API runtime inputs must have float32 or float64 dtype"
        )
    return array


def _creation_dtype(dtype: object) -> str:
    return "float64" if dtype is None else _dtype_name(dtype)


def _creation_device(device: object) -> None:
    if device is not None:
        raise ValueError(
            "experimental creation only admits device=None (CPU reference); "
            "device selection requires a qualified backend"
        )


def full(
    shape: int | tuple[int, ...],
    fill_value: object,
    *,
    dtype: object = None,
    device: object = None,
) -> typing.Any:
    """Create a finite uniform array, or capture an exact TensorIR constant."""
    _creation_device(device)
    target = _namespace._creation_shape(shape)
    name = _creation_dtype(dtype)
    if dtype is None and type(fill_value) in (bool, int):
        raise TypeError(
            "full with integer/bool fill_value requires an integer/bool dtype "
            "not supported by this preview; specify a floating dtype explicitly"
        )
    factor = _namespace._generic_scalar(fill_value, "full fill value")
    if _active_capture():
        return _namespace.full(target, factor, dtype=name)
    return _eager_compute(np.full, target, float(factor), dtype=np.dtype(name))


def zeros(
    shape: int | tuple[int, ...], *, dtype: object = None, device: object = None
) -> typing.Any:
    return full(shape, 0, dtype=_creation_dtype(dtype), device=device)


def ones(
    shape: int | tuple[int, ...], *, dtype: object = None, device: object = None
) -> typing.Any:
    return full(shape, 1, dtype=_creation_dtype(dtype), device=device)


def full_like(
    x: object,
    fill_value: object,
    *,
    dtype: object = None,
    device: object = None,
) -> typing.Any:
    """Create a uniform generic array inheriting the input's shape/dtype."""
    _creation_device(device)
    factor = _namespace._generic_scalar(fill_value, "full_like fill value")
    if isinstance(x, VibeArray):
        name = x.dtype if dtype is None else _dtype_name(dtype)
        return _namespace.full_like(x, factor, dtype=name)
    array = _eager_array(x)
    name = array.dtype.name if dtype is None else _dtype_name(dtype)
    return _eager_compute(np.full_like, array, float(factor), dtype=np.dtype(name))


def zeros_like(x: object, *, dtype: object = None, device: object = None) -> typing.Any:
    return full_like(x, 0, dtype=dtype, device=device)


def ones_like(x: object, *, dtype: object = None, device: object = None) -> typing.Any:
    return full_like(x, 1, dtype=dtype, device=device)


def _generic_spec(array: np.ndarray, *, differentiable: bool) -> TensorSpec:
    return TensorSpec(
        _namespace._generic_indices(tuple(int(extent) for extent in array.shape)),
        dtype=array.dtype.name,
        role="parameter" if differentiable else "input",
        differentiable=differentiable,
    )


class CompiledFunction:
    """Shape/dtype-specialized TensorIR capture with reference execution.

    ``compile`` currently means compile Python array expressions to canonical
    TensorIR. Native CPU/CUDA execution remains an explicit follow-on capability;
    this first public path uses the independent NumPy TensorIR interpreter.
    """

    def __init__(
        self,
        function: typing.Callable[..., object],
        *,
        backend: str,
        differentiable: tuple[str, ...],
    ) -> None:
        if not callable(function):
            raise TypeError("compile requires a callable")
        if backend != "reference":
            raise ValueError(
                "experimental compile currently supports backend='reference'"
            )
        signature = inspect.signature(function)
        unsupported = {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        if any(
            parameter.kind in unsupported for parameter in signature.parameters.values()
        ):
            raise TypeError(
                "compiled array functions require named, non-variadic parameters"
            )
        if not isinstance(differentiable, tuple) or any(
            not isinstance(name, str) for name in differentiable
        ):
            raise TypeError("differentiable must be a tuple of parameter names")
        if len(set(differentiable)) != len(differentiable):
            raise ValueError("differentiable parameter names must be unique")
        unknown = set(differentiable) - set(signature.parameters)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown differentiable parameter(s): {names}")
        self._function = function
        self._signature = signature
        self._backend = backend
        self._differentiable = frozenset(differentiable)
        self._programs: dict[tuple[tuple[str, tuple[int, ...], str], ...], Program] = {}
        functools.update_wrapper(self, function)

    def _prepare(
        self, *args: object, **kwargs: object
    ) -> tuple[Program, dict[str, np.ndarray]]:
        bound = self._signature.bind(*args, **kwargs)
        bound.apply_defaults()
        feeds: dict[str, np.ndarray] = {}
        specs: dict[str, TensorSpec] = {}
        key_rows = []
        for name, value in bound.arguments.items():
            array = asarray(value)
            if isinstance(array, VibeArray):
                raise TypeError("compiled runtime arguments must be concrete arrays")
            assert isinstance(array, np.ndarray)
            feeds[name] = array
            specs[name] = _generic_spec(
                array, differentiable=name in self._differentiable
            )
            key_rows.append(
                (name, tuple(int(x) for x in array.shape), array.dtype.name)
            )
        key = tuple(key_rows)
        program = self._programs.get(key)
        if program is None:
            program = trace(
                self._function,
                specs,
                provenance={
                    "public_array_api_version": API_VERSION,
                    "generic_array_semantics": 1,
                },
            )
            self._programs[key] = program
        return program, feeds

    def lower(self, *args: object, **kwargs: object) -> Program:
        """Return the canonical TensorIR specialization without executing it."""
        program, _ = self._prepare(*args, **kwargs)
        return program

    def __call__(self, *args: object, **kwargs: object) -> typing.Any:
        program, feeds = self._prepare(*args, **kwargs)
        execution = _execute(program, feeds)
        if tuple(program.outputs) == ("output",):
            return execution.outputs["output"]
        return execution.outputs


@typing.overload
def compile(
    function: typing.Callable[..., object],
    *,
    backend: str = "reference",
    differentiable: tuple[str, ...] = (),
) -> CompiledFunction: ...


@typing.overload
def compile(
    function: None = None,
    *,
    backend: str = "reference",
    differentiable: tuple[str, ...] = (),
) -> typing.Callable[[typing.Callable[..., object]], CompiledFunction]: ...


def compile(
    function: typing.Callable[..., object] | None = None,
    *,
    backend: str = "reference",
    differentiable: tuple[str, ...] = (),
) -> (
    CompiledFunction | typing.Callable[[typing.Callable[..., object]], CompiledFunction]
):
    """Capture a normal array function lazily from its first concrete signature."""
    if function is None:
        return lambda target: CompiledFunction(
            target, backend=backend, differentiable=differentiable
        )
    return CompiledFunction(function, backend=backend, differentiable=differentiable)


def capabilities() -> dict[str, object]:
    """Return the detached capability contract for this public preview."""
    report = _compiler_capabilities()
    functions = set(typing.cast("tuple[str, ...]", report["functions"]))
    functions.update({"asarray", "compile", "matrix_transpose"})
    report.update(
        {
            "public_api_version": API_VERSION,
            "surface": "array-api-shaped-experimental-public-preview",
            "stability": "experimental",
            "import_path": "generativeqc.experimental.array_api",
            "functions": tuple(sorted(functions)),
            "implicit_broadcast": True,
            "reshape_requires_explicit_indices": False,
            "broadcast_requires_explicit_indices_and_axes": False,
            "scientific_metadata_requires_explicit_indices": True,
            "compiled_call": "shape-dtype-specialized-tensorir-reference",
            "compiled_differentiability": "explicit-parameter-names",
            "namespace_dispatch": "numpy-eager-or-symbolic-tensorir",
            "runtime_array": "numpy-host-float32-float64",
        }
    )
    return report


__all__ = [
    "API_VERSION",
    "DLPACK_INTEROP_VERSION",
    "FRONTEND_VERSION",
    "SUPPORTED_FUNCTIONS",
    "CompiledFunction",
    "DLPackDevice",
    "DLPackImport",
    "DLPackInteropError",
    "ExactScalar",
    "Index",
    "IndexSpace",
    "Program",
    "TensorSpec",
    "VibeArray",
    "add",
    "asarray",
    "broadcast_arrays",
    "broadcast_shapes",
    "broadcast_to",
    "capabilities",
    "compile",
    "divide",
    "dlpack_device",
    "einsum",
    "exp",
    "expand_dims",
    "flip",
    "float32",
    "float64",
    "full",
    "full_like",
    "import_dlpack",
    "input_array",
    "log",
    "matmul",
    "matrix_transpose",
    "mean",
    "moveaxis",
    "multiply",
    "negative",
    "ones",
    "ones_like",
    "permute_dims",
    "pow",
    "reciprocal",
    "reshape",
    "slice",
    "sqrt",
    "square",
    "squeeze",
    "subtract",
    "sum",
    "take",
    "trace",
    "zeros",
    "zeros_like",
]
