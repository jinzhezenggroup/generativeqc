"""Array API 2025.12 on CPU, with an explicitly bounded TensorIR compiler.

Eager arrays use NumPy-owned CPU storage and implement standardized discovery,
dtypes, IEEE behavior, data-dependent shapes, linalg and fft. Symbolic arrays
retain scientific index metadata and reject operations outside TensorIR's
qualified subset; eager conformance does not imply native CPU/CUDA compilation.
"""

from __future__ import annotations

import builtins
import functools
import inspect
import typing
from dataclasses import replace

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

from . import _array_api_eager as _eager
from . import _array_api_symbolic as _symbolic_backend
from ._array_api_eager import Array

API_VERSION = 1
__array_api_version__ = "2025.12"

bool = np.bool_
int8 = np.int8
int16 = np.int16
int32 = np.int32
int64 = np.int64
uint8 = np.uint8
uint16 = np.uint16
uint32 = np.uint32
uint64 = np.uint64
float32 = np.float32
float64 = np.float64
complex64 = np.complex64
complex128 = np.complex128

e = np.e
pi = np.pi
inf = np.inf
nan = np.nan
newaxis = None

_symbolic_functions = {
    name: getattr(_symbolic_backend, name, None) for name in _eager.FUNCTION_NAMES
}


def _contains_symbolic(values: object) -> builtins.bool:
    """Find symbolic operands inside standard nested array argument lists."""
    pending = [(values, 0)]
    seen: set[int] = set()
    inspected = 0
    while pending:
        value, depth = pending.pop()
        inspected += 1
        if inspected > 1_000_000:
            raise ValueError("host container inspection exceeds 1000000 items")
        if isinstance(value, VibeArray):
            return True
        if isinstance(value, (list, tuple)) and id(value) not in seen:
            if depth >= 64:
                raise ValueError("host container nesting exceeds 64 levels")
            seen.add(id(value))
            pending.extend((item, depth + 1) for item in value)
    return False


def _dispatch(name: str, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Keep standard eager execution out of the symbolic capture boundary."""
    if _active_capture() or _contains_symbolic((*args, *kwargs.values())):
        symbolic = _symbolic_functions.get(name)
        if symbolic is None:
            raise NotImplementedError(f"{name} is not supported by TensorIR capture")
        accepted = inspect.signature(symbolic).parameters
        for key in tuple(kwargs):
            if key not in accepted:
                if kwargs[key] is not None:
                    raise NotImplementedError(f"TensorIR {name} does not support {key}")
                del kwargs[key]
        return symbolic(*args, **kwargs)
    return _eager.call(name, *args, **kwargs)


def __array_namespace_info__() -> typing.Any:
    """Inspect CPU devices, dtype defaults and standard eager capabilities."""
    return _eager.namespace_info()


def arange(
    start: typing.Any,
    /,
    stop: typing.Any = None,
    step: typing.Any = 1,
    *,
    dtype: typing.Any = None,
    device: typing.Any = None,
) -> typing.Any:
    """Execute standard arange on CPU, or its bounded TensorIR lowering."""
    return _dispatch("arange", start, stop, step, dtype=dtype, device=device)


def asarray(
    obj: typing.Any,
    /,
    *,
    dtype: typing.Any = None,
    device: typing.Any = None,
    copy: builtins.bool | None = None,
) -> typing.Any:
    """Execute standard asarray on CPU, or its bounded TensorIR lowering."""
    return _dispatch("asarray", obj, dtype=dtype, device=device, copy=copy)


def empty(
    shape: typing.Any, *, dtype: typing.Any = None, device: typing.Any = None
) -> typing.Any:
    """Execute standard empty on CPU, or its bounded TensorIR lowering."""
    return _dispatch("empty", shape, dtype=dtype, device=device)


def empty_like(
    x: typing.Any, /, *, dtype: typing.Any = None, device: typing.Any = None
) -> typing.Any:
    """Execute standard empty_like on CPU, or its bounded TensorIR lowering."""
    return _dispatch("empty_like", x, dtype=dtype, device=device)


def eye(
    n_rows: typing.Any,
    n_cols: typing.Any = None,
    /,
    *,
    k: typing.Any = 0,
    dtype: typing.Any = None,
    device: typing.Any = None,
) -> typing.Any:
    """Execute standard eye on CPU, or its bounded TensorIR lowering."""
    return _dispatch("eye", n_rows, n_cols, k=k, dtype=dtype, device=device)


def from_dlpack(
    x: typing.Any, /, *, device: typing.Any = None, copy: builtins.bool | None = None
) -> typing.Any:
    """Execute standard from_dlpack on CPU, or its bounded TensorIR lowering."""
    return _dispatch("from_dlpack", x, device=device, copy=copy)


def full(
    shape: typing.Any,
    fill_value: typing.Any,
    *,
    dtype: typing.Any = None,
    device: typing.Any = None,
) -> typing.Any:
    """Execute standard full on CPU, or its bounded TensorIR lowering."""
    return _dispatch("full", shape, fill_value, dtype=dtype, device=device)


def full_like(
    x: typing.Any,
    /,
    fill_value: typing.Any,
    *,
    dtype: typing.Any = None,
    device: typing.Any = None,
) -> typing.Any:
    """Execute standard full_like on CPU, or its bounded TensorIR lowering."""
    return _dispatch("full_like", x, fill_value, dtype=dtype, device=device)


def linspace(
    start: typing.Any,
    stop: typing.Any,
    /,
    num: typing.Any,
    *,
    dtype: typing.Any = None,
    device: typing.Any = None,
    endpoint: typing.Any = True,
) -> typing.Any:
    """Execute standard linspace on CPU, or its bounded TensorIR lowering."""
    return _dispatch(
        "linspace", start, stop, num, dtype=dtype, device=device, endpoint=endpoint
    )


def meshgrid(*arrays: typing.Any, indexing: typing.Any = "xy") -> typing.Any:
    """Execute standard meshgrid on CPU, or its bounded TensorIR lowering."""
    return _dispatch("meshgrid", *arrays, indexing=indexing)


def ones(
    shape: typing.Any, *, dtype: typing.Any = None, device: typing.Any = None
) -> typing.Any:
    """Execute standard ones on CPU, or its bounded TensorIR lowering."""
    return _dispatch("ones", shape, dtype=dtype, device=device)


def ones_like(
    x: typing.Any, /, *, dtype: typing.Any = None, device: typing.Any = None
) -> typing.Any:
    """Execute standard ones_like on CPU, or its bounded TensorIR lowering."""
    return _dispatch("ones_like", x, dtype=dtype, device=device)


def tril(x: typing.Any, /, *, k: typing.Any = 0) -> typing.Any:
    """Execute standard tril on CPU, or its bounded TensorIR lowering."""
    return _dispatch("tril", x, k=k)


def triu(x: typing.Any, /, *, k: typing.Any = 0) -> typing.Any:
    """Execute standard triu on CPU, or its bounded TensorIR lowering."""
    return _dispatch("triu", x, k=k)


def zeros(
    shape: typing.Any, *, dtype: typing.Any = None, device: typing.Any = None
) -> typing.Any:
    """Execute standard zeros on CPU, or its bounded TensorIR lowering."""
    return _dispatch("zeros", shape, dtype=dtype, device=device)


def zeros_like(
    x: typing.Any, /, *, dtype: typing.Any = None, device: typing.Any = None
) -> typing.Any:
    """Execute standard zeros_like on CPU, or its bounded TensorIR lowering."""
    return _dispatch("zeros_like", x, dtype=dtype, device=device)


def astype(
    x: typing.Any,
    dtype: typing.Any,
    /,
    *,
    copy: builtins.bool = True,
    device: typing.Any = None,
) -> typing.Any:
    """Execute standard astype on CPU, or its bounded TensorIR lowering."""
    return _dispatch("astype", x, dtype, copy=copy, device=device)


def can_cast(from_: typing.Any, to: typing.Any, /) -> builtins.bool:
    """Execute standard can_cast on CPU, or its bounded TensorIR lowering."""
    return _dispatch("can_cast", from_, to)


def finfo(type: typing.Any, /) -> typing.Any:
    """Execute standard finfo on CPU, or its bounded TensorIR lowering."""
    return _dispatch("finfo", type)


def iinfo(type: typing.Any, /) -> typing.Any:
    """Execute standard iinfo on CPU, or its bounded TensorIR lowering."""
    return _dispatch("iinfo", type)


def isdtype(dtype: typing.Any, kind: typing.Any) -> builtins.bool:
    """Execute standard isdtype on CPU, or its bounded TensorIR lowering."""
    return _dispatch("isdtype", dtype, kind)


def result_type(*arrays_and_dtypes: typing.Any) -> typing.Any:
    """Execute standard result_type on CPU, or its bounded TensorIR lowering."""
    return _dispatch("result_type", *arrays_and_dtypes)


def abs(x: typing.Any, /) -> typing.Any:
    """Execute standard abs on CPU, or its bounded TensorIR lowering."""
    return _dispatch("abs", x)


def acos(x: typing.Any, /) -> typing.Any:
    """Execute standard acos on CPU, or its bounded TensorIR lowering."""
    return _dispatch("acos", x)


def acosh(x: typing.Any, /) -> typing.Any:
    """Execute standard acosh on CPU, or its bounded TensorIR lowering."""
    return _dispatch("acosh", x)


def add(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard add on CPU, or its bounded TensorIR lowering."""
    return _dispatch("add", x1, x2)


def asin(x: typing.Any, /) -> typing.Any:
    """Execute standard asin on CPU, or its bounded TensorIR lowering."""
    return _dispatch("asin", x)


def asinh(x: typing.Any, /) -> typing.Any:
    """Execute standard asinh on CPU, or its bounded TensorIR lowering."""
    return _dispatch("asinh", x)


def atan(x: typing.Any, /) -> typing.Any:
    """Execute standard atan on CPU, or its bounded TensorIR lowering."""
    return _dispatch("atan", x)


def atan2(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard atan2 on CPU, or its bounded TensorIR lowering."""
    return _dispatch("atan2", x1, x2)


def atanh(x: typing.Any, /) -> typing.Any:
    """Execute standard atanh on CPU, or its bounded TensorIR lowering."""
    return _dispatch("atanh", x)


def bitwise_and(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard bitwise_and on CPU, or its bounded TensorIR lowering."""
    return _dispatch("bitwise_and", x1, x2)


def bitwise_left_shift(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard bitwise_left_shift on CPU, or its bounded TensorIR lowering."""
    return _dispatch("bitwise_left_shift", x1, x2)


def bitwise_invert(x: typing.Any, /) -> typing.Any:
    """Execute standard bitwise_invert on CPU, or its bounded TensorIR lowering."""
    return _dispatch("bitwise_invert", x)


def bitwise_or(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard bitwise_or on CPU, or its bounded TensorIR lowering."""
    return _dispatch("bitwise_or", x1, x2)


def bitwise_right_shift(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard bitwise_right_shift on CPU, or its bounded TensorIR lowering."""
    return _dispatch("bitwise_right_shift", x1, x2)


def bitwise_xor(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard bitwise_xor on CPU, or its bounded TensorIR lowering."""
    return _dispatch("bitwise_xor", x1, x2)


def ceil(x: typing.Any, /) -> typing.Any:
    """Execute standard ceil on CPU, or its bounded TensorIR lowering."""
    return _dispatch("ceil", x)


def clip(
    x: typing.Any, /, min: typing.Any = None, max: typing.Any = None
) -> typing.Any:
    """Execute standard clip on CPU, or its bounded TensorIR lowering."""
    return _dispatch("clip", x, min, max)


def conj(x: typing.Any, /) -> typing.Any:
    """Execute standard conj on CPU, or its bounded TensorIR lowering."""
    return _dispatch("conj", x)


def copysign(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard copysign on CPU, or its bounded TensorIR lowering."""
    return _dispatch("copysign", x1, x2)


def cos(x: typing.Any, /) -> typing.Any:
    """Execute standard cos on CPU, or its bounded TensorIR lowering."""
    return _dispatch("cos", x)


def cosh(x: typing.Any, /) -> typing.Any:
    """Execute standard cosh on CPU, or its bounded TensorIR lowering."""
    return _dispatch("cosh", x)


def divide(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard divide on CPU, or its bounded TensorIR lowering."""
    return _dispatch("divide", x1, x2)


def equal(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard equal on CPU, or its bounded TensorIR lowering."""
    return _dispatch("equal", x1, x2)


def exp(x: typing.Any, /) -> typing.Any:
    """Execute standard exp on CPU, or its bounded TensorIR lowering."""
    return _dispatch("exp", x)


def expm1(x: typing.Any, /) -> typing.Any:
    """Execute standard expm1 on CPU, or its bounded TensorIR lowering."""
    return _dispatch("expm1", x)


def floor(x: typing.Any, /) -> typing.Any:
    """Execute standard floor on CPU, or its bounded TensorIR lowering."""
    return _dispatch("floor", x)


def floor_divide(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard floor_divide on CPU, or its bounded TensorIR lowering."""
    return _dispatch("floor_divide", x1, x2)


def greater(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard greater on CPU, or its bounded TensorIR lowering."""
    return _dispatch("greater", x1, x2)


def greater_equal(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard greater_equal on CPU, or its bounded TensorIR lowering."""
    return _dispatch("greater_equal", x1, x2)


def hypot(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard hypot on CPU, or its bounded TensorIR lowering."""
    return _dispatch("hypot", x1, x2)


def imag(x: typing.Any, /) -> typing.Any:
    """Execute standard imag on CPU, or its bounded TensorIR lowering."""
    return _dispatch("imag", x)


def isfinite(x: typing.Any, /) -> typing.Any:
    """Execute standard isfinite on CPU, or its bounded TensorIR lowering."""
    return _dispatch("isfinite", x)


def isinf(x: typing.Any, /) -> typing.Any:
    """Execute standard isinf on CPU, or its bounded TensorIR lowering."""
    return _dispatch("isinf", x)


def isnan(x: typing.Any, /) -> typing.Any:
    """Execute standard isnan on CPU, or its bounded TensorIR lowering."""
    return _dispatch("isnan", x)


def less(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard less on CPU, or its bounded TensorIR lowering."""
    return _dispatch("less", x1, x2)


def less_equal(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard less_equal on CPU, or its bounded TensorIR lowering."""
    return _dispatch("less_equal", x1, x2)


def log(x: typing.Any, /) -> typing.Any:
    """Execute standard log on CPU, or its bounded TensorIR lowering."""
    return _dispatch("log", x)


def log1p(x: typing.Any, /) -> typing.Any:
    """Execute standard log1p on CPU, or its bounded TensorIR lowering."""
    return _dispatch("log1p", x)


def log2(x: typing.Any, /) -> typing.Any:
    """Execute standard log2 on CPU, or its bounded TensorIR lowering."""
    return _dispatch("log2", x)


def log10(x: typing.Any, /) -> typing.Any:
    """Execute standard log10 on CPU, or its bounded TensorIR lowering."""
    return _dispatch("log10", x)


def logaddexp(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard logaddexp on CPU, or its bounded TensorIR lowering."""
    return _dispatch("logaddexp", x1, x2)


def logical_and(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard logical_and on CPU, or its bounded TensorIR lowering."""
    return _dispatch("logical_and", x1, x2)


def logical_not(x: typing.Any, /) -> typing.Any:
    """Execute standard logical_not on CPU, or its bounded TensorIR lowering."""
    return _dispatch("logical_not", x)


def logical_or(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard logical_or on CPU, or its bounded TensorIR lowering."""
    return _dispatch("logical_or", x1, x2)


def logical_xor(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard logical_xor on CPU, or its bounded TensorIR lowering."""
    return _dispatch("logical_xor", x1, x2)


def maximum(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard maximum on CPU, or its bounded TensorIR lowering."""
    return _dispatch("maximum", x1, x2)


def minimum(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard minimum on CPU, or its bounded TensorIR lowering."""
    return _dispatch("minimum", x1, x2)


def multiply(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard multiply on CPU, or its bounded TensorIR lowering."""
    return _dispatch("multiply", x1, x2)


def negative(x: typing.Any, /) -> typing.Any:
    """Execute standard negative on CPU, or its bounded TensorIR lowering."""
    return _dispatch("negative", x)


def nextafter(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard nextafter on CPU, or its bounded TensorIR lowering."""
    return _dispatch("nextafter", x1, x2)


def not_equal(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard not_equal on CPU, or its bounded TensorIR lowering."""
    return _dispatch("not_equal", x1, x2)


def positive(x: typing.Any, /) -> typing.Any:
    """Execute standard positive on CPU, or its bounded TensorIR lowering."""
    return _dispatch("positive", x)


def pow(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard pow on CPU, or its bounded TensorIR lowering."""
    return _dispatch("pow", x1, x2)


def real(x: typing.Any, /) -> typing.Any:
    """Execute standard real on CPU, or its bounded TensorIR lowering."""
    return _dispatch("real", x)


def reciprocal(x: typing.Any, /) -> typing.Any:
    """Execute standard reciprocal on CPU, or its bounded TensorIR lowering."""
    return _dispatch("reciprocal", x)


def remainder(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard remainder on CPU, or its bounded TensorIR lowering."""
    return _dispatch("remainder", x1, x2)


def round(x: typing.Any, /) -> typing.Any:
    """Execute standard round on CPU, or its bounded TensorIR lowering."""
    return _dispatch("round", x)


def sign(x: typing.Any, /) -> typing.Any:
    """Execute standard sign on CPU, or its bounded TensorIR lowering."""
    return _dispatch("sign", x)


def signbit(x: typing.Any, /) -> typing.Any:
    """Execute standard signbit on CPU, or its bounded TensorIR lowering."""
    return _dispatch("signbit", x)


def sin(x: typing.Any, /) -> typing.Any:
    """Execute standard sin on CPU, or its bounded TensorIR lowering."""
    return _dispatch("sin", x)


def sinh(x: typing.Any, /) -> typing.Any:
    """Execute standard sinh on CPU, or its bounded TensorIR lowering."""
    return _dispatch("sinh", x)


def square(x: typing.Any, /) -> typing.Any:
    """Execute standard square on CPU, or its bounded TensorIR lowering."""
    return _dispatch("square", x)


def sqrt(x: typing.Any, /) -> typing.Any:
    """Execute standard sqrt on CPU, or its bounded TensorIR lowering."""
    return _dispatch("sqrt", x)


def subtract(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard subtract on CPU, or its bounded TensorIR lowering."""
    return _dispatch("subtract", x1, x2)


def tan(x: typing.Any, /) -> typing.Any:
    """Execute standard tan on CPU, or its bounded TensorIR lowering."""
    return _dispatch("tan", x)


def tanh(x: typing.Any, /) -> typing.Any:
    """Execute standard tanh on CPU, or its bounded TensorIR lowering."""
    return _dispatch("tanh", x)


def trunc(x: typing.Any, /) -> typing.Any:
    """Execute standard trunc on CPU, or its bounded TensorIR lowering."""
    return _dispatch("trunc", x)


def take(
    x: typing.Any, indices: typing.Any, /, *, axis: typing.Any = None
) -> typing.Any:
    """Execute standard take on CPU, or its bounded TensorIR lowering."""
    return _dispatch("take", x, indices, axis=axis)


def take_along_axis(
    x: typing.Any, indices: typing.Any, /, *, axis: typing.Any = -1
) -> typing.Any:
    """Execute standard take_along_axis on CPU, or its bounded TensorIR lowering."""
    return _dispatch("take_along_axis", x, indices, axis=axis)


def matmul(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard matmul on CPU, or its bounded TensorIR lowering."""
    return _dispatch("matmul", x1, x2)


def matrix_transpose(x: typing.Any, /) -> typing.Any:
    """Execute standard matrix_transpose on CPU, or its bounded TensorIR lowering."""
    return _dispatch("matrix_transpose", x)


def tensordot(x1: typing.Any, x2: typing.Any, /, *, axes: typing.Any = 2) -> typing.Any:
    """Execute standard tensordot on CPU, or its bounded TensorIR lowering."""
    return _dispatch("tensordot", x1, x2, axes=axes)


def vecdot(x1: typing.Any, x2: typing.Any, /, *, axis: typing.Any = -1) -> typing.Any:
    """Execute standard vecdot on CPU, or its bounded TensorIR lowering."""
    return _dispatch("vecdot", x1, x2, axis=axis)


def broadcast_arrays(*arrays: typing.Any) -> typing.Any:
    """Execute standard broadcast_arrays on CPU, or its bounded TensorIR lowering."""
    return _dispatch("broadcast_arrays", *arrays)


def broadcast_shapes(*shapes: typing.Any) -> typing.Any:
    """Execute standard broadcast_shapes on CPU, or its bounded TensorIR lowering."""
    return _dispatch("broadcast_shapes", *shapes)


def broadcast_to(
    x: typing.Any,
    /,
    shape: typing.Any,
    *,
    indices: typing.Any = None,
    axes: typing.Any = None,
) -> typing.Any:
    """Execute standard broadcast_to on CPU, or its bounded TensorIR lowering."""
    return _dispatch("broadcast_to", x, shape, indices=indices, axes=axes)


def concat(arrays: typing.Any, /, *, axis: typing.Any = 0) -> typing.Any:
    """Execute standard concat on CPU, or its bounded TensorIR lowering."""
    return _dispatch("concat", arrays, axis=axis)


def expand_dims(x: typing.Any, /, axis: typing.Any) -> typing.Any:
    """Execute standard expand_dims on CPU, or its bounded TensorIR lowering."""
    return _dispatch("expand_dims", x, axis)


def flip(x: typing.Any, /, *, axis: typing.Any = None) -> typing.Any:
    """Execute standard flip on CPU, or its bounded TensorIR lowering."""
    return _dispatch("flip", x, axis=axis)


def moveaxis(
    x: typing.Any, source: typing.Any, destination: typing.Any, /
) -> typing.Any:
    """Execute standard moveaxis on CPU, or its bounded TensorIR lowering."""
    return _dispatch("moveaxis", x, source, destination)


def permute_dims(x: typing.Any, /, axes: typing.Any) -> typing.Any:
    """Execute standard permute_dims on CPU, or its bounded TensorIR lowering."""
    return _dispatch("permute_dims", x, axes)


def repeat(
    x: typing.Any, repeats: typing.Any, /, *, axis: typing.Any = None
) -> typing.Any:
    """Execute standard repeat on CPU, or its bounded TensorIR lowering."""
    return _dispatch("repeat", x, repeats, axis=axis)


def reshape(
    x: typing.Any,
    /,
    shape: typing.Any,
    *,
    copy: builtins.bool | None = None,
    indices: typing.Any = None,
) -> typing.Any:
    """Execute standard reshape on CPU, or its bounded TensorIR lowering."""
    return _dispatch("reshape", x, shape, copy=copy, indices=indices)


def roll(x: typing.Any, /, shift: typing.Any, *, axis: typing.Any = None) -> typing.Any:
    """Execute standard roll on CPU, or its bounded TensorIR lowering."""
    return _dispatch("roll", x, shift, axis=axis)


def squeeze(x: typing.Any, /, axis: typing.Any) -> typing.Any:
    """Execute standard squeeze on CPU, or its bounded TensorIR lowering."""
    return _dispatch("squeeze", x, axis)


def stack(arrays: typing.Any, /, *, axis: typing.Any = 0) -> typing.Any:
    """Execute standard stack on CPU, or its bounded TensorIR lowering."""
    return _dispatch("stack", arrays, axis=axis)


def tile(x: typing.Any, repetitions: typing.Any, /) -> typing.Any:
    """Execute standard tile on CPU, or its bounded TensorIR lowering."""
    return _dispatch("tile", x, repetitions)


def unstack(x: typing.Any, /, *, axis: typing.Any = 0) -> typing.Any:
    """Execute standard unstack on CPU, or its bounded TensorIR lowering."""
    return _dispatch("unstack", x, axis=axis)


def argmax(
    x: typing.Any, /, *, axis: typing.Any = None, keepdims: builtins.bool = False
) -> typing.Any:
    """Execute standard argmax on CPU, or its bounded TensorIR lowering."""
    return _dispatch("argmax", x, axis=axis, keepdims=keepdims)


def argmin(
    x: typing.Any, /, *, axis: typing.Any = None, keepdims: builtins.bool = False
) -> typing.Any:
    """Execute standard argmin on CPU, or its bounded TensorIR lowering."""
    return _dispatch("argmin", x, axis=axis, keepdims=keepdims)


def count_nonzero(
    x: typing.Any, /, *, axis: typing.Any = None, keepdims: builtins.bool = False
) -> typing.Any:
    """Execute standard count_nonzero on CPU, or its bounded TensorIR lowering."""
    return _dispatch("count_nonzero", x, axis=axis, keepdims=keepdims)


def nonzero(x: typing.Any, /) -> typing.Any:
    """Execute standard nonzero on CPU, or its bounded TensorIR lowering."""
    return _dispatch("nonzero", x)


def searchsorted(
    x1: typing.Any,
    x2: typing.Any,
    /,
    *,
    side: typing.Any = "left",
    sorter: typing.Any = None,
) -> typing.Any:
    """Execute standard searchsorted on CPU, or its bounded TensorIR lowering."""
    return _dispatch("searchsorted", x1, x2, side=side, sorter=sorter)


def where(condition: typing.Any, x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard where on CPU, or its bounded TensorIR lowering."""
    return _dispatch("where", condition, x1, x2)


def isin(
    x1: typing.Any, x2: typing.Any, /, *, invert: typing.Any = False
) -> typing.Any:
    """Execute standard isin on CPU, or its bounded TensorIR lowering."""
    return _dispatch("isin", x1, x2, invert=invert)


def unique_all(x: typing.Any, /) -> typing.Any:
    """Execute standard unique_all on CPU, or its bounded TensorIR lowering."""
    return _dispatch("unique_all", x)


def unique_counts(x: typing.Any, /) -> typing.Any:
    """Execute standard unique_counts on CPU, or its bounded TensorIR lowering."""
    return _dispatch("unique_counts", x)


def unique_inverse(x: typing.Any, /) -> typing.Any:
    """Execute standard unique_inverse on CPU, or its bounded TensorIR lowering."""
    return _dispatch("unique_inverse", x)


def unique_values(x: typing.Any, /) -> typing.Any:
    """Execute standard unique_values on CPU, or its bounded TensorIR lowering."""
    return _dispatch("unique_values", x)


def argsort(
    x: typing.Any,
    /,
    *,
    axis: typing.Any = -1,
    descending: builtins.bool = False,
    stable: builtins.bool = True,
) -> typing.Any:
    """Execute standard argsort on CPU, or its bounded TensorIR lowering."""
    return _dispatch("argsort", x, axis=axis, descending=descending, stable=stable)


def sort(
    x: typing.Any,
    /,
    *,
    axis: typing.Any = -1,
    descending: builtins.bool = False,
    stable: builtins.bool = True,
) -> typing.Any:
    """Execute standard sort on CPU, or its bounded TensorIR lowering."""
    return _dispatch("sort", x, axis=axis, descending=descending, stable=stable)


def cumulative_prod(
    x: typing.Any,
    /,
    *,
    axis: typing.Any = None,
    dtype: typing.Any = None,
    include_initial: builtins.bool = False,
) -> typing.Any:
    """Execute standard cumulative_prod on CPU, or its bounded TensorIR lowering."""
    return _dispatch(
        "cumulative_prod", x, axis=axis, dtype=dtype, include_initial=include_initial
    )


def cumulative_sum(
    x: typing.Any,
    /,
    *,
    axis: typing.Any = None,
    dtype: typing.Any = None,
    include_initial: builtins.bool = False,
) -> typing.Any:
    """Execute standard cumulative_sum on CPU, or its bounded TensorIR lowering."""
    return _dispatch(
        "cumulative_sum", x, axis=axis, dtype=dtype, include_initial=include_initial
    )


def max(
    x: typing.Any, /, *, axis: typing.Any = None, keepdims: builtins.bool = False
) -> typing.Any:
    """Execute standard max on CPU, or its bounded TensorIR lowering."""
    return _dispatch("max", x, axis=axis, keepdims=keepdims)


def mean(
    x: typing.Any, /, *, axis: typing.Any = None, keepdims: builtins.bool = False
) -> typing.Any:
    """Execute standard mean on CPU, or its bounded TensorIR lowering."""
    return _dispatch("mean", x, axis=axis, keepdims=keepdims)


def min(
    x: typing.Any, /, *, axis: typing.Any = None, keepdims: builtins.bool = False
) -> typing.Any:
    """Execute standard min on CPU, or its bounded TensorIR lowering."""
    return _dispatch("min", x, axis=axis, keepdims=keepdims)


def prod(
    x: typing.Any,
    /,
    *,
    axis: typing.Any = None,
    dtype: typing.Any = None,
    keepdims: builtins.bool = False,
) -> typing.Any:
    """Execute standard prod on CPU, or its bounded TensorIR lowering."""
    return _dispatch("prod", x, axis=axis, dtype=dtype, keepdims=keepdims)


def std(
    x: typing.Any,
    /,
    *,
    axis: typing.Any = None,
    correction: typing.Any = 0.0,
    keepdims: builtins.bool = False,
) -> typing.Any:
    """Execute standard std on CPU, or its bounded TensorIR lowering."""
    return _dispatch("std", x, axis=axis, correction=correction, keepdims=keepdims)


def sum(
    x: typing.Any,
    /,
    *,
    axis: typing.Any = None,
    dtype: typing.Any = None,
    keepdims: builtins.bool = False,
) -> typing.Any:
    """Execute standard sum on CPU, or its bounded TensorIR lowering."""
    return _dispatch("sum", x, axis=axis, dtype=dtype, keepdims=keepdims)


def var(
    x: typing.Any,
    /,
    *,
    axis: typing.Any = None,
    correction: typing.Any = 0.0,
    keepdims: builtins.bool = False,
) -> typing.Any:
    """Execute standard var on CPU, or its bounded TensorIR lowering."""
    return _dispatch("var", x, axis=axis, correction=correction, keepdims=keepdims)


def all(
    x: typing.Any, /, *, axis: typing.Any = None, keepdims: builtins.bool = False
) -> typing.Any:
    """Execute standard all on CPU, or its bounded TensorIR lowering."""
    return _dispatch("all", x, axis=axis, keepdims=keepdims)


def any(
    x: typing.Any, /, *, axis: typing.Any = None, keepdims: builtins.bool = False
) -> typing.Any:
    """Execute standard any on CPU, or its bounded TensorIR lowering."""
    return _dispatch("any", x, axis=axis, keepdims=keepdims)


def diff(
    x: typing.Any,
    /,
    *,
    axis: typing.Any = -1,
    n: typing.Any = 1,
    prepend: typing.Any = None,
    append: typing.Any = None,
) -> typing.Any:
    """Execute standard diff on CPU, or its bounded TensorIR lowering."""
    return _dispatch("diff", x, axis=axis, n=n, prepend=prepend, append=append)


def slice(x: object, ranges: tuple[tuple[int, int], ...]) -> typing.Any:
    """GenerativeQC extension: exact static half-open scientific slices."""
    return _eager.wrap(_symbolic_backend.slice(x, ranges))


def einsum(
    equation: str,
    *operands: object,
    coefficient: ExactScalar = 1,
) -> typing.Any:
    """GenerativeQC extension: explicit-output finite scientific contractions."""
    return _eager.wrap(
        _symbolic_backend.einsum(equation, *operands, coefficient=coefficient)
    )


def _generic_spec(array: np.ndarray, *, differentiable: builtins.bool) -> TensorSpec:
    return TensorSpec(
        _namespace._generic_indices(tuple(int(extent) for extent in array.shape)),
        dtype=array.dtype.name,
        role="parameter" if differentiable else "input",
        differentiable=differentiable,
    )


class CompiledFunction:
    """Shape/dtype-specialized quantum-chemistry TensorIR compiler.

    CPU execution is compiled C++ with float64 semantics. CUDA uses the existing
    TensorIR planner and native executor with explicit host staging. Reference
    interpretation is opt-in; unsupported lowering never selects it implicitly.
    Optional TensorSpecs retain AO/orbital/spin domains instead of inferring
    scientific compatibility from equal axis extents.
    """

    def __init__(
        self,
        function: typing.Callable[..., object],
        *,
        backend: str,
        differentiable: tuple[str, ...],
        specs: typing.Mapping[str, TensorSpec] | None = None,
        **options: typing.Any,
    ) -> None:
        """Bind a function, semantic input types, backend and finite resource limits."""
        if not callable(function):
            raise TypeError("compile requires a callable")
        if backend not in ("cpu", "cuda", "reference"):
            raise ValueError("backend must be 'cpu', 'cuda' or 'reference'")
        symbolic = _symbolic_functions.get(getattr(function, "__name__", ""))
        signature = inspect.signature(
            symbolic
            if symbolic is not None and function is globals().get(function.__name__)
            else function
        )
        unsupported = {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        if builtins.any(
            parameter.kind in unsupported for parameter in signature.parameters.values()
        ):
            raise TypeError(
                "compiled array functions require named, non-variadic parameters"
            )
        if not isinstance(differentiable, tuple) or builtins.any(
            not isinstance(name, str) for name in differentiable
        ):
            raise TypeError("differentiable must be a tuple of parameter names")
        if len(set(differentiable)) != len(differentiable):
            raise ValueError("differentiable parameter names must be unique")
        unknown = set(differentiable) - set(signature.parameters)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown differentiable parameter(s): {names}")
        self._function = (
            symbolic
            if symbolic is not None and function is globals().get(function.__name__)
            else function
        )
        self._signature = signature
        self._backend = backend
        self._differentiable = frozenset(differentiable)
        self._specs = dict(specs or {})
        if set(self._specs) - set(signature.parameters):
            raise ValueError("input specs must name function parameters")
        if builtins.any(
            not isinstance(spec, TensorSpec) for spec in self._specs.values()
        ):
            raise TypeError("input specs must be TensorSpecs")
        from ._array_api_execution import _Executor

        self._execution = _Executor(backend, options)
        self._options = self._execution.options
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
            array = _symbolic_backend.asarray(value)
            if isinstance(array, VibeArray):
                raise TypeError("compiled runtime arguments must be concrete arrays")
            assert isinstance(array, np.ndarray)
            if array.dtype.name not in ("bool", "float32", "float64"):
                raise TypeError(
                    "TensorIR compilation supports bool, float32 or float64 inputs"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"non-finite compiled input: {name}")
            feeds[name] = array
            spec = self._specs.get(name)
            if spec is not None:
                if spec.shape != array.shape or spec.dtype != array.dtype.name:
                    raise ValueError(
                        f"input {name} does not match its scientific TensorSpec"
                    )
                specs[name] = replace(spec, differentiable=name in self._differentiable)
            else:
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
        """Compile/cache and execute the whole program on the selected backend."""
        program, feeds = self._prepare(*args, **kwargs)
        outputs = self._execution.execute(program, feeds)
        if tuple(program.outputs) == ("output",):
            return _eager.wrap(outputs["output"])
        return {name: _eager.wrap(value) for name, value in outputs.items()}

    def _variant(
        self, names: tuple[str, ...], backend: str | None = None
    ) -> CompiledFunction:
        return CompiledFunction(
            self._function,
            backend=backend or self._backend,
            differentiable=names,
            specs=self._specs,
            **self._options,
        )

    def compilation_info(self) -> dict[str, object]:
        """Inspect backend, cached specializations and actual native executable count."""
        return {
            "backend": self._backend,
            "specializations": len(self._programs),
            "native_executables": len(self._execution.executables),
            "program_hashes": tuple(self._execution.executables),
        }

    def close(self) -> None:
        """Release native execution owners; subsequent calls fail rather than rebuild."""
        self._execution.close()


@typing.overload
def compile(
    function: typing.Callable[..., object],
    *,
    backend: str = "cpu",
    differentiable: tuple[str, ...] = (),
    **options: typing.Any,
) -> CompiledFunction: ...


@typing.overload
def compile(
    function: None = None,
    *,
    backend: str = "cpu",
    differentiable: tuple[str, ...] = (),
    **options: typing.Any,
) -> typing.Callable[[typing.Callable[..., object]], CompiledFunction]: ...


def compile(
    function: typing.Callable[..., object] | None = None,
    *,
    backend: str = "cpu",
    differentiable: tuple[str, ...] = (),
    specs: typing.Mapping[str, TensorSpec] | None = None,
    cache: typing.Any = None,
    max_bytes: int = 64 * 1024 * 1024,
    max_work: int = 100_000_000,
    max_nodes: int = 4096,
    compile_timeout: float = 300.0,
    cuda_arch: str = "sm_120",
    device: int = 0,
) -> (
    CompiledFunction | typing.Callable[[typing.Callable[..., object]], CompiledFunction]
):
    """Compile Array API expressions to native CPU/CUDA code through TensorIR.

    Native CPU currently requires float64; reference and CUDA have their own
    explicitly bounded primitive/dtype admission. Shapes specialize on first
    call. Specs optionally declare scientific domains. Cache, byte/work/node
    budgets and finite compiler timeouts apply to native execution. CUDA stages
    host inputs and returns detached CPU arrays; it is not a GPU eager namespace.
    """
    options = {
        "cache": cache,
        "max_bytes": max_bytes,
        "max_work": max_work,
        "max_nodes": max_nodes,
        "compile_timeout": compile_timeout,
        "cuda_arch": cuda_arch,
        "device": device,
    }
    if function is None:
        return lambda target: CompiledFunction(
            target,
            backend=backend,
            differentiable=differentiable,
            specs=specs,
            **options,
        )
    return CompiledFunction(
        function, backend=backend, differentiable=differentiable, specs=specs, **options
    )


def grad(
    function: typing.Callable[..., object],
    *,
    argnums: int | tuple[int, ...] = 0,
    backend: str | None = None,
    **options: typing.Any,
) -> typing.Any:
    """Compile a scalar function's reverse derivative for the selected arguments.

    An integer argnums returns one Array; a tuple returns a tuple in that order.
    Existing compiled functions retain their backend, budgets and scientific
    specs. Plain functions use native CPU unless a backend is explicitly chosen.
    """
    from ._array_api_autodiff import _GradientFunction

    return _GradientFunction(function, argnums, backend, options, with_value=False)


def value_and_grad(
    function: typing.Callable[..., object],
    *,
    argnums: int | tuple[int, ...] = 0,
    backend: str | None = None,
    **options: typing.Any,
) -> typing.Any:
    """Compile scalar energy and selected gradients as one shared TensorIR program.

    Primal and derivative nodes share the same native endpoint, specialization
    cache and resource gates. This is not numerical differentiation or a call to
    the reference autodiff evaluator.
    """
    from ._array_api_autodiff import _GradientFunction

    return _GradientFunction(function, argnums, backend, options, with_value=True)


def jvp(
    function: typing.Callable[..., object],
    primals: tuple[object, ...],
    tangents: tuple[object, ...],
    *,
    backend: str | None = None,
    **options: typing.Any,
) -> tuple[typing.Any, typing.Any]:
    """Compile a function and forward directional derivative on one backend.

    Primals/tangents are matching tuples of positional array arguments. Returned
    primal and tangent outputs preserve the function's array/dictionary structure.
    """
    from ._array_api_autodiff import _jvp

    return _jvp(function, primals, tangents, backend, options)


def vjp(
    function: typing.Callable[..., object],
    *primals: object,
    backend: str | None = None,
    **options: typing.Any,
) -> typing.Any:
    """Return native primal outputs and a reusable compiled cotangent pullback.

    The pullback snapshots its primal inputs, validates cotangent shapes/dtypes,
    and returns gradients in positional argument order. Close it to release its
    execution owner, especially for CUDA.
    """
    from ._array_api_autodiff import _vjp

    return _vjp(function, primals, backend, options)


def capabilities() -> dict[str, object]:
    """Report eager standard support separately from compiler qualification.

    Compiler metadata and operation_matrix describe only the bounded symbolic
    subset. Eager dtypes/functions/devices and namespace discovery describe CPU
    execution. Native/AD metadata names explicit compiler paths, not full
    standard coverage or GPU eager-array conformance.
    """
    report = _compiler_capabilities()
    report.update(
        {
            "compiled_functions": report["functions"],
            "compiled_array_namespace_protocol": False,
            "public_api_version": API_VERSION,
            "surface": "array-api-standard-eager-with-bounded-tensorir-capture",
            "stability": "experimental",
            "import_path": "generativeqc.experimental.array_api",
            "array_api_version": __array_api_version__,
            "array_namespace_protocol": True,
            "functions": tuple(
                sorted(
                    set(_eager.FUNCTION_NAMES)
                    | {
                        "compile",
                        "trace",
                        "einsum",
                        "slice",
                        "grad",
                        "value_and_grad",
                        "jvp",
                        "vjp",
                    }
                )
            ),
            "implicit_broadcast": True,
            "reshape_requires_explicit_indices": False,
            "broadcast_requires_explicit_indices_and_axes": False,
            "scientific_metadata_requires_explicit_indices": True,
            "compiled_call": "shape-dtype-specialized-native-tensorir",
            "compiled_backends": ("cpu", "cuda", "reference"),
            "native_cpu_dtypes": ("float64",),
            "autodiff": ("grad", "value_and_grad", "jvp", "vjp"),
            "compiled_differentiability": "explicit-parameter-names",
            "namespace_dispatch": "array-api-cpu-eager-or-symbolic-tensorir",
            "runtime_array": "numpy-backed-array-api-cpu",
            "eager_dtypes": _eager.DTYPE_NAMES,
            "eager_functions": tuple(_eager.FUNCTION_NAMES),
            "eager_device": "cpu",
            "eager_extensions": ("linalg", "fft"),
        }
    )
    return report


from . import _array_api_fft as fft
from . import _array_api_linalg as linalg

__all__ = [
    "API_VERSION",
    "DLPACK_INTEROP_VERSION",
    "FRONTEND_VERSION",
    "SUPPORTED_FUNCTIONS",
    "Array",
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
    "__array_api_version__",
    "__array_namespace_info__",
    "abs",
    "acos",
    "acosh",
    "add",
    "all",
    "any",
    "arange",
    "argmax",
    "argmin",
    "argsort",
    "asarray",
    "asin",
    "asinh",
    "astype",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "bool",
    "broadcast_arrays",
    "broadcast_shapes",
    "broadcast_to",
    "can_cast",
    "capabilities",
    "ceil",
    "clip",
    "compile",
    "complex64",
    "complex128",
    "concat",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "count_nonzero",
    "cumulative_prod",
    "cumulative_sum",
    "diff",
    "divide",
    "dlpack_device",
    "e",
    "einsum",
    "empty",
    "empty_like",
    "equal",
    "exp",
    "expand_dims",
    "expm1",
    "eye",
    "fft",
    "finfo",
    "flip",
    "float32",
    "float64",
    "floor",
    "floor_divide",
    "from_dlpack",
    "full",
    "full_like",
    "grad",
    "greater",
    "greater_equal",
    "hypot",
    "iinfo",
    "imag",
    "import_dlpack",
    "inf",
    "input_array",
    "int8",
    "int16",
    "int32",
    "int64",
    "isdtype",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "jvp",
    "less",
    "less_equal",
    "linalg",
    "linspace",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "matmul",
    "matrix_transpose",
    "max",
    "maximum",
    "mean",
    "meshgrid",
    "min",
    "minimum",
    "moveaxis",
    "multiply",
    "nan",
    "negative",
    "newaxis",
    "nextafter",
    "nonzero",
    "not_equal",
    "ones",
    "ones_like",
    "permute_dims",
    "pi",
    "positive",
    "pow",
    "prod",
    "real",
    "reciprocal",
    "remainder",
    "repeat",
    "reshape",
    "result_type",
    "roll",
    "round",
    "searchsorted",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "slice",
    "sort",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "subtract",
    "sum",
    "take",
    "take_along_axis",
    "tan",
    "tanh",
    "tensordot",
    "tile",
    "trace",
    "tril",
    "triu",
    "trunc",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "unstack",
    "value_and_grad",
    "var",
    "vecdot",
    "vjp",
    "where",
    "zeros",
    "zeros_like",
]
