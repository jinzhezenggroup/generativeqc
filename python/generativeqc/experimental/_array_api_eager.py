"""CPU Array API execution, separate from the bounded symbolic compiler.

NumPy owns storage and numerical execution; array-api-compat supplies standard
function semantics, not a scientific reference oracle. Keeping the adapter here
prevents eager dtype/device capabilities from leaking into TensorIR capture.
"""

from __future__ import annotations

import builtins
import inspect
import sys
from fractions import Fraction
from types import ModuleType, SimpleNamespace
from typing import Any

import array_api_compat.numpy as _numpy
import numpy as np
from typing_extensions import Self

ARRAY_API_VERSION = "2025.12"
DTYPE_NAMES = (
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "complex64",
    "complex128",
)
FUNCTION_NAMES = [
    "arange",
    "asarray",
    "empty",
    "empty_like",
    "eye",
    "from_dlpack",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
    "astype",
    "can_cast",
    "finfo",
    "iinfo",
    "isdtype",
    "result_type",
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_invert",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "clip",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "hypot",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "multiply",
    "negative",
    "nextafter",
    "not_equal",
    "positive",
    "pow",
    "real",
    "reciprocal",
    "remainder",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "square",
    "sqrt",
    "subtract",
    "tan",
    "tanh",
    "trunc",
    "take",
    "take_along_axis",
    "matmul",
    "matrix_transpose",
    "tensordot",
    "vecdot",
    "broadcast_arrays",
    "broadcast_shapes",
    "broadcast_to",
    "concat",
    "expand_dims",
    "flip",
    "moveaxis",
    "permute_dims",
    "repeat",
    "reshape",
    "roll",
    "squeeze",
    "stack",
    "tile",
    "unstack",
    "argmax",
    "argmin",
    "count_nonzero",
    "nonzero",
    "searchsorted",
    "where",
    "isin",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "argsort",
    "sort",
    "cumulative_prod",
    "cumulative_sum",
    "max",
    "mean",
    "min",
    "prod",
    "std",
    "sum",
    "var",
    "all",
    "any",
    "diff",
]


def check_host_values(value: object) -> None:
    """Bound nested inspection and reject hidden implicit device transfers."""
    pending = [(value, 0, frozenset())]
    inspected = 0
    while pending:
        item, depth, ancestors = pending.pop()
        inspected += 1
        if inspected > 1_000_000:
            raise ValueError("host container inspection exceeds 1000000 items")
        if isinstance(item, np.ndarray):
            if not item.dtype.hasobject:
                continue
            children = item.flat
            count = item.size
        elif isinstance(item, (list, tuple)):
            children = item
            count = len(item)
        else:
            if isinstance(item, np.generic) or (
                isinstance(item, type) and issubclass(item, np.generic)
            ):
                continue
            if (
                callable(getattr(item, "__dlpack_device__", None))
                or inspect.getattr_static(item, "__cuda_array_interface__", None)
                is not None
            ):
                raise TypeError(
                    "asarray does not perform implicit external-device transfer; use from_dlpack or import_dlpack for an explicit handoff"
                )
            if depth and callable(getattr(item, "__array__", None)):
                raise TypeError(
                    "nested host containers require scalar or NumPy array leaves"
                )
            continue
        identity = id(item)
        if identity in ancestors:
            raise ValueError("cyclic host containers are unsupported")
        if count and depth >= 64:
            raise ValueError("host container nesting exceeds 64 levels")
        if inspected + len(pending) + count > 1_000_000:
            raise ValueError("host container inspection exceeds 1000000 items")
        lineage = ancestors | {identity}
        pending.extend((child, depth + 1, lineage) for child in children)


def _unwrap(value: Any) -> Any:
    """Strip only our storage wrapper, preserving nested argument structure."""
    if isinstance(value, np.ndarray) and type(value) is not np.ndarray:
        return np.ndarray.view(value, np.ndarray)
    if isinstance(value, tuple):
        return tuple(_unwrap(item) for item in value)
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    return value


def wrap(value: Any) -> Any:
    """Return arrays (including zero-dimensional results) with our protocol."""
    if isinstance(value, Array):
        return value
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.name not in DTYPE_NAMES:
            raise TypeError(f"unsupported Array API dtype: {value.dtype}")
        return np.asarray(value).view(Array)
    if isinstance(value, tuple):
        items = tuple(wrap(item) for item in value)
        return type(value)(*items) if hasattr(value, "_fields") else items
    if isinstance(value, list):
        return [wrap(item) for item in value]
    return value


class Array(np.ndarray):
    """NumPy-backed CPU array exposing the standardized discovery protocol.

    Views preserve storage ownership, so copy=False never disguises a transfer.
    Python scalar conversions are intentionally restricted to rank-zero arrays,
    rather than inheriting NumPy's deprecated size-one conversion behavior.
    """

    def __array_namespace__(self, *, api_version: str | None = None) -> ModuleType:
        """Discover the public namespace, rejecting unqualified versions."""
        if api_version not in (None, ARRAY_API_VERSION):
            raise ValueError(f"unsupported Array API version: {api_version!r}")
        return sys.modules["generativeqc.experimental.array_api"]

    @property
    def device(self) -> str:
        """The eager backend only allocates CPU storage."""
        return "cpu"

    def to_device(self, device: object, /, *, stream: object = None) -> Array:
        """Preserve CPU storage; never implicitly stage another device."""
        if device != "cpu":
            raise ValueError("the Array API eager backend only supports device='cpu'")
        if stream is not None:
            raise ValueError("CPU arrays do not support execution streams")
        return self

    @property
    def T(self) -> Array:
        """Transpose a matrix; higher ranks use mT or permute_dims explicitly."""
        if self.ndim != 2:
            raise ValueError("T requires a two-dimensional array")
        return wrap(self.view(np.ndarray).T)

    @property
    def mT(self) -> Array:
        """Transpose the final two dimensions of a stack of matrices."""
        if self.ndim < 2:
            raise ValueError("mT requires at least two dimensions")
        return wrap(np.swapaxes(self.view(np.ndarray), -1, -2))

    def __getitem__(self, key: Any) -> Array:
        """Indexing always returns an array, even when selecting one scalar."""
        check_host_values(key)
        return wrap(self.view(np.ndarray)[_unwrap(key)])

    def _scalar(self) -> Any:
        if self.ndim != 0:
            raise TypeError(
                "Python scalar conversion requires a zero-dimensional array"
            )
        return self.item()

    def __bool__(self) -> builtins.bool:
        """Convert a rank-zero value to a Python Boolean."""
        return builtins.bool(self._scalar())

    def __int__(self) -> int:
        """Convert a rank-zero real value to a Python integer."""
        return int(self._scalar())

    def __float__(self) -> float:
        """Convert a rank-zero real value to a Python float."""
        return float(self._scalar())

    def __complex__(self) -> complex:
        """Convert a rank-zero numeric value to a Python complex number."""
        return complex(self._scalar())

    def __index__(self) -> int:
        """Use only a rank-zero integer as a Python index."""
        if self.dtype.kind not in "iu":
            raise TypeError("integer indexing requires an integer array")
        return int(self._scalar())

    def __setitem__(self, key: Any, value: Any) -> None:
        """Assign into existing CPU storage without implicit foreign transfers."""
        check_host_values(key)
        check_host_values(value)
        self.view(np.ndarray)[_unwrap(key)] = _unwrap(value)

    def _inplace(self, name: str, other: object) -> Self:
        """Publish only shape/dtype-preserving results into existing storage."""
        result = call(name, self, other)
        if result.dtype != self.dtype:
            raise TypeError("in-place operations cannot change the array dtype")
        if result.shape != self.shape:
            raise ValueError("in-place operations cannot change the array shape")
        self[...] = result
        return self

    def __abs__(self) -> Array:
        """Apply standard abs with array-valued results."""
        return call("abs", self)

    def __neg__(self) -> Array:
        """Apply standard negative with array-valued results."""
        return call("negative", self)

    def __pos__(self) -> Array:
        """Apply standard positive with array-valued results."""
        return call("positive", self)

    def __invert__(self) -> Array:
        """Apply standard bitwise_invert with array-valued results."""
        return call("bitwise_invert", self)

    def __add__(self, other: object) -> Array:
        """Apply standard add with array-valued results."""
        return call("add", self, other)

    def __radd__(self, other: object) -> Array:
        """Apply reflected standard add with array-valued results."""
        return call("add", other, self)

    def __iadd__(self, other: object) -> Self:
        """Apply standard add without changing this storage dtype."""
        return self._inplace("add", other)

    def __sub__(self, other: object) -> Array:
        """Apply standard subtract with array-valued results."""
        return call("subtract", self, other)

    def __rsub__(self, other: object) -> Array:
        """Apply reflected standard subtract with array-valued results."""
        return call("subtract", other, self)

    def __isub__(self, other: object) -> Self:
        """Apply standard subtract without changing this storage dtype."""
        return self._inplace("subtract", other)

    def __mul__(self, other: object) -> Array:
        """Apply standard multiply with array-valued results."""
        return call("multiply", self, other)

    def __rmul__(self, other: object) -> Array:
        """Apply reflected standard multiply with array-valued results."""
        return call("multiply", other, self)

    def __imul__(self, other: object) -> Self:
        """Apply standard multiply without changing this storage dtype."""
        return self._inplace("multiply", other)

    def __truediv__(self, other: object) -> Array:
        """Apply standard divide with array-valued results."""
        return call("divide", self, other)

    def __rtruediv__(self, other: object) -> Array:
        """Apply reflected standard divide with array-valued results."""
        return call("divide", other, self)

    def __itruediv__(self, other: object) -> Self:
        """Apply standard divide without changing this storage dtype."""
        return self._inplace("divide", other)

    def __floordiv__(self, other: object) -> Array:
        """Apply standard floor_divide with array-valued results."""
        return call("floor_divide", self, other)

    def __rfloordiv__(self, other: object) -> Array:
        """Apply reflected standard floor_divide with array-valued results."""
        return call("floor_divide", other, self)

    def __ifloordiv__(self, other: object) -> Self:
        """Apply standard floor_divide without changing this storage dtype."""
        return self._inplace("floor_divide", other)

    def __mod__(self, other: object) -> Array:
        """Apply standard remainder with array-valued results."""
        return call("remainder", self, other)

    def __rmod__(self, other: object) -> Array:
        """Apply reflected standard remainder with array-valued results."""
        return call("remainder", other, self)

    def __imod__(self, other: object) -> Self:
        """Apply standard remainder without changing this storage dtype."""
        return self._inplace("remainder", other)

    def __pow__(self, other: object) -> Array:
        """Apply standard pow with array-valued results."""
        return call("pow", self, other)

    def __rpow__(self, other: object) -> Array:
        """Apply reflected standard pow with array-valued results."""
        return call("pow", other, self)

    def __ipow__(self, other: object) -> Self:
        """Apply standard pow without changing this storage dtype."""
        return self._inplace("pow", other)

    def __matmul__(self, other: object) -> Array:
        """Apply standard matmul with array-valued results."""
        return call("matmul", self, other)

    def __rmatmul__(self, other: object) -> Array:
        """Apply reflected standard matmul with array-valued results."""
        return call("matmul", other, self)

    def __imatmul__(self, other: object) -> Self:
        """Apply standard matmul without changing this storage dtype."""
        return self._inplace("matmul", other)

    def __and__(self, other: object) -> Array:
        """Apply standard bitwise_and with array-valued results."""
        return call("bitwise_and", self, other)

    def __rand__(self, other: object) -> Array:
        """Apply reflected standard bitwise_and with array-valued results."""
        return call("bitwise_and", other, self)

    def __iand__(self, other: object) -> Self:
        """Apply standard bitwise_and without changing this storage dtype."""
        return self._inplace("bitwise_and", other)

    def __or__(self, other: object) -> Array:
        """Apply standard bitwise_or with array-valued results."""
        return call("bitwise_or", self, other)

    def __ror__(self, other: object) -> Array:
        """Apply reflected standard bitwise_or with array-valued results."""
        return call("bitwise_or", other, self)

    def __ior__(self, other: object) -> Self:
        """Apply standard bitwise_or without changing this storage dtype."""
        return self._inplace("bitwise_or", other)

    def __xor__(self, other: object) -> Array:
        """Apply standard bitwise_xor with array-valued results."""
        return call("bitwise_xor", self, other)

    def __rxor__(self, other: object) -> Array:
        """Apply reflected standard bitwise_xor with array-valued results."""
        return call("bitwise_xor", other, self)

    def __ixor__(self, other: object) -> Self:
        """Apply standard bitwise_xor without changing this storage dtype."""
        return self._inplace("bitwise_xor", other)

    def __lshift__(self, other: object) -> Array:
        """Apply standard bitwise_left_shift with array-valued results."""
        return call("bitwise_left_shift", self, other)

    def __rlshift__(self, other: object) -> Array:
        """Apply reflected standard bitwise_left_shift with array-valued results."""
        return call("bitwise_left_shift", other, self)

    def __ilshift__(self, other: object) -> Self:
        """Apply standard bitwise_left_shift without changing this storage dtype."""
        return self._inplace("bitwise_left_shift", other)

    def __rshift__(self, other: object) -> Array:
        """Apply standard bitwise_right_shift with array-valued results."""
        return call("bitwise_right_shift", self, other)

    def __rrshift__(self, other: object) -> Array:
        """Apply reflected standard bitwise_right_shift with array-valued results."""
        return call("bitwise_right_shift", other, self)

    def __irshift__(self, other: object) -> Self:
        """Apply standard bitwise_right_shift without changing this storage dtype."""
        return self._inplace("bitwise_right_shift", other)

    def __eq__(self, other: object) -> Array:
        """Apply standard equal with array-valued results."""
        return call("equal", self, other)

    def __ne__(self, other: object) -> Array:
        """Apply standard not_equal with array-valued results."""
        return call("not_equal", self, other)

    def __lt__(self, other: object) -> Array:
        """Apply standard less with array-valued results."""
        return call("less", self, other)

    def __le__(self, other: object) -> Array:
        """Apply standard less_equal with array-valued results."""
        return call("less_equal", self, other)

    def __gt__(self, other: object) -> Array:
        """Apply standard greater with array-valued results."""
        return call("greater", self, other)

    def __ge__(self, other: object) -> Array:
        """Apply standard greater_equal with array-valued results."""
        return call("greater_equal", self, other)


def call(name: str, *args: Any, **kwargs: Any) -> Any:
    """Execute one standardized operation without invoking subclass hooks."""
    if name != "from_dlpack":
        for value in (*args, *kwargs.values()):
            check_host_values(value)
            if (
                name != "asarray"
                and isinstance(value, np.ndarray)
                and value.dtype.name not in DTYPE_NAMES
            ):
                raise TypeError(f"unsupported Array API dtype: {value.dtype}")
    if (
        name in ("take", "take_along_axis")
        and np.asarray(args[1]).dtype.kind not in "iu"
    ):
        if isinstance(args[1], (list, tuple)) and not args[1]:
            args = (args[0], np.asarray(args[1], dtype=np.int64), *args[2:])
        else:
            raise TypeError("take indices must have an integer dtype")
    for key in ("copy", "keepdims", "descending", "stable", "include_initial"):
        value = kwargs.get(key)
        if value is not None and type(value) is not builtins.bool:
            raise TypeError(f"{key} must be a bool")
    for key in ("axis", "axes", "source", "destination"):
        value = kwargs.get(key)
        if type(value) is builtins.bool or (
            isinstance(value, tuple)
            and any(type(axis) is builtins.bool for axis in value)
        ):
            raise TypeError(f"{key} must contain integer axes, not bool")
    if name in ("expand_dims", "squeeze", "moveaxis", "permute_dims"):
        for value in args[1:]:
            if type(value) is builtins.bool or (
                isinstance(value, tuple)
                and any(type(axis) is builtins.bool for axis in value)
            ):
                raise TypeError("axes must be integers, not bool")
    if name == "broadcast_shapes":
        for shape in args:
            if not isinstance(shape, tuple) or any(
                type(extent) is not int or extent < 0 for extent in shape
            ):
                raise TypeError("shapes require nonnegative integer dimensions")
    if (
        name == "astype"
        and np.asarray(args[0]).dtype.kind == "c"
        and np.dtype(args[1]).kind not in "cb"
    ):
        raise TypeError(
            "complex arrays must select real or imag before casting to a real dtype"
        )
    if (
        name == "astype"
        and kwargs.get("copy", True) is False
        and isinstance(args[0], Array)
        and args[0].dtype == np.dtype(args[1])
    ):
        device = kwargs.get("device")
        if device not in (None, "cpu"):
            raise ValueError("the eager backend only supports device='cpu'")
        return args[0]
    # Exact rational scalars remain a GenerativeQC extension to standard calls.
    scalar_operations = {
        "add",
        "subtract",
        "multiply",
        "divide",
        "pow",
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
    }
    arguments = tuple(
        float(Fraction(value))
        if isinstance(value, Fraction)
        or (name in scalar_operations and isinstance(value, str))
        else value
        for value in args
    )
    if (
        name == "asarray"
        and not isinstance(
            args[0],
            (np.ndarray, np.generic, list, tuple, int, float, complex, builtins.bool),
        )
        and callable(getattr(args[0], "__array__", None))
    ):
        arguments = (_numpy.asarray(args[0], copy=kwargs.get("copy")),)
        # An array protocol can expose object storage only after conversion.
        # Inspect its leaves before a requested dtype can invoke scalar hooks.
        check_host_values(arguments[0])
    if name == "can_cast":
        source = np.dtype(args[0].dtype if isinstance(args[0], np.ndarray) else args[0])
        target = np.dtype(args[1])
        if (source.kind == "b") != (target.kind == "b"):
            return False
    if name == "result_type":
        dtypes = [
            np.dtype(value.dtype if isinstance(value, np.ndarray) else value)
            for value in args
            if not isinstance(value, (int, float, complex, builtins.bool))
        ]
        if not dtypes:
            raise TypeError("result_type requires an array or dtype")
        if any(dtype.kind == "b" for dtype in dtypes) and (
            any(dtype.kind != "b" for dtype in dtypes)
            or any(type(value) in (int, float, complex) for value in args)
        ):
            raise TypeError("Boolean/numeric dtype promotion is unsupported")
    comparisons = {
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
    }
    if name in comparisons and all(
        isinstance(value, (int, float, complex, Fraction, str)) for value in arguments
    ):
        raise TypeError("comparison requires an array operand")
    numeric_operations = (scalar_operations - comparisons) | {
        "abs",
        "negative",
        "positive",
        "square",
        "reciprocal",
        "floor_divide",
        "remainder",
        "matmul",
        "sum",
        "prod",
    }
    floating_operations = {
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "cos",
        "cosh",
        "exp",
        "expm1",
        "log",
        "log1p",
        "log2",
        "log10",
        "logaddexp",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "mean",
        "std",
        "var",
    }
    if name in numeric_operations | floating_operations:
        for value in arguments:
            if np.asarray(value).dtype.kind == "b":
                raise TypeError("floating arithmetic does not accept Boolean arrays")
            if (
                name in floating_operations
                and not isinstance(value, (int, float, complex))
                and np.asarray(value).dtype.kind not in "fc"
            ):
                raise TypeError(
                    "floating arithmetic requires float32 or float64 or complex arrays"
                )
    if name == "diff":
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in ("prepend", "append") or value is not None
        }
    if name in ("reshape", "broadcast_to"):
        for key in ("indices", "axes"):
            if kwargs.pop(key, None) is not None:
                raise ValueError("explicit TensorIR indices require a symbolic array")
    if name == "from_dlpack" and args[0].__dlpack_device__()[0] != 1:
        raise ValueError("the Array API eager backend only supports CPU DLPack storage")
    backend_args = tuple(_unwrap(value) for value in arguments)
    with np.errstate(all="ignore"):
        if name == "floor_divide" and any(
            np.asarray(value).dtype.kind == "f" for value in backend_args
        ):
            # NumPy's Python-compatible // differs from the standard at infinity
            # and signed zero. Use floor of the IEEE quotient for floating data.
            result = np.floor(np.divide(*backend_args))
        elif name == "acosh" and np.asarray(backend_args[0]).dtype.kind == "c":
            array = np.asarray(backend_args[0])
            result = np.asarray(np.arccosh(array))
            # Older NumPy loses the defined imaginary limit when only the
            # imaginary input is NaN and the real input is zero.
            exceptional = (array.real == 0) & np.isnan(array.imag)
            result.imag[exceptional] = np.copysign(np.pi / 2, array.imag[exceptional])
        elif name == "expm1" and np.asarray(backend_args[0]).dtype.kind == "c":
            array = np.asarray(backend_args[0])
            result = np.expm1(array)
            exceptional = ~np.isfinite(array)
            result = np.asarray(result)
            if np.any(exceptional):
                replacement = np.exp(array[exceptional])
                result.real[exceptional] = replacement.real - 1
                result.imag[exceptional] = replacement.imag
            result.real[(array.real == 0) & (array.imag == 0)] = 0.0
        elif name == "tanh" and np.asarray(backend_args[0]).dtype.kind == "c":
            array = np.asarray(backend_args[0])
            result = np.asarray(np.tanh(array))
            # At an infinite real part, the standard specifies the sign of the
            # zero imaginary limit from the input, not sin(2*imaginary).
            exceptional = np.isinf(array.real) & np.isfinite(array.imag)
            result.imag[exceptional] = np.copysign(0.0, array.imag[exceptional])
        else:
            backend = _numpy
            function_name = name
            if "." in name:
                module_name, function_name = name.split(".", 1)
                backend = getattr(_numpy, module_name)
            result = getattr(backend, function_name)(
                *backend_args,
                **{key: _unwrap(value) for key, value in kwargs.items()},
            )
    if name == "finfo":
        return SimpleNamespace(
            bits=int(result.bits),
            eps=float(result.eps),
            max=float(result.max),
            min=float(result.min),
            smallest_normal=float(result.smallest_normal),
            dtype=result.dtype,
        )
    if name in (
        "can_cast",
        "finfo",
        "iinfo",
        "isdtype",
        "result_type",
        "broadcast_shapes",
    ):
        return result
    if name == "astype" and result is args[0]:
        return result
    return wrap(result)


def namespace_info() -> Any:
    """Report NumPy's CPU devices, dtype defaults and data-dependent shapes."""
    return _numpy.__array_namespace_info__()
