"""Experimental public Array-API-shaped frontend over canonical TensorIR.

The ordinary path is shape/dtype based and deliberately hides TensorIR metadata.
Advanced users may still call ``trace`` with explicit scientific index spaces.
This preview does not implement ``__array_namespace__`` and does not claim Array
API conformance.
"""

from __future__ import annotations

import functools
import inspect
import typing
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

add = _namespace.add
broadcast_to = _namespace.broadcast_to
divide = _namespace.divide
einsum = _namespace.einsum
exp = _namespace.exp
log = _namespace.log
matmul = _namespace.matmul
matrix_transpose = _namespace.matrix_transpose
multiply = _namespace.multiply
negative = _namespace.negative
permute_dims = _namespace.permute_dims
pow = _namespace.pow
reshape = _namespace.reshape
slice = _namespace.slice
sqrt = _namespace.sqrt
subtract = _namespace.subtract
sum = _namespace.sum
take = _namespace.take


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
    if not isinstance(value, np.ndarray) and callable(
        getattr(value, "__dlpack_device__", None)
    ):
        raise TypeError(
            "asarray does not perform implicit external-device transfer; "
            "use import_dlpack for an explicit handoff"
        )
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


def _generic_spec(array: np.ndarray) -> TensorSpec:
    return TensorSpec(
        _namespace._generic_indices(tuple(int(extent) for extent in array.shape)),
        dtype=array.dtype.name,
        role="input",
    )


class CompiledFunction:
    """Shape/dtype-specialized TensorIR capture with reference execution.

    ``compile`` currently means compile Python array expressions to canonical
    TensorIR. Native CPU/CUDA execution remains an explicit follow-on capability;
    this first public path uses the independent NumPy TensorIR interpreter.
    """

    def __init__(self, function: typing.Callable[..., object], *, backend: str) -> None:
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
        self._function = function
        self._signature = signature
        self._backend = backend
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
            specs[name] = _generic_spec(array)
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
) -> CompiledFunction: ...


@typing.overload
def compile(
    function: None = None,
    *,
    backend: str = "reference",
) -> typing.Callable[[typing.Callable[..., object]], CompiledFunction]: ...


def compile(
    function: typing.Callable[..., object] | None = None,
    *,
    backend: str = "reference",
) -> CompiledFunction | typing.Callable[[typing.Callable[..., object]], CompiledFunction]:
    """Capture a normal array function lazily from its first concrete signature."""
    if function is None:
        return lambda target: CompiledFunction(target, backend=backend)
    return CompiledFunction(function, backend=backend)


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
    "broadcast_to",
    "capabilities",
    "compile",
    "divide",
    "dlpack_device",
    "einsum",
    "exp",
    "float32",
    "float64",
    "import_dlpack",
    "input_array",
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
    "trace",
]
