"""Experimental public symbolic Array-API-shaped frontend.

This module is a curated facade over the canonical compiler-owned frontend. It
is intentionally not an Array API conformance surface: symbolic arrays do not
implement ``__array_namespace__`` and unsupported semantics fail closed.
"""

from __future__ import annotations

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
from generativeqc_compiler.tensor import Index, IndexSpace, Program, TensorSpec

API_VERSION = 1

add = _namespace.add
broadcast_to = _namespace.broadcast_to
divide = _namespace.divide
einsum = _namespace.einsum
exp = _namespace.exp
log = _namespace.log
matmul = _namespace.matmul
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


def capabilities() -> dict[str, object]:
    """Return the detached capability contract for this public preview."""
    report = _compiler_capabilities()
    report.update(
        {
            "public_api_version": API_VERSION,
            "surface": "array-api-shaped-experimental-public-preview",
            "stability": "experimental",
            "import_path": "generativeqc.experimental.array_api",
        }
    )
    return report


__all__ = [
    "API_VERSION",
    "DLPACK_INTEROP_VERSION",
    "FRONTEND_VERSION",
    "SUPPORTED_FUNCTIONS",
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
    "broadcast_to",
    "capabilities",
    "divide",
    "dlpack_device",
    "einsum",
    "exp",
    "import_dlpack",
    "input_array",
    "log",
    "matmul",
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
