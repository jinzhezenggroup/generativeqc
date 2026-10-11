"""Declared issue-#633 frontend subset; unsupported behavior fails closed."""

from __future__ import annotations

from .interop import DLPACK_INTEROP_VERSION

FRONTEND_VERSION = 1

## The 2025.12 standard is a comparison target, not an implementation claim.
# Group definitions own the bounded preview inventory and its matrix together.
_OPERATION_GROUPS: dict[str, tuple[str, ...]] = {
    "real-elementwise": (
        "add",
        "subtract",
        "multiply",
        "divide",
        "negative",
        "pow",
        "exp",
        "log",
        "sqrt",
        "square",
        "reciprocal",
    ),
    "real-comparison": (
        "equal",
        "not_equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
    ),
    "array-creation": (
        "zeros",
        "ones",
        "full",
        "zeros_like",
        "ones_like",
        "full_like",
    ),
    "dtype-conversion": (
        "astype",
    ),
    "real-reduction": (
        "sum",
        "mean",
    ),
    "shape-transform": (
        "reshape_explicit_indices",
        "permute_dims",
        "reshape",
        "broadcast_to",
        "broadcast_shapes",
        "broadcast_arrays",
        "expand_dims",
        "squeeze",
        "moveaxis",
        "flip",
        "broadcast_to_explicit_indices",
        "matrix_transpose",
    ),
    "static-indexing": (
        "slice",
        "take",
        "slice_static",
        "take_static",
    ),
    "tensor-algebra": (
        "matmul",
        "einsum_extension",
    ),
}

# Keep the public export statically resolvable by the API contract checker.
# test_array_api_coverage_matrix verifies this literal against _OPERATION_GROUPS.
SUPPORTED_FUNCTIONS = frozenset(
    {
        "add",
        "equal",
        "not_equal",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "astype",
        "zeros",
        "ones",
        "full",
        "zeros_like",
        "ones_like",
        "full_like",
        "subtract",
        "multiply",
        "divide",
        "negative",
        "pow",
        "exp",
        "log",
        "sqrt",
        "square",
        "reciprocal",
        "sum",
        "mean",
        "reshape_explicit_indices",
        "permute_dims",
        "reshape",
        "broadcast_to",
        "broadcast_shapes",
        "broadcast_arrays",
        "expand_dims",
        "squeeze",
        "moveaxis",
        "flip",
        "slice",
        "take",
        "broadcast_to_explicit_indices",
        "slice_static",
        "take_static",
        "matmul",
        "matrix_transpose",
        "einsum_extension",
    }
)

_OPERATION_DTYPES: dict[str, str] = {
    "real-elementwise": "float32|float64",
    "real-comparison": "float32|float64 -> bool",
    "array-creation": "bool|float32|float64 (operation-specific admission)",
    "dtype-conversion": "bool-identity|float32|float64",
    "real-reduction": "float32|float64",
    "shape-transform": "bool|float32|float64 (generic-array restrictions)",
    "static-indexing": "bool|float32|float64 (generic-array restrictions)",
    "tensor-algebra": "float32|float64",
}

_AD_STATUS: dict[str, str] = {
    "real-elementwise": "TensorIR-rule-if-explicitly-differentiable",
    "real-comparison": "non-differentiable",
    "array-creation": "constant-no-input-adjoint",
    "dtype-conversion": "TensorIR-rule-if-explicitly-differentiable",
    "real-reduction": "TensorIR-rule-if-explicitly-differentiable",
    "shape-transform": "operation-dependent-TensorIR-rule",
    "static-indexing": "operation-dependent-TensorIR-rule",
    "tensor-algebra": "TensorIR-rule-if-explicitly-differentiable",
}

# Only reviewed keyword constraints are enumerated; empty means not audited.
_AUDITED_KEYWORDS: dict[str, dict[str, str]] = {
    "sum": {
        "axis": "static-int|tuple|None",
        "keepdims": "generic-arrays-only",
        "dtype": "conversion-unsupported",
    },
    "mean": {
        "axis": "static-int|tuple|None",
        "keepdims": "generic-arrays-only",
    },
    "reshape": {
        "shape": "static; at most one -1",
    },
    "expand_dims": {
        "axis": "static-int|tuple",
    },
    "flip": {
        "axis": "static-int|tuple|None",
        "capture": "gather extent <= 65536",
    },
    "astype": {
        "dtype": "bool identity or real float32/float64",
        "copy": "eager-copy-policy",
        "device": "None-only",
    },
    "full": {
        "dtype": "bool|float32|float64; unsupported integer default",
        "device": "None-only",
    },
    "take": {
        "indices": "static nonnegative in-bounds tuple",
        "axis": "static",
    },
    "slice": {
        "ranges": "static in-bounds half-open",
    },
    "einsum_extension": {
        "equation": "explicit-output; no ellipsis",
    },
}


def _operation_matrix() -> dict[str, dict[str, object]]:
    """Describe admitted reference operations without implying backend conformance."""
    return {
        name: {
            "category": category,
            "dtypes": _OPERATION_DTYPES[category],
            "device": "device-none-cpu-reference",
            "eager": "numpy-reference",
            "capture": "tensorir-reference",
            "ad": _AD_STATUS[category],
            "native_cpu": False,
            "native_cuda": False,
            "audited_keywords": dict(_AUDITED_KEYWORDS.get(name, {})),
        }
        for category, names in _OPERATION_GROUPS.items()
        for name in names
    }


def capabilities() -> dict[str, object]:
    """Return a detached capability description for diagnostics/tests."""
    return {
        "frontend_version": FRONTEND_VERSION,
        "surface": "array-api-shaped-internal-preview",
        "array_api_version": None,
        "target_standard_for_gap_analysis": "2025.12",
        "operation_matrix_schema": "generativeqc.array-api-operation-coverage.v1",
        "operation_matrix": _operation_matrix(),
        "array_namespace_protocol": False,
        "implicit_broadcast": "generic-arrays",
        "reshape_requires_explicit_indices": "scientific-arrays-only",
        "broadcast_requires_explicit_indices_and_axes": "scientific-arrays-only",
        "generic_indexing": "integers-slices-newaxis-ellipsis-static",
        "generic_shape_manipulation": "static-reshape-transpose-gather",
        "permute_negative_axes": True,
        "flip_gather_axis_budget": 65536,
        "generic_slice_steps": "static-nonzero",
        "scientific_indexing": "rank-preserving-static-slices",
        "scientific_slice_ranges": "static-half-open-unit-step",
        "take_indices": "static-int-tuple",
        "dtype_promotion": "generic-float32-float64-explicit-cast",
        "scientific_dtype_promotion": False,
        "dtype_helpers": "bool-identity-or-real-float32-float64; finfo-real-only",
        "boolean_data": "typed-input-constant-comparison-reference-only",
        "comparison_inputs": "finite-real-float32-float64-only",
        "boolean_native_lowering": False,
        "astype_devices": "device-none-only-cpu-reference",
        "uniform_creation": "typed-bool-or-exact-real-scalar-tensorir-broadcast",
        "creation_devices": "device-none-only-cpu-reference",
        "full_integer_default": "requires-unsupported-integer-dtype",
        "scientific_like_creation": False,
        "generic_reduction_keepdims": True,
        "scientific_reduction_keepdims": False,
        "empty_mean": "unsupported-nonfinite",
        "dynamic_shapes": False,
        "python_control_flow": False,
        "functions": tuple(sorted(SUPPORTED_FUNCTIONS)),
        "dlpack_interop": {
            "version": DLPACK_INTEROP_VERSION,
            "import": "same-device-zero-copy",
            "device_transfer": False,
            "stream_handoff": "consumer-owned-protocol",
            "raw_capsule_ownership": "not-retained",
        },
        "tensorir_metadata": (
            "index_spaces",
            "representation",
            "symmetry",
            "exact_coefficients",
            "role",
            "differentiability",
        ),
    }
