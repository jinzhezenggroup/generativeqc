"""Declared issue-#633 frontend subset; unsupported behavior fails closed."""

from __future__ import annotations

from .interop import DLPACK_INTEROP_VERSION

FRONTEND_VERSION = 1

SUPPORTED_FUNCTIONS = frozenset(
    {
        "add",
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


def capabilities() -> dict[str, object]:
    """Return a detached capability description for diagnostics/tests."""
    return {
        "frontend_version": FRONTEND_VERSION,
        "surface": "array-api-shaped-internal-preview",
        "array_api_version": None,
        "array_namespace_protocol": False,
        "implicit_broadcast": "generic-arrays",
        "reshape_requires_explicit_indices": "scientific-arrays-only",
        "broadcast_requires_explicit_indices_and_axes": "scientific-arrays-only",
        "generic_indexing": "integers-slices-newaxis-ellipsis-static",
        "generic_slice_steps": "static-nonzero",
        "scientific_indexing": "rank-preserving-static-slices",
        "scientific_slice_ranges": "static-half-open-unit-step",
        "take_indices": "static-int-tuple",
        "dtype_promotion": False,
        "uniform_creation": "exact-scalar-tensorir-broadcast",
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
