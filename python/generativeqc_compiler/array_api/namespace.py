"""Bounded Array-API-like namespace lowering directly to TensorIR."""

from __future__ import annotations

import builtins
import typing
from fractions import Fraction
from math import copysign, isfinite
from math import prod as _shape_product
from string import ascii_letters
from struct import pack, unpack

from generativeqc_compiler.tensor import ir as tensor_ir
from generativeqc_compiler.tensor.types import (
    Index,
    IndexSpace,
    TensorSpec,
    checked_shape,
)

from .array import ExactScalar, VibeArray

_GENERIC_SPACE_PREFIX = "_array_extent_"
_MAX_STATIC_GATHER_POSITIONS = 65_536


def _array(value: object, name: str = "operand") -> VibeArray:
    if not isinstance(value, VibeArray):
        raise TypeError(f"{name} must be a symbolic VibeArray")
    return value


def _exact(value: object, name: str = "scalar") -> Fraction:
    if type(value) not in (int, str, Fraction):
        raise TypeError(
            f"{name} must be an exact integer, Fraction, or rational string; "
            "floating-point scalar spelling is not accepted"
        )
    return Fraction(typing.cast("ExactScalar", value))


def _binary_arrays(
    left: object, right: object, name: str
) -> tuple[VibeArray, VibeArray]:
    return _array(left, f"{name} left operand"), _array(right, f"{name} right operand")


def _shape(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or any(
        type(extent) is not int or extent < 0 for extent in value
    ):
        raise TypeError(f"{name} shape must be a tuple of nonnegative integers")
    return value


def _reshape_shape(value: object, size: int) -> tuple[int, ...]:
    if not isinstance(value, tuple) or any(type(extent) is not int for extent in value):
        raise TypeError("reshape shape must be a tuple of integers")
    missing = [position for position, extent in enumerate(value) if extent == -1]
    if len(missing) > 1 or any(extent < -1 for extent in value):
        raise ValueError("reshape permits at most one inferred -1 dimension")
    if not missing:
        shape = _shape(value, "reshape")
        product = 1
        for extent in shape:
            product *= extent
        if product != size:
            raise ValueError("reshape must preserve element count")
        return shape
    known = 1
    for extent in value:
        if extent != -1:
            known *= extent
    if known == 0 or size % known:
        raise ValueError("reshape inferred dimension is not integral")
    shape = list(value)
    shape[missing[0]] = size // known
    return tuple(shape)


def _target_indices(shape: object, indices: object, name: str) -> tuple[Index, ...]:
    target_shape = _shape(shape, name)
    if not isinstance(indices, tuple) or any(
        not isinstance(index, Index) for index in indices
    ):
        raise TypeError(f"{name} requires an explicit tuple of TensorIR Index objects")
    if tuple(index.extent for index in indices) != target_shape:
        raise ValueError(f"{name} shape must match the explicit target indices")
    return indices


def _generic_space(extent: int) -> IndexSpace:
    return IndexSpace(f"{_GENERIC_SPACE_PREFIX}{extent}", "matrix", extent)


def _generic_indices(shape: tuple[int, ...]) -> tuple[Index, ...]:
    return tuple(
        Index(f"axis_{axis}", _generic_space(extent))
        for axis, extent in enumerate(shape)
    )


def _creation_shape(value: object) -> tuple[int, ...]:
    """Validate scalar/tuple shapes without accepting bool or negative extents."""
    if type(value) is int:
        value = (value,)
    return _shape(value, "creation")


def full(
    shape: int | tuple[int, ...], fill_value: object, *, dtype: str = "float64"
) -> VibeArray:
    """Construct a symbolic uniform array without expanding its literal payload."""
    target = _creation_shape(shape)
    if dtype not in ("float32", "float64", "bool"):
        raise TypeError("symbolic full supports bool, float32 or float64")
    if dtype == "bool":
        if type(fill_value) is not bool:
            raise TypeError("symbolic bool full requires a bool fill value")
        factor = fill_value
    else:
        factor = _generic_scalar(fill_value, "full fill value")
    scalar = tensor_ir.constant(factor, TensorSpec(dtype=dtype, role="constant"))
    if not target:
        return VibeArray(scalar)
    return VibeArray(tensor_ir.broadcast(scalar, _generic_indices(target), ()))


def zeros(shape: int | tuple[int, ...], *, dtype: str = "float64") -> VibeArray:
    return full(shape, False if dtype == "bool" else 0, dtype=dtype)


def ones(shape: int | tuple[int, ...], *, dtype: str = "float64") -> VibeArray:
    return full(shape, True if dtype == "bool" else 1, dtype=dtype)


def full_like(x: object, fill_value: object, *, dtype: str | None = None) -> VibeArray:
    """Create a generic constant with the same shape/dtype, not a QC relabeling."""
    value = _array(x)
    if not _is_generic_array(value):
        raise TypeError(
            "full_like for scientifically annotated arrays requires explicit "
            "TensorIR index and representation metadata"
        )
    return full(value.shape, fill_value, dtype=value.dtype if dtype is None else dtype)


def zeros_like(x: object, *, dtype: str | None = None) -> VibeArray:
    value = _array(x)
    target = value.dtype if dtype is None else dtype
    return full_like(value, False if target == "bool" else 0, dtype=target)


def ones_like(x: object, *, dtype: str | None = None) -> VibeArray:
    value = _array(x)
    target = value.dtype if dtype is None else dtype
    return full_like(value, True if target == "bool" else 1, dtype=target)


def _is_generic_array(value: VibeArray) -> bool:
    return all(
        index.space.kind == "matrix"
        and index.space.name == f"{_GENERIC_SPACE_PREFIX}{index.space.size}"
        for index in value.node.spec.indices
    )


def _canonical_generic(value: VibeArray) -> VibeArray:
    if not _is_generic_array(value):
        return value
    indices = _generic_indices(value.shape)
    if tuple(index.domain for index in value.node.spec.indices) == tuple(
        index.domain for index in indices
    ):
        return value
    return VibeArray(tensor_ir.reshape(value.node, indices))


def _broadcast_shape(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    rank = builtins.max(len(left), len(right))
    a = (1,) * (rank - len(left)) + left
    b = (1,) * (rank - len(right)) + right
    result = []
    for first, second in zip(a, b, strict=True):
        if first == second:
            result.append(first)
        elif first == 1:
            result.append(second)
        elif second == 1:
            result.append(first)
        else:
            raise ValueError(
                f"operands with shapes {left} and {right} cannot be broadcast together"
            )
    return tuple(result)


def broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Static-shape broadcasting without creating arrays or runtime work."""
    result: tuple[int, ...] = ()
    for shape in shapes:
        result = _broadcast_shape(result, _shape(shape, "broadcast_shapes"))
    return result


def broadcast_arrays(*arrays: object) -> tuple[VibeArray, ...]:
    """Broadcast each generic symbolic input explicitly, keeping its own dtype."""
    values = tuple(_array(item, "broadcast_arrays operand") for item in arrays)
    if any(not _is_generic_array(item) for item in values):
        raise TypeError(
            "broadcast_arrays needs generic arrays; scientific domains require "
            "explicit TensorIR axis mappings"
        )
    target = broadcast_shapes(*(value.shape for value in values))
    return tuple(_broadcast_generic(value, target) for value in values)


def _broadcast_generic(value: VibeArray, target_shape: tuple[int, ...]) -> VibeArray:
    value = _canonical_generic(value)
    if len(value.shape) > len(target_shape):
        raise ValueError("cannot broadcast to fewer dimensions")
    padded = (1,) * (len(target_shape) - len(value.shape)) + value.shape
    for source, target in zip(padded, target_shape, strict=True):
        if source != target and source != 1:
            raise ValueError(
                f"array with shape {value.shape} cannot broadcast to {target_shape}"
            )
    target_indices = _generic_indices(target_shape)
    offset = len(target_shape) - value.ndim
    kept_indices = []
    kept_axes = []
    for source_axis, source_index in enumerate(value.node.spec.indices):
        target_axis = offset + source_axis
        source_extent = value.shape[source_axis]
        target_extent = target_shape[target_axis]
        if source_extent == target_extent:
            kept_indices.append(source_index)
            kept_axes.append(target_axis)
        elif source_extent != 1:
            raise ValueError(
                f"array with shape {value.shape} cannot broadcast to {target_shape}"
            )
    node = value.node
    if len(kept_indices) != value.ndim:
        node = tensor_ir.reshape(node, tuple(kept_indices))
    if tuple(index.domain for index in node.spec.indices) == tuple(
        target_indices[axis].domain for axis in kept_axes
    ) and len(kept_axes) == len(target_shape):
        return VibeArray(node)
    return VibeArray(tensor_ir.broadcast(node, target_indices, tuple(kept_axes)))


def astype(x: object, dtype: str, *, copy: bool = True) -> VibeArray:
    """Lower an explicit real-valued cast, preserving scientific index spaces."""
    value = _array(x)
    if dtype not in ("float32", "float64"):
        raise TypeError("astype supports only float32 and float64")
    if type(copy) is not bool:
        raise TypeError("astype copy must be a bool")
    if not copy and value.dtype == dtype:
        return value
    return VibeArray(tensor_ir.cast(value.node, dtype))


def _promote_generic_arrays(*values: VibeArray) -> tuple[VibeArray, ...]:
    """Promote supported generic floats by inserting explicit TensorIR casts."""
    if any(not _is_generic_array(value) for value in values):
        raise TypeError("promotion requires generic arrays")
    if any(value.dtype not in ("float32", "float64") for value in values):
        raise TypeError("floating arithmetic requires real float32 or float64 arrays")
    target = (
        "float64" if any(value.dtype == "float64" for value in values) else "float32"
    )
    return tuple(astype(value, target, copy=False) for value in values)


def _generic_binary(
    left: VibeArray, right: VibeArray
) -> tuple[VibeArray, VibeArray] | None:
    if not (_is_generic_array(left) and _is_generic_array(right)):
        return None
    left, right = _promote_generic_arrays(left, right)
    shape = _broadcast_shape(left.shape, right.shape)
    return _broadcast_generic(left, shape), _broadcast_generic(right, shape)


def _comparison(op: str, x1: object, x2: object) -> VibeArray:
    if isinstance(x1, VibeArray) and isinstance(x2, VibeArray):
        left, right = x1, x2
        operands = _generic_binary(left, right)
        if operands is not None:
            left, right = operands
    elif isinstance(x1, VibeArray):
        left, right = _generic_array_and_scalar(x1, x2, name=op)
    elif isinstance(x2, VibeArray):
        right, left = _generic_array_and_scalar(x2, x1, name=op)
    else:
        raise TypeError(f"{op} requires at least one symbolic VibeArray")
    return _canonical_generic(VibeArray(tensor_ir.compare(op, left.node, right.node)))


def where(condition: object, x1: object, x2: object) -> VibeArray:
    """Capture finite generic real selection; condition and branches broadcast separately."""
    if type(condition) is bool:
        condition = full((), condition, dtype="bool")
    predicate = _array(condition, "where condition")
    if predicate.dtype != "bool":
        raise TypeError("where condition must be bool data")
    arrays = tuple(value for value in (x1, x2) if isinstance(value, VibeArray))
    if any(
        not _is_generic_array(value)
        or value.node.spec.representation != "general"
        or value.node.spec.symmetries
        or any(index.space.spin is not None for index in value.node.spec.indices)
        for value in (predicate, *arrays)
    ):
        raise TypeError(
            "where supports only generic arrays without scientific annotations"
        )
    dtype = (
        "float64"
        if any(value.dtype == "float64" for value in arrays) or not arrays
        else "float32"
    )
    branches = tuple(
        value
        if isinstance(value, VibeArray)
        else _generic_exact_scalar(value, dtype=dtype, name="where branch scalar")
        for value in (x1, x2)
    )
    left, right = _promote_generic_arrays(*branches)
    predicate, left, right = broadcast_arrays(predicate, left, right)
    return VibeArray(tensor_ir.where(predicate.node, left.node, right.node))


def _extremum(x1: object, x2: object, *, name: str) -> VibeArray:
    arrays = tuple(value for value in (x1, x2) if isinstance(value, VibeArray))
    if not arrays:
        raise TypeError(f"{name} requires at least one symbolic VibeArray")
    for value in arrays:
        if value.dtype not in ("float32", "float64"):
            raise TypeError(f"{name} requires real float32 or float64 arrays")
        if (
            not _is_generic_array(value)
            or value.node.spec.representation != "general"
            or value.node.spec.symmetries
            or any(index.space.spin is not None for index in value.node.spec.indices)
        ):
            raise TypeError(
                f"{name} supports only generic arrays without scientific annotations"
            )
    dtype = (
        "float64" if any(value.dtype == "float64" for value in arrays) else "float32"
    )
    operands = []
    for value in (x1, x2):
        if isinstance(value, VibeArray):
            operands.append(value)
        else:
            factor = _generic_scalar(value, f"{name} scalar")
            try:
                rounded = float(factor)
                if dtype == "float32":
                    rounded = unpack("f", pack("f", rounded))[0]
            except (ValueError, OverflowError) as error:
                raise ValueError(f"{name} scalar must be finite in {dtype}") from error
            if not isfinite(rounded):
                raise ValueError(f"{name} scalar must be finite in {dtype}")
            operands.append(
                _generic_exact_scalar(value, dtype=dtype, name=f"{name} scalar")
            )
    left, right = _promote_generic_arrays(*operands)
    # Share the broadcast values between predicate and branches. Materializing
    # duplicate broadcasts would consume the reference retained-byte budget.
    left, right = broadcast_arrays(left, right)
    predicate = less(right, left) if name == "minimum" else greater(right, left)
    return where(predicate, right, left)


def minimum(x1: object, x2: object) -> VibeArray:
    """Select the finite generic minimum; equality retains the first operand.

    Signed-zero order is implementation-defined by Array API 2025.12. The
    comparison-selector AD opt-in follows this branch at ties, without claiming
    mathematical differentiability there.
    """
    return _extremum(x1, x2, name="minimum")


def maximum(x1: object, x2: object) -> VibeArray:
    """Select the finite generic maximum with the same tie policy as minimum."""
    return _extremum(x1, x2, name="maximum")


def clip(x: object, /, min: object = None, max: object = None) -> VibeArray:
    """Clamp finite generic real arrays using existing comparisons and selection.

    Array bounds must have x's dtype. Apply the lower bound before the upper
    bound, keeping the current value on equality; reversed bounds select max.
    Comparisons define a selected-branch AD convention, not a derivative at ties.
    """
    value = _array(x, "clip x")
    if value.dtype not in ("float32", "float64"):
        raise TypeError("clip requires real float32 or float64 arrays")
    bounds = []
    for name, bound in (("min", min), ("max", max)):
        if bound is None:
            bounds.append(None)
        elif isinstance(bound, VibeArray):
            if bound.dtype != value.dtype:
                raise TypeError("clip array bounds must have the same dtype as x")
            bounds.append(bound)
        else:
            factor = _generic_scalar(bound, f"clip {name}")
            try:
                rounded = float(factor)
                if value.dtype == "float32":
                    rounded = unpack("f", pack("f", rounded))[0]
            except (ValueError, OverflowError) as error:
                raise ValueError(
                    f"clip {name} must be finite in {value.dtype}"
                ) from error
            if not isfinite(rounded):
                raise ValueError(f"clip {name} must be finite in {value.dtype}")
            bounds.append(
                VibeArray(
                    tensor_ir.constant(
                        factor, TensorSpec(dtype=value.dtype, role="constant")
                    )
                )
            )
    for operand in (value, *bounds):
        if operand is not None and (
            not _is_generic_array(operand)
            or operand.node.spec.representation != "general"
            or operand.node.spec.symmetries
            or any(index.space.spin is not None for index in operand.node.spec.indices)
        ):
            raise TypeError(
                "clip supports only generic arrays without scientific annotations"
            )
    # Validate the joint shape even when one bound would mask every value.
    shape = broadcast_shapes(
        value.shape, *(bound.shape for bound in bounds if bound is not None)
    )
    value = _broadcast_generic(value, shape)
    # Share each bound between its comparison and selected real branch. The
    # reference interpreter charges every live logical array before any CSE.
    lower, upper = (
        _broadcast_generic(bound, shape) if bound is not None else None
        for bound in bounds
    )
    if lower is not None:
        value = where(less(value, lower), lower, value)
    if upper is not None:
        value = where(greater(value, upper), upper, value)
    return value


def equal(x1: object, x2: object) -> VibeArray:
    return _comparison("equal", x1, x2)


def not_equal(x1: object, x2: object) -> VibeArray:
    return _comparison("not_equal", x1, x2)


def greater(x1: object, x2: object) -> VibeArray:
    return _comparison("greater", x1, x2)


def greater_equal(x1: object, x2: object) -> VibeArray:
    return _comparison("greater_equal", x1, x2)


def less(x1: object, x2: object) -> VibeArray:
    return _comparison("less", x1, x2)


def less_equal(x1: object, x2: object) -> VibeArray:
    return _comparison("less_equal", x1, x2)


def _generic_scalar(value: object, name: str) -> Fraction:
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value == 0 and copysign(1.0, value) < 0:
            raise ValueError(f"{name} cannot represent a negative-zero literal")
        return Fraction.from_float(value)
    return _exact(value, name)


def _generic_exact_scalar(value: object, *, dtype: str, name: str) -> VibeArray:
    factor = _generic_scalar(value, name)
    spec = TensorSpec(dtype=dtype, role="constant")
    return VibeArray(tensor_ir.constant(factor, spec))


def _generic_array_and_scalar(
    array: VibeArray, scalar: object, *, name: str
) -> tuple[VibeArray, VibeArray]:
    if not _is_generic_array(array):
        raise TypeError(f"{name} requires two symbolic arrays for scientific domains")
    scalar_array = _generic_exact_scalar(
        scalar, dtype=array.dtype, name=f"{name} scalar"
    )
    return array, _broadcast_generic(scalar_array, array.shape)


def add(x1: object, x2: object) -> VibeArray:
    """Elementwise add with standard broadcasting for generic public arrays."""
    if isinstance(x1, VibeArray) and isinstance(x2, VibeArray):
        left, right = x1, x2
        operands = _generic_binary(left, right)
        if operands is not None:
            left, right = operands
    elif isinstance(x1, VibeArray):
        left, right = _generic_array_and_scalar(x1, x2, name="add")
    elif isinstance(x2, VibeArray):
        right, left = _generic_array_and_scalar(x2, x1, name="add")
    else:
        raise TypeError("add requires at least one symbolic VibeArray")
    return _canonical_generic(VibeArray(tensor_ir.add(left.node, right.node)))


def subtract(x1: object, x2: object) -> VibeArray:
    """Elementwise subtraction with standard broadcasting for generic arrays."""
    if isinstance(x1, VibeArray) and isinstance(x2, VibeArray):
        left, right = x1, x2
        operands = _generic_binary(left, right)
        if operands is not None:
            left, right = operands
    elif isinstance(x1, VibeArray):
        left, right = _generic_array_and_scalar(x1, x2, name="subtract")
    elif isinstance(x2, VibeArray):
        right, left = _generic_array_and_scalar(x2, x1, name="subtract")
    else:
        raise TypeError("subtract requires at least one symbolic VibeArray")
    return _canonical_generic(
        VibeArray(tensor_ir.add(left.node, right.node, coefficients=(1, -1)))
    )


def multiply(x1: object, x2: object) -> VibeArray:
    """Multiply arrays elementwise or scale one array by an exact scalar."""
    if isinstance(x1, VibeArray) and isinstance(x2, VibeArray):
        operands = _generic_binary(x1, x2)
        left, right = operands if operands is not None else (x1, x2)
        return _canonical_generic(VibeArray(tensor_ir.multiply(left.node, right.node)))
    if isinstance(x1, VibeArray):
        if _is_generic_array(x1):
            left, right = _generic_array_and_scalar(x1, x2, name="multiply")
            return _canonical_generic(
                VibeArray(tensor_ir.multiply(left.node, right.node))
            )
        factor = _exact(x2)
        return VibeArray(tensor_ir.add(x1.node, coefficients=(factor,)))
    if isinstance(x2, VibeArray):
        if _is_generic_array(x2):
            right, left = _generic_array_and_scalar(x2, x1, name="multiply")
            return _canonical_generic(
                VibeArray(tensor_ir.multiply(left.node, right.node))
            )
        factor = _exact(x1)
        return VibeArray(tensor_ir.add(x2.node, coefficients=(factor,)))
    raise TypeError("multiply requires at least one symbolic VibeArray")


def divide(x1: object, x2: object) -> VibeArray:
    """Divide arrays elementwise or divide one array by an exact scalar."""
    if isinstance(x1, VibeArray) and isinstance(x2, VibeArray):
        operands = _generic_binary(x1, x2)
        left, right = operands if operands is not None else (x1, x2)
        return _canonical_generic(VibeArray(tensor_ir.divide(left.node, right.node)))
    if isinstance(x1, VibeArray):
        if _is_generic_array(x1):
            left, right = _generic_array_and_scalar(x1, x2, name="divide")
            return _canonical_generic(
                VibeArray(tensor_ir.divide(left.node, right.node))
            )
        denominator = _exact(x2, "divisor")
        if denominator == 0:
            raise ZeroDivisionError("exact scalar divisor cannot be zero")
        return VibeArray(
            tensor_ir.add(x1.node, coefficients=(Fraction(1, 1) / denominator,))
        )
    if isinstance(x2, VibeArray):
        right, left = _generic_array_and_scalar(x2, x1, name="divide")
        return _canonical_generic(VibeArray(tensor_ir.divide(left.node, right.node)))
    raise TypeError("divide requires at least one symbolic VibeArray")


def positive(x: object) -> VibeArray:
    """Preserve a real symbolic array and its scientific index metadata."""
    value = _array(x)
    if value.dtype not in ("float32", "float64"):
        raise TypeError("positive supports only float32 and float64")
    return value


def negative(x: object) -> VibeArray:
    value = _array(x)
    return VibeArray(tensor_ir.add(value.node, coefficients=(-1,)))


def pow(x: object, exponent: object) -> VibeArray:
    value = _array(x)
    factor = (
        _generic_scalar(exponent, "exponent")
        if _is_generic_array(value)
        else _exact(exponent, "exponent")
    )
    return VibeArray(tensor_ir.power(value.node, factor))


def exp(x: object) -> VibeArray:
    return VibeArray(tensor_ir.exp(_array(x).node))


def log(x: object) -> VibeArray:
    return VibeArray(tensor_ir.log(_array(x).node))


def sqrt(x: object) -> VibeArray:
    return VibeArray(tensor_ir.sqrt(_array(x).node))


def square(x: object) -> VibeArray:
    """Square without imposing the strictly positive domain of generic power."""
    value = _array(x)
    return _canonical_generic(VibeArray(tensor_ir.multiply(value.node, value.node)))


def reciprocal(x: object) -> VibeArray:
    """Elementwise reciprocal; generic arrays admit exact scalar broadcasting."""
    value = _array(x)
    if not _is_generic_array(value):
        raise TypeError(
            "reciprocal requires a generic array; scalar broadcasting into "
            "scientifically annotated domains is unsupported"
        )
    return divide(1, value)


def _normalized_axes(axis: object, rank: int, operation: str) -> tuple[int, ...]:
    """Normalize unique signed axes, rejecting bools and duplicate positions."""
    if type(axis) is int:
        axes = (axis,)
    elif isinstance(axis, tuple) and all(type(item) is int for item in axis):
        axes = axis
    else:
        raise TypeError(f"{operation} axis must be an integer or tuple of integers")
    normalized = tuple(_axis(item, rank, operation) for item in axes)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{operation} axes must be unique")
    return normalized


def _expand_shape(shape: tuple[int, ...], axis: object) -> tuple[int, ...]:
    count = 1 if type(axis) is int else len(axis) if isinstance(axis, tuple) else 0
    new_rank = len(shape) + count
    inserted = set(_normalized_axes(axis, new_rank, "expand_dims"))
    if len(inserted) != count:
        raise ValueError("expand_dims axis count is inconsistent")
    original = iter(shape)
    return tuple(
        1 if position in inserted else next(original) for position in range(new_rank)
    )


def expand_dims(x: object, axis: int | tuple[int, ...]) -> VibeArray:
    value = _array(x)
    if not _is_generic_array(value):
        raise ValueError(
            "expand_dims scientific arrays require explicit TensorIR indices"
        )
    return reshape(value, _expand_shape(value.shape, axis))


def _squeezed_shape(
    shape: tuple[int, ...], axis: int | tuple[int, ...]
) -> tuple[int, ...]:
    axes = set(_normalized_axes(axis, len(shape), "squeeze"))
    if any(shape[position] != 1 for position in axes):
        raise ValueError("squeeze requires singleton dimensions at specified axes")
    return tuple(
        extent for position, extent in enumerate(shape) if position not in axes
    )


def squeeze(x: object, axis: int | tuple[int, ...]) -> VibeArray:
    value = _array(x)
    if not _is_generic_array(value):
        raise ValueError("squeeze scientific arrays require explicit TensorIR indices")
    return reshape(value, _squeezed_shape(value.shape, axis))


def _moveaxis_order(
    rank: int, source: int | tuple[int, ...], destination: int | tuple[int, ...]
) -> tuple[int, ...]:
    src = _normalized_axes(source, rank, "moveaxis source")
    dst = _normalized_axes(destination, rank, "moveaxis destination")
    if len(src) != len(dst):
        raise ValueError("moveaxis source and destination must have matching lengths")
    order = [axis for axis in range(rank) if axis not in src]
    for target, original in sorted(zip(dst, src, strict=True)):
        order.insert(target, original)
    return tuple(order)


def moveaxis(
    x: object, source: int | tuple[int, ...], destination: int | tuple[int, ...]
) -> VibeArray:
    value = _array(x)
    return permute_dims(value, _moveaxis_order(value.ndim, source, destination))


def flip(x: object, *, axis: int | tuple[int, ...] | None = None) -> VibeArray:
    """Flip generic arrays using bounded, explicit TensorIR gather maps."""
    value = _array(x)
    if not _is_generic_array(value):
        raise ValueError("flip scientific arrays require explicit TensorIR index maps")
    axes = (
        tuple(range(value.ndim))
        if axis is None
        else _normalized_axes(axis, value.ndim, "flip")
    )
    if any(value.shape[position] > _MAX_STATIC_GATHER_POSITIONS for position in axes):
        raise ValueError("flip exceeds the bounded static gather index budget")
    node = value.node
    for position in sorted(axes):
        node = tensor_ir.gather(
            node, position, range(value.shape[position] - 1, -1, -1)
        )
    return _canonical_generic(VibeArray(node))


def roll(
    x: object,
    shift: int | tuple[int, ...],
    *,
    axis: int | tuple[int, ...] | None = None,
) -> VibeArray:
    """Roll generic arrays with a bounded total of static gather positions."""
    value = _array(x)
    if not _is_generic_array(value):
        raise ValueError("roll scientific arrays require explicit TensorIR index maps")
    if type(shift) is int:
        shifts = (shift,)
    elif isinstance(shift, tuple) and all(type(item) is int for item in shift):
        shifts = shift
    else:
        raise TypeError("roll shift must be an integer or tuple of integers")
    if axis is None:
        flattened = reshape(value, (value.size,))
        return reshape(roll(flattened, builtins.sum(shifts), axis=0), value.shape)
    if type(axis) is int:
        axes = (axis,)
    elif isinstance(axis, tuple) and all(type(item) is int for item in axis):
        axes = axis
    else:
        raise TypeError("roll axis must be an integer or tuple of integers")
    axes = tuple(_axis(item, value.ndim, "roll") for item in axes)
    if len(shifts) != len(axes):
        if len(shifts) == 1:
            shifts = shifts * len(axes)
        elif len(axes) == 1:
            axes = axes * len(shifts)
        else:
            raise ValueError("roll shift and axis must have broadcastable lengths")
    offsets: dict[int, int] = {}
    for position, offset in zip(axes, shifts, strict=True):
        extent = value.shape[position]
        offsets[position] = (offsets.get(position, 0) + offset) % (extent or 1)
    # Check the complete map budget before constructing any gather positions.
    if (
        builtins.sum(
            value.shape[position] for position, offset in offsets.items() if offset
        )
        > _MAX_STATIC_GATHER_POSITIONS
    ):
        raise ValueError("roll exceeds the bounded static gather index budget")
    result = value
    for position, offset in sorted(offsets.items()):
        if offset:
            extent = value.shape[position]
            result = take(
                result,
                tuple((index - offset) % extent for index in range(extent)),
                axis=position,
            )
    return result


def repeat(x: object, repeats: int, *, axis: int | None = None) -> VibeArray:
    """Repeat generic elements with static reshape/broadcast metadata."""
    value = _array(x)
    if not _is_generic_array(value):
        raise ValueError(
            "repeat scientific arrays require explicit TensorIR index maps"
        )
    if type(repeats) is not int:
        raise TypeError("repeat requires a static integer repetition count")
    if repeats < 0:
        raise ValueError("repeat count must be nonnegative")
    if axis is None:
        value = reshape(value, (value.size,))
        position = 0
    else:
        position = _axis(axis, value.ndim, "repeat")
    if repeats == 1:
        return value
    shape = value.shape
    # Collapse the leading and trailing groups so a valid rank-64 input
    # does not need a rank-65 intermediate. Empty inputs use a harmless
    # unit trailing group without multiplying possibly huge empty extents.
    inner = _shape_product(shape[position + 1 :]) if value.size else 1
    outer = value.size // inner
    expanded = (outer, 1, inner)
    repeated = (outer, repeats, inner)
    output = shape[:position] + (shape[position] * repeats,) + shape[position + 1 :]
    return reshape(broadcast_to(reshape(value, expanded), repeated), output)


def tile(x: object, repetitions: tuple[int, ...]) -> VibeArray:
    """Tile generic arrays with static block repeats, without explicit index maps."""
    value = _array(x)
    if not _is_generic_array(value):
        raise ValueError("tile scientific arrays require explicit TensorIR index maps")
    if not isinstance(repetitions, tuple) or any(
        type(count) is not int for count in repetitions
    ):
        raise TypeError("tile repetitions must be a static tuple of integers")
    if any(count < 0 for count in repetitions):
        raise ValueError("tile repetitions must be nonnegative")
    rank = builtins.max(value.ndim, len(repetitions))
    shape = (1,) * (rank - value.ndim) + value.shape
    counts = (1,) * (rank - len(repetitions)) + repetitions
    output = tuple(extent * count for extent, count in zip(shape, counts, strict=True))
    checked_shape(output, value.node.spec.itemsize)
    if value.size == 0:
        return reshape(value, output)
    if 0 in output:
        # Empty results must not materialize any earlier nonzero repetitions.
        empty = broadcast_to(reshape(value, (1, value.size)), (0, value.size))
        return reshape(empty, output)
    result = reshape(value, shape) if value.shape != shape else value
    for axis, count in enumerate(counts):
        if count == 1:
            continue
        shape = result.shape
        outer = _shape_product(shape[:axis])
        inner = result.size // outer
        # Split into three groups so rank-64 arrays do not require a
        # rank-65 intermediate. Repeat whole axis blocks, not elements.
        blocks = broadcast_to(reshape(result, (outer, 1, inner)), (outer, count, inner))
        result = reshape(
            blocks, shape[:axis] + (shape[axis] * count,) + shape[axis + 1 :]
        )
    return result


def reshape(
    x: object,
    shape: tuple[int, ...],
    *,
    indices: tuple[Index, ...] | None = None,
) -> VibeArray:
    """Reshape generic arrays by shape; scientific arrays require explicit indices."""
    value = _array(x)
    if indices is None:
        if not _is_generic_array(value):
            raise ValueError(
                "frontend reshape requires explicit TensorIR indices for "
                "scientifically annotated arrays"
            )
        target_shape = _reshape_shape(shape, value.size)
        return VibeArray(tensor_ir.reshape(value.node, _generic_indices(target_shape)))
    target = _target_indices(shape, indices, "reshape")
    return VibeArray(tensor_ir.reshape(value.node, target))


def broadcast_to(
    x: object,
    shape: tuple[int, ...],
    *,
    indices: tuple[Index, ...] | None = None,
    axes: tuple[int, ...] | None = None,
) -> VibeArray:
    """Broadcast generic arrays by shape or use explicit scientific axis semantics."""
    value = _array(x)
    if indices is None and axes is None:
        if not _is_generic_array(value):
            raise ValueError(
                "frontend broadcast_to requires explicit TensorIR indices and axes "
                "for scientifically annotated arrays"
            )
        return _broadcast_generic(value, _shape(shape, "broadcast_to"))
    if indices is None or axes is None:
        raise ValueError(
            "frontend broadcast_to requires both explicit TensorIR indices and axes"
        )
    target = _target_indices(shape, indices, "broadcast_to")
    if not isinstance(axes, tuple) or any(type(axis) is not int for axis in axes):
        raise TypeError("broadcast_to axes must be a tuple of integers")
    return VibeArray(tensor_ir.broadcast(value.node, target, axes))


def slice(x: object, ranges: tuple[tuple[int, int], ...]) -> VibeArray:
    """Static unit-step half-open slicing."""
    value = _array(x)
    if not isinstance(ranges, tuple) or any(
        not isinstance(bounds, tuple)
        or len(bounds) != 2
        or any(type(bound) is not int for bound in bounds)
        for bounds in ranges
    ):
        raise TypeError("slice ranges must be a static tuple of (start, stop) pairs")
    result = VibeArray(tensor_ir.slice_tensor(value.node, ranges))
    return _canonical_generic(result)


def _take_axis(axis: object, rank: int) -> int:
    if axis is None:
        if rank != 1:
            raise ValueError("take requires a one-dimensional input when axis is None")
        return 0
    return _axis(axis, rank, "take")


def _take_indices(indices: object, extent: int) -> tuple[int, ...]:
    if not isinstance(indices, tuple) or any(
        type(index) is not int for index in indices
    ):
        raise TypeError("take indices must be a static tuple of integers")
    if any(not -extent <= index < extent for index in indices):
        raise ValueError("gather position is outside its input axis")
    return tuple(index + extent if index < 0 else index for index in indices)


def take(
    x: object,
    indices: tuple[int, ...],
    *,
    axis: int | None = None,
) -> VibeArray:
    """Static gather; omitted axis selects the sole axis of a vector."""
    value = _array(x)
    position = _take_axis(axis, value.ndim)
    positions = _take_indices(indices, value.shape[position])
    result = VibeArray(tensor_ir.gather(value.node, position, positions))
    return _canonical_generic(result)


def _reduction_axes(
    axis: int | tuple[int, ...] | None, rank: int, operation: str
) -> tuple[int, ...]:
    if axis is None:
        axes = tuple(range(rank))
    elif type(axis) is int:
        axes = (_axis(axis, rank, operation),)
    elif isinstance(axis, tuple):
        axes = tuple(_axis(item, rank, operation) for item in axis)
    else:
        raise TypeError("axis must be an int, tuple of ints, or None")
    if len(set(axes)) != len(axes):
        raise ValueError(f"{operation} axes must be unique")
    return tuple(sorted(axes))


def _reduction_dtype(value: VibeArray, dtype: object, name: str) -> str:
    """Validate explicit reduction precision before any metadata-preserving cast."""
    if (
        not _is_generic_array(value)
        or value.node.spec.representation != "general"
        or value.node.spec.symmetries
        or any(index.space.spin is not None for index in value.node.spec.indices)
    ):
        raise ValueError(
            f"{name} dtype conversions require generic arrays without scientific annotations"
        )
    if (
        value.dtype not in ("float32", "float64")
        or type(dtype) is not str
        or dtype not in ("float32", "float64")
    ):
        raise TypeError(f"{name} dtype requires real float32 or float64")
    return dtype


def sum(
    x: object,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: object = None,
    keepdims: bool = False,
) -> VibeArray:
    """Reduce selected axes; generic arrays may retain singleton reduced axes."""
    value = _array(x)
    target = value.dtype if dtype is None else _reduction_dtype(value, dtype, "sum")
    if type(keepdims) is not bool:
        raise TypeError("keepdims must be a bool")
    axes = _reduction_axes(axis, value.ndim, "sum")
    if keepdims and not _is_generic_array(value):
        raise ValueError(
            "keepdims for scientifically annotated arrays requires explicit "
            "TensorIR index metadata"
        )
    if target != value.dtype:
        # Include the complete cast buffer and reduction/possible output reshape
        # before creating any conversion node. Backend limits still cover inputs.
        itemsize = 4 if target == "float32" else 8
        output_size = _shape_product(
            n for i, n in enumerate(value.shape) if i not in axes
        )
        if (value.size + 2 * output_size) * itemsize > 256 * 1024 * 1024:
            raise ValueError("sum exceeds the static reduction graph budget")
        value = astype(value, target, copy=False)
    result = VibeArray(tensor_ir.reduce_sum(value.node, axes=axes))
    if keepdims:
        shape = tuple(
            1 if position in axes else extent
            for position, extent in enumerate(value.shape)
        )
        return reshape(result, shape)
    return _canonical_generic(result)


def _prod_plan(
    shape: tuple[int, ...],
    axes: tuple[int, ...],
    itemsize: int,
    output: tuple[int, ...],
    *,
    cast_bytes: int | None = None,
) -> tuple[int, int, int]:
    """Bound new retained tensors, nodes and gather positions before allocation.

    Each take may retain both a gather and a canonical reshape. Multiplication
    retains one more tensor. Odd tails have extent one and stay separate until
    the axis is finished; they are never duplicated or divided out.
    Existing input graphs and returned copies remain subject to backend budgets.
    """
    current = list(shape)
    retained = (cast_bytes or 0) + _shape_product(output) * itemsize  # final reshape
    nodes, positions, rounds = 1 + (cast_bytes is not None), 0, 0
    empty = any(shape[i] == 0 for i in axes)
    if empty:
        retained += itemsize  # scalar one plus optional output broadcast
        nodes += 2
    elif _shape_product(shape):
        for axis in reversed(axes):
            tails = 0
            while current[axis] > 1:
                extent = current[axis]
                if extent % 2:
                    tail_shape = current.copy()
                    tail_shape[axis] = 1
                    retained += (
                        _shape_product(tail_shape) * itemsize * (2 + bool(tails))
                    )
                    nodes += 2 + bool(tails)
                    positions += 1
                    tails += 1
                current[axis] = extent // 2
                retained += _shape_product(current) * itemsize * 5
                nodes += 5
                positions += 2 * current[axis]
                rounds += 1
            if tails:
                retained += _shape_product(current) * itemsize
                nodes += 1
    if positions > 65_536 or rounds > 64 or nodes > 512 or retained > 256 * 1024 * 1024:
        raise ValueError("prod exceeds the static reduction graph budget")
    return retained, nodes, positions


def prod(
    x: object,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: object = None,
    keepdims: bool = False,
) -> VibeArray:
    """Bounded finite generic product using zero-safe polynomial multiplication."""
    value = _array(x, "prod")
    if value.dtype not in ("float32", "float64"):
        raise TypeError("prod requires real float32 or float64 arrays")
    if (
        not _is_generic_array(value)
        or value.node.spec.representation != "general"
        or value.node.spec.symmetries
        or any(index.space.spin is not None for index in value.node.spec.indices)
    ):
        raise TypeError(
            "prod supports only generic arrays without scientific annotations"
        )
    target = value.dtype if dtype is None else _reduction_dtype(value, dtype, "prod")
    if type(keepdims) is not bool:
        raise TypeError("keepdims must be a bool")
    axes = _reduction_axes(axis, value.ndim, "prod")
    output = (
        tuple(1 if i in axes else n for i, n in enumerate(value.shape))
        if keepdims
        else tuple(n for i, n in enumerate(value.shape) if i not in axes)
    )
    itemsize = 4 if target == "float32" else 8
    _prod_plan(
        value.shape,
        axes,
        itemsize,
        output,
        cast_bytes=value.size * itemsize if target != value.dtype else None,
    )
    if target != value.dtype:
        value = astype(value, target, copy=False)
    if any(value.shape[i] == 0 for i in axes):
        return full(output, 1, dtype=value.dtype)
    if value.size == 0:
        return reshape(value, output)
    result = value
    for position in reversed(axes):
        carry = None
        while result.shape[position] > 1:
            extent = result.shape[position]
            if extent % 2:
                tail = take(result, (extent - 1,), axis=position)
                carry = tail if carry is None else multiply(carry, tail)
            left = take(result, tuple(range(0, extent - 1, 2)), axis=position)
            right = take(result, tuple(range(1, extent, 2)), axis=position)
            result = multiply(left, right)
        if carry is not None:
            result = multiply(result, carry)
    return reshape(result, output)


def _reduce_extremum(
    x: object, *, axis: int | tuple[int, ...] | None, keepdims: bool, name: str
) -> VibeArray:
    value = _array(x, name)
    # Validate metadata even for identities and empty outputs.
    if value.dtype not in ("float32", "float64"):
        raise TypeError(f"{name} requires real float32 or float64 arrays")
    if (
        not _is_generic_array(value)
        or value.node.spec.representation != "general"
        or value.node.spec.symmetries
        or any(index.space.spin is not None for index in value.node.spec.indices)
    ):
        raise TypeError(
            f"{name} supports only generic arrays without scientific annotations"
        )
    if type(keepdims) is not bool:
        raise TypeError("keepdims must be a bool")
    axes = _reduction_axes(axis, value.ndim, name)
    if any(value.shape[position] == 0 for position in axes):
        raise ValueError(f"{name} of an empty reduction is unsupported")
    target = tuple(
        1 if position in axes else extent for position, extent in enumerate(value.shape)
    )
    output = (
        target
        if keepdims
        else tuple(
            extent
            for position, extent in enumerate(value.shape)
            if position not in axes
        )
    )
    if value.size == 0:
        return reshape(value, output)

    # Preflight the whole construction before materializing any index tuples.
    # Adjacent pairs preserve the earliest C-order winner (also for signed zero).
    # Each round halves an axis, giving logarithmic depth rather than a node per
    # input scalar. Existing backend budgets still apply to the resulting graph.
    positions = rounds = 0
    shape = list(value.shape)
    itemsize = 4 if value.dtype == "float32" else 8
    # Include the final output reshape, including identity reductions. Each
    # pairwise round retains two gathers, their canonical reshapes and where.
    retained = _shape_product(output) * itemsize
    if retained > 256 * 1024 * 1024:
        raise ValueError(f"{name} exceeds the static reduction graph budget")
    for position in reversed(axes):
        while shape[position] > 1:
            shape[position] = (shape[position] + 1) // 2
            positions += 2 * shape[position]
            rounds += 1
            retained += _shape_product(shape) * (5 * itemsize + 1)
            if positions > 65_536 or rounds > 64 or retained > 256 * 1024 * 1024:
                raise ValueError(f"{name} exceeds the static reduction graph budget")
    result = value
    operation = maximum if name == "max" else minimum
    for position in reversed(axes):
        while result.shape[position] > 1:
            extent = result.shape[position]
            left = take(result, tuple(range(0, extent, 2)), axis=position)
            right = take(
                result,
                tuple(builtins.min(i + 1, extent - 1) for i in range(0, extent, 2)),
                axis=position,
            )
            result = operation(left, right)
    return reshape(result, output)


def max(
    x: object, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> VibeArray:
    """Bounded finite generic maximum; ties retain the earliest C-order value."""
    return _reduce_extremum(x, axis=axis, keepdims=keepdims, name="max")


def min(
    x: object, /, *, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
) -> VibeArray:
    """Bounded finite generic minimum with the same tie convention as max."""
    return _reduce_extremum(x, axis=axis, keepdims=keepdims, name="min")


def mean(
    x: object,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> VibeArray:
    """Arithmetic mean with an exact static reduction count."""
    value = _array(x)
    axes = _reduction_axes(axis, value.ndim, "mean")
    count = 1
    for position in axes:
        count *= value.shape[position]
    if count == 0:
        raise ValueError("mean of an empty reduction is unsupported")
    result = sum(value, axis=axes, keepdims=keepdims)
    if count == 1:
        return result
    return _canonical_generic(
        VibeArray(tensor_ir.add(result.node, coefficients=(Fraction(1, count),)))
    )


def _var_plan(
    shape: tuple[int, ...], axes: tuple[int, ...], itemsize: int
) -> tuple[int, int]:
    """Bound the centered two-pass graph before creating any new nodes.

    Eight input-sized slots cover canonicalization, mean broadcast, subtraction
    and squaring (including their possible canonical reshapes). Ten reduced
    slots cover both sums, mean scaling, singleton insertion/removal, final
    scaling and their canonical reshapes. Identity and zero-byte nodes still
    consume slots. Existing graphs and AD retain their backend resource gates.
    """
    reduced = tuple(n for i, n in enumerate(shape) if i not in axes)
    singleton = tuple(1 if i in axes else n for i, n in enumerate(shape))
    checked_shape(reduced)
    checked_shape(singleton)
    retained = (8 * _shape_product(shape) + 10 * _shape_product(reduced)) * itemsize
    nodes = 18
    if retained > 256 * 1024 * 1024:
        raise ValueError("var exceeds the static reduction graph budget")
    return retained, nodes


def var(
    x: object,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: float = 0.0,
    keepdims: bool = False,
) -> VibeArray:
    """Finite generic variance via centered squares and existing first-order AD.

    Empty reductions and nonpositive divisors are outside this finite capture
    subset. Eager IEEE behavior is a separate public contract.
    """
    value = _array(x, "var")
    if value.dtype not in ("float32", "float64"):
        raise TypeError("var requires real float32 or float64 arrays")
    if (
        not _is_generic_array(value)
        or value.node.spec.representation != "general"
        or value.node.spec.symmetries
        or any(index.space.spin is not None for index in value.node.spec.indices)
    ):
        raise TypeError(
            "var supports only generic arrays without scientific annotations"
        )
    if type(keepdims) is not bool:
        raise TypeError("keepdims must be a bool")
    axes = _reduction_axes(axis, value.ndim, "var")
    if type(correction) not in (int, float):
        raise TypeError("var correction must be a static int or float")
    if type(correction) is float and not isfinite(correction):
        raise ValueError("var correction must be finite")
    count = _shape_product(value.shape[i] for i in axes)
    divisor = count - Fraction(correction)
    if count == 0 or divisor <= 0:
        raise ValueError("var requires a nonempty reduction and positive divisor")
    factor = 1 / divisor
    try:
        rounded = float(factor)
        if value.dtype == "float32":
            rounded = unpack("f", pack("f", rounded))[0]
    except (ValueError, OverflowError) as error:
        raise ValueError(
            "var coefficient must be finite and nonzero in input dtype"
        ) from error
    if not isfinite(rounded) or rounded == 0:
        raise ValueError("var coefficient must be finite and nonzero in input dtype")
    _var_plan(value.shape, axes, 4 if value.dtype == "float32" else 8)
    center = subtract(value, mean(value, axis=axes, keepdims=True))
    squares = multiply(center, center)
    total = sum(squares, axis=axes, keepdims=keepdims)
    return _canonical_generic(
        VibeArray(tensor_ir.add(total.node, coefficients=(factor,)))
    )


def _permutation(axes: object, rank: int) -> tuple[int, ...]:
    if not isinstance(axes, tuple) or len(axes) != rank:
        raise ValueError("permute_dims axes must be a full-rank tuple")
    normalized = _normalized_axes(axes, rank, "permute_dims")
    if len(normalized) != rank:
        raise ValueError("permute_dims axes must be a full permutation")
    return normalized


def permute_dims(x: object, axes: tuple[int, ...]) -> VibeArray:
    value = _array(x)
    result = VibeArray(tensor_ir.transpose(value.node, _permutation(axes, value.ndim)))
    return _canonical_generic(result)


def matrix_transpose(x: object) -> VibeArray:
    """Transpose the final two dimensions, preserving leading batch axes."""
    value = _array(x)
    if value.ndim < 2:
        raise ValueError(
            "matrix_transpose requires an array with at least two dimensions"
        )
    axes = list(range(value.ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return permute_dims(value, tuple(axes))


def _matmul_generic(left: VibeArray, right: VibeArray) -> VibeArray:
    if left.ndim == 0 or right.ndim == 0:
        raise ValueError("matmul requires arrays with at least one dimension")
    left_vector = left.ndim == 1
    right_vector = right.ndim == 1
    left_work = reshape(left, (1, left.shape[0])) if left_vector else left
    right_work = reshape(right, (right.shape[0], 1)) if right_vector else right
    if left_work.shape[-1] != right_work.shape[-2]:
        raise ValueError("matmul core dimensions do not agree")
    batch = _broadcast_shape(left_work.shape[:-2], right_work.shape[:-2])
    left_shape = batch + left_work.shape[-2:]
    right_shape = batch + right_work.shape[-2:]
    left_work = _broadcast_generic(left_work, left_shape)
    right_work = _broadcast_generic(right_work, right_shape)
    batch_rank = len(batch)
    if batch_rank + 3 > len(ascii_letters):
        raise ValueError("matmul rank exceeds the bounded symbolic label inventory")
    labels = iter(ascii_letters)
    batch_labels = "".join(next(labels) for _ in range(batch_rank))
    m, k, n = next(labels), next(labels), next(labels)
    equation = f"{batch_labels}{m}{k},{batch_labels}{k}{n}->{batch_labels}{m}{n}"
    result = VibeArray(tensor_ir.einsum(equation, left_work.node, right_work.node))
    if left_vector and right_vector:
        final_shape: tuple[int, ...] = batch
    elif left_vector:
        final_shape = batch + (right_work.shape[-1],)
    elif right_vector:
        final_shape = batch + (left_work.shape[-2],)
    else:
        final_shape = batch + (left_work.shape[-2], right_work.shape[-1])
    if result.shape != final_shape:
        result = reshape(result, final_shape)
    return _canonical_generic(result)


def matmul(x1: object, x2: object) -> VibeArray:
    """Array-API-style matmul for generic arrays; strict rank-2 for scientific IR."""
    left, right = _binary_arrays(x1, x2, "matmul")
    if _is_generic_array(left) and _is_generic_array(right):
        promoted_left, promoted_right = _promote_generic_arrays(left, right)
        return _matmul_generic(promoted_left, promoted_right)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(
            "scientifically annotated matmul currently supports rank-2 arrays only"
        )
    return VibeArray(tensor_ir.einsum("ik,kj->ij", left.node, right.node))


def _contraction_arrays(
    x1: object, x2: object, operation: str
) -> tuple[VibeArray, VibeArray]:
    left, right = _binary_arrays(x1, x2, operation)
    if not (_is_generic_array(left) and _is_generic_array(right)):
        raise TypeError(
            f"{operation} requires generic arrays; scientific domains need explicit einsum"
        )
    left, right = _promote_generic_arrays(left, right)
    return left, right


def _contraction_axes(
    axes: object, left_rank: int, right_rank: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if type(axes) is int:
        if not 0 <= axes <= builtins.min(left_rank, right_rank):
            raise ValueError("tensordot axes count is out of range")
        return tuple(range(left_rank - axes, left_rank)), tuple(range(axes))
    if not isinstance(axes, (tuple, list)) or len(axes) != 2:
        raise TypeError("tensordot axes must be an integer or pair of axis sequences")
    normalized = []
    for selected, rank in zip(axes, (left_rank, right_rank), strict=True):
        if not isinstance(selected, (tuple, list)):
            raise TypeError("tensordot requires static axis sequences")
        values = tuple(_axis(axis, rank, "tensordot") for axis in selected)
        if len(set(values)) != len(values):
            raise ValueError("tensordot axes must be unique")
        normalized.append(values)
    if len(normalized[0]) != len(normalized[1]):
        raise ValueError("tensordot axis sequences must have equal lengths")
    return normalized[0], normalized[1]


def tensordot(x1: object, x2: object, /, *, axes: object = 2) -> VibeArray:
    """Contract static generic real axes, keeping left then right free axes."""
    left, right = _contraction_arrays(x1, x2, "tensordot")
    left_axes, right_axes = _contraction_axes(axes, left.ndim, right.ndim)
    if any(
        left.shape[a] != right.shape[b]
        for a, b in zip(left_axes, right_axes, strict=True)
    ):
        raise ValueError("tensordot contracted extents must match")
    if left.ndim + right.ndim - len(left_axes) > len(ascii_letters):
        raise ValueError("tensordot exceeds the 52-label contraction limit")
    left_labels = list(ascii_letters[: left.ndim])
    right_labels = [""] * right.ndim
    for a, b in zip(left_axes, right_axes, strict=True):
        right_labels[b] = left_labels[a]
    unused = iter(ascii_letters[left.ndim :])
    for axis in range(right.ndim):
        if not right_labels[axis]:
            right_labels[axis] = next(unused)
    output = "".join(
        label for axis, label in enumerate(left_labels) if axis not in left_axes
    )
    output += "".join(
        label for axis, label in enumerate(right_labels) if axis not in right_axes
    )
    return einsum(
        f"{''.join(left_labels)},{''.join(right_labels)}->{output}", left, right
    )


def vecdot(x1: object, x2: object, /, *, axis: object = -1) -> VibeArray:
    """Contract equal vector extents while broadcasting the remaining axes."""
    left, right = _contraction_arrays(x1, x2, "vecdot")
    left_axis = _axis(axis, left.ndim, "vecdot")
    right_axis = _axis(axis, right.ndim, "vecdot")
    if left.shape[left_axis] != right.shape[right_axis]:
        raise ValueError("vecdot contracted extents must match")
    left = moveaxis(left, left_axis, -1)
    right = moveaxis(right, right_axis, -1)
    batch = _broadcast_shape(left.shape[:-1], right.shape[:-1])
    shape = batch + (left.shape[-1],)
    if len(shape) > len(ascii_letters):
        raise ValueError("vecdot exceeds the 52-label contraction limit")
    left, right = _broadcast_generic(left, shape), _broadcast_generic(right, shape)
    labels = ascii_letters[: len(shape)]
    return einsum(f"{labels},{labels}->{labels[:-1]}", left, right)


def einsum(
    equation: str,
    *operands: object,
    coefficient: ExactScalar = 1,
) -> VibeArray:
    """GenerativeQC extension for general contractions absent from the core subset."""
    arrays = tuple(_array(value, "einsum operand") for value in operands)
    if arrays and all(_is_generic_array(value) for value in arrays):
        arrays = _promote_generic_arrays(*arrays)
    result = VibeArray(
        tensor_ir.einsum(
            equation,
            *(value.node for value in arrays),
            coefficient=_exact(coefficient, "einsum coefficient"),
        )
    )
    return _canonical_generic(result)


def _axis(axis: object, rank: int, operation: str) -> int:
    if type(axis) is not int:
        raise TypeError(f"{operation} axis must be an integer")
    normalized = axis + rank if axis < 0 else axis
    if not 0 <= normalized < rank:
        raise ValueError(f"{operation} axis is out of range")
    return normalized


def _expand_index_key(key: object, rank: int) -> tuple[object, ...]:
    items = list(key if isinstance(key, tuple) else (key,))
    ellipses = [position for position, item in enumerate(items) if item is Ellipsis]
    if len(ellipses) > 1:
        raise IndexError("an index can contain at most one ellipsis")
    consumed = builtins.sum(item is not None and item is not Ellipsis for item in items)
    if consumed > rank:
        raise IndexError("too many indices for symbolic VibeArray")
    fill = rank - consumed
    if ellipses:
        position = ellipses[0]
        items[position : position + 1] = [builtins.slice(None)] * fill
    else:
        items.extend(builtins.slice(None) for _ in range(fill))
    return tuple(items)


def _getitem_generic(value: VibeArray, key: object) -> VibeArray:
    items = _expand_index_key(key, value.ndim)
    node = value.node
    source_axis = 0
    output_shape: list[int] = []
    for item in items:
        if item is None:
            output_shape.append(1)
            continue
        extent = node.spec.shape[source_axis]
        if type(item) is int:
            position = item + extent if item < 0 else item
            if not 0 <= position < extent:
                raise IndexError("symbolic VibeArray index is out of range")
            node = tensor_ir.gather(node, source_axis, (position,))
            source_axis += 1
            continue
        if not isinstance(item, builtins.slice):
            raise TypeError(
                "generic symbolic indexing supports integers, slices, None, and ellipsis"
            )
        start, stop, step = item.indices(extent)
        if step == 1:
            # A contiguous slice needs only its bounds, not an index tuple.
            selected_count = builtins.max(0, stop - start)
            if selected_count:
                ranges = tuple(
                    (start, stop) if axis == source_axis else (0, node.spec.shape[axis])
                    for axis in range(len(node.spec.shape))
                )
                node = tensor_ir.slice_tensor(node, ranges)
            else:
                node = tensor_ir.gather(node, source_axis, ())
        else:
            # Non-unit strides require an explicit gather map. Check its extent
            # before TensorIR materializes the map as an immutable tuple.
            positions = range(start, stop, step)
            selected_count = len(positions)
            if selected_count > _MAX_STATIC_GATHER_POSITIONS:
                raise ValueError(
                    "symbolic subscript exceeds the bounded static gather index budget"
                )
            node = tensor_ir.gather(node, source_axis, positions)
        output_shape.append(selected_count)
        source_axis += 1
    return VibeArray(tensor_ir.reshape(node, _generic_indices(tuple(output_shape))))


def _getitem(x: object, key: object) -> VibeArray:
    """Index generic arrays naturally while preserving strict scientific slices."""
    value = _array(x)
    if _is_generic_array(value):
        return _getitem_generic(value, key)
    items = key if isinstance(key, tuple) else (key,)
    if len(items) > value.ndim:
        raise IndexError("too many indices for symbolic VibeArray")
    items = (*items, *(builtins.slice(None) for _ in range(value.ndim - len(items))))
    ranges: list[tuple[int, int]] = []
    for item, extent in zip(items, value.shape, strict=True):
        if not isinstance(item, builtins.slice):
            raise TypeError(
                "scientifically annotated symbolic indexing supports "
                "rank-preserving slices only"
            )
        if item.step is not None and (type(item.step) is not int or item.step != 1):
            raise ValueError("scientific symbolic VibeArray slices require unit step")
        start = 0 if item.start is None else item.start
        stop = extent if item.stop is None else item.stop
        if type(start) is not int or type(stop) is not int:
            raise TypeError("symbolic VibeArray slice bounds must be integers or None")
        if start < 0 or stop < 0:
            raise ValueError("scientific symbolic slice bounds must be nonnegative")
        ranges.append((start, stop))
    return VibeArray(tensor_ir.slice_tensor(value.node, tuple(ranges)))
