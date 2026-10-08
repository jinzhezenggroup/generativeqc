"""Bounded Array-API-like namespace lowering directly to TensorIR."""

from __future__ import annotations

import builtins
import typing
from fractions import Fraction
from math import copysign, isfinite
from string import ascii_letters

from generativeqc_compiler.tensor import ir as tensor_ir
from generativeqc_compiler.tensor.types import Index, IndexSpace, TensorSpec

from .array import ExactScalar, VibeArray

_GENERIC_SPACE_PREFIX = "_array_extent_"


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
    if dtype not in ("float32", "float64"):
        raise TypeError("symbolic full supports float32 or float64")
    factor = _generic_scalar(fill_value, "full fill value")
    scalar = tensor_ir.constant(factor, TensorSpec(dtype=dtype, role="constant"))
    if not target:
        return VibeArray(scalar)
    return VibeArray(tensor_ir.broadcast(scalar, _generic_indices(target), ()))


def zeros(shape: int | tuple[int, ...], *, dtype: str = "float64") -> VibeArray:
    return full(shape, 0, dtype=dtype)


def ones(shape: int | tuple[int, ...], *, dtype: str = "float64") -> VibeArray:
    return full(shape, 1, dtype=dtype)


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
    return full_like(x, 0, dtype=dtype)


def ones_like(x: object, *, dtype: str | None = None) -> VibeArray:
    return full_like(x, 1, dtype=dtype)


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
    rank = max(len(left), len(right))
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


def _generic_binary(
    left: VibeArray, right: VibeArray
) -> tuple[VibeArray, VibeArray] | None:
    if not (_is_generic_array(left) and _is_generic_array(right)):
        return None
    shape = _broadcast_shape(left.shape, right.shape)
    return _broadcast_generic(left, shape), _broadcast_generic(right, shape)


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


def take(
    x: object,
    indices: tuple[int, ...],
    *,
    axis: int,
) -> VibeArray:
    """Static gather along one axis."""
    value = _array(x)
    if type(axis) is not int:
        raise TypeError("take axis must be an integer")
    if not isinstance(indices, tuple) or any(
        type(index) is not int for index in indices
    ):
        raise TypeError("take indices must be a static tuple of integers")
    result = VibeArray(
        tensor_ir.gather(value.node, _axis(axis, value.ndim, "take"), indices)
    )
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


def sum(
    x: object,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: object = None,
    keepdims: bool = False,
) -> VibeArray:
    """Reduce selected axes; generic arrays may retain singleton reduced axes."""
    value = _array(x)
    if dtype is not None:
        raise ValueError("frontend sum does not insert dtype conversions")
    if type(keepdims) is not bool:
        raise TypeError("keepdims must be a bool")
    axes = _reduction_axes(axis, value.ndim, "sum")
    if keepdims and not _is_generic_array(value):
        raise ValueError(
            "keepdims for scientifically annotated arrays requires explicit "
            "TensorIR index metadata"
        )
    result = VibeArray(tensor_ir.reduce_sum(value.node, axes=axes))
    if keepdims:
        shape = tuple(
            1 if position in axes else extent
            for position, extent in enumerate(value.shape)
        )
        return reshape(result, shape)
    return _canonical_generic(result)


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


def permute_dims(x: object, axes: tuple[int, ...]) -> VibeArray:
    value = _array(x)
    result = VibeArray(tensor_ir.transpose(value.node, axes))
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
        return _matmul_generic(left, right)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError(
            "scientifically annotated matmul currently supports rank-2 arrays only"
        )
    return VibeArray(tensor_ir.einsum("ik,kj->ij", left.node, right.node))


def einsum(
    equation: str,
    *operands: object,
    coefficient: ExactScalar = 1,
) -> VibeArray:
    """GenerativeQC extension for general contractions absent from the core subset."""
    arrays = tuple(_array(value, "einsum operand") for value in operands)
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
        positions = tuple(range(start, stop, step))
        if step == 1 and positions:
            ranges = tuple(
                (start, stop) if axis == source_axis else (0, node.spec.shape[axis])
                for axis in range(len(node.spec.shape))
            )
            node = tensor_ir.slice_tensor(node, ranges)
        else:
            node = tensor_ir.gather(node, source_axis, positions)
        output_shape.append(len(positions))
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
    for item, extent in zip(items, value.shape):
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
