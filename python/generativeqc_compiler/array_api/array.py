"""Symbolic array values that retain TensorIR scientific semantics."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from fractions import Fraction
from math import prod

from generativeqc_compiler.tensor.ir import Node

ExactScalar: typing.TypeAlias = int | str | Fraction


@dataclass(frozen=True, eq=False)
class VibeArray:
    """One symbolic Array-API value backed by an ordinary TensorIR node."""

    node: Node

    def __post_init__(self) -> None:
        """Require a TensorIR Node as the symbolic array value."""
        if not isinstance(self.node, Node):
            raise TypeError("VibeArray requires a TensorIR Node")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the symbolic node's tensor dimensions."""
        return self.node.spec.shape

    @property
    def dtype(self) -> str:
        """Return the symbolic node's tensor dtype name."""
        return self.node.spec.dtype

    @property
    def ndim(self) -> int:
        """Return the number of tensor axes."""
        return len(self.node.spec.indices)

    @property
    def size(self) -> int:
        """Return the product of symbolic array dimensions."""
        return prod(self.shape)

    @property
    def T(self) -> VibeArray:
        """Reverse all axes, matching the standard array transpose attribute."""
        from . import namespace

        return namespace.permute_dims(self, tuple(reversed(range(self.ndim))))

    @property
    def mT(self) -> VibeArray:
        """Transpose the last two axes while preserving leading batch axes."""
        from . import namespace

        return namespace.matrix_transpose(self)

    def __bool__(self) -> bool:
        """Reject use of symbolic array values in Python control flow."""
        raise TypeError("symbolic VibeArray values cannot drive Python control flow")

    def __eq__(self, other: object) -> bool:
        """Reject symbolic equality comparisons."""
        raise TypeError("symbolic VibeArray comparisons are not supported")

    def __ne__(self, other: object) -> bool:
        """Reject symbolic inequality comparisons."""
        raise TypeError("symbolic VibeArray comparisons are not supported")

    def __add__(self, other: object) -> VibeArray:
        """Build symbolic addition with this array as the left operand."""
        from . import namespace

        return namespace.add(self, other)

    def __radd__(self, other: object) -> VibeArray:
        """Build symbolic addition with this array as the right operand."""
        from . import namespace

        return namespace.add(other, self)

    def __sub__(self, other: object) -> VibeArray:
        """Build symbolic subtraction with this array as the left operand."""
        from . import namespace

        return namespace.subtract(self, other)

    def __rsub__(self, other: object) -> VibeArray:
        """Build symbolic subtraction with this array as the right operand."""
        from . import namespace

        return namespace.subtract(other, self)

    def __mul__(self, other: object) -> VibeArray:
        """Build symbolic multiplication with this array as the left operand."""
        from . import namespace

        return namespace.multiply(self, other)

    def __rmul__(self, other: object) -> VibeArray:
        """Build symbolic multiplication with this array as the right operand."""
        from . import namespace

        return namespace.multiply(other, self)

    def __truediv__(self, other: object) -> VibeArray:
        """Build symbolic division with this array as the left operand."""
        from . import namespace

        return namespace.divide(self, other)

    def __rtruediv__(self, other: object) -> VibeArray:
        """Build symbolic division with this array as the right operand."""
        from . import namespace

        return namespace.divide(other, self)

    def __matmul__(self, other: object) -> VibeArray:
        """Build symbolic matrix multiplication with this array as the left operand."""
        from . import namespace

        return namespace.matmul(self, other)

    def __rmatmul__(self, other: object) -> VibeArray:
        """Build symbolic matrix multiplication with this array as the right operand."""
        from . import namespace

        return namespace.matmul(other, self)

    def __pow__(self, exponent: object) -> VibeArray:
        """Build a symbolic power through the Array API namespace."""
        from . import namespace

        return namespace.pow(self, exponent)

    def __neg__(self) -> VibeArray:
        """Build the symbolic elementwise negative through the Array API namespace."""
        from . import namespace

        return namespace.negative(self)

    def __pos__(self) -> VibeArray:
        """Return this symbolic array unchanged for unary plus."""
        return self

    def __getitem__(self, key: object) -> VibeArray:
        """Build symbolic indexing through the Array API namespace."""
        from . import namespace

        return namespace._getitem(self, key)
