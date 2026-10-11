"""CPU Array API linalg extension; no implicit TensorIR lowering."""

from __future__ import annotations

import typing


def _dispatch(name: str, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Resolve the public dispatcher lazily to avoid a namespace import cycle."""
    from .array_api import _dispatch as dispatch

    return dispatch(name, *args, **kwargs)


def cholesky(x: typing.Any, /, *, upper: typing.Any = False) -> typing.Any:
    """Execute standard CPU linalg.cholesky, returning protocol-aware arrays."""
    return _dispatch("linalg.cholesky", x, upper=upper)


def cross(x1: typing.Any, x2: typing.Any, /, *, axis: typing.Any = -1) -> typing.Any:
    """Execute standard CPU linalg.cross, returning protocol-aware arrays."""
    return _dispatch("linalg.cross", x1, x2, axis=axis)


def det(x: typing.Any, /) -> typing.Any:
    """Execute standard CPU linalg.det, returning protocol-aware arrays."""
    return _dispatch("linalg.det", x)


def diagonal(x: typing.Any, /, *, offset: typing.Any = 0) -> typing.Any:
    """Execute standard CPU linalg.diagonal, returning protocol-aware arrays."""
    return _dispatch("linalg.diagonal", x, offset=offset)


def eigh(x: typing.Any, /) -> typing.Any:
    """Execute standard CPU linalg.eigh, returning protocol-aware arrays."""
    return _dispatch("linalg.eigh", x)


def eigvalsh(x: typing.Any, /) -> typing.Any:
    """Execute standard CPU linalg.eigvalsh, returning protocol-aware arrays."""
    return _dispatch("linalg.eigvalsh", x)


def eig(x: typing.Any, /) -> typing.Any:
    """Execute standard CPU linalg.eig, returning protocol-aware arrays."""
    return _dispatch("linalg.eig", x)


def eigvals(x: typing.Any, /) -> typing.Any:
    """Execute standard CPU linalg.eigvals, returning protocol-aware arrays."""
    return _dispatch("linalg.eigvals", x)


def inv(x: typing.Any, /) -> typing.Any:
    """Execute standard CPU linalg.inv, returning protocol-aware arrays."""
    return _dispatch("linalg.inv", x)


def matmul(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard CPU linalg.matmul, returning protocol-aware arrays."""
    return _dispatch("linalg.matmul", x1, x2)


def matrix_norm(
    x: typing.Any, /, *, keepdims: typing.Any = False, ord: typing.Any = "fro"
) -> typing.Any:
    """Execute standard CPU linalg.matrix_norm, returning protocol-aware arrays."""
    return _dispatch("linalg.matrix_norm", x, keepdims=keepdims, ord=ord)


def matrix_power(x: typing.Any, n: typing.Any, /) -> typing.Any:
    """Execute standard CPU linalg.matrix_power, returning protocol-aware arrays."""
    return _dispatch("linalg.matrix_power", x, n)


def matrix_rank(x: typing.Any, /, *, rtol: typing.Any = None) -> typing.Any:
    """Execute standard CPU linalg.matrix_rank, returning protocol-aware arrays."""
    return _dispatch("linalg.matrix_rank", x, rtol=rtol)


def matrix_transpose(x: typing.Any, /) -> typing.Any:
    """Execute standard CPU linalg.matrix_transpose, returning protocol-aware arrays."""
    return _dispatch("linalg.matrix_transpose", x)


def outer(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard CPU linalg.outer, returning protocol-aware arrays."""
    return _dispatch("linalg.outer", x1, x2)


def pinv(x: typing.Any, /, *, rtol: typing.Any = None) -> typing.Any:
    """Execute standard CPU linalg.pinv, returning protocol-aware arrays."""
    return _dispatch("linalg.pinv", x, rtol=rtol)


def qr(x: typing.Any, /, *, mode: typing.Any = "reduced") -> typing.Any:
    """Execute standard CPU linalg.qr, returning protocol-aware arrays."""
    return _dispatch("linalg.qr", x, mode=mode)


def slogdet(x: typing.Any, /) -> typing.Any:
    """Execute standard CPU linalg.slogdet, returning protocol-aware arrays."""
    return _dispatch("linalg.slogdet", x)


def solve(x1: typing.Any, x2: typing.Any, /) -> typing.Any:
    """Execute standard CPU linalg.solve, returning protocol-aware arrays."""
    return _dispatch("linalg.solve", x1, x2)


def svd(x: typing.Any, /, *, full_matrices: typing.Any = True) -> typing.Any:
    """Execute standard CPU linalg.svd, returning protocol-aware arrays."""
    return _dispatch("linalg.svd", x, full_matrices=full_matrices)


def svdvals(x: typing.Any, /) -> typing.Any:
    """Execute standard CPU linalg.svdvals, returning protocol-aware arrays."""
    return _dispatch("linalg.svdvals", x)


def tensordot(x1: typing.Any, x2: typing.Any, /, *, axes: typing.Any = 2) -> typing.Any:
    """Execute standard CPU linalg.tensordot, returning protocol-aware arrays."""
    return _dispatch("linalg.tensordot", x1, x2, axes=axes)


def trace(
    x: typing.Any, /, *, offset: typing.Any = 0, dtype: typing.Any = None
) -> typing.Any:
    """Execute standard CPU linalg.trace, returning protocol-aware arrays."""
    return _dispatch("linalg.trace", x, offset=offset, dtype=dtype)


def vecdot(x1: typing.Any, x2: typing.Any, /, *, axis: typing.Any = -1) -> typing.Any:
    """Execute standard CPU linalg.vecdot, returning protocol-aware arrays."""
    return _dispatch("linalg.vecdot", x1, x2, axis=axis)


def vector_norm(
    x: typing.Any,
    /,
    *,
    axis: typing.Any = None,
    keepdims: typing.Any = False,
    ord: typing.Any = 2,
) -> typing.Any:
    """Execute standard CPU linalg.vector_norm, returning protocol-aware arrays."""
    return _dispatch("linalg.vector_norm", x, axis=axis, keepdims=keepdims, ord=ord)
