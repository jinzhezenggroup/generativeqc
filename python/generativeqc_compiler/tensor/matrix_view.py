"""Prove direct matrix views of canonical affine contractions without packing.

Provider adapters reuse this physical proof without replacing einsum modes,
scientific identity or admitted precision. Kernel/toolkit legality and algorithm
choice remain responsibilities of each provider.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod
from typing import TYPE_CHECKING

from generativeqc_compiler.common.provenance import canonical_hash

if TYPE_CHECKING:
    from generativeqc_compiler.common.lowering_contract import OperandLayout
    from generativeqc_compiler.common.lowering_provider import LoweringRequest


@dataclass(frozen=True)
class MatrixLayout:
    """A logical matrix with native row/column storage; strides are elements."""

    rows: int
    columns: int
    order: str
    leading_dimension: int
    batch_stride: int


@dataclass(frozen=True)
class MatrixContraction:
    """Execution recipe retaining the canonical request and physical layouts.

    A and B have logical shapes M,K and K,N; D has M,N. Row/column orders
    describe the existing memory directly, including a transposed result. There
    is no generated packing, output scatter, broadcast or epilogue in this offer.
    """

    request_identity: str
    batches: int
    layouts: tuple[MatrixLayout, ...]

    @property
    def identity(self) -> str:
        return canonical_hash(asdict(self))


def _collapsed_axis(view: OperandLayout, modes: tuple[int, ...]) -> tuple[int, int]:
    """Prove lexicographic flattening preserves every nonunit axis address.

    Groups use the same semantic mode order in every operand. Equal dimension
    products alone would incorrectly admit swapped reduction axes. Unit axes
    have no observable stride and do not constrain the physical matrix cut.
    """
    assert view.strides is not None
    extent, stride = 1, None
    for mode in reversed(modes):
        axis = view.modes.index(mode)
        size, physical = view.shape[axis], view.strides[axis]
        if size != 1:
            if stride is None:
                stride = physical
            if physical != extent * stride:
                raise ValueError("matrix view mode group is not physically contiguous")
        extent *= size
    return extent, 1 if stride is None else stride


def _matrix_layout(
    view: OperandLayout,
    row: tuple[int, ...],
    column: tuple[int, ...],
    batch: tuple[int, ...],
    batches: int,
) -> MatrixLayout:
    """Prove the view is addressable without copying or overlapping batches."""
    if view.strides is None or any(s <= 0 or s > (1 << 63) - 1 for s in view.strides):
        raise ValueError("matrix view requires materialized positive matrix strides")
    rows, rs = _collapsed_axis(view, row)
    columns, cs = _collapsed_axis(view, column)
    # A unit axis has no observable stride. Resolve it deterministically without
    # rejecting equivalent dense or padded views emitted by the storage owner.
    if (columns == 1 or cs == 1) and (rows == 1 or rs >= columns):
        order, ld = "row", rs if rows > 1 else columns
    elif (rows == 1 or rs == 1) and (columns == 1 or cs >= rows):
        order, ld = "column", cs if columns > 1 else rows
    else:
        raise ValueError("matrix view requires a native row/column matrix layout")
    stride = _collapsed_axis(view, batch)[1] if batch else 0
    span = (rows - 1) * rs + (columns - 1) * cs + 1
    if batches > 1 and stride < span:
        raise ValueError("matrix view adapter requires nonoverlapping matrix batches")
    if max(rows, columns, ld, batches) > (1 << 31) - 1:
        raise ValueError(
            "matrix view adapter matrix dimensions exceed its native bound"
        )
    return MatrixLayout(rows, columns, order, ld, stride)


def matrix_contraction(request: LoweringRequest) -> MatrixContraction:
    """Recognize affine matmul while retaining the original einsum identity.

    M/N/K and batch may each group existing modes only when their ordered
    strides prove a matrix view without packing. One-sided reductions,
    diagonals, broadcast and scientific symmetry remain explicit rejections.
    """
    if request.backend != "cuda" or request.operation != "einsum":
        raise ValueError("matrix view requires the canonical CUDA einsum request")
    if request.scientific_identity is None or len(request.operands) != 3:
        raise ValueError(
            "matrix view requires two inputs, one output and scientific identity"
        )
    a, b, c = request.operands
    if (a.access, b.access, c.access) != ("read", "read", "write"):
        raise ValueError("matrix view adapter requires two reads and a fresh output")
    if c.alias_group is not None and c.alias_group in (a.alias_group, b.alias_group):
        raise ValueError("matrix view adapter cannot overwrite a borrowed input")
    sizes: dict[int, int] = {}
    for view in request.operands:
        if len(view.modes) > 8:
            raise ValueError(
                "matrix view operand exceeds the native rank bound of eight"
            )
        if view.triangle != "full" or not view.shape or not all(view.shape):
            raise ValueError("matrix view requires positive full matrix operands")
        if len(set(view.modes)) != len(view.modes):
            raise ValueError(
                "matrix view adapter does not implement repeated-mode diagonals"
            )
        for mode, size in zip(view.modes, view.shape, strict=True):
            if sizes.setdefault(mode, size) != size:
                raise ValueError("matrix view semantic mode extents disagree")
    am, bm, cm = (set(view.modes) for view in request.operands)
    batch, m, n, k = am & bm & cm, (am & cm) - bm, (bm & cm) - am, (am & bm) - cm
    if (
        not m
        or not n
        or not k
        or am != batch | m | k
        or bm != batch | k | n
        or cm != batch | m | n
    ):
        raise ValueError(
            "matrix view adapter requires M/N/K groups and shared batch modes"
        )
    mi = tuple(mode for mode in c.modes if mode in m)
    ni = tuple(mode for mode in c.modes if mode in n)
    ki = tuple(mode for mode in a.modes if mode in k)
    bi = tuple(mode for mode in c.modes if mode in batch)
    batches = prod(sizes[mode] for mode in bi)
    return MatrixContraction(
        request.identity,
        batches,
        tuple(
            _matrix_layout(view, row, column, bi, batches)
            for view, row, column in ((a, mi, ki), (b, ki, ni), (c, mi, ni))
        ),
    )
