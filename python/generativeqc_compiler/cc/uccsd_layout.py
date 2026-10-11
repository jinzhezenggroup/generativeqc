"""Exact real-FP64 UCCSD amplitude storage; no equations or solver policy.

Logical axes are ia for singles and ijab for doubles. Same-spin doubles
store i<j, a<b with independent occupied/virtual exchange signs; mixed-spin
axes are alpha,beta,alpha,beta and have no within-block exchange symmetry.
See docs/developer/uccsd_layout.md for operator and metric conventions.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from fractions import Fraction
from math import prod

import numpy as np

from generativeqc_compiler.common.layout import DenseLayout
from generativeqc_compiler.common.resources import byte_product, checked_bytes
from generativeqc_compiler.tensor.types import Index, IndexSpace, Symmetry, TensorSpec

BLOCK_NAMES = ("t1a", "t1b", "t2aa", "t2ab", "t2bb")


def _pairs(n: int) -> int:
    return n * (n - 1) // 2


def _pair_slot(i: int, j: int, n: int) -> int:
    return i * (2 * n - i - 1) // 2 + j - i - 1


def _pair_indices(slot: int, n: int) -> tuple[int, int]:
    # Integer binary search keeps the inverse exact beyond FP64 integer precision.
    low, high = 0, n - 1
    while low + 1 < high:
        middle = (low + high) // 2
        if middle * (2 * n - middle - 1) // 2 <= slot:
            low = middle
        else:
            high = middle
    return low, low + 1 + slot - low * (2 * n - low - 1) // 2


def _fp64(value: typing.Any, shape: tuple[int, ...]) -> np.ndarray:
    # Subclasses can override finiteness/equality (notably masked arrays).
    if type(value) is not np.ndarray or value.dtype != np.dtype("float64"):
        raise ValueError("amplitudes must be base NumPy arrays of native real-FP64")
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError("amplitude shape mismatch or nonfinite value")
    return value


@dataclass(frozen=True)
class UCCSDBlock:
    """One of five spin blocks, with derived logical and physical contracts.

    Populations are (oa,ob,va,vb). All metadata is derived from these checked
    dimensions and the block name; there is no independently editable symmetry.
    UCCSDLayout.block_slices addresses the concatenated packed vector.
    """

    name: str
    populations: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        """Validate identifiers and full/packed FP64 sizes before conversion."""
        if not isinstance(self.name, str) or self.name not in BLOCK_NAMES:
            raise ValueError("unknown UCCSD amplitude block")
        populations = tuple(self.populations)
        if len(populations) != 4:
            raise ValueError("UCCSD requires oa, ob, va, vb")
        for n in populations:
            checked_bytes(n, "UCCSD population")
        object.__setattr__(self, "populations", populations)
        self.full_layout.storage_bytes(8)
        self.packed_layout.storage_bytes(8)

    @property
    def same_spin(self) -> bool:
        """Whether independent occupied and virtual antisymmetries apply."""
        return self.name in ("t2aa", "t2bb")

    @property
    def spec(self) -> TensorSpec:
        """Return spin-labelled logical axes and exact TensorIR symmetries."""
        oa, ob, va, vb = self.populations
        spaces = {
            "oa": IndexSpace("oa", "occupied", oa, "alpha"),
            "ob": IndexSpace("ob", "occupied", ob, "beta"),
            "va": IndexSpace("va", "virtual", va, "alpha"),
            "vb": IndexSpace("vb", "virtual", vb, "beta"),
        }
        domains = {
            "t1a": ("oa", "va"),
            "t1b": ("ob", "vb"),
            "t2aa": ("oa", "oa", "va", "va"),
            "t2ab": ("oa", "ob", "va", "vb"),
            "t2bb": ("ob", "ob", "vb", "vb"),
        }[self.name]
        axes = "ia" if self.name.startswith("t1") else "ijab"
        return TensorSpec(
            tuple(Index(axis, spaces[space]) for axis, space in zip(axes, domains)),
            symmetries=(Symmetry((1, 0, 2, 3), -1), Symmetry((0, 1, 3, 2), -1))
            if self.same_spin
            else (),
            representation="spin_orbital",
            role="input",
        )

    @property
    def full_layout(self) -> DenseLayout:
        """Return the C-order dense logical layout (including empty domains)."""
        return DenseLayout(self.spec.shape)

    @property
    def packed_layout(self) -> DenseLayout:
        """Return pair-by-pair same-spin storage or unchanged dense axes."""
        shape = self.spec.shape
        return (
            DenseLayout((_pairs(shape[0]), _pairs(shape[2])))
            if self.same_spin
            else DenseLayout(shape)
        )

    @property
    def full_metric_weight(self) -> int:
        """Multiplicity for the Frobenius metric of the five full blocks."""
        return 4 if self.same_spin else 1

    @property
    def operator_prefactor(self) -> Fraction:
        """Coefficient when summing this full block's ordered excitation terms."""
        return Fraction(1, 4) if self.same_spin else Fraction(1)

    def coordinate(self, indices: tuple[int, ...]) -> tuple[int | None, int]:
        """Map logical indices to a local packed slot and permutation sign.

        Repeated same-spin indices return (None,0). Mixed-spin axes are never
        sorted, including when alpha and beta populations happen to coincide.
        """
        indices = tuple(indices)
        shape = self.spec.shape
        if len(indices) != len(shape) or any(
            type(i) is not int or not 0 <= i < n for i, n in zip(indices, shape)
        ):
            raise ValueError("UCCSD coordinate out of range")
        if not self.same_spin:
            slot = 0
            for i, n in zip(indices, shape):
                slot = slot * n + i
            return slot, 1
        i, j, a, b = indices
        if i == j or a == b:
            return None, 0
        sign = (1 if i < j else -1) * (1 if a < b else -1)
        i, j = min(i, j), max(i, j)
        a, b = min(a, b), max(a, b)
        return _pair_slot(i, j, shape[0]) * _pairs(shape[2]) + _pair_slot(
            a, b, shape[2]
        ), sign

    def representative(self, slot: int) -> tuple[int, ...]:
        """Invert a checked local slot to its canonical logical representative."""
        count = self.packed_layout.storage_elements
        if type(slot) is not int or not 0 <= slot < count:
            raise ValueError("UCCSD packed slot out of range")
        shape = self.spec.shape
        if self.same_spin:
            occupied, virtual = divmod(slot, _pairs(shape[2]))
            return (
                *_pair_indices(occupied, shape[0]),
                *_pair_indices(virtual, shape[2]),
            )
        result = []
        for n in reversed(shape):
            slot, index = divmod(slot, n)
            result.append(index)
        return tuple(reversed(result))

    def pack(self, full: np.ndarray) -> np.ndarray:
        """Copy exact admitted amplitudes, rejecting rather than projecting."""
        full = _fp64(full, self.spec.shape)
        if not self.same_spin:
            return full.ravel(order="C").copy()
        if not np.array_equal(full, -full.swapaxes(0, 1)) or not np.array_equal(
            full, -full.swapaxes(2, 3)
        ):
            raise ValueError("same-spin amplitudes must be exactly antisymmetric")
        # Antisymmetry implies diagonal zero over the real numbers; check it
        # explicitly so that this invariant is not tied to a future tolerance.
        if np.any(np.diagonal(full, axis1=0, axis2=1) != 0) or np.any(
            np.diagonal(full, axis1=2, axis2=3) != 0
        ):
            raise ValueError("repeated same-spin indices must be zero")
        return np.array(
            [
                full[self.representative(slot)]
                for slot in range(self.packed_layout.storage_elements)
            ],
            dtype=np.float64,
        )

    def unpack(self, packed: np.ndarray) -> np.ndarray:
        """Expand raw representative values with exact signs and zero diagonals."""
        packed = _fp64(packed, (self.packed_layout.storage_elements,))
        if not self.same_spin:
            return packed.reshape(self.spec.shape, order="C").copy()
        full = np.zeros(self.spec.shape, dtype=np.float64)
        for slot, value in enumerate(packed):
            i, j, a, b = self.representative(slot)
            full[i, j, a, b] = full[j, i, b, a] = value
            full[j, i, a, b] = full[i, j, b, a] = -value
        return full


@dataclass(frozen=True)
class UCCSDLayout:
    """Five independently sized real-FP64 amplitude blocks and packed vector.

    This CPU conversion contract accepts zero/empty and one-spin populations.
    It makes no statement about reference admission or UCCSD solver support.
    """

    oa: int
    ob: int
    va: int
    vb: int

    def __post_init__(self) -> None:
        """Validate every block and the total dense/packed int64 byte counts."""
        blocks = self.blocks
        byte_product(sum(block.packed_layout.storage_elements for block in blocks), 8)
        byte_product(sum(prod(block.spec.shape) for block in blocks), 8)

    @property
    def blocks(self) -> tuple[UCCSDBlock, ...]:
        """Return descriptors in stable t1a,t1b,t2aa,t2ab,t2bb order."""
        return tuple(
            UCCSDBlock(name, (self.oa, self.ob, self.va, self.vb))
            for name in BLOCK_NAMES
        )

    @property
    def packed_size(self) -> int:
        """Total number of unscaled real-FP64 representatives."""
        return sum(block.packed_layout.storage_elements for block in self.blocks)

    @property
    def block_slices(self) -> dict[str, slice]:
        """Offsets in the C-order concatenated packed vector, without allocation."""
        offset, result = 0, {}
        for block in self.blocks:
            end = offset + block.packed_layout.storage_elements
            result[block.name] = slice(offset, end)
            offset = end
        return result

    def pack(self, full: typing.Mapping[str, np.ndarray]) -> np.ndarray:
        """Validate all five named full blocks and return a fresh packed vector."""
        if set(full) != set(BLOCK_NAMES):
            raise ValueError(
                "exactly the five named UCCSD amplitude blocks are required"
            )
        return np.concatenate([block.pack(full[block.name]) for block in self.blocks])

    def unpack(self, packed: np.ndarray) -> dict[str, np.ndarray]:
        """Validate the packed vector and return fresh full spin-block arrays."""
        packed = _fp64(packed, (self.packed_size,))
        slices = self.block_slices
        return {
            block.name: block.unpack(packed[slices[block.name]])
            for block in self.blocks
        }
