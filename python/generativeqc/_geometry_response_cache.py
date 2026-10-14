"""Bounded exact-density retention inside one immutable geometry owner."""

from __future__ import annotations

import typing

import numpy as np


class ExactGeometryResponseCache:
    """Own one density key and its successful density-only force components.

    The prepared caller owns geometry, operator and grid identity and validates
    the live snapshot before lookup and publication. Weighted-density/Pulay
    components must never enter this cache. Equality compares every FP64 bit,
    including signed zero. The allowance also covers NumPy's largest temporary
    boolean array during equality and finiteness checks.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self._storage: np.ndarray | None = None
        self._density_shape: tuple[int, ...] = ()
        self._key_size = 0
        self._parts: dict[str, np.ndarray] = {}
        self.peak_bytes = 0

    @property
    def retained_bytes(self) -> int:
        return 0 if self._storage is None else self._storage.nbytes

    def lookup(
        self, density: np.ndarray, maximum_bytes: int
    ) -> dict[str, np.ndarray] | None:
        """Return immutable component views only for a complete exact key."""
        if self.peak_bytes > maximum_bytes:
            self.clear()
        if (
            self._storage is None
            or density.dtype != np.dtype(np.float64)
            or not density.flags.c_contiguous
            or density.shape != self._density_shape
        ):
            return None
        try:
            equal = np.array_equal(
                density.reshape(-1).view(np.uint64),
                self._storage[: self._key_size].view(np.uint64),
            )
            return self._parts.copy() if equal else None
        except MemoryError:
            self.clear()
            return None

    def store(
        self,
        density: np.ndarray,
        parts: typing.Mapping[str, np.ndarray],
        maximum_bytes: int,
    ) -> bool:
        """Retain finite computed parts, or clear retention on resource refusal.

        Inputs must be independently owned computed arrays, not this cache's
        own views. Clear before allocation so old and new keys never coexist.
        """
        self.clear()
        if (
            density.dtype != np.dtype(np.float64)
            or not density.flags.c_contiguous
            or not density.size
            or not parts
            or any(part.dtype != np.dtype(np.float64) for part in parts.values())
        ):
            return False
        elements = density.size + sum(part.size for part in parts.values())
        peak = 8 * elements + max(density.size, *(part.size for part in parts.values()))
        if peak > maximum_bytes:
            return False
        try:
            if not np.isfinite(density).all() or any(
                not np.isfinite(part).all() for part in parts.values()
            ):
                return False
            storage = np.empty(elements, dtype=np.float64)
            storage[: density.size] = density.reshape(-1)
            views = {}
            offset = density.size
            for name, part in parts.items():
                view = storage[offset : offset + part.size].reshape(part.shape)
                view[...] = part
                view.setflags(write=False)
                views[name] = view
                offset += part.size
            storage.setflags(write=False)
        except MemoryError:
            return False
        self._storage = storage
        self._density_shape = density.shape
        self._key_size = density.size
        self._parts = views
        self.peak_bytes = peak
        return True
