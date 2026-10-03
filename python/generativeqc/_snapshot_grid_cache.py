"""Bounded reuse of exact immutable grids decoded from native KS snapshots."""

from __future__ import annotations

import sys

import numpy as np
from generativeqc_compiler.dft.grid import ExplicitGrid

SNAPSHOT_GRID_CACHE_BYTES = 128 << 20


class SnapshotGridCache:
    """Retain one grid; current snapshot contents remain the authority.

    Every lookup compares all point, weight, and owner values before reuse.
    Native state leases, density validation, and geometry admission still run
    independently. A miss replaces the previous entry, so ragged batches and
    moving geometries cannot accumulate retained grids. The byte bound includes
    the owned FP64 buffers and Python owner storage; over-budget grids use the
    ordinary uncached construction path.
    """

    def __init__(self, *, max_bytes: int = SNAPSHOT_GRID_CACHE_BYTES) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("snapshot grid cache budget must be nonnegative")
        self.max_bytes = max_bytes
        self.retained_bytes = 0
        self.grid: ExplicitGrid | None = None
        self.owner: int | None = None

    def clear(self) -> None:
        """Release the previous entry before allocating a replacement grid."""
        self.grid = None
        self.owner = None
        self.retained_bytes = 0

    def resolve(
        self,
        points: np.ndarray,
        weights: np.ndarray,
        owners: np.ndarray,
        *,
        owner: int,
    ) -> tuple[ExplicitGrid, bool]:
        """Return an exactly matching grid and whether serialization was reused."""
        if (
            self.grid is not None
            and self.owner == owner
            and np.array_equal(
                self.grid.points.view(np.uint64),
                np.asarray(points, dtype=np.float64).view(np.uint64),
            )
            and np.array_equal(
                self.grid.weights.view(np.uint64),
                np.asarray(weights, dtype=np.float64).view(np.uint64),
            )
            and np.array_equal(self.grid.owners, owners)
        ):
            return self.grid, True
        self.clear()
        grid = ExplicitGrid(
            points,
            weights,
            tuple(map(int, owners)),
            {"source": "native-ks-snapshot-v1", "owner": owner},
        )
        retained_bytes = (
            sys.getsizeof(grid)
            + sys.getsizeof(grid.points)
            + grid.points.nbytes
            + sys.getsizeof(grid.weights)
            + grid.weights.nbytes
            + sys.getsizeof(grid.owners)
            + sum(
                sys.getsizeof(value)
                for value in {id(value): value for value in grid.owners}.values()
            )
            + sys.getsizeof(grid.provenance)
            + sys.getsizeof(grid._provenance_json)
            + sys.getsizeof(grid.identity)
            + 4096
        )
        if retained_bytes <= self.max_bytes:
            self.grid = grid
            self.owner = owner
            self.retained_bytes = retained_bytes
        return grid, False
