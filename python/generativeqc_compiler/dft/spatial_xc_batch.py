"""Bounded optional batching for artifact-bound spatial XC consumers.

The resource planner can always retain the single-tile implementation. Mixed
matrix arithmetic is explicit; scalar XC, integration and scatter remain FP64.
"""

from __future__ import annotations

import ctypes as ct
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.native_call import checked_native_call
from generativeqc_compiler.common.resources import (
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    byte_product,
)

from .ao import pointer


@dataclass(frozen=True)
class SpatialXcOptions:
    """Execution choices independent of the immutable quadrature and AO maps.

    ``batch_tiles=1`` disables the queue. The default queue holds at most eight
    tiles and 32 MiB; a tighter resource budget or allocation failure selects
    single-tile execution. ``fp32`` lowers only compact matrix contractions,
    retaining double point evaluation, storage, totals and global scatter.
    """

    batch_tiles: int = 8
    max_batch_bytes: int = 32 << 20
    matrix_precision: Literal["fp64", "fp32"] = "fp64"

    def __post_init__(self) -> None:
        if type(self.batch_tiles) is not int or not 1 <= self.batch_tiles <= 32:
            raise ValueError("spatial XC batch_tiles must be in [1, 32]")
        if type(self.max_batch_bytes) is not int or self.max_batch_bytes < 0:
            raise ValueError("spatial XC max_batch_bytes must be nonnegative")
        if self.matrix_precision not in ("fp64", "fp32"):
            raise ValueError("spatial XC matrix_precision must be fp64 or fp32")


def batch_request(owner: Any, options: SpatialXcOptions) -> ResourceRequest:
    """Charge every retained numeric panel, map, descriptor and error word."""
    plan = owner.plan
    active, points = plan.active_ao_capacity or plan.nao, plan.tile_points
    jets = 1 if plan.order == 0 else 4
    candidates = []
    columns = (max(active, 16) + 15) // 16
    admitted = columns * (columns + 1) // 2 <= 65535 and points <= 16 * 65535
    # Tiny tiles keep the already qualified scalar/individual matrix path.
    if admitted and active >= 16 and points >= 16:
        for tiles in range(options.batch_tiles, 1, -1):
            elements = tiles * (
                (jets + 8) * points * active + 28 * points + 2 * active**2
            )
            device = (
                byte_product(8, elements + 6 + tiles * active)
                + 56 * tiles
                + 4 * (tiles + 1)
            )
            if device > options.max_batch_bytes:
                continue
            candidates.append(
                ResourceCandidate(
                    f"batch_{tiles}",
                    "tiled",
                    (
                        ResourceEstimate(
                            "spatial_xc_batch",
                            device,
                            f"device:{owner.device_id}",
                            0,
                            1,
                            kind="persistent",
                        ),
                        ResourceEstimate(
                            "spatial_xc_descriptors",
                            56 * tiles,
                            "pageable",
                            0,
                            1,
                            kind="persistent",
                        ),
                    ),
                    relative_cost=32 - tiles,
                )
            )
    candidates.append(
        ResourceCandidate(
            "single",
            "streamed",
            (
                ResourceEstimate(
                    "spatial_xc_batch", 0, f"device:{owner.device_id}", 0, 1
                ),
            ),
            relative_cost=32,
        )
    )
    return ResourceRequest(
        "spatial_xc_batch",
        ResourceIdentity(
            "dft",
            "spatial_xc_batch",
            "cuda",
            options.matrix_precision,
            json.dumps({"active": active, "points": points, "jets": jets}),
            ("energy", "potential"),
            "bounded_tiles",
        ),
        tuple(candidates),
        ("Python/C++ object headers and allocator rounding",),
    )


class SpatialXcBatch:
    """One prepared consumer's optional queue, owned natively by its grid.

    Retaining the grid prevents a Python lifetime inversion. Native grid close
    also destroys all queues, so closing this wrapper afterwards is harmless.
    """

    def __init__(
        self, owner: Any, tiles: int, device_bytes: int, options: SpatialXcOptions
    ) -> None:
        self.owner, self.tiles, self.device_bytes = owner, tiles, device_bytes
        self.handle = ct.c_void_p()
        with owner._lock:
            owner._check_open()
            checked_native_call(
                owner._library.grid_cuda_xc_batch_create_v1,
                owner._handle,
                tiles,
                int(options.matrix_precision == "fp32"),
                device_bytes,
                ct.byref(self.handle),
                status_error_types={7: MemoryError},
            )

    def enqueue(
        self,
        lease: Any,
        weights: Any,
        capability: Any,
        *,
        restricted: bool,
        reset: bool,
        download: bool,
    ) -> tuple[Any, Any]:
        """Copy a live generation and publish only after the final flush."""
        if lease._owner is not self.owner or not self.handle or not self.owner._handle:
            raise ValueError("stale spatial XC batch owner")
        view = lease.view
        if self.owner.xc_capability(capability) is None:
            raise ValueError("incompatible spatial XC batch artifact")
        weights = immutable(weights, shape=(view.npoint,))
        integrals = np.empty(3)
        potential = np.empty((2, view.nao, view.nao)) if download else None
        self.owner._call(
            "grid_cuda_xc_batch_enqueue_v1",
            self.owner._handle,
            self.handle,
            view.generation,
            capability.evaluator,
            int(restricted),
            int(reset),
            int(download),
            pointer(weights),
            len(weights),
            pointer(integrals),
            None if potential is None else pointer(potential),
        )
        return immutable(integrals), None if potential is None else immutable(potential)

    def close(self) -> None:
        """Release queue storage without closing the borrowed grid."""
        with self.owner._lock:
            if self.handle and self.owner._handle:
                self.owner._call(
                    "grid_cuda_xc_batch_destroy_v1", self.owner._handle, self.handle
                )
            self.handle = ct.c_void_p()

    def __del__(self) -> None:
        # A prepared consumer can be dropped while its borrowed grid survives.
        # Release its own queue then, rather than retaining slots until grid close.
        if hasattr(self, "handle"):
            self.close()
