"""Compose resident semilocal and nonlocal geometry consumers on one stream.

This module owns scheduling only. AO/XC pullbacks and VV10 mathematics remain
in their existing generated/native owners. The final SCF rho/grad-rho are
seeded D2D into the resident VV10 owner, so both geometry consumers share one
bounded AO/grid collocation pass without retaining O(N*AO) jets.
"""

from __future__ import annotations

import typing
from time import perf_counter

import numpy as np


def resident_nonlocal_geometry(
    *,
    grid: typing.Any,
    sources: typing.Any,
    nonlocal_sources: typing.Any,
    nonlocal_owner: typing.Any,
    state: typing.Any,
    raw_weights: typing.Any,
    tile_points: int,
    ao_count: int,
    functional: int,
    ingredients: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, typing.Any]]:
    """Seed pairs from the live SCF state and consume each grid lease once.

    The caller has reset both stationary accumulators and enqueued the nuclear
    term only on the semilocal source. The first grid lease supplies the shared
    CUDA stream used to D2D-seed the full-grid resident VV10 owner from the exact
    current KS snapshot. Pair seeds are then available before that same lease
    reaches the nonlocal geometry consumer. No AO jet survives the lease.
    """
    if type(tile_points) is not int or tile_points <= 0:
        raise ValueError("resident nonlocal tile_points must be a positive integer")
    if not callable(getattr(nonlocal_sources, "geometry_external_device", None)):
        raise TypeError("nonlocal stationary owner lacks the resident seed consumer")
    if not callable(getattr(nonlocal_owner, "seed_from_snapshot", None)):
        raise TypeError("resident nonlocal owner lacks the final-state feature handoff")
    points = state.grid.points
    count = len(points)
    if count == 0 or nonlocal_owner.point_count != count:
        raise ValueError("resident nonlocal owner/grid point count differs")
    if len(raw_weights) != count:
        raise ValueError("resident nonlocal raw quadrature size differs")
    if type(ao_count) is not int or ao_count <= 0:
        raise ValueError("resident nonlocal AO count must be a positive integer")

    began = perf_counter()
    diagnostic = nonlocal_owner.diagnostic()
    if diagnostic.executed:
        nonlocal_owner.reset()
    elif diagnostic.collected_points:
        raise ValueError(
            "resident nonlocal owner contains an incomplete previous collection"
        )
    reset_seconds = perf_counter() - began

    pass_began = perf_counter()
    seed_seconds = 0.0
    pair_seconds = 0.0
    seeds = None
    for begin in range(0, count, tile_points):
        end = min(begin + tile_points, count)
        with grid.feature_task(
            points[begin:end],
            None,
            ingredients,
            defer_error_to_consumer=True,
        ) as task:
            owners = np.asarray(state.grid.owners[begin:end], dtype=np.int64)
            weights = state.grid.weights[begin:end]
            raw = raw_weights[begin:end]
            sources.geometry(
                task,
                owners,
                weights,
                raw,
                functional=functional,
            )
            if begin == 0:
                began = perf_counter()
                nonlocal_owner.seed_from_snapshot(state._source, task)
                seed_seconds = perf_counter() - began
                began = perf_counter()
                seeds = nonlocal_owner.execute()
                pair_seconds = perf_counter() - began
                if (
                    not seeds.pointer
                    or seeds.stride != count
                    or not seeds.stream
                    or seeds.generation <= 0
                ):
                    raise ValueError(
                        "resident nonlocal producer returned an invalid seed lease"
                    )
            if seeds is None:
                raise RuntimeError("resident nonlocal pair seed was not initialized")
            if task.view.stream != seeds.stream:
                raise ValueError("resident nonlocal seed and geometry streams differ")
            # Inactive MolecularV1 rows have all six seeds zeroed by the native
            # producer. Original weights are therefore correct without a host
            # active mask; no density or seed values cross this boundary.
            nonlocal_sources.geometry_external_device(
                task,
                owners,
                weights,
                raw,
                seeds.pointer,
                seeds.stride,
                begin,
            )

    local = sources.finish()
    nonlocal_parts = nonlocal_sources.finish()
    components = {
        name: local[name] for name in ("xc_ao", "xc_grid", "xc_weight", "nuclear")
    }
    for suffix in ("ao", "grid", "weight"):
        components["nonlocal_" + suffix] = nonlocal_parts["xc_" + suffix]

    # These host-wall intervals are additive: the enclosing pass includes the
    # separately reported seed and pair submission intervals. Keep asynchronous
    # pair execution/drain in the shared tail, not in the enqueue measurements.
    geometry_and_drain_seconds = (
        perf_counter() - pass_began - seed_seconds - pair_seconds
    )
    seconds = {
        "nonlocal_reset": reset_seconds,
        "resident_feature_seed_enqueue": seed_seconds,
        "vv10_pair_enqueue": pair_seconds,
        "single_pass_geometry_and_pair_drain": geometry_and_drain_seconds,
    }
    work = {
        "nonlocal_execution": "resident-full-grid-device-seeds",
        "nonlocal_feature_source": "exact-final-scf-device-binding",
        "nonlocal_feature_d2d_bytes": 4 * count * 8,
        "nonlocal_feature_d2h_bytes": 0,
        "nonlocal_seed_h2d_bytes": 0,
        "nonlocal_dense_pair_capacity": count * count,
        "nonlocal_seed_generation": seeds.generation,
        "nonlocal_active_count_scope": "device-only; not measured by host scheduler",
        "ao_collocation_point_visits": count,
        "geometry_point_visits": 2 * count,
    }
    return components, seconds, work
