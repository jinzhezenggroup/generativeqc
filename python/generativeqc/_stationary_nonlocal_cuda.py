"""Compose resident semilocal and nonlocal geometry consumers on one stream.

This module owns scheduling only. AO/XC pullbacks and VV10 mathematics remain
in their existing generated/native owners. No feature or seed array is copied
to the host. The bounded fallback collocates twice because nonlocal seeds
require the complete density grid before any tile can consume them.
"""

from __future__ import annotations

import typing
from time import perf_counter

import numpy as np


def resident_nonlocal_geometry(
    *,
    grid: typing.Any,
    sources: typing.Any,
    nonlocal_owner: typing.Any,
    state: typing.Any,
    raw_weights: typing.Any,
    tile_points: int,
    functional: int,
    ingredients: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, typing.Any]]:
    """Collect once, evaluate pairs, and consume device seeds transactionally.

    The caller has already reset ``sources`` and enqueued its nuclear term.
    All borrowed task/seed storage remains live through ``sources.finish()``.
    The caller must close/discard both owners on failure before attempting a
    replay. Returned component dictionaries contain no partial publication.
    """
    if type(tile_points) is not int or tile_points <= 0:
        raise ValueError("resident nonlocal tile_points must be a positive integer")
    if not callable(getattr(sources, "geometry_external_device", None)):
        raise TypeError("stationary owner lacks the resident nonlocal seed consumer")
    points = state.grid.points
    count = len(points)
    if count == 0 or nonlocal_owner.point_count != count:
        raise ValueError("resident nonlocal owner/grid point count differs")
    if len(raw_weights) != count:
        raise ValueError("resident nonlocal raw quadrature size differs")
    began = perf_counter()
    diagnostic = nonlocal_owner.diagnostic()
    if diagnostic.executed:
        nonlocal_owner.reset()
    elif diagnostic.collected_points:
        raise ValueError(
            "resident nonlocal owner contains an incomplete previous collection"
        )
    reset_seconds = perf_counter() - began
    began = perf_counter()
    for begin in range(0, count, tile_points):
        end = min(begin + tile_points, count)
        with grid.feature_task(
            points[begin:end],
            None,
            ingredients,
            defer_error_to_consumer=True,
        ) as task:
            # Both operations consume this exact lease before the grid can
            # overwrite its tile. The producer checks device/stream identity.
            nonlocal_owner.collect(task, begin)
            sources.geometry(
                task,
                np.asarray(state.grid.owners[begin:end], dtype=np.int64),
                state.grid.weights[begin:end],
                raw_weights[begin:end],
                functional=functional,
            )
    local = sources.finish()
    components = {
        name: local[name] for name in ("xc_ao", "xc_grid", "xc_weight", "nuclear")
    }
    seconds = {
        "nonlocal_reset": reset_seconds,
        "semilocal_geometry_and_features": perf_counter() - began,
    }

    # Finish/reset the stationary consumer BEFORE submitting pair work. Reset
    # may fence borrowed streams; doing it after execute() would add an avoidable
    # pair-completion fence between the resident producer and its consumer.
    sources.reset(
        state._source.grid_spec.coincident_tolerance,
        state.density,
        state.weighted_density,
    )
    began = perf_counter()
    seeds = nonlocal_owner.execute()
    if (
        not seeds.pointer
        or seeds.stride != count
        or not seeds.stream
        or seeds.generation <= 0
    ):
        raise ValueError("resident nonlocal producer returned an invalid seed lease")
    seconds["vv10_pair_enqueue"] = perf_counter() - began
    began = perf_counter()
    for begin in range(0, count, tile_points):
        end = min(begin + tile_points, count)
        with grid.feature_task(
            points[begin:end],
            None,
            ingredients,
            defer_error_to_consumer=True,
        ) as task:
            if task.view.stream != seeds.stream:
                raise ValueError("resident nonlocal seed and geometry streams differ")
            # Inactive MolecularV1 rows have all six seeds zeroed by the native
            # producer. Original weights are therefore correct without a host
            # active mask; no density or seed values cross this boundary.
            sources.geometry_external_device(
                task,
                np.asarray(state.grid.owners[begin:end], dtype=np.int64),
                state.grid.weights[begin:end],
                raw_weights[begin:end],
                seeds.pointer,
                seeds.stride,
                begin,
            )
    nonlocal_parts = sources.finish()
    for suffix in ("ao", "grid", "weight"):
        components["nonlocal_" + suffix] = nonlocal_parts["xc_" + suffix]
    seconds["nonlocal_geometry_and_pair_drain"] = perf_counter() - began
    work = {
        "nonlocal_execution": "resident-full-grid-device-seeds",
        "nonlocal_feature_d2h_bytes": 0,
        "nonlocal_seed_h2d_bytes": 0,
        "nonlocal_dense_pair_capacity": count * count,
        "nonlocal_seed_generation": seeds.generation,
        "nonlocal_active_count_scope": "device-only; not measured by host scheduler",
        "ao_collocation_point_visits": 2 * count,
        "geometry_point_visits": 2 * count,
    }
    return components, seconds, work
