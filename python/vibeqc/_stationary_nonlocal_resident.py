"""Compose shared semilocal and nonlocal geometry on resident feature leases.

The native nonlocal owner owns the full-grid rho/gradient and seed storage.
This module owns only sequencing. It never downloads a feature/seed array and
never supplies independent nonlocal mathematics or a functional-name selector.
"""

from __future__ import annotations

import typing
from time import perf_counter

import numpy as np


def resident_nonlocal_geometry(
    grid: typing.Any,
    sources: typing.Any,
    nonlocal_owner: typing.Any,
    *,
    points: np.ndarray,
    weights: np.ndarray,
    owners: np.ndarray,
    atomic_weights: np.ndarray,
    ao_ids: np.ndarray,
    tile_points: int,
    density: np.ndarray,
    weighted_density: np.ndarray,
    coincident_tolerance: float,
    functional: int,
    ingredients: tuple[str, ...] = ("rho", "gradient", "tau"),
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, typing.Any]]:
    """Consume a fresh/reset stationary owner, preserving its nuclear source.

    All borrowed work uses the grid stream. ``finish`` drains geometry before
    scratch reset, result publication, or reuse by a later force evaluation.
    The caller must close both owners on failure, in nonlocal-before-grid order.
    Two bounded AO traversals are retained; the nonlocal pair operator needs
    the complete density grid before its adjoints can be consumed.
    """
    if type(tile_points) is not int or tile_points <= 0:
        raise ValueError("resident geometry tile_points must be positive")
    count = len(points)
    if (
        count == 0
        or nonlocal_owner.point_count != count
        or len(weights) != count
        or len(owners) != count
        or len(atomic_weights) != count
    ):
        raise ValueError("resident geometry and nonlocal grid shapes differ")
    if not callable(getattr(sources, "geometry_external_device", None)):
        raise TypeError(
            "resident nonlocal geometry requires the device-seed consumer"
        )

    diagnostic = nonlocal_owner.diagnostic()
    if diagnostic.executed:
        nonlocal_owner.reset()
    elif diagnostic.collected_points:
        raise ValueError("resident nonlocal owner contains an incomplete previous grid")

    started = perf_counter()
    stream = None
    batches = 0
    for begin in range(0, count, tile_points):
        end = min(begin + tile_points, count)
        with grid.feature_task(points[begin:end], ao_ids, ingredients) as task:
            if stream is None:
                stream = task.view.stream
            elif task.view.stream != stream:
                raise ValueError("resident geometry grid stream changed")
            nonlocal_owner.collect(task, begin)
            sources.geometry(
                task,
                np.asarray(owners[begin:end], dtype=np.int64),
                weights[begin:end],
                atomic_weights[begin:end],
                functional=functional,
            )
            batches += 1
    local = sources.finish()
    components = {
        name: local[name] for name in ("xc_ao", "xc_grid", "xc_weight", "nuclear")
    }
    timings = {"semilocal_geometry_and_features": perf_counter() - started}

    started = perf_counter()
    seeds = nonlocal_owner.execute()
    if seeds.stream != stream or seeds.stride != count or seeds.generation <= 0:
        raise ValueError("resident nonlocal seed stream, shape, or generation mismatch")
    # This is enqueue time, not completed pair-kernel time. Its drain is included
    # in the nonlocal geometry boundary below, and the distinction is explicit.
    timings["vv10_enqueue"] = perf_counter() - started
    started = perf_counter()
    sources.reset(coincident_tolerance, density, weighted_density)
    for begin in range(0, count, tile_points):
        end = min(begin + tile_points, count)
        with grid.feature_task(points[begin:end], ao_ids, ingredients) as task:
            if task.view.stream != seeds.stream:
                raise ValueError("resident nonlocal consumer grid stream changed")
            # Inactive device seed rows are exactly zero. Original quadrature
            # weights are therefore correct for every AO/grid pullback; no host
            # active-mask download or reweighting is needed.
            sources.geometry_external_device(
                task,
                np.asarray(owners[begin:end], dtype=np.int64),
                weights[begin:end],
                atomic_weights[begin:end],
                seeds.pointer,
                seeds.stride,
                begin,
            )
    nonlocal_parts = sources.finish()
    for suffix in ("ao", "grid", "weight"):
        components["nonlocal_" + suffix] = nonlocal_parts["xc_" + suffix]
    timings["nonlocal_geometry_and_pair_drain"] = perf_counter() - started
    final = nonlocal_owner.diagnostic()
    if not final.executed or final.generation != seeds.generation:
        raise ValueError("resident nonlocal seed generation changed before publication")
    work = {
        "nonlocal_execution": "resident-device-seeds",
        "nonlocal_feature_d2h_bytes": 0,
        "nonlocal_feature_h2d_bytes": 0,
        "nonlocal_seed_d2h_bytes": 0,
        "nonlocal_seed_h2d_bytes": 0,
        "nonlocal_owned_device_bytes": final.device_bytes,
        "nonlocal_seed_generation": seeds.generation,
        "nonlocal_active_points": None,
        "nonlocal_pair_evaluations": None,
        "nonlocal_pair_evaluations_upper_bound": count * count,
        "ao_collocation_point_visits": 2 * count,
        "geometry_point_visits": 2 * count,
        "geometry_tile_batches": 2 * batches,
        "nonlocal_timing_boundary": "pair enqueue separate; completion charged to geometry drain",
    }
    return components, timings, work
