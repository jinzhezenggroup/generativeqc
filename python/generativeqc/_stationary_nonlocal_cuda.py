"""Compose resident semilocal and nonlocal geometry consumers on one stream.

This module owns scheduling only. AO/XC pullbacks and VV10 mathematics remain
in their existing generated/native owners. The final SCF rho/grad-rho are
seeded D2D into the resident VV10 owner, so both geometry consumers share one
bounded AO/grid collocation pass without retaining O(N*AO) jets. If that
optional binding is unavailable, two bounded passes collect device features
and consume device seeds without exporting either array to the host.
"""

from __future__ import annotations

import typing
from time import perf_counter

if typing.TYPE_CHECKING:
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
    """Prefer live SCF features, retaining the bounded device collection fallback.

    The caller has reset both stationary accumulators and enqueued the nuclear
    term only on the semilocal source. The first grid lease supplies the shared
    CUDA stream used to D2D-seed the full-grid resident VV10 owner from the exact
    current KS snapshot. Pair seeds are then available before that same lease
    reaches the nonlocal geometry consumer. A capability miss instead collects
    features alongside semilocal geometry and consumes pairs in a second pass.
    Token, device, allocation and numerical errors never select that fallback.
    No AO jet survives its lease; failed calls publish no component dictionary.
    """
    if type(tile_points) is not int or tile_points <= 0:
        raise ValueError("resident nonlocal tile_points must be a positive integer")
    if not callable(
        getattr(
            nonlocal_sources,
            "geometry_external_device_molecular_resident_weights",
            None,
        )
    ):
        raise TypeError("nonlocal stationary owner lacks the resident seed consumer")
    if not callable(getattr(nonlocal_owner, "seed_from_snapshot", None)):
        raise TypeError("resident nonlocal owner lacks the final-state feature handoff")
    points = state.grid.points
    count = len(points)
    if count == 0 or nonlocal_owner.point_count != count:
        raise ValueError("resident nonlocal owner/grid point count differs")
    resident_grid = state._source.cuda_resident_grid()
    if resident_grid is None:
        raise NotImplementedError(
            "resident nonlocal geometry requires the CUDA molecular-grid lease"
        )
    if resident_grid.device != grid.device_id or resident_grid.point_count != count:
        raise ValueError("resident molecular-grid lease differs from stationary grid")
    if count % sources.natom:
        raise ValueError("molecular grid point count is not atom-major uniform")
    points_per_atom = count // sources.natom
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
    local = None
    snapshot_seeded = True
    for phase in range(2):
        if phase == 1 and snapshot_seeded:
            break
        for begin in range(0, count, tile_points):
            end = min(begin + tile_points, count)
            point_pointer = resident_grid.points + 3 * begin * 8
            with grid.feature_task_device_points(
                point_pointer,
                end - begin,
                None,
                ingredients,
            ) as task:
                weights = state.grid.weights[begin:end]
                device_weights = resident_grid.weights + begin * 8
                raw = raw_weights[begin:end]
                if phase == 0:
                    sources.geometry_molecular_resident_weights(
                        task,
                        begin,
                        points_per_atom,
                        device_weights,
                        weights,
                        raw,
                        functional=functional,
                    )
                    if begin == 0:
                        began = perf_counter()
                        try:
                            nonlocal_owner.seed_from_snapshot(state._source, task)
                        except NotImplementedError:
                            # The native bridge checks the exact token before
                            # returning a capability miss, without seeding or
                            # enqueueing copies. Do not catch execution errors.
                            diagnostic = nonlocal_owner.diagnostic()
                            if diagnostic.executed or diagnostic.collected_points:
                                raise RuntimeError(
                                    "feature capability miss modified "
                                    "the nonlocal owner"
                                ) from None
                            snapshot_seeded = False
                        seed_seconds = perf_counter() - began
                    if not snapshot_seeded:
                        nonlocal_owner.collect(task, begin)
                        continue
                if seeds is None:
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
                if task.view.stream != seeds.stream:
                    raise ValueError(
                        "resident nonlocal seed and geometry streams differ"
                    )
                # Inactive MolecularV1 rows have all six seeds zeroed by the
                # native producer, so no host active mask is needed.
                nonlocal_sources.geometry_external_device_molecular_resident_weights(
                    task,
                    begin,
                    points_per_atom,
                    device_weights,
                    weights,
                    raw,
                    seeds.pointer,
                    seeds.stride,
                    begin,
                )
        if phase == 0:
            local = sources.finish()

    if local is None or seeds is None:
        raise RuntimeError("resident nonlocal geometry did not complete")
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
    geometry_passes = 1 if snapshot_seeded else 2
    geometry_timing = (
        "single_pass_geometry_and_pair_drain"
        if snapshot_seeded
        else "two_pass_geometry_and_pair_drain"
    )
    seconds = {
        "nonlocal_reset": reset_seconds,
        "resident_feature_seed_enqueue": seed_seconds,
        "vv10_pair_enqueue": pair_seconds,
        geometry_timing: geometry_and_drain_seconds,
    }
    work = {
        "nonlocal_execution": "resident-full-grid-device-seeds",
        "nonlocal_feature_source": (
            "exact-final-scf-device-binding"
            if snapshot_seeded
            else "bounded-grid-feature-collection"
        ),
        # D2D copy enqueues are distinct from the fallback collection kernel.
        "nonlocal_feature_d2d_bytes": 4 * count * 8 if snapshot_seeded else 0,
        "nonlocal_feature_collection_point_visits": 0 if snapshot_seeded else count,
        "nonlocal_feature_d2h_bytes": 0,
        "nonlocal_seed_h2d_bytes": 0,
        "grid_owner_source": "implicit-atom-major-index",
        "grid_owner_h2d_bytes": 0,
        "grid_point_source": "exact-native-resident-grid",
        "grid_point_h2d_bytes": 0,
        "grid_weight_source": "exact-native-resident-grid",
        "grid_weight_h2d_bytes": 0,
        "nonlocal_dense_pair_capacity": count * count,
        "nonlocal_seed_generation": seeds.generation,
        "nonlocal_active_count_scope": "device-only; not measured by host scheduler",
        "ao_collocation_point_visits": geometry_passes * count,
        "geometry_point_visits": 2 * count,
    }
    return components, seconds, work
