"""Opt-in real shared-owner geometry gate; not a complete molecular endpoint."""

from __future__ import annotations

import ctypes as ct
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.method.stationary_resources import (
    plan_stationary_cuda_resources,
)


class GridTaskView(ct.Structure):
    """Version-one borrowed grid ABI; every pointed-to buffer outlives drain."""

    _fields_ = [
        (name, ct.c_uint64 if name in {"version", "generation"} else ct.c_size_t)
        for name in ("version", "generation", "npoint", "nao", "nactive", "jets")
    ] + [
        (name, ct.c_void_p)
        for name in (
            "ao_ids",
            "points",
            "ao",
            "features",
            "local_potential",
            "potential",
            "stream",
            "error",
        )
    ]


@pytest.fixture(scope="module")
def native() -> SimpleNamespace:
    path = os.environ.get("GENERATIVEQC_PHASED_OWNER_PROBE")
    if not path:
        pytest.skip("shared-owner qualification artifact not configured")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("real-GPU tests require a finite Slurm allocation")
    cupy = pytest.importorskip("cupy")
    library = ct.CDLL(str(Path(path).resolve()))
    tail = [ct.c_char_p, ct.c_size_t]
    pointer, size = ct.c_void_p, ct.c_size_t
    signatures = {
        "stationary_create": [ct.c_int] * 3 + [size] * 10 + [ct.POINTER(pointer)],
        "stationary_configure_becke": [pointer, size, size],
        "stationary_configure_phased_becke_v1": [pointer, size],
        "stationary_topology": [pointer] * 5,
        "stationary_geometry_reset": [pointer, pointer, ct.c_double],
        "stationary_geometry_enqueue": [pointer, ct.POINTER(GridTaskView)]
        + [pointer] * 4,
        "stationary_geometry_external_device_enqueue": [
            pointer,
            ct.POINTER(GridTaskView),
            *([pointer] * 5),
            size,
            size,
        ],
        "stationary_geometry_molecular_resident_weights_enqueue": [
            pointer,
            ct.POINTER(GridTaskView),
            pointer,
            size,
            size,
            pointer,
            pointer,
        ],
        "stationary_geometry_external_device_molecular_resident_weights_enqueue": [
            pointer,
            ct.POINTER(GridTaskView),
            pointer,
            size,
            size,
            pointer,
            pointer,
            pointer,
            size,
            size,
        ],
        "stationary_finish": [pointer, pointer, size],
        "stationary_profile": [pointer],
    }
    if hasattr(library, "stationary_configure_becke_primitive_v1"):
        signatures["stationary_configure_becke_primitive_v1"] = [pointer, ct.c_int]
    if hasattr(library, "stationary_configure_becke_zero_seed_v1"):
        signatures["stationary_configure_becke_zero_seed_v1"] = [pointer, ct.c_int]
    if hasattr(library, "stationary_configure_becke_normalize_v1"):
        signatures["stationary_configure_becke_normalize_v1"] = [pointer, ct.c_int]
    for name, arguments in signatures.items():
        getattr(library, name).argtypes = [*arguments, *tail]
    library.stationary_destroy.argtypes = [pointer]
    library.stationary_metrics.argtypes = [pointer, ct.POINTER(ct.c_uint64), size]
    library.stationary_phased_becke_metrics_v1.argtypes = [
        pointer,
        ct.POINTER(ct.c_uint64),
        size,
    ]
    if hasattr(library, "stationary_becke_primitive_metrics_v1"):
        library.stationary_becke_primitive_metrics_v1.argtypes = [
            pointer,
            ct.POINTER(ct.c_uint64),
            size,
        ]
    for name, scalar in (
        ("stationary_becke_phase_metrics_v1", ct.c_uint64),
        ("stationary_becke_phase_profile_v1", ct.c_double),
    ):
        if hasattr(library, name):
            getattr(library, name).argtypes = [pointer, ct.POINTER(scalar), size]

    def call(name: str, *arguments: object) -> None:
        error = ct.create_string_buffer(4096)
        status = getattr(library, name)(*arguments, error, len(error))
        if status:
            raise RuntimeError(error.value.decode())

    return SimpleNamespace(cupy=cupy, library=library, call=call)


@pytest.mark.parametrize("atoms", [3, 4, 48])
@pytest.mark.parametrize("phased", [False, True])
def test_restricted_point_binding_preserves_sources_and_reports_actual_selection(
    native: SimpleNamespace, atoms: int, phased: bool
) -> None:
    """Qualify synthetic equal-spin owner routing, not a molecular endpoint.

    The small-atom and nonphased cases must retain the general kernel despite
    a valid proof. Independent point references and real producer publication
    are separate gates; these supplied features/work panels are explicit mocks.
    """
    library = native.library
    enqueue = getattr(
        library, "stationary_geometry_molecular_resident_weights_enqueue_v2", None
    )
    counters = getattr(library, "stationary_point_binding_metrics_v1", None)
    if enqueue is None or counters is None:
        pytest.skip("artifact predates restricted point binding")
    enqueue.argtypes = [
        ct.c_void_p,
        ct.POINTER(GridTaskView),
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        ct.c_void_p,
        ct.c_void_p,
        ct.c_uint64,
        ct.c_uint64,
        ct.c_char_p,
        ct.c_size_t,
    ]
    counters.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint64), ct.c_size_t]
    cupy = native.cupy
    rng = np.random.default_rng(702900 + atoms)
    centers = rng.normal(size=(atoms, 3)) * 3
    primitives = np.array([[1.0, 1.0]] * 4)
    ranges = np.column_stack((np.arange(4), np.ones(4))).astype(np.int64)
    norms = np.ones(4)
    ao_atoms = np.array([0, 1, atoms - 2, atoms - 1], dtype=np.int64)
    capacity, points = 31, atoms * 3
    plan = plan_stationary_cuda_resources(
        atoms=atoms,
        aos=4,
        primitives=4,
        points=capacity,
        tasks=1,
        spins=1,
        sources=8,
        target=cuda_target_info("sm_120"),
        budget_bytes=256 << 20,
        phased_becke=phased,
    )
    handle = ct.c_void_p()
    native.call(
        "stationary_create",
        0,
        12,
        0,
        atoms,
        4,
        4,
        capacity,
        1,
        1,
        1,
        plan.allocation_bytes,
        plan.geometry_lanes,
        plan.geometry_threads,
        ct.byref(handle),
    )
    stream = cupy.cuda.Stream(non_blocking=True)

    def metrics() -> tuple[int, ...]:
        values = (ct.c_uint64 * 5)()
        assert counters(handle, values, 5) == 0
        return tuple(values)

    try:
        if metrics()[0] != 1:
            pytest.skip("artifact is not the qualified unpolarized PBE0 composition")
        native.call(
            "stationary_configure_becke",
            handle,
            plan.becke_threads_per_point,
            plan.becke_shared_bytes,
        )
        if plan.phased_becke_bytes:
            native.call(
                "stationary_configure_phased_becke_v1", handle, plan.phased_becke_bytes
            )
        native.call(
            "stationary_topology",
            handle,
            primitives.ctypes.data,
            ranges.ctypes.data,
            norms.ctypes.data,
            ao_atoms.ctypes.data,
        )
        point_owners = np.arange(points) // 3
        xyz = centers[point_owners] + rng.normal(size=(points, 3)) * 0.3
        features = rng.random((5, points)) * 0.1
        features[0] += 0.2
        features = np.tile(features, (2, 1))
        ao = rng.normal(size=(10, points, 4)) * 0.2
        work = np.tile(rng.normal(size=(4, points, 4)) * 0.1, (2, 1, 1))
        results = []
        counts = []
        for proof in (None, 0, 1):
            native.call("stationary_geometry_reset", handle, centers.ctypes.data, 1e-12)
            before = metrics()
            for begin in range(0, points, capacity):
                end = min(points, begin + capacity)
                with stream:
                    device = {
                        "points": cupy.asarray(xyz[begin:end]),
                        "features": cupy.asarray(
                            np.ascontiguousarray(features[:, begin:end])
                        ),
                        "ao": cupy.asarray(np.ascontiguousarray(ao[:, begin:end])),
                        "work": cupy.asarray(np.ascontiguousarray(work[:, begin:end])),
                        "weights": cupy.full(end - begin, 0.4),
                        "raw": cupy.full(end - begin, 0.7),
                        "error": cupy.zeros(1, dtype=np.int32),
                    }
                    view = GridTaskView(
                        1,
                        17,
                        end - begin,
                        4,
                        4,
                        10,
                        None,
                        device["points"].data.ptr,
                        device["ao"].data.ptr,
                        device["features"].data.ptr,
                        None,
                        None,
                        stream.ptr,
                        device["error"].data.ptr,
                    )
                    arguments = [
                        handle,
                        ct.byref(view),
                        device["work"].data.ptr,
                        begin,
                        3,
                        device["weights"].data.ptr,
                        device["raw"].data.ptr,
                    ]
                    if proof is None:
                        native.call(
                            "stationary_geometry_molecular_resident_weights_enqueue",
                            *arguments,
                        )
                    else:
                        native.call(
                            "stationary_geometry_molecular_resident_weights_enqueue_v2",
                            *arguments,
                            17,
                            proof,
                        )
                stream.synchronize()
            output = np.empty((8, atoms, 3))
            native.call("stationary_finish", handle, output.ctypes.data, output.size)
            results.append(output)
            delta = tuple(
                after - earlier
                for after, earlier in zip(metrics()[1:], before[1:], strict=True)
            )
            selected = proof == 1 and bool(plan.phased_becke_bytes) and atoms >= 4
            batches = (points + capacity - 1) // capacity
            assert delta == (
                (batches, points, 0, 0) if selected else (0, 0, batches, points)
            )
            counts.append(delta)
        for output in results[1:]:
            np.testing.assert_allclose(output, results[0], rtol=5e-12, atol=2e-11)
        for generation, proof in ((16, 1), (17, 2)):
            native.call("stationary_geometry_reset", handle, centers.ctypes.data, 1e-12)
            before = metrics()
            with pytest.raises(RuntimeError, match="density binding proof"):
                native.call(
                    "stationary_geometry_molecular_resident_weights_enqueue_v2",
                    *arguments,
                    generation,
                    proof,
                )
            assert metrics() == before
        evidence = os.environ.get("GENERATIVEQC_RESTRICTED_POINT_EVIDENCE")
        if evidence:
            with Path(evidence).open("a") as output:
                output.write(
                    json.dumps(
                        {
                            "scope": "synthetic owner routing and source parity, not endpoint performance",
                            "atoms": atoms,
                            "phased_requested": phased,
                            "phased": bool(plan.phased_becke_bytes),
                            "points": points,
                            "point_counts": counts,
                            "max_source_error": float(
                                np.max(np.abs(results[2] - results[0]))
                            ),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    finally:
        library.stationary_destroy(handle)


@pytest.mark.parametrize("atoms", [48, 96, 128])
@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("selection", ["full", "subset", "empty"])
@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("primitive", [False, True, 2, "2-off", "2-serial"])
@pytest.mark.parametrize("profiled", [False, True])
def test_shared_owner_phases_preserve_sources_and_work(
    native: SimpleNamespace,
    atoms: int,
    implicit: bool,
    selection: str,
    external: bool,
    primitive: bool | int | str,
    profiled: bool,
) -> None:
    """Replay identical AO/XC inputs through the actual bounded/phased owner.

    Synthetic AO/features isolate routing and lifetime; they are not a molecular
    oracle. Independent Becke Decimal tests and complete E/F gates are separate.
    Include the actual 128-atom primitive cap, not only the two endpoint sizes.
    """
    zero_seed_off = primitive == "2-off"
    serial_normalize = primitive == "2-serial"
    if zero_seed_off and not hasattr(
        native.library, "stationary_configure_becke_zero_seed_v1"
    ):
        pytest.skip("artifact predates zero-seed qualification opt-out")
    if serial_normalize and not hasattr(
        native.library, "stationary_configure_becke_normalize_v1"
    ):
        pytest.skip("artifact predates normalization qualification control")
    if zero_seed_off or serial_normalize:
        primitive = 2
    if primitive and not hasattr(
        native.library, "stationary_configure_becke_primitive_v1"
    ):
        pytest.skip("artifact predates whole-domain primitive admission")
    if primitive == 2 and not hasattr(
        native.library, "stationary_configure_becke_normalized_adjoint_v1"
    ):
        pytest.skip("artifact predates generated normalized atom adjoints")
    if profiled and not hasattr(native.library, "stationary_becke_phase_profile_v1"):
        pytest.skip("artifact predates separate Becke phase profiling")
    cupy = native.cupy
    rng = np.random.default_rng(183000 + atoms)
    centers = rng.normal(size=(atoms, 3)) * 3
    base_centers = centers.copy()
    owners = (
        np.arange(257, dtype=np.int64) // 7
        if implicit
        else rng.integers(atoms, size=257, dtype=np.int64)
    )
    offsets = rng.normal(size=(257, 3)) * 0.3
    selected = (
        np.arange(4, dtype=np.uintp)
        if selection == "full"
        else np.array([3, 1], dtype=np.uintp)
        if selection == "subset"
        else np.empty(0, dtype=np.uintp)
    )
    host_primitives = np.array([[1.0, 1.0]] * 4)
    host_ranges = np.column_stack((np.arange(4), np.ones(4))).astype(np.int64)
    host_norms = np.ones(4)
    host_atoms = np.array([0, 1, atoms - 2, atoms - 1], dtype=np.int64)
    baseline = []
    for phased in (False, True):
        max_abs_error = 0.0
        max_scaled_error = 0.0
        plan = plan_stationary_cuda_resources(
            atoms=atoms,
            aos=4,
            primitives=4,
            points=256,
            tasks=1,
            spins=1,
            sources=8,
            target=cuda_target_info("sm_120"),
            budget_bytes=256 << 20,
            phased_becke=phased,
        )
        handle = ct.c_void_p()
        native.call(
            "stationary_create",
            0,
            12,
            0,
            atoms,
            4,
            4,
            256,
            1,
            1,
            1,
            plan.allocation_bytes,
            plan.geometry_lanes,
            plan.geometry_threads,
            ct.byref(handle),
        )
        stream = cupy.cuda.Stream(non_blocking=True)
        try:
            if profiled:
                native.call("stationary_profile", handle)
            native.call(
                "stationary_configure_becke",
                handle,
                plan.becke_threads_per_point,
                plan.becke_shared_bytes,
            )
            if phased:
                assert plan.phased_becke_bytes > 0
                native.call(
                    "stationary_configure_phased_becke_v1",
                    handle,
                    plan.phased_becke_bytes,
                )
            if hasattr(native.library, "stationary_configure_becke_primitive_v1"):
                native.call(
                    "stationary_configure_becke_primitive_v1", handle, int(primitive)
                )
            if zero_seed_off:
                native.call("stationary_configure_becke_zero_seed_v1", handle, 0)
            if serial_normalize:
                native.call("stationary_configure_becke_normalize_v1", handle, 0)
            native.call(
                "stationary_topology",
                handle,
                host_primitives.ctypes.data,
                host_ranges.ctypes.data,
                host_norms.ctypes.data,
                host_atoms.ctypes.data,
            )
            for geometry in range(3):
                centers = base_centers.copy()
                if geometry == 1:
                    centers[:, 1] += 0.03 * np.sin(np.arange(atoms))
                native.call(
                    "stationary_geometry_reset",
                    handle,
                    centers.ctypes.data,
                    1e-14 if geometry == 1 else 1e-12,
                )
                buffers = []
                with stream:
                    for begin, end in ((0, 256), (256, 257)):
                        count = end - begin
                        features = np.zeros((13, count))
                        features[0] = 0.8
                        features[1:4] = np.array([0.02, -0.015, 0.01])[:, None]
                        features[4] = 0.1
                        active = len(selected)
                        raw = np.full(count, 0.7)
                        # Mixed zero/nonzero rows and a subsequent all-zero
                        # replay exercise uninitialized and stale pair panels.
                        zero_rows = (
                            np.ones(count, dtype=bool)
                            if geometry == 2
                            else np.arange(begin, end) % (3 if geometry == 0 else 5)
                            == 0
                        )
                        raw[zero_rows] = 0
                        if geometry == 0:
                            raw[(np.arange(begin, end) % 7 == 1) & ~zero_rows] = 1e-300
                        device = {
                            "points": cupy.asarray(
                                centers[owners[begin:end]] + offsets[begin:end]
                            ),
                            "features": cupy.asarray(features),
                            "ao": cupy.full(max(1, 10 * count * active), 0.1),
                            "work": cupy.full(max(1, 8 * count * active), 0.03),
                            "ids": cupy.asarray(selected),
                            "weights": cupy.full(count, 0.4),
                            "raw": cupy.asarray(raw),
                            "error": cupy.zeros(1, dtype=np.int32),
                            "external": cupy.asarray(
                                np.linspace(-0.3, 0.5, 6 * 300).reshape(6, 300)
                            ),
                        }
                        buffers.append(device)
                        view = GridTaskView(
                            1,
                            geometry + 1,
                            count,
                            4,
                            active,
                            10,
                            None if selection == "full" else device["ids"].data.ptr,
                            device["points"].data.ptr,
                            device["ao"].data.ptr,
                            device["features"].data.ptr,
                            None,
                            None,
                            stream.ptr,
                            device["error"].data.ptr,
                        )
                        if implicit:
                            method = (
                                "stationary_geometry_external_device_molecular_resident_weights_enqueue"
                                if external
                                else "stationary_geometry_molecular_resident_weights_enqueue"
                            )
                            arguments = [
                                device["work"].data.ptr,
                                begin,
                                7,
                                device["weights"].data.ptr,
                                device["raw"].data.ptr,
                            ]
                        else:
                            host_weights = np.full(count, 0.4)
                            host_raw = raw
                            device["host_weights"] = host_weights
                            device["host_raw"] = host_raw
                            method = (
                                "stationary_geometry_external_device_enqueue"
                                if external
                                else "stationary_geometry_enqueue"
                            )
                            arguments = [
                                device["work"].data.ptr,
                                owners[begin:end].ctypes.data,
                                host_weights.ctypes.data,
                                host_raw.ctypes.data,
                            ]
                        if external:
                            arguments.extend(
                                [device["external"].data.ptr, 300, begin + 11]
                            )
                        native.call(method, handle, ct.byref(view), *arguments)
                result = np.empty((8, atoms, 3))
                native.call(
                    "stationary_finish", handle, result.ctypes.data, result.size
                )
                if phased:
                    difference = np.abs(result - baseline[geometry])
                    max_abs_error = max(max_abs_error, float(np.max(difference)))
                    max_scaled_error = max(
                        max_scaled_error,
                        float(
                            np.max(
                                difference
                                / (2e-11 + 5e-12 * np.abs(baseline[geometry]))
                            )
                        ),
                    )
                    np.testing.assert_allclose(
                        result, baseline[geometry], rtol=5e-12, atol=2e-11
                    )
                else:
                    baseline.append(result)
            metrics = (ct.c_uint64 * 24)()
            phase_metrics = (ct.c_uint64 * 2)()
            assert native.library.stationary_metrics(handle, metrics, 24) == 0
            assert (
                native.library.stationary_phased_becke_metrics_v1(
                    handle, phase_metrics, 2
                )
                == 0
            )
            assert metrics[23] == 3 * 257 * atoms * (atoms - 1) // (2 if phased else 1)
            assert phase_metrics[0] == plan.phased_becke_bytes
            assert phase_metrics[1] == (6 if phased else 0)
            assert metrics[0] == plan.allocation_bytes
            zero_metrics = getattr(
                native.library, "stationary_becke_zero_seed_metrics_v1", None
            )
            if zero_metrics is not None:
                zero_metrics.argtypes = [
                    ct.c_void_p,
                    ct.POINTER(ct.c_uint64),
                    ct.c_size_t,
                ]
                zero_work = (ct.c_uint64 * 2)()
                assert zero_metrics(handle, zero_work, 2) == 0
                # The 1e-14 reset deliberately retains the generic arithmetic.
                assert tuple(zero_work) == (
                    int(primitive == 2 and phased and not zero_seed_off),
                    (86 + 257)
                    if primitive == 2 and phased and not zero_seed_off
                    else 0,
                )
            if hasattr(native.library, "stationary_becke_phase_metrics_v1"):
                phase_work = (ct.c_uint64 * 17)()
                phase_times = (ct.c_double * 7)()
                assert (
                    native.library.stationary_becke_phase_metrics_v1(
                        handle, phase_work, 17
                    )
                    == 0
                )
                assert (
                    native.library.stationary_becke_phase_profile_v1(
                        handle, phase_times, 7
                    )
                    == 0
                )
                points = 3 * 257 if phased else 0
                pairs = points * atoms * (atoms - 1) // 2
                words = 2 if primitive == 1 and phased else 4
                profiled_batches = 6 if profiled and phased else 0
                assert tuple(phase_work) == (
                    6 if phased else 0,
                    points,
                    points * atoms,
                    pairs,
                    2 * pairs,
                    points * atoms,
                    pairs,
                    2 * pairs,
                    points * atoms,
                    42 if phased else 0,
                    4 * 8 * pairs,
                    words * 8 * pairs,
                    2 * words * 8 * pairs,
                    2 * 3 * 8 * pairs if primitive == 1 and phased else 0,
                    profiled_batches,
                    8 * profiled_batches,
                    profiled_batches,
                )
                if profiled_batches:
                    assert all(
                        np.isfinite(value) and value > 0 for value in phase_times
                    )
                else:
                    assert tuple(phase_times) == (0.0,) * 7
                if path := os.environ.get("GENERATIVEQC_BECKE_PHASE_RECORD"):
                    with Path(path).open("a") as records:
                        records.write(
                            json.dumps(
                                {
                                    "atoms": atoms,
                                    "implicit": implicit,
                                    "selection": selection,
                                    "external": external,
                                    "primitive_requested": primitive,
                                    "phased": phased,
                                    "profiled": profiled,
                                    "phase_work": list(phase_work),
                                    "phase_ms": list(phase_times),
                                    "generic_ad_max_abs_error": max_abs_error
                                    if phased
                                    else None,
                                    "generic_ad_max_scaled_error": max_scaled_error
                                    if phased
                                    else None,
                                    "generic_ad_tolerance": {
                                        "atol": 2e-11,
                                        "rtol": 5e-12,
                                    },
                                    "scope": "synthetic shared-owner qualification; not endpoint",
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
            if hasattr(native.library, "stationary_becke_primitive_metrics_v1"):
                primitive_metrics = (ct.c_uint64 * 4)()
                assert (
                    native.library.stationary_becke_primitive_metrics_v1(
                        handle, primitive_metrics, 4
                    )
                    == 0
                )
                assert tuple(primitive_metrics) == (
                    int(primitive),
                    int(primitive) if phased else 0,
                    6 if primitive and phased else 0,
                    3 * 257 * atoms * (atoms - 1) // 2 if primitive and phased else 0,
                )
            valid_points = device["points"].get()
            valid_tail = None
            for invalid in (False, "nan", "coincident", False):
                native.call(
                    "stationary_geometry_reset", handle, centers.ctypes.data, 1e-12
                )
                with stream:
                    device["points"].set(valid_points, stream=stream)
                    if invalid == "nan":
                        device["points"][0, 0] = np.nan
                    elif invalid == "coincident":
                        device["points"][0] = cupy.asarray(centers[owners[256]])
                    native.call(method, handle, ct.byref(view), *arguments)
                result = np.full((8, atoms, 3), 31415.0)
                if invalid:
                    with pytest.raises(RuntimeError, match="nonfinite or invalid"):
                        native.call(
                            "stationary_finish", handle, result.ctypes.data, result.size
                        )
                    np.testing.assert_array_equal(result, 31415.0)
                else:
                    native.call(
                        "stationary_finish", handle, result.ctypes.data, result.size
                    )
                    assert np.isfinite(result).all()
                    if valid_tail is None:
                        valid_tail = result
                    else:
                        np.testing.assert_array_equal(result, valid_tail)
            if hasattr(native.library, "stationary_configure_becke_zero_seed_v1"):
                with pytest.raises(RuntimeError, match="once before topology"):
                    native.call("stationary_configure_becke_zero_seed_v1", handle, 1)
        finally:
            native.library.stationary_destroy(handle)
