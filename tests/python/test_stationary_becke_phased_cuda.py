"""Opt-in real shared-owner geometry gate; not a complete molecular endpoint."""

from __future__ import annotations

import ctypes as ct
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
    if not os.environ.get("SLURM_STEP_ID") or not os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ):
        pytest.fail(
            "real-GPU tests require srun and scheduler-assigned device visibility"
        )
    cupy = pytest.importorskip("cupy")
    library = ct.CDLL(str(Path(path).resolve()))
    tail = [ct.c_char_p, ct.c_size_t]
    pointer, size = ct.c_void_p, ct.c_size_t
    signatures = {
        "stationary_create": [ct.c_int] * 3 + [size] * 10 + [ct.POINTER(pointer)],
        "stationary_configure_becke": [pointer, size, size],
        "stationary_configure_geometry_ao_v1": [pointer, ct.c_uint],
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
    }
    for name, arguments in signatures.items():
        getattr(library, name).argtypes = [*arguments, *tail]
    library.stationary_destroy.argtypes = [pointer]
    library.stationary_metrics.argtypes = [pointer, ct.POINTER(ct.c_uint64), size]
    library.stationary_phased_becke_metrics_v1.argtypes = [
        pointer,
        ct.POINTER(ct.c_uint64),
        size,
    ]

    def call(name: str, *arguments: object) -> None:
        error = ct.create_string_buffer(4096)
        status = getattr(library, name)(*arguments, error, len(error))
        if status:
            raise RuntimeError(error.value.decode())

    return SimpleNamespace(cupy=cupy, library=library, call=call)


@pytest.mark.parametrize(
    "atoms,aos", [(48, 4), (96, 4), (48, 384), (96, 768), (96, 900)]
)
@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("selection", ["full", "subset", "empty"])
@pytest.mark.parametrize("external", [False, True])
def test_shared_owner_phases_preserve_sources_and_work(
    native: SimpleNamespace,
    atoms: int,
    aos: int,
    implicit: bool,
    selection: str,
    external: bool,
) -> None:
    """Replay identical AO/XC inputs through the actual bounded/phased owner.

    Synthetic AO/features isolate routing and lifetime; they are not a molecular
    oracle. Independent Becke Decimal tests and complete E/F gates are separate.
    """
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
        np.arange(aos, dtype=np.uintp)
        if selection == "full"
        else rng.permutation(aos).astype(np.uintp)[: max(1, aos // 2)]
        if selection == "subset"
        else np.empty(0, dtype=np.uintp)
    )
    host_primitives = np.array([[1.0, 1.0]] * aos)
    host_ranges = np.column_stack((np.arange(aos), np.ones(aos))).astype(np.int64)
    host_norms = np.ones(aos)
    host_atoms = (np.arange(aos, dtype=np.int64) * 7) % atoms
    # Freeze the same cancellation-sensitive FP64 inputs across all schedules.
    ao_values = rng.normal(scale=0.1, size=(10, 257, len(selected)))
    work_values = rng.normal(scale=0.03, size=(8, 257, len(selected)))
    baseline = {}
    for ao_schedule, phased in (
        (schedule, phase) for schedule in range(4) for phase in (False, True)
    ):
        plan = plan_stationary_cuda_resources(
            atoms=atoms,
            aos=aos,
            primitives=aos,
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
            aos,
            aos,
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
            native.call(
                "stationary_configure_becke",
                handle,
                plan.becke_threads_per_point,
                plan.becke_shared_bytes,
            )
            native.call("stationary_configure_geometry_ao_v1", handle, ao_schedule)
            if phased:
                assert plan.phased_becke_bytes > 0
                native.call(
                    "stationary_configure_phased_becke_v1",
                    handle,
                    plan.phased_becke_bytes,
                )
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
                    "stationary_geometry_reset", handle, centers.ctypes.data, 1e-12
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
                        device = {
                            "points": cupy.asarray(
                                centers[owners[begin:end]] + offsets[begin:end]
                            ),
                            "features": cupy.asarray(features),
                            "ao": cupy.asarray(
                                np.ascontiguousarray(ao_values[:, begin:end]).reshape(
                                    -1
                                )
                                if active
                                else np.zeros(1)
                            ),
                            "work": cupy.asarray(
                                np.ascontiguousarray(work_values[:, begin:end]).reshape(
                                    -1
                                )
                                if active
                                else np.zeros(1)
                            ),
                            "ids": cupy.asarray(selected),
                            "weights": cupy.full(count, 0.4),
                            "raw": cupy.full(count, 0.7),
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
                            aos,
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
                            host_raw = np.full(count, 0.7)
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
                    np.testing.assert_allclose(
                        result, baseline[False, geometry], rtol=5e-12, atol=2e-11
                    )
                if ao_schedule == 0:
                    baseline[phased, geometry] = result.copy()
                else:
                    np.testing.assert_array_equal(
                        result.view(np.uint64),
                        baseline[phased, geometry].view(np.uint64),
                    )
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
            valid_points = device["points"].get()
            valid_tail = None
            for invalid in (False, True, False):
                native.call(
                    "stationary_geometry_reset", handle, centers.ctypes.data, 1e-12
                )
                with stream:
                    device["points"].set(valid_points, stream=stream)
                    if invalid:
                        device["points"][0, 0] = np.nan
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
        finally:
            native.library.stationary_destroy(handle)


@pytest.mark.parametrize(
    "schedule,topology", [(4, False), (2**32 - 1, False), (1, True)]
)
def test_geometry_ao_schedule_rejects_unknown_and_topology_mutation(
    native: SimpleNamespace, schedule: int, topology: bool
) -> None:
    """A diagnostic schedule is bounded and cannot change a live owner's contract."""
    plan = plan_stationary_cuda_resources(
        atoms=48,
        aos=4,
        primitives=4,
        points=256,
        tasks=1,
        spins=1,
        sources=8,
        target=cuda_target_info("sm_120"),
        budget_bytes=256 << 20,
    )
    handle = ct.c_void_p()
    native.call(
        "stationary_create",
        0,
        12,
        0,
        48,
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
    try:
        if topology:
            primitives = np.ones((4, 2))
            ranges = np.column_stack((np.arange(4), np.ones(4))).astype(np.int64)
            norms = np.ones(4)
            atoms = np.arange(4, dtype=np.int64)
            native.call(
                "stationary_topology",
                handle,
                primitives.ctypes.data,
                ranges.ctypes.data,
                norms.ctypes.data,
                atoms.ctypes.data,
            )
        with pytest.raises(
            RuntimeError, match="schedule must be 0..3.*before topology"
        ):
            native.call("stationary_configure_geometry_ao_v1", handle, schedule)
    finally:
        native.library.stationary_destroy(handle)
