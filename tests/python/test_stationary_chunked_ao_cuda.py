"""Real-device chunk/fallback parity; molecular E/F oracles are separate gates."""

from __future__ import annotations

import ctypes as ct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.method.stationary_resources import (
    plan_stationary_cuda_resources,
)
from test_stationary_becke_phased_cuda import GridTaskView
from test_stationary_becke_phased_cuda import native as _native_fixture


@pytest.fixture(scope="module")
def native() -> SimpleNamespace:
    return _native_fixture.__wrapped__()


@pytest.mark.parametrize(("atoms", "aos"), [(48, 544), (96, 1088)])
@pytest.mark.parametrize("indexed", [False, True])
def test_chunked_ao_matches_scalar_sources(
    native: SimpleNamespace, atoms: int, aos: int, indexed: bool
) -> None:
    """Cross the shared-panel boundary with signed jets and unordered AO maps.

    Run with a freshly built r2SCAN RKS standalone stationary artifact. Comparing
    scalar, cooperative and phased schedules isolates the chunking reduction;
    the synthetic panels do not qualify XC physics or endpoint performance.
    """
    cp = native.cupy
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    major, minor = int(props["major"]), int(props["minor"])
    target = cuda_target_info(f"sm_{major}{minor}")
    rng = np.random.default_rng(8123 + atoms)
    points = 17
    centers = rng.normal(size=(atoms, 3)) * 4
    primitives = np.column_stack((np.ones(aos), np.full(aos, 0.7)))
    ranges = np.column_stack((np.arange(aos), np.ones(aos))).astype(np.int64)
    norms = np.ones(aos)
    labels = (np.arange(aos, dtype=np.int64) * 7) % atoms
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        xyz = cp.asarray(rng.normal(size=(points, 3)))
        owners = np.arange(points, dtype=np.int64) % atoms
        ids = cp.asarray(rng.permutation(aos).astype(np.uint64)) if indexed else None
        ao = cp.asarray(rng.normal(size=(10, points, aos)) * 0.1)
        work = cp.asarray(rng.normal(size=(8, points, aos)) * 0.03)
        host_features = np.zeros((13, points))
        host_features[0] = host_features[5] = 0.5
        host_features[4] = host_features[9] = 0.2
        features = cp.asarray(host_features)
        weights = np.full(points, 0.013)
        raw = np.full(points, 0.019)
        error = cp.asarray(np.zeros(1, dtype=np.int32))
    stream.synchronize()
    results = []
    for schedule in ("scalar", "cooperative", "phased"):
        plan = plan_stationary_cuda_resources(
            atoms=atoms,
            aos=aos,
            primitives=aos,
            points=points,
            tasks=1,
            spins=1,
            sources=7,
            target=target,
            budget_bytes=256 << 20,
            cooperative_becke=schedule != "scalar",
            phased_becke=schedule == "phased",
        )
        handle = ct.c_void_p()
        native.call(
            "stationary_create",
            device.id,
            major,
            minor,
            atoms,
            aos,
            aos,
            points,
            1,
            1,
            1,
            plan.allocation_bytes,
            plan.geometry_lanes,
            plan.geometry_threads,
            ct.byref(handle),
        )
        try:
            if schedule != "scalar":
                native.call(
                    "stationary_configure_becke",
                    handle,
                    plan.becke_threads_per_point,
                    plan.becke_shared_bytes,
                )
            if schedule == "phased":
                assert plan.phased_becke_bytes
                native.call(
                    "stationary_configure_phased_becke_v1",
                    handle,
                    plan.phased_becke_bytes,
                )
            native.call(
                "stationary_topology",
                handle,
                primitives.ctypes.data,
                ranges.ctypes.data,
                norms.ctypes.data,
                labels.ctypes.data,
            )
            native.call("stationary_geometry_reset", handle, centers.ctypes.data, 1e-12)
            view = GridTaskView(
                1,
                1,
                points,
                aos,
                aos,
                10,
                ids.data.ptr if ids is not None else None,
                xyz.data.ptr,
                ao.data.ptr,
                features.data.ptr,
                None,
                None,
                stream.ptr,
                error.data.ptr,
            )
            native.call(
                "stationary_geometry_enqueue",
                handle,
                ct.byref(view),
                work.data.ptr,
                owners.ctypes.data,
                weights.ctypes.data,
                raw.ctypes.data,
            )
            output = np.full((7, atoms, 3), np.nan)
            native.call("stationary_finish", handle, output.ctypes.data, output.size)
            results.append(output)
        finally:
            native.library.stationary_destroy(handle)
    for actual in results[1:]:
        np.testing.assert_allclose(actual, results[0], rtol=3e-12, atol=3e-11)
