"""Device-free contracts for the resident nonlocal CUDA force owner (#1482)."""

from __future__ import annotations

import ctypes
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from generativeqc.nonlocal_runtime import _ResidentNonlocalForceOwner
from generativeqc_compiler.method import original_nonlocal_correlation

ROOT = Path(__file__).resolve().parents[2]


class _View(ctypes.Structure):
    _fields_ = [("npoint", ctypes.c_size_t), ("stream", ctypes.c_void_p)]


def _fake_library() -> tuple[SimpleNamespace, list[tuple[object, ...]]]:
    calls: list[tuple[object, ...]] = []

    def create(
        _context: object,
        _model: object,
        _points: object,
        _point_count: int,
        _weights: object,
        _weight_count: int,
        _threshold: float,
        output: object,
    ) -> int:
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(
            0x1111
        )
        calls.append(("create",))
        return 0

    def collect(_owner: object, _view: object, offset: int) -> int:
        calls.append(("collect", int(offset)))
        return 0

    def seed_snapshot(
        _batch: object, _snapshot: object, _owner: object, _view: object
    ) -> int:
        calls.append(("seed_snapshot",))
        return 0

    def execute(_owner: object) -> int:
        calls.append(("execute",))
        return 0

    def seed_view(
        _owner: object,
        pointer: object,
        stride: object,
        stream: object,
        generation: object,
    ) -> int:
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(
            0xCAFE
        )
        ctypes.cast(stride, ctypes.POINTER(ctypes.c_size_t))[0] = 4
        ctypes.cast(stream, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(
            0xBEEF
        )
        ctypes.cast(generation, ctypes.POINTER(ctypes.c_uint64))[0] = 3
        calls.append(("seed_view",))
        return 0

    def reset(_owner: object) -> int:
        calls.append(("reset",))
        return 0

    def metrics(_owner: object, values: object, count: int) -> int:
        assert int(count) == 6
        target = ctypes.cast(values, ctypes.POINTER(ctypes.c_uint64))
        for i, value in enumerate((928, 4, 4, 3, 1, 1)):
            target[i] = value
        calls.append(("metrics",))
        return 0

    def destroy(_owner: object) -> None:
        calls.append(("destroy",))

    library = SimpleNamespace(
        generativeqc_internal_nonlocal_cuda_force_create_v1=MagicMock(
            side_effect=create
        ),
        generativeqc_internal_nonlocal_cuda_force_destroy_v1=MagicMock(
            side_effect=destroy
        ),
        generativeqc_internal_nonlocal_cuda_force_collect_v1=MagicMock(
            side_effect=collect
        ),
        generativeqc_ks_snapshot_cuda_seed_nonlocal_force_v1=MagicMock(
            side_effect=seed_snapshot
        ),
        generativeqc_internal_nonlocal_cuda_force_execute_v1=MagicMock(
            side_effect=execute
        ),
        generativeqc_internal_nonlocal_cuda_force_seed_view_v1=MagicMock(
            side_effect=seed_view
        ),
        generativeqc_internal_nonlocal_cuda_force_reset_v1=MagicMock(side_effect=reset),
        generativeqc_internal_nonlocal_cuda_force_metrics_v1=MagicMock(
            side_effect=metrics
        ),
    )
    return library, calls


def test_resident_force_owner_publishes_only_borrowed_device_identity() -> None:
    library, calls = _fake_library()
    spec = original_nonlocal_correlation("vv10")
    points = np.arange(12, dtype=np.float64).reshape(4, 3) / 10.0
    weights = np.full(4, 0.25)
    context = ctypes.c_void_p(0x2222)

    owner = _ResidentNonlocalForceOwner(
        spec,
        points,
        weights,
        coefficient=Fraction(1),
        tile_points=2,
        maximum_bytes=1 << 20,
        density_threshold=1.0e-12,
        device_id=0,
        context=context,
        library=library,
    )
    task = SimpleNamespace(
        view=_View(2, ctypes.c_void_p(0x3333)),
        _owner=SimpleNamespace(device_id=0),
    )
    owner.collect(task, 0)
    owner.collect(task, 2)
    seed = owner.execute()
    diagnostic = owner.diagnostic()

    assert seed.pointer == 0xCAFE
    assert seed.stride == 4
    assert seed.stream == 0xBEEF
    assert seed.generation == 3
    assert diagnostic.device_bytes == 928
    assert diagnostic.point_count == diagnostic.collected_points == 4
    assert diagnostic.generation == 3
    assert diagnostic.executed and diagnostic.stream_bound
    assert calls[:5] == [
        ("create",),
        ("collect", 0),
        ("collect", 2),
        ("execute",),
        ("seed_view",),
    ]

    owner.reset()
    owner.close()
    assert calls[-2:] == [("reset",), ("destroy",)]


def test_resident_force_seeds_from_live_snapshot_without_host_materialization() -> None:
    library, calls = _fake_library()
    owner = _ResidentNonlocalForceOwner(
        original_nonlocal_correlation("vv10"),
        np.arange(12, dtype=np.float64).reshape(4, 3) / 10.0,
        np.full(4, 0.25),
        coefficient=Fraction(1),
        tile_points=2,
        maximum_bytes=1 << 20,
        density_threshold=1.0e-12,
        device_id=0,
        context=ctypes.c_void_p(0x2222),
        library=library,
    )
    task = SimpleNamespace(
        view=_View(2, ctypes.c_void_p(0x3333)),
        _owner=SimpleNamespace(device_id=0),
    )
    metadata = [0] * 16
    metadata[12] = 0
    snapshot = SimpleNamespace(
        backend="cuda",
        metadata=tuple(metadata),
        _handle=0x4444,
        _batch=SimpleNamespace(_batch=ctypes.c_void_p(0x5555)),
    )

    owner.seed_from_snapshot(snapshot, task)
    assert calls[-1] == ("seed_snapshot",)
    owner.close()


def test_resident_force_hot_path_has_no_host_transfer_or_fence() -> None:
    source = (ROOT / "src/api/c_api_nonlocal.cpp").read_text()
    hot = source.split(
        "GENERATIVEQC_API generativeqc_status generativeqc_internal_nonlocal_cuda_force_collect_v1(",
        1,
    )[1].split(
        "GENERATIVEQC_API generativeqc_status generativeqc_internal_nonlocal_cuda_force_seed_view_v1(",
        1,
    )[0]
    for forbidden in (
        "cudaMemcpyDeviceToHost",
        "cudaMemcpyHostToDevice",
        "cudaStreamSynchronize",
        "std::vector",
    ):
        assert forbidden not in hot
    assert "enqueue_vv10_collect_total_features_cuda" in hot
    assert "cudaMemcpyDeviceToDevice" in hot
    assert "owner->source_ready.record(source_stream)" in hot
    assert "cudaStreamWaitEvent(owner->stream, owner->source_ready.get(), 0)" in hot
    assert "enqueue_vv10_molecular_domain_cuda" in hot
    assert "enqueue_vv10_cuda_device" in hot
    assert "enqueue_vv10_pack_force_seeds_cuda" in hot


def test_resident_force_helpers_are_all_device_only() -> None:
    source = (ROOT / "src/dft/nonlocal_correlation/vv10_runtime_cuda.cu").read_text()
    body = source.split("void enqueue_vv10_collect_total_features_cuda(", 1)[1].split(
        "Vv10CudaDeviceLayout vv10_cuda_device_layout(", 1
    )[0]
    for forbidden in (
        "OwnedCudaBuffer",
        "OwnedCudaStream",
        "cudaMemcpy",
        "cudaStreamSynchronize",
    ):
        assert forbidden not in body
    assert "collect_total_features_kernel" in body
    assert "pack_force_seeds_kernel" in body
    assert "effective_weights" in body


def test_resident_feature_handoff_imports_cuda_stream_wait_event() -> None:
    cmake = (ROOT / "cmake/GenerativeQCCudaImplib.cmake").read_text()
    assert "cudaStreamWaitEvent" in cmake


def test_resident_feature_handoff_failure_cleanup_is_not_on_success_path() -> None:
    source = (ROOT / "src/api/c_api_nonlocal.cpp").read_text()
    helper = source.split("void drain_failed_seed_source(", 1)[1].split("}  // namespace", 1)[0]
    assert "cudaStreamSynchronize" in helper
    hot = source.split(
        "GENERATIVEQC_API generativeqc_status generativeqc_internal_nonlocal_cuda_force_seed_device_v1(",
        1,
    )[1].split(
        "GENERATIVEQC_API generativeqc_status\ngenerativeqc_internal_nonlocal_cuda_force_execute_v1(",
        1,
    )[0]
    assert "drain_failed_seed_source(source_stream);" in hot
    assert "catch (...)" in hot
