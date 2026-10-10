"""Journal adapters fail closed; optional profiler tests use real host heap calls.

Synthetic protocol tests are not scientific/GPU qualification. The Memray tests
require the separately pinned optional profiling environment, not production
compiler/runtime dependencies.
"""

from __future__ import annotations

import copy
import ctypes
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from tools.audit_replay_allocations import InvalidReceipt
from tools.replay_allocation_capture import (
    JournalWindows,
    normalize_heap_records,
    read_memray_heap,
)

if TYPE_CHECKING:
    from conftest import NativeCxx

ROOT = Path(__file__).resolve().parents[2]
ALLOCATORS = {1: "MALLOC", 2: "FREE", 3: "MMAP"}


def raw(kind: int, address: int, size: int = 0, count: int = 1) -> Any:
    return SimpleNamespace(
        allocator=kind, address=address, size=size, n_allocations=count
    )


def journal() -> dict[str, Any]:
    return normalize_heap_records(
        [raw(1, 16, 31), raw(1, 32, 47), raw(2, 16), raw(2, 32), raw(2, 900)],
        ALLOCATORS,
        markers={"boundary": (32, 47)},
    )


def test_heap_requests_peak_and_initial_frees_remain_distinct() -> None:
    capture = journal()
    assert capture["event_end"] == 4
    assert capture["raw_count"] == 5
    assert capture["excluded"] == {"pre-tracker/untracked free": 1}
    assert capture["marker_positions"] == {"boundary": 1}
    windows = JournalWindows()
    setup = windows.take(capture, 2)
    assert setup["metrics"] == {
        "allocation_count": 2,
        "requested_bytes": 78,
        "peak_live_bytes": 78,
        "live_bytes": 78,
    }
    replay = windows.take(capture, 3)
    assert replay["metrics"] == {
        "allocation_count": 0,
        "requested_bytes": 0,
        "peak_live_bytes": 78,
        "live_bytes": 47,
    }
    publication = windows.take(capture)
    assert publication["event_begin"] == replay["event_end"] == 3
    assert publication["metrics"]["live_bytes"] == 0


@pytest.mark.parametrize(
    "records,markers,limit,match",
    [
        ([raw(1, 16, 8), raw(1, 16, 8)], {}, 10, "reused"),
        ([raw(1, 16, 8), raw(2, 16), raw(1, 16, 8)], {"end": (16, 8)}, 10, "ambiguous"),
        ([raw(1, 16, 8)], {"end": (32, 8)}, 10, "unobserved"),
        ([raw(1, 16, 8), raw(2, 16)], {}, 1, "finite"),
        ([raw(1, 16, 8, 2)], {}, 10, "aggregated"),
        ([raw(1, 16, 8, True)], {}, 10, "uint64"),
        ([raw(9, 16, 8)], {}, 10, "unsupported"),
        ([raw(1, 0, 8)], {}, 10, "null"),
    ],
)
def test_heap_adapter_refuses_missing_or_ambiguous_capture(
    records: list[Any], markers: dict, limit: int, match: str
) -> None:
    with pytest.raises(InvalidReceipt, match=match):
        normalize_heap_records(records, ALLOCATORS, markers=markers, max_events=limit)


def test_mapped_memory_is_not_silently_added_to_heap_ownership() -> None:
    capture = normalize_heap_records([raw(3, 16, 4096)], ALLOCATORS, markers={})
    assert capture["events"] == []
    assert capture["excluded"] == {"MMAP": 1}


def test_window_history_and_published_events_do_not_alias_mutable_input() -> None:
    capture = journal()
    windows = JournalWindows()
    setup = windows.take(capture, 2)
    capture["events"][0]["requested_bytes"] = 32
    assert setup["events"][0]["requested_bytes"] == 31
    with pytest.raises(InvalidReceipt, match="history changed"):
        windows.take(capture)


@pytest.mark.parametrize(
    "mutation", ("dropped", "rewritten", "owner", "initial", "range")
)
def test_windows_refuse_capture_loss_and_history_changes(mutation: str) -> None:
    capture = journal()
    windows = JournalWindows()
    windows.take(capture, 2)
    altered = copy.deepcopy(capture)
    if mutation == "dropped":
        altered["dropped_events"] = 1
    elif mutation == "rewritten":
        altered["events"][0]["requested_bytes"] = 32
    elif mutation == "owner":
        altered["domain"]["owner"] = "other"
    elif mutation == "initial":
        altered["initial_live"] = {"hidden": 99}
    with pytest.raises(InvalidReceipt):
        windows.take(altered, 1 if mutation == "range" else None)
    assert windows.cursor == 2
    assert windows.take(capture)["metrics"]["live_bytes"] == 0


@pytest.fixture(scope="module")
def marker_library(
    tmp_path_factory: pytest.TempPathFactory, required_native_cxx: NativeCxx
) -> Any:
    """Compile the real observer's PLT marker; no GPU or allocator replacement."""
    folder = tmp_path_factory.mktemp("allocation-marker")
    path = required_native_cxx.build_shared(
        [ROOT / "tools/allocation_audit_marker.cpp"],
        folder / "marker.so",
        compile_args=("-std=c++20", "-O2"),
    )
    library = ctypes.CDLL(str(path))
    library.generativeqc_audit_marker_allocate.argtypes = [ctypes.c_size_t]
    library.generativeqc_audit_marker_allocate.restype = ctypes.c_void_p
    library.generativeqc_audit_marker_release.argtypes = [ctypes.c_void_p]
    library.generativeqc_audit_marker_release.restype = None
    return library


def test_real_native_heap_capture_has_exact_ownership_and_visible_markers(
    marker_library: Any, tmp_path: Path
) -> None:
    memray = pytest.importorskip("memray")
    assert memray.__version__ == "1.20.0"
    capture = tmp_path / "host.bin"
    allocate = marker_library.generativeqc_audit_marker_allocate
    release = marker_library.generativeqc_audit_marker_release
    retained = allocate(11)
    with memray.Tracker(str(capture), file_format=memray.FileFormat.ALL_ALLOCATIONS):
        first = allocate(31)
        second = allocate(47)
        release(first)
        release(second)
        release(retained)
    actual = read_memray_heap(
        capture, markers={"first": (first, 31), "second": (second, 47)}
    )
    observed = JournalWindows().take(actual)
    assert observed["metrics"]["allocation_count"] >= 2
    requests = {
        event["requested_bytes"]
        for event in actual["events"]
        if event["kind"] == "allocate"
    }
    assert {31, 47} <= requests
    assert actual["excluded"]["pre-tracker/untracked free"] >= 1


def test_production_dense_oracle_hoist_reduces_actual_heap_requests(
    tmp_path: Path,
) -> None:
    memray = pytest.importorskip("memray")
    from generativeqc.response_operator import _BaseResponseOperator

    class DiagonalOperator:
        dimension = 1024
        to_dense = _BaseResponseOperator.to_dense

        def apply(self, vector: np.ndarray) -> np.ndarray:
            return 2.0 * vector

    operator = DiagonalOperator()

    def per_iteration_reference() -> np.ndarray:
        dense = np.empty((operator.dimension, operator.dimension))
        for column in range(operator.dimension):
            impulse = np.zeros(operator.dimension)
            impulse[column] = 1.0
            dense[:, column] = operator.apply(impulse)
        return dense

    counts = []
    for name, endpoint in (
        ("per-iteration", per_iteration_reference),
        ("hoisted", operator.to_dense),
    ):
        capture = tmp_path / f"{name}.bin"
        with memray.Tracker(
            str(capture), file_format=memray.FileFormat.ALL_ALLOCATIONS
        ):
            dense = endpoint()
        np.testing.assert_array_equal(dense, 2.0 * np.eye(operator.dimension))
        with memray.FileReader(str(capture)) as reader:
            counts.append(
                sum(
                    record.allocator == memray.AllocatorType.CALLOC
                    and record.size == operator.dimension * np.dtype("float64").itemsize
                    for record in reader.get_allocation_records()
                )
            )
        read_memray_heap(capture, markers={})
    assert counts == [operator.dimension, 1]
