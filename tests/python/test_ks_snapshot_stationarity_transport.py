"""Exercise native allowance transport without loading a library or a GPU.

These are Python ABI/ownership controls, not numerical PBE0 qualification.
The real snapshot constructor consumes a minimal fake native ABI.
"""

from __future__ import annotations

import ctypes as ct
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc._ks_snapshot import NativeKsSnapshot
from generativeqc.batch import PreparedBatch
from generativeqc.ks import KsOptions, resolve_ks_options


def native_batch(
    allowance: float | None,
    *,
    failure: str | None = None,
    domain: int = 1,
    device: int = 0,
    version: int = 3,
    method: str = "pbe-rks",
) -> tuple[PreparedBatch, list[str]]:
    """Model only the documented snapshot ABI and lifetime boundary."""
    events: list[str] = []

    def create(batch: Any, index: int, output: Any, metadata: Any, size: int) -> int:
        assert batch.value == 11 and index == 0 and size == 16
        events.append("create")
        ct.cast(output, ct.POINTER(ct.c_void_p))[0] = 23
        metadata[0] = version
        metadata[7] = domain
        metadata[12] = device
        metadata[15] = 2
        return 0

    def tolerance(batch: Any, handle: int, output: Any) -> int:
        assert batch.value == 11 and handle == 23
        events.append("allowance")
        if failure == "allowance":
            return 1
        ct.cast(output, ct.POINTER(ct.c_double))[0] = allowance
        return 0

    def copy(batch: Any, handle: int, output: Any, size: int) -> int:
        assert batch.value == 11 and handle == 23 and size == 2
        events.append("copy")
        output[0], output[1] = 7.0, 9.0
        return 1 if failure == "copy" else 0

    def destroy(handle: int) -> None:
        assert handle == 23
        events.append("destroy")

    library = SimpleNamespace(
        generativeqc_batch_get_system_count=lambda _: 1,
        generativeqc_ks_snapshot_create_v1=create,
        generativeqc_ks_snapshot_check_v1=lambda *_: 0,
        generativeqc_ks_snapshot_copy_v1=copy,
        generativeqc_ks_snapshot_destroy_v1=destroy,
        generativeqc_status_message=lambda _: b"native snapshot failure",
        generativeqc_context_get_last_detail=lambda _: b"stale native token",
    )
    if allowance is not None:
        library.generativeqc_ks_snapshot_stationarity_tolerance_v1 = tolerance
    batch = object.__new__(PreparedBatch)
    batch._batch = ct.c_void_p(11)
    batch._context = ct.c_void_p(17)
    batch._library = library
    batch._calculator = SimpleNamespace(
        ks_options=resolve_ks_options(method, KsOptions()),
        _density_fitting_mode=0,
    )
    return batch, events


def assert_native_owner_released(events: list[str]) -> None:
    # ABI lifetime matters; independent validation/copy operations may reorder.
    assert events[0] == "create"
    assert events[-1] == "destroy"
    assert events.count("destroy") == 1


@pytest.mark.parametrize("allowance", [None, 1e-12, 1e-8, 1e-5, 1e-4])
def test_snapshot_transports_native_allowance_and_keeps_legacy_strict(
    allowance: float | None,
) -> None:
    batch, events = native_batch(allowance)
    snapshot = NativeKsSnapshot(batch, 0)
    try:
        assert snapshot.stationarity_tolerance == (
            1e-8 if allowance is None else allowance
        )
        np.testing.assert_array_equal(snapshot.values, [7.0, 9.0])
        assert not snapshot.values.flags.writeable
        assert snapshot.backend == "cuda"
        with pytest.raises(AttributeError, match="immutable"):
            snapshot.stationarity_tolerance = 1e-4
        with pytest.raises(AttributeError, match="immutable"):
            del snapshot.stationarity_tolerance
    finally:
        snapshot.close()
    assert "copy" in events
    assert ("allowance" in events) is (allowance is not None)
    assert_native_owner_released(events)
    snapshot.close()
    assert events.count("destroy") == 1


@pytest.mark.parametrize(
    "allowance", [0.0, -1e-8, np.nan, np.inf, -np.inf, np.nextafter(1e-4, np.inf)]
)
def test_invalid_allowance_is_not_published_and_releases_native_snapshot(
    allowance: float,
) -> None:
    batch, events = native_batch(allowance)
    snapshot = object.__new__(NativeKsSnapshot)
    with pytest.raises(ValueError, match="invalid native stationary force allowance"):
        snapshot.__init__(batch, 0)
    assert_native_owner_released(events)


@pytest.mark.parametrize("failure", ["allowance", "copy"])
def test_snapshot_native_errors_keep_identity_and_release_owner(failure: str) -> None:
    batch, events = native_batch(1e-5, failure=failure)
    snapshot = object.__new__(NativeKsSnapshot)
    with pytest.raises(RuntimeError, match="stale native token"):
        snapshot.__init__(batch, 0)
    assert failure in events
    assert_native_owner_released(events)


@pytest.mark.parametrize(
    "domain,device,error", [(99, 0, NotImplementedError), (1, 2**64 - 1, ValueError)]
)
def test_allowance_does_not_bypass_domain_or_backend_provenance(
    domain: int, device: int, error: type[Exception]
) -> None:
    batch, events = native_batch(1e-5, domain=domain, device=device)
    snapshot = object.__new__(NativeKsSnapshot)
    with pytest.raises(error):
        snapshot.__init__(batch, 0)
    assert_native_owner_released(events)


@pytest.mark.parametrize("index", [-1, 1, True, 0.0])
def test_invalid_snapshot_index_never_queries_native_allowance(index: Any) -> None:
    batch, events = native_batch(1e-5)
    with pytest.raises(ValueError, match="in-range batch index"):
        NativeKsSnapshot(batch, index)
    assert not events


@pytest.mark.parametrize("version", range(1, 10))
@pytest.mark.parametrize("matching_device", [False, True])
def test_snapshot_version_keeps_backend_device_binding(
    version: int, matching_device: bool
) -> None:
    cpu = version in (2, 4, 6, 7)
    device = 2**64 - 1 if cpu == matching_device else 0
    batch, events = native_batch(1e-8, version=version, device=device)
    snapshot = object.__new__(NativeKsSnapshot)
    try:
        if matching_device:
            snapshot.__init__(batch, 0)
            assert snapshot.backend == ("cpu" if cpu else "cuda")
        else:
            with pytest.raises(ValueError, match="backend/device mismatch"):
                snapshot.__init__(batch, 0)
            assert "destroy" in events and "copy" not in events
    finally:
        snapshot.close()


@pytest.mark.parametrize("version", [0, 10, 2**64 - 1])
def test_snapshot_rejects_unknown_wire_versions(version: int) -> None:
    batch, events = native_batch(1e-8, version=version)
    snapshot = object.__new__(NativeKsSnapshot)
    with pytest.raises(NotImplementedError, match="snapshot/domain version"):
        snapshot.__init__(batch, 0)
    assert "destroy" in events and "copy" not in events


@pytest.mark.parametrize(
    "method,expected_domain",
    [("libxc:LDA_C_VWN_4", 5), ("libxc:GGA_X_APBE", 4), ("libxc:MGGA_X_LTA", 4)],
)
@pytest.mark.parametrize("domain", [1, 4, 5, 99])
def test_snapshot_automatic_functionals_require_their_exact_domain(
    method: str, expected_domain: int, domain: int
) -> None:
    batch, events = native_batch(1e-8, method=method, domain=domain)
    snapshot = object.__new__(NativeKsSnapshot)
    try:
        if domain == expected_domain:
            snapshot.__init__(batch, 0)
            assert "copy" in events
        else:
            with pytest.raises(NotImplementedError, match="snapshot/domain version"):
                snapshot.__init__(batch, 0)
            assert "destroy" in events and "copy" not in events
    finally:
        snapshot.close()
