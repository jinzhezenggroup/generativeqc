"""Behavioral ownership/compatibility contracts; no GPU science is simulated here."""

import ctypes
import typing
from types import SimpleNamespace

import pytest
from generativeqc import _native
from generativeqc.resources_native import NativeDeviceLedger, observe_method_call


def forbidden(*args: typing.Any) -> typing.NoReturn:
    raise AssertionError("frontend observation/binding or compatibility retry was used")


def test_native_owner_does_not_need_frontend_observation_or_binding() -> None:
    library = SimpleNamespace(
        generativeqc_resource_tracking_begin_v1=forbidden,
        generativeqc_resource_tracking_end_v1=forbidden,
        generativeqc_resource_ledger_bind_v1=forbidden,
    )
    snapshot = {
        "device_ledger": {"peak_bytes": 64},
        "sampled_item_peak_host_bytes": 17,
        "samples": 1,
        "sampled_cuda_arena_peak_bytes": None,
        "cuda_arena_samples": None,
    }
    ledger = SimpleNamespace(native_observation=lambda: snapshot)
    plan = SimpleNamespace(
        to_dict=lambda: {"identity": "typed-plan"},
        peak_bytes={"host": 32, "device": 64},
    )
    status, diagnostic = observe_method_call(
        library,
        plan,
        ledger,
        lambda: _native.STATUS_SUCCESS,
        owner="hf",
        native_managed=True,
    )
    assert status == _native.STATUS_SUCCESS
    assert diagnostic["observation"]["scope_owner"] == "native-prepared-owner"
    assert diagnostic["observation"]["sampled_item_peak_host_bytes"] == 17
    assert diagnostic["observation"]["sampled_cuda_arena_peak_bytes"] is None
    assert diagnostic["observation"]["device_ledger"] == {"peak_bytes": 64}
    assert diagnostic["observation"]["complete_plan_peak"] is False


def test_native_ownership_preserves_failure_evidence_and_capacity_checks() -> None:
    snapshot = {
        "device_ledger": {"peak_bytes": 65},
        "sampled_item_peak_host_bytes": None,
        "samples": None,
        "sampled_cuda_arena_peak_bytes": None,
        "cuda_arena_samples": None,
    }
    plan = SimpleNamespace(to_dict=dict, peak_bytes={"host": 32, "device": 64})
    ledger = SimpleNamespace(native_observation=lambda: snapshot)
    with pytest.raises(RuntimeError, match="owned CUDA allocation peak") as failed:
        observe_method_call(
            SimpleNamespace(), plan, ledger, lambda: 0, owner="hf", native_managed=True
        )
    assert (
        failed.value.resource_diagnostics["observation"]["device_ledger"]["peak_bytes"]
        == 65
    )


@pytest.mark.parametrize("kind", ["batch", "calculation"])
def test_only_missing_native_prepare_symbol_allows_compatibility(kind: str) -> None:
    ledger = NativeDeviceLedger.__new__(NativeDeviceLedger)
    ledger.library = SimpleNamespace()
    assert ledger.native_prepare(kind) is None
    setattr(
        ledger.library, f"generativeqc_{kind}_prepare_with_device_resources", forbidden
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        ledger.native_prepare(kind)
    setattr(ledger.library, f"generativeqc_{kind}_get_device_resources", forbidden)
    ledger.library.generativeqc_device_resources_read = forbidden
    ledger.library.generativeqc_device_resources_create = forbidden
    ledger.library.generativeqc_device_resources_destroy = forbidden
    assert ledger.native_prepare(kind) is forbidden


def prepared_ledger(
    *, status: int = 0, changes: dict[str, int] | None = None
) -> NativeDeviceLedger:
    """Build a typed snapshot fixture without suggesting any real GPU execution."""

    def read(handle: typing.Any, output: typing.Any) -> int:
        if status:
            return status
        snapshot = ctypes.cast(
            output, ctypes.POINTER(_native.DeviceResourceDiagnostic)
        ).contents
        snapshot.numeric_limit = 64
        snapshot.declared_device_peak = 80
        snapshot.numeric_live_bytes = 16
        snapshot.numeric_peak_bytes = 32
        snapshot.allocations = 2
        snapshot.combined_peak_bytes = 32
        snapshot.device = 0
        snapshot.requested_bytes_available = 1
        snapshot.requested_bytes = 40
        for key, value in (changes or {}).items():
            setattr(snapshot, key, value)
        return 0

    ledger = NativeDeviceLedger.__new__(NativeDeviceLedger)
    ledger.handle = 17
    ledger._destroy = lambda handle: None
    ledger.owner = "hf"
    ledger.device = 0
    ledger.limit = 64
    ledger._capacity_admission = {"declared_device_peak": 80}
    ledger.library = SimpleNamespace(
        generativeqc_device_resources_read=read,
        generativeqc_resource_ledger_read_v1=forbidden,
        generativeqc_resource_ledger_destroy_v1=lambda handle: None,
    )
    return ledger


def test_native_snapshot_preserves_unavailable_samples_and_requested_byte_overflow() -> (
    None
):
    ledger = prepared_ledger(
        changes={"requested_bytes_available": 0, "requested_bytes": 0}
    )
    result = ledger.native_observation()
    assert result["samples"] is None and result["sampled_item_peak_host_bytes"] is None
    assert (
        result["cuda_arena_samples"] is None
        and result["sampled_cuda_arena_peak_bytes"] is None
    )
    assert result["device_ledger"]["requested_bytes"] is None
    assert result["device_ledger"]["live_bytes"] == 16
    assert result["device_ledger"]["peak_bytes"] == 32


def test_native_snapshot_refusal_does_not_retry_an_older_reader() -> None:
    ledger = prepared_ledger(status=1)
    with pytest.raises(RuntimeError, match="unavailable"):
        ledger.native_observation()


@pytest.mark.parametrize(
    "changes",
    [
        {"struct_size": 0},
        {"abi_version": _native.ABI_VERSION + 1},
        {"device": 7},
        {"numeric_limit": 65},
        {"declared_device_peak": 79},
        {"numeric_live_bytes": 33},
        {"numeric_peak_bytes": 65},
        {"combined_peak_bytes": 65},
        {"provider_reserved_bytes": 1},
        {"provider_reservation_peak_bytes": 33},
        {"provider_reserved_bytes": 17, "provider_reservation_peak_bytes": 17},
        {"requested_bytes_available": 2},
        {"admission": 1},
        {"admission": 2},
        {"capacity_reserved_bytes": 49},
    ],
)
def test_malformed_native_snapshot_never_falls_back(changes: dict[str, int]) -> None:
    ledger = prepared_ledger(changes=changes)
    with pytest.raises(ValueError, match="inconsistent"):
        ledger.native_observation()


def test_unused_capacity_is_distinct_from_live_and_peak_allocations() -> None:
    ledger = prepared_ledger(changes={"capacity_reserved_bytes": 48})
    snapshot = ledger.native_observation()["device_ledger"]
    assert snapshot["admission"]["capacity_reserved_bytes"] == 48
    assert snapshot["live_bytes"] == 16
    assert snapshot["peak_bytes"] == 32
    assert snapshot["requested_bytes"] == 40


def inventory_plan() -> SimpleNamespace:
    """A selected capacity inventory; global planning is outside these adapter tests."""
    estimate = SimpleNamespace(
        bytes=64,
        space="device:7",
        first_phase=0,
        last_phase=1,
        accounting="capacity_bound",
    )
    candidate = SimpleNamespace(name="selected", estimates=(estimate,))
    return SimpleNamespace(
        require_feasible=lambda: None,
        requests=(
            SimpleNamespace(
                name="hf",
                identity=SimpleNamespace(backend="cuda"),
                candidates=(candidate,),
            ),
        ),
        selections=(("hf", "selected"),),
    )


@pytest.mark.parametrize("kind", ["calculation", "batch"])
@pytest.mark.parametrize(
    "failure",
    [
        None,
        "create",
        "missing-handle",
        "read",
        "device",
        "abi",
        "live",
        "mode",
        "promise",
    ],
)
def test_public_creation_needs_no_private_owner_abi_and_never_retries(
    kind: str,
    failure: str | None,
) -> None:
    released = []

    def create(descriptor: typing.Any, output: typing.Any) -> int:
        selected = ctypes.cast(
            descriptor, ctypes.POINTER(_native.DeviceResourceDescriptor)
        ).contents
        assert selected.struct_size == ctypes.sizeof(_native.DeviceResourceDescriptor)
        assert selected.abi_version == _native.ABI_VERSION
        assert selected.buffer_count == 1
        assert selected.admission == 0
        assert selected.buffers[0].bytes == 64 and selected.buffers[0].device == 7
        if failure != "missing-handle":
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 17
        return _native.STATUS_NOT_IMPLEMENTED if failure == "create" else 0

    def read(handle: typing.Any, output: typing.Any) -> int:
        snapshot = ctypes.cast(
            output, ctypes.POINTER(_native.DeviceResourceDiagnostic)
        ).contents
        snapshot.numeric_limit = snapshot.declared_device_peak = 64
        snapshot.device = 8 if failure == "device" else 7
        if failure == "abi":
            snapshot.abi_version = _native.ABI_VERSION + 1
        if failure == "live":
            snapshot.numeric_live_bytes = 1
        if failure == "mode":
            snapshot.admission = 1
        if failure == "promise":
            snapshot.capacity_reserved_bytes = 1
        snapshot.requested_bytes_available = 1
        return _native.STATUS_NOT_IMPLEMENTED if failure == "read" else 0

    def destroy(handle: typing.Any) -> None:
        released.append(handle.value if isinstance(handle, ctypes.c_void_p) else handle)

    library = SimpleNamespace(
        generativeqc_device_resources_create=create,
        generativeqc_device_resources_read=read,
        generativeqc_device_resources_destroy=destroy,
    )
    setattr(library, f"generativeqc_{kind}_prepare_with_device_resources", forbidden)
    setattr(library, f"generativeqc_{kind}_get_device_resources", forbidden)
    if failure:
        with pytest.raises((ValueError, NotImplementedError)):
            NativeDeviceLedger(library, inventory_plan(), prepared_kind=kind)
    else:
        ledger = NativeDeviceLedger(library, inventory_plan(), prepared_kind=kind)
        assert ledger.device == 7 and ledger.limit == 64
        assert ledger.to_dict()["capacity_admission"]["declared_device_peak"] == 64
        ledger.close()
        ledger.close()
    assert released == ([] if failure == "missing-handle" else [17])


@pytest.mark.parametrize("admission", ("public_native", "native_inventory", "legacy"))
def test_fault_injection_changes_the_selected_constructor_input(
    monkeypatch: typing.Any, admission: str
) -> None:
    """Keep fault injection on the admission ABI, not detached Python metadata."""
    from _resource_admission_test_support import limit_device_admission

    handle = 1 << 40
    seen = []

    @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    def public_create(descriptor: typing.Any, output: typing.Any) -> int:
        selected = ctypes.cast(
            descriptor, ctypes.POINTER(_native.DeviceResourceDescriptor)
        ).contents
        row = selected.buffers[0]
        seen.append((selected.buffer_count, row.bytes, row.space, row.device))
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = handle
        return 0

    @ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_size_t,
    )
    def inventory_create(
        buffers: typing.Any,
        count: int,
        output: typing.Any,
        receipt: typing.Any,
        receipt_count: int,
    ) -> int:
        seen.append((count, *buffers[:count]))
        output[0] = handle
        return 0

    @ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    def legacy_create(capacity: int, device: int) -> int:
        seen.append((capacity, device))
        return handle

    library = SimpleNamespace(
        generativeqc_device_resources_create=public_create,
        generativeqc_resource_ledger_create_from_buffers_v1=inventory_create,
        generativeqc_resource_ledger_create_v1=legacy_create,
    )
    limit_device_admission(monkeypatch, library, 1, admission)
    output = ctypes.c_void_p()
    if admission == "public_native":
        assert (
            library.generativeqc_device_resources_create(None, ctypes.byref(output))
            == 0
        )
        assert seen == [(1, 1, 4, 0)]
    elif admission == "native_inventory":
        assert (
            library.generativeqc_resource_ledger_create_from_buffers_v1(
                None, 0, ctypes.byref(output), None, 0
            )
            == 0
        )
        assert seen == [(6, 1, 4, 0, 0, 0, 0)]
    else:
        output.value = library.generativeqc_resource_ledger_create_v1(4096, 3)
        assert seen == [(1, 3)]
        assert library.generativeqc_resource_ledger_create_from_buffers_v1 is None
    assert output.value == handle
