"""Private native accounting for accepted prepared resource requests."""

import ctypes
import typing

from generativeqc_compiler.common.resources import ResourceAllocationError, _account


def observe_method_call(
    library: typing.Any,
    plan: typing.Any,
    ledger: typing.Any,
    callback: typing.Any,
    *,
    owner: typing.Any,
    phase: typing.Any = "observation",
    previous: typing.Any = None,
    native_managed: bool = False,
) -> typing.Any:
    """Bind the same prepared owner around setup and each synchronous replay.

    Preparation can allocate persistent scientific buffers. Its evidence must
    survive subsequent execution scopes and failures, while each replay resets
    the ledger peak to the buffers still owned by the prepared calculation.
    """
    from generativeqc_compiler.common.resources import CpuResourceObservation

    if native_managed and ledger is None:
        raise ValueError(
            "native prepared observation requires its resource metadata handle"
        )
    observed = CpuResourceObservation(
        library,
        enabled=not native_managed,
        cpu_workers=1,
        ledger=None if native_managed else ledger,
    )
    diagnostics = dict(previous or {})
    diagnostics.update(plan=plan.to_dict(), owner=owner, phase=phase)

    def evidence() -> typing.Any:
        if native_managed:
            snapshot = ledger.native_observation()
            observed.device_observation = snapshot["device_ledger"]
            observed.peak_bytes = snapshot["sampled_item_peak_host_bytes"]
            observed.samples = snapshot["samples"]
            observed.cuda_arena_peak_bytes = snapshot["sampled_cuda_arena_peak_bytes"]
            observed.cuda_arena_samples = snapshot["cuda_arena_samples"]
        record = observed.to_dict()
        if native_managed:
            record["scope_owner"] = "native-prepared-owner"
        if owner == "ks":
            record["cuda_scope"] = (
                "common direct-J provider arena samples; complete explicit KS device capacities are in device_ledger"
            )
            record["scope"] = (
                "explicit KS grid/basis/provider/SCF capacities with retained fleet and warm buffers"
            )
            record["excludes"] = [
                "unsampled setup/XC/recurrence/eigensolver temporaries",
                "object metadata and runtime overhead",
            ]
        return record

    try:
        with observed:
            status = callback()
        diagnostics[phase] = evidence()
        observed.verify(plan)
    except Exception as error:
        diagnostics[phase] = evidence()
        error.resource_diagnostics = diagnostics
        raise
    return status, diagnostics


def check_resource_status(
    library: typing.Any, status: typing.Any, diagnostics: typing.Any
) -> None:
    """Keep resource evidence on failed native calls without guessing OOM space.

    CPU allocation failures are host failures. For CUDA, only an actual ledger
    rejection identifies a device failure; opaque library/host allocation
    errors retain unknown placement and cannot trigger a space-specific retry.
    """
    from . import _native

    try:
        _native.check(library, status)
    except RuntimeError as error:
        failure = error
        if status == _native.STATUS_OUT_OF_MEMORY:
            observation = diagnostics.get(diagnostics.get("phase", "observation"), {})
            ledger = observation.get("device_ledger")
            backend = next(
                r["identity"]["backend"]
                for r in diagnostics["plan"]["requests"]
                if r["name"] == diagnostics.get("owner", "hf")
            )
            space = (
                "host"
                if backend == "cpu"
                else (
                    f"device:{ledger['device']}"
                    if ledger and ledger["rejected_allocations"]
                    else None
                )
            )
            failure = (
                ResourceAllocationError(space, str(error))
                if space
                else MemoryError(str(error))
            )
        failure.resource_diagnostics = diagnostics
        if failure is error:
            raise
        raise failure from error


class NativeDeviceLedger:
    """Persist charges across warm calls and release them with native buffers.

    Driver, graph, pool and library-internal numeric allocations remain outside
    the ledger. Native retained-provider reservations share its admission limit
    and v3 reports them separately from actual allocations and requested bytes.
    Each ledger belongs to one prepared request on one visible CUDA device.
    """

    def __init__(
        self,
        library: typing.Any,
        plan: typing.Any,
        *,
        owner: typing.Any = "hf",
        prepared_kind: str | None = None,
        component_owners: tuple[str, ...] = (),
    ) -> None:
        plan.require_feasible()
        self.owner = owner
        request = next(r for r in plan.requests if r.name == owner)
        if request.identity.backend != "cuda":
            raise ValueError("native device ledger requires a CUDA request")
        selected = dict(plan.selections)[owner]
        candidate = next(c for c in request.candidates if c.name == selected)
        self._native_complete_cap = (
            dict(candidate.decisions).get("native_d4_complete_cap") == "1"
        )
        if self._native_complete_cap and (
            owner != "ks" or component_owners != ("d4",) or prepared_kind != "batch"
        ):
            raise ValueError("native D4 capacity requires its complete prepared owner")
        # A complete native program owns its corrections in the same lifetime
        # scope. Their selected capacities must be charged to this ledger too;
        # global planning still retains separate, inspectable request identities.
        estimates = candidate.estimates
        if len({owner, *component_owners}) != 1 + len(component_owners):
            raise ValueError("native resource component owners must be distinct")
        for component in component_owners:
            request = next(r for r in plan.requests if r.name == component)
            selected = dict(plan.selections)[component]
            estimates += next(
                c for c in request.candidates if c.name == selected
            ).estimates
        self.library = library
        self.handle = None
        self._capacity_admission = None
        self.prepared_call = (
            self.native_prepare(prepared_kind) if prepared_kind is not None else None
        )
        if self.prepared_call is not None:
            self._destroy = library.generativeqc_device_resources_destroy
            self._create_public_inventory(estimates)
            return
        create = getattr(
            library, "generativeqc_resource_ledger_create_from_buffers_v1", None
        )
        required = ("destroy", "bind", "read")
        if create is None:
            required = ("create", *required)
        for name in required:
            if not hasattr(library, f"generativeqc_resource_ledger_{name}_v1"):
                raise NotImplementedError(
                    "native library has no persistent allocation ledger v1"
                )
        library.generativeqc_resource_ledger_destroy_v1.argtypes = [ctypes.c_void_p]
        library.generativeqc_resource_ledger_destroy_v1.restype = None
        self._destroy = library.generativeqc_resource_ledger_destroy_v1
        library.generativeqc_resource_ledger_bind_v1.argtypes = [ctypes.c_void_p]
        library.generativeqc_resource_ledger_bind_v1.restype = ctypes.c_int
        library.generativeqc_resource_ledger_read_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        library.generativeqc_resource_ledger_read_v1.restype = ctypes.c_int
        self._read_v2 = getattr(library, "generativeqc_resource_ledger_read_v2", None)
        if self._read_v2 is not None:
            self._read_v2.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
            ]
            self._read_v2.restype = ctypes.c_int
        self._read_v3 = getattr(library, "generativeqc_resource_ledger_read_v3", None)
        if self._read_v3 is not None:
            self._read_v3.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
            ]
            self._read_v3.restype = ctypes.c_int
        if create is not None:
            self._create_from_inventory(create, estimates)
        else:
            devices = {
                estimate.space
                for estimate in estimates
                if estimate.space.startswith("device:")
            }
            if len(devices) != 1:
                raise NotImplementedError(
                    "one native prepared owner must use one visible CUDA device"
                )
            self.device = int(devices.pop().split(":")[1])
            if self.device > 2**31 - 1:
                raise ValueError(
                    "device ledger ordinal exceeds the native device domain"
                )
            admitted = tuple(
                estimate
                for estimate in estimates
                if estimate.accounting != "runtime_allowance"
            )
            self.limit = _account(admitted)[0].get(f"device:{self.device}", 0)
            library.generativeqc_resource_ledger_create_v1.argtypes = [
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            library.generativeqc_resource_ledger_create_v1.restype = ctypes.c_void_p
            self.handle = library.generativeqc_resource_ledger_create_v1(
                self.limit, self.device
            )
            if not self.handle:
                raise MemoryError("could not allocate native resource ledger metadata")

    def _create_public_inventory(self, estimates: typing.Any) -> None:
        """Use only the selected public owner ABI; rejection never retries private creation."""
        from . import _native

        if not 0 < len(estimates) <= 2**32 - 1:
            raise ValueError("native selected inventory count exceeds uint32")
        rows = (_native.DeviceResourceBuffer * len(estimates))(
            *(
                _native.DeviceResourceBuffer(
                    estimate.bytes,
                    4
                    if estimate.space.startswith("device:")
                    else {"pinned": 2, "pageable": 3}[estimate.space],
                    int(estimate.space.split(":")[1])
                    if estimate.space.startswith("device:")
                    else 0,
                    estimate.first_phase,
                    estimate.last_phase,
                    int(estimate.accounting == "runtime_allowance"),
                )
                for estimate in estimates
            )
        )
        if self._native_complete_cap:
            # The validated candidate owns exactly KS and D4. Its private cap
            # includes provider allowances because native promises convert those
            # into leases; no other global request's bytes enter this handle.
            devices = {row.device for row in rows if row.space == 4}
            if len(devices) != 1:
                raise ValueError("native D4 capacity requires one visible device")
            device = devices.pop()
            cap = _account(estimates)[0][f"device:{device}"]
            rows = (_native.DeviceResourceBuffer * 1)(
                _native.DeviceResourceBuffer(cap, 4, device, 0, 0, 0)
            )
        descriptor = _native.DeviceResourceDescriptor(
            ctypes.sizeof(_native.DeviceResourceDescriptor),
            _native.ABI_VERSION,
            rows,
            len(rows),
            int(self._native_complete_cap),
        )
        handle = ctypes.c_void_p()
        status = self.library.generativeqc_device_resources_create(
            ctypes.byref(descriptor), ctypes.byref(handle)
        )
        if status or not handle.value:
            if handle.value:
                self._destroy(handle)
            if status == _native.STATUS_OUT_OF_MEMORY:
                raise MemoryError("native prepared resource metadata allocation failed")
            if status == _native.STATUS_NOT_IMPLEMENTED:
                raise NotImplementedError(
                    "native prepared resource inventory is unsupported"
                )
            raise ValueError("native prepared resource inventory was rejected")
        try:
            snapshot = _native.DeviceResourceDiagnostic(
                ctypes.sizeof(_native.DeviceResourceDiagnostic), _native.ABI_VERSION
            )
            if self.library.generativeqc_device_resources_read(
                handle, ctypes.byref(snapshot)
            ):
                raise ValueError(
                    "native prepared resource creation returned no diagnostic"
                )
            devices = {row.device for row in rows if row.space == 4}
            if devices != {snapshot.device} or not (
                0 <= snapshot.device <= 2**31 - 1
                and snapshot.numeric_limit <= snapshot.declared_device_peak <= 2**63 - 1
                and snapshot.struct_size
                == ctypes.sizeof(_native.DeviceResourceDiagnostic)
                and snapshot.abi_version == _native.ABI_VERSION
                and snapshot.requested_bytes_available == 1
                and snapshot.admission == int(self._native_complete_cap)
                and not any(
                    getattr(snapshot, name)
                    for name in (
                        "numeric_live_bytes",
                        "numeric_peak_bytes",
                        "allocations",
                        "rejected_allocations",
                        "requested_bytes",
                        "provider_reserved_bytes",
                        "provider_reservation_peak_bytes",
                        "combined_peak_bytes",
                        "capacity_reserved_bytes",
                        "sampled_host_peak_bytes",
                        "host_samples",
                        "sampled_cuda_arena_peak_bytes",
                        "cuda_arena_samples",
                    )
                )
            ):
                raise ValueError(
                    "native prepared resource creation returned invalid admission"
                )
            self.device, self.limit = snapshot.device, snapshot.numeric_limit
            self._capacity_admission = {
                "source": "native-device-ledger-plan-v1",
                "declared_device_peak": snapshot.declared_device_peak,
            }
        except BaseException:
            self._destroy(handle)
            raise
        self.handle = handle.value

    def _create_from_inventory(self, create: typing.Any, estimates: typing.Any) -> None:
        """Bind only a complete native receipt, never retrying rejected metadata."""
        rows = []
        for estimate in estimates:
            device = estimate.space.startswith("device:")
            rows.extend(
                (
                    estimate.bytes,
                    4 if device else {"pinned": 2, "pageable": 3}[estimate.space],
                    int(estimate.space.split(":")[1]) if device else 0,
                    estimate.first_phase,
                    estimate.last_phase,
                    int(estimate.accounting == "runtime_allowance"),
                )
            )
        buffers = (ctypes.c_uint64 * len(rows))(*rows)
        handle = ctypes.c_void_p()
        receipt = (ctypes.c_uint64 * 3)()
        create.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        create.restype = ctypes.c_int
        status = create(
            buffers, len(buffers), ctypes.byref(handle), receipt, len(receipt)
        )
        if status or not handle.value:
            if handle.value:
                self.library.generativeqc_resource_ledger_destroy_v1(handle)
            if status == 3:
                raise MemoryError("could not allocate native resource ledger metadata")
            if status == 4:
                raise NotImplementedError(
                    "one native prepared owner must use one visible CUDA device"
                )
            raise ValueError("native resource ledger rejected the selected inventory")
        try:
            device_labels = {
                rows[index + 2]
                for index in range(0, len(rows), 6)
                if rows[index + 1] == 4
            }
            if (
                device_labels != {receipt[0]}
                or receipt[0] > 2**31 - 1
                or not receipt[1] <= receipt[2] <= 2**63 - 1
            ):
                raise ValueError(
                    "native resource ledger returned an invalid capacity receipt"
                )
            device, limit = int(receipt[0]), int(receipt[1])
            admission = {
                "source": "native-device-ledger-plan-v1",
                "declared_device_peak": int(receipt[2]),
            }
        except BaseException:
            self.library.generativeqc_resource_ledger_destroy_v1(handle)
            raise
        self.device, self.limit = device, limit
        self._capacity_admission = admission
        self.handle = handle.value

    def native_prepare(self, kind: str) -> typing.Any:
        """Select the native lifecycle ABI once; only a missing symbol permits compatibility."""
        if kind not in ("batch", "calculation"):
            raise ValueError("unknown native prepared resource owner kind")
        prepare = getattr(
            self.library, f"generativeqc_{kind}_prepare_with_device_resources", None
        )
        if prepare is not None and not all(
            hasattr(self.library, name)
            for name in (
                f"generativeqc_{kind}_get_device_resources",
                "generativeqc_device_resources_read",
                "generativeqc_device_resources_create",
                "generativeqc_device_resources_destroy",
            )
        ):
            raise RuntimeError("native prepared resource ABI is incomplete")
        return prepare

    def native_observation(self) -> dict[str, typing.Any]:
        """Present completed native-operation samples without binding frontend observers."""
        from . import _native

        if not self.handle:
            raise RuntimeError("native resource ledger is closed")
        snapshot = _native.DeviceResourceDiagnostic(
            ctypes.sizeof(_native.DeviceResourceDiagnostic), _native.ABI_VERSION
        )
        status = self.library.generativeqc_device_resources_read(
            self.handle, ctypes.byref(snapshot)
        )
        if status:
            raise RuntimeError("native prepared resource observation is unavailable")
        if (
            snapshot.struct_size != ctypes.sizeof(_native.DeviceResourceDiagnostic)
            or snapshot.abi_version != _native.ABI_VERSION
            or snapshot.device != self.device
            or snapshot.numeric_limit != self.limit
            or snapshot.requested_bytes_available not in (0, 1)
            or snapshot.admission != int(self._native_complete_cap)
            or not snapshot.numeric_live_bytes
            <= snapshot.numeric_peak_bytes
            <= snapshot.numeric_limit
            or snapshot.numeric_live_bytes
            + snapshot.provider_reserved_bytes
            + snapshot.capacity_reserved_bytes
            > snapshot.numeric_limit
            or snapshot.combined_peak_bytes > snapshot.numeric_limit
            or snapshot.numeric_live_bytes + snapshot.provider_reserved_bytes
            > snapshot.combined_peak_bytes
            or snapshot.numeric_peak_bytes > snapshot.combined_peak_bytes
            or snapshot.provider_reservation_peak_bytes > snapshot.combined_peak_bytes
            or snapshot.provider_reserved_bytes
            > snapshot.provider_reservation_peak_bytes
            or snapshot.declared_device_peak < snapshot.numeric_limit
            or (
                self._capacity_admission is not None
                and snapshot.declared_device_peak
                != self._capacity_admission["declared_device_peak"]
            )
        ):
            raise ValueError("native prepared resource observation is inconsistent")
        device = {
            "device": self.device,
            "limit_bytes": self.limit,
            "live_bytes": snapshot.numeric_live_bytes,
            "peak_bytes": snapshot.numeric_peak_bytes,
            "allocations": snapshot.allocations,
            "rejected_allocations": snapshot.rejected_allocations,
            "owner": self.owner,
            "scope": "owned CUDA buffer capacities; excludes driver/graph/pool and library-internal allocations",
            "requested_bytes": snapshot.requested_bytes
            if snapshot.requested_bytes_available
            else None,
            "admission": {
                name: getattr(snapshot, name)
                for name in (
                    "numeric_live_bytes",
                    "numeric_peak_bytes",
                    "provider_reserved_bytes",
                    "provider_reservation_peak_bytes",
                    "combined_peak_bytes",
                    "capacity_reserved_bytes",
                )
            },
        }
        device["admission"]["limit_bytes"] = snapshot.numeric_limit
        device["admission"]["mode"] = (
            "numeric-cap" if self._native_complete_cap else "selected-inventory"
        )
        if self._capacity_admission is not None:
            device["capacity_admission"] = dict(self._capacity_admission)
        return {
            "device_ledger": device,
            "sampled_item_peak_host_bytes": snapshot.sampled_host_peak_bytes
            if snapshot.host_samples
            else None,
            "samples": snapshot.host_samples or None,
            "sampled_cuda_arena_peak_bytes": snapshot.sampled_cuda_arena_peak_bytes
            if snapshot.cuda_arena_samples
            else None,
            "cuda_arena_samples": snapshot.cuda_arena_samples or None,
        }

    def to_dict(self) -> typing.Any:
        """Read numeric capacities, v2 requested bytes and additive v3 admission.

        Older libraries retain the exact v1 shape: absent requested-byte
        evidence must not be inferred from the peak or outstanding storage.
        """
        if not self.handle:
            raise RuntimeError("native resource ledger is closed")
        if self.prepared_call is not None:
            return self.native_observation()["device_ledger"]
        count = 5 if self._read_v2 is not None else 4
        values = (ctypes.c_uint64 * count)()
        status = (
            self._read_v2(self.handle, values, count)
            if self._read_v2 is not None
            else self.library.generativeqc_resource_ledger_read_v1(self.handle, values)
        )
        if status:
            raise RuntimeError("native resource ledger is unavailable")
        result = {
            "device": self.device,
            "limit_bytes": self.limit,
            "live_bytes": values[0],
            "peak_bytes": values[1],
            "allocations": values[2],
            "rejected_allocations": values[3],
            "owner": self.owner,
            "scope": "owned CUDA buffer capacities; excludes driver/graph/pool and library-internal allocations",
        }
        if self._capacity_admission is not None:
            result["capacity_admission"] = dict(self._capacity_admission)
        if self._read_v2 is not None:
            result["requested_bytes"] = values[4]
        if self._read_v3 is not None:
            admission = (ctypes.c_uint64 * 6)()
            if self._read_v3(self.handle, admission, len(admission)):
                raise RuntimeError("native resource admission ledger is unavailable")
            result["admission"] = dict(
                zip(
                    (
                        "limit_bytes",
                        "numeric_live_bytes",
                        "numeric_peak_bytes",
                        "provider_reserved_bytes",
                        "provider_reservation_peak_bytes",
                        "combined_peak_bytes",
                    ),
                    admission,
                    strict=True,
                )
            )
        return result

    def close(self) -> None:
        """Release the observation handle; any live native buffers keep charges."""
        if self.handle:
            self._destroy(self.handle)
            self.handle = None

    def __del__(self) -> None:
        if getattr(self, "handle", None):
            self.close()


class NativeDeviceJournal:
    """Bounded event capture for one existing prepared owned-device ledger.

    Start between synchronous observation scopes. Retained owners become the
    initial live map, not fabricated allocation events. The capture survives
    ledger-handle closure so final native buffer releases remain observable.
    Snapshots alone do not authenticate source, phases or host coverage; any
    dropped event prevents a strict complete-coverage receipt.
    """

    def __init__(self, ledger: NativeDeviceLedger, *, capacity: int = 4096) -> None:
        self.handle = None
        if type(capacity) is not int or not 1 <= capacity <= 1 << 20:
            raise ValueError("journal capacity must be an integer in [1, 1048576]")
        if not ledger.handle:
            raise RuntimeError("native resource ledger is closed")
        self.library = ledger.library
        self.owner = ledger.owner
        self.device = ledger.device
        self.capacity = capacity
        for name in ("create", "read", "destroy"):
            if not hasattr(self.library, f"generativeqc_resource_journal_{name}_v1"):
                raise NotImplementedError("native library has no allocation journal v1")
        self.library.generativeqc_resource_journal_create_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.library.generativeqc_resource_journal_create_v1.restype = ctypes.c_void_p
        self.library.generativeqc_resource_journal_read_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.library.generativeqc_resource_journal_read_v1.restype = ctypes.c_int
        self.library.generativeqc_resource_journal_destroy_v1.argtypes = [
            ctypes.c_void_p
        ]
        self.library.generativeqc_resource_journal_destroy_v1.restype = None
        self.handle = self.library.generativeqc_resource_journal_create_v1(
            ledger.handle, capacity
        )
        if not self.handle:
            raise RuntimeError(
                "journal could not start: active/already captured owner or unavailable memory"
            )

    def snapshot(self) -> dict[str, typing.Any]:
        """Read actual generation-based records, including explicit capture loss."""
        if not self.handle:
            raise RuntimeError("native allocation journal is closed")
        records = (ctypes.c_uint64 * (6 * self.capacity))()
        state = (ctypes.c_uint64 * 4)()
        status = self.library.generativeqc_resource_journal_read_v1(
            self.handle, records, len(records), state
        )
        initial_count, event_count, dropped, recording = state
        if status or max(initial_count, event_count) > self.capacity or recording != 1:
            raise RuntimeError("native allocation journal is unavailable")
        initial_live = {}
        events = []
        for index in range(initial_count + event_count):
            kind, generation, size = records[3 * index : 3 * index + 3]
            if kind not in (0, 1) or generation == 0:
                raise RuntimeError("native allocation journal contains invalid records")
            name = f"allocation-{generation}"
            if index < initial_count:
                if kind != 0 or name in initial_live:
                    raise RuntimeError(
                        "native allocation journal has invalid retained owners"
                    )
                initial_live[name] = size
            else:
                events.append(
                    {
                        "sequence": index - initial_count,
                        "kind": "allocate" if kind == 0 else "release",
                        "allocation_id": name,
                        "requested_bytes": size,
                    }
                )
        return {
            "domain": {
                "owner": self.owner,
                "space": f"device:{self.device}",
                "counter": "native-device-journal.v1",
            },
            "initial_live": initial_live,
            "event_end": event_count,
            "events": events,
            "dropped_events": dropped,
        }

    def close(self) -> None:
        """Stop capture without freeing or reassigning any scientific buffer."""
        if self.handle:
            self.library.generativeqc_resource_journal_destroy_v1(self.handle)
            self.handle = None

    def __del__(self) -> None:
        if getattr(self, "handle", None):
            self.close()
