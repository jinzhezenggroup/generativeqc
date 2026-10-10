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
) -> typing.Any:
    """Bind the same prepared owner around setup and each synchronous replay.

    Preparation can allocate persistent scientific buffers. Its evidence must
    survive subsequent execution scopes and failures, while each replay resets
    the ledger peak to the buffers still owned by the prepared calculation.
    """
    from generativeqc_compiler.common.resources import CpuResourceObservation

    observed = CpuResourceObservation(library, cpu_workers=1, ledger=ledger)
    diagnostics = dict(previous or {})
    diagnostics.update(plan=plan.to_dict(), owner=owner, phase=phase)

    def evidence() -> typing.Any:
        record = observed.to_dict()
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

    Driver, graph, pool and library-internal allocations are outside this
    numeric-buffer ledger; the common plan reports their allowances/exclusions.
    Each ledger belongs to one prepared request on one visible CUDA device.
    """

    def __init__(
        self, library: typing.Any, plan: typing.Any, *, owner: typing.Any = "hf"
    ) -> None:
        plan.require_feasible()
        self.owner = owner
        request = next(r for r in plan.requests if r.name == owner)
        if request.identity.backend != "cuda":
            raise ValueError("native device ledger requires a CUDA request")
        selected = dict(plan.selections)[owner]
        candidate = next(c for c in request.candidates if c.name == selected)
        devices = {
            e.space for e in candidate.estimates if e.space.startswith("device:")
        }
        if len(devices) != 1:
            raise NotImplementedError(
                "one native prepared owner must use one visible CUDA device"
            )
        self.device = int(devices.pop().split(":")[1])
        numeric = tuple(
            e for e in candidate.estimates if e.accounting != "runtime_allowance"
        )
        self.limit = _account(numeric)[0][f"device:{self.device}"]
        self.library = library
        self.handle = None
        for name in ("create", "destroy", "bind", "read"):
            if not hasattr(library, f"generativeqc_resource_ledger_{name}_v1"):
                raise NotImplementedError(
                    "native library has no persistent allocation ledger v1"
                )
        library.generativeqc_resource_ledger_create_v1.argtypes = [
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        library.generativeqc_resource_ledger_create_v1.restype = ctypes.c_void_p
        library.generativeqc_resource_ledger_destroy_v1.argtypes = [ctypes.c_void_p]
        library.generativeqc_resource_ledger_destroy_v1.restype = None
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
        self.handle = library.generativeqc_resource_ledger_create_v1(
            self.limit, self.device
        )
        if not self.handle:
            raise MemoryError("could not allocate native resource ledger metadata")

    def to_dict(self) -> typing.Any:
        """Read owned capacities and, with v2, successful requested bytes.

        Older libraries retain the exact v1 shape: absent requested-byte
        evidence must not be inferred from the peak or outstanding storage.
        """
        if not self.handle:
            raise RuntimeError("native resource ledger is closed")
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
        if self._read_v2 is not None:
            result["requested_bytes"] = values[4]
        return result

    def close(self) -> None:
        """Release the observation handle; any live native buffers keep charges."""
        if self.handle:
            self.library.generativeqc_resource_ledger_destroy_v1(self.handle)
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
