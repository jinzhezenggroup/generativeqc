"""Join actual CUPTI work to source-owned roles and named payload instances.

Native boundary records describe submissions, not GPU activity or dependencies.
Only exact external-correlation joins attach them to observed copies/waits.
Unknown or unannotated work is retained; even intact annotations cannot certify
all graph/implicit waits or manufacture D2H-to-H2D host-transform links.
"""

from __future__ import annotations

import ctypes
import os
from typing import TYPE_CHECKING, Any

from tools.audit_replay_allocations import _integer, _require
from tools.cupti_graph_inventory import GraphInventory
from tools.cupti_residency_capture import summarize

if TYPE_CHECKING:
    from pathlib import Path

FIELDS = (
    "version",
    "kind",
    "operation_id",
    "execution_id",
    "owner",
    "role",
    "site",
    "payload",
    "payload_instance",
    "dependency",
    "operation_kind",
    "direction",
    "bytes",
    "status",
)
METADATA = (
    "record_count",
    "capacity",
    "dropped_records",
    "errors",
    "outstanding_operations",
    "stopped",
    "cupti_version",
    "source_version",
)
CATEGORIES = {"owner": 1, "role": 2, "site": 3, "payload": 4}
ROLES = {
    "prepare",
    "iteration",
    "tile",
    "publication",
    "oracle",
    "compatibility",
    "lifetime",
}
DIRECTIONS = {1: "h2d", 2: "d2h", 3: "d2d"}
OPERATIONS = {1: "transfer", 2: "stream_sync", 3: "event_sync"}
WORK_COUNTERS = (
    "source_transfer_calls",
    "source_zero_byte_calls",
    "source_stream_fences",
    "source_event_fences",
    "h2d_bytes",
    "d2h_bytes",
    "d2d_bytes",
    "activity_sync_records",
    "activity_event_wait_records",
)


def _add_source_work(bucket: dict[str, int], operation: dict[str, Any]) -> None:
    """Project the same joined work once, separating API calls from activity."""
    accepted = operation["status"] == 0 and operation["source_call_verified"]
    bucket["source_transfer_calls"] += int(accepted and operation["kind"] == "transfer")
    bucket["source_zero_byte_calls"] += int(
        accepted
        and operation["kind"] == "transfer"
        and not operation["requested_bytes"]
    )
    bucket["source_stream_fences"] += int(
        accepted and operation["kind"] == "stream_sync"
    )
    bucket["source_event_fences"] += int(accepted and operation["kind"] == "event_sync")
    for event in operation["events"]:
        if event["kind"] == "memcpy":
            bucket[event["direction"] + "_bytes"] += event["bytes"]
        elif event["kind"] == "sync":
            bucket[
                "activity_event_wait_records"
                if event["sync_type"] == "event_wait"
                else "activity_sync_records"
            ] += 1


class SourceInventory:
    """One native callback multiplexes graph and source observations.

    Both stores are bounded before production work begins. Read native tag names
    from the actual production DSO after detaching, never from a shadow Python
    registry. Borrowed pointers are neither recorded nor compared.
    """

    def __init__(
        self,
        path: Path,
        native: ctypes.CDLL,
        *,
        capacity: int = 1 << 18,
        node_limit: int = 4096,
        depth_limit: int = 16,
    ) -> None:
        _require(bool(os.environ.get("SLURM_JOB_ID")), "source capture requires Slurm")
        _require(
            type(capacity) is int and 1 <= capacity <= 1 << 20,
            "invalid source capacity",
        )
        self.library = ctypes.CDLL(str(path))
        self.native = native
        self.capacity = capacity
        _require(
            hasattr(native, "generativeqc_residency_boundary_name_v1"),
            "native library lacks source tag ABI",
        )
        native.generativeqc_residency_boundary_name_v1.argtypes = [ctypes.c_uint64] * 2
        native.generativeqc_residency_boundary_name_v1.restype = ctypes.c_char_p
        self.library.generativeqc_cupti_source_begin_v1.argtypes = [ctypes.c_uint64]
        self.library.generativeqc_cupti_source_begin_v1.restype = ctypes.c_int
        self.library.generativeqc_cupti_source_stop_v1.argtypes = []
        self.library.generativeqc_cupti_source_stop_v1.restype = ctypes.c_int
        self.library.generativeqc_cupti_source_read_v1.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64,
        ]
        self.library.generativeqc_cupti_source_read_v1.restype = ctypes.c_int
        _require(
            self.library.generativeqc_cupti_source_begin_v1(capacity) == 0,
            "source capture initialization failed",
        )
        self.graphs = GraphInventory(
            path,
            native,
            capacity=capacity,
            node_limit=node_limit,
            depth_limit=depth_limit,
            callback_name="generativeqc_cupti_residency_observe_v1",
        )
        self.stopped = False

    def stop(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Detach callbacks and return (graph journal, source journal), in that order."""
        _require(not self.stopped, "source inventory already stopped")
        graph_raw = self.graphs.stop()
        self.stopped = True
        _require(
            self.library.generativeqc_cupti_source_stop_v1() == 0, "source stop failed"
        )
        metadata = (ctypes.c_uint64 * len(METADATA))()
        _require(
            self.library.generativeqc_cupti_source_read_v1(
                None, 0, metadata, len(metadata)
            )
            == 0,
            "source metadata query failed",
        )
        values = dict(zip(METADATA, metadata, strict=True))
        _require(values["record_count"] <= self.capacity, "source capacity exceeded")
        storage = (ctypes.c_uint64 * (values["record_count"] * len(FIELDS)))()
        _require(
            self.library.generativeqc_cupti_source_read_v1(
                storage, values["record_count"], metadata, len(metadata)
            )
            == 0,
            "source record read failed",
        )
        _require(
            values == dict(zip(METADATA, metadata, strict=True)),
            "source history changed",
        )
        records = [
            dict(
                zip(
                    FIELDS,
                    storage[index * len(FIELDS) : (index + 1) * len(FIELDS)],
                    strict=True,
                )
            )
            for index in range(values["record_count"])
        ]
        catalog: dict[str, dict[str, str | None]] = {name: {} for name in CATEGORIES}
        for category, tag in CATEGORIES.items():
            for identity in {row[category] for row in records}:
                label = self.native.generativeqc_residency_boundary_name_v1(
                    tag, identity
                )
                catalog[category][str(identity)] = label.decode() if label else None
        return graph_raw, {
            "schema": "generativeqc.cupti-source-boundaries.v1",
            "metadata": values,
            "source_dispatch_errors": graph_raw["source_dispatch_errors"],
            "catalog": catalog,
            "records": records,
        }


def summarize_sources(
    raw: dict[str, Any], activity: dict[str, Any], regions: dict[int, dict[str, str]]
) -> dict[str, Any]:
    """Resolve two independent correlation domains, never times/bytes/addresses.

    Runtime/driver envelopes cannot both count as copies. Zero-byte API calls
    are retained as source calls but not transfer events. A successful existing
    cudaStreamSynchronize call is an explicit source fence, separate from CUPTI
    synchronization records and from any assertion about implicit/graph waits.
    """
    observed = summarize(activity, regions)
    _require(
        type(raw) is dict
        and set(raw)
        == {"schema", "metadata", "source_dispatch_errors", "catalog", "records"}
        and raw["schema"] == "generativeqc.cupti-source-boundaries.v1",
        "invalid source capture schema",
    )
    metadata = raw["metadata"]
    _require(
        type(metadata) is dict and set(metadata) == set(METADATA),
        "invalid source metadata",
    )
    for name, value in metadata.items():
        _integer(value, name)
    _integer(raw["source_dispatch_errors"], "source dispatch errors")
    _require(type(raw["records"]) is list, "invalid source records")
    _require(
        metadata["record_count"] == len(raw["records"])
        and 1 <= metadata["capacity"] <= 1 << 20
        and metadata["record_count"] <= metadata["capacity"]
        and metadata["stopped"] == 1
        and metadata["cupti_version"] == 28
        and metadata["source_version"] == 2,
        "invalid source bounds/version/stop",
    )
    catalog = raw["catalog"]
    _require(
        type(catalog) is dict and set(catalog) == set(CATEGORIES),
        "invalid source tag catalog",
    )
    for values in catalog.values():
        _require(type(values) is dict, "invalid native tag category")
        for identity, label in values.items():
            _require(
                type(identity) is str
                and identity.isdigit()
                and str(int(identity)) == identity,
                "invalid native tag ID",
            )
            _require(
                label is None or (type(label) is str and bool(label.strip())),
                "invalid native tag name",
            )
    issues = [
        f"{name}={metadata[name]}"
        for name in ("dropped_records", "errors", "outstanding_operations")
        if metadata[name]
    ]
    if raw["source_dispatch_errors"]:
        issues.append(f"source_dispatch_errors={raw['source_dispatch_errors']}")
    if not observed["activity_stream_intact"]:
        issues.append("CUPTI activity stream incomplete")
    phase_by_correlation: dict[int, int] = {}
    source_by_correlation: dict[int, int] = {}
    for row in activity["records"]:
        if row["kind"] == 39 and row["subtype"] in (3, 5):
            selected = (
                phase_by_correlation if row["subtype"] == 3 else source_by_correlation
            )
            correlation, identity = row["correlation_id"], row["external_id"]
            _require(
                correlation not in selected or selected[correlation] == identity,
                "conflicting source correlation",
            )
            selected[correlation] = identity
    apis: dict[int, list[dict[str, Any]]] = {}
    for row in activity["records"]:
        if row["kind"] in (4, 5, 48) and row["correlation_id"] in source_by_correlation:
            apis.setdefault(source_by_correlation[row["correlation_id"]], []).append(
                row
            )
    work: dict[int, list[dict[str, Any]]] = {}
    unannotated = []
    for event in observed["events"]:
        selected = {
            source_by_correlation[key]
            for key in (event["correlation_id"], event["runtime_correlation_id"])
            if key and key in source_by_correlation
        }
        _require(len(selected) <= 1, "runtime/driver source disagreement")
        if selected:
            work.setdefault(next(iter(selected)), []).append(event)
        elif (
            event["region_id"] not in regions
            or regions[event["region_id"]]["role"] != "observer"
        ):
            unannotated.append(event)
    executions: dict[int, dict[str, Any]] = {}
    pending: dict[int, dict[str, Any]] = {}
    seen: set[int] = set()
    completed: list[dict[str, Any]] = []
    for index, row in enumerate(raw["records"]):
        _require(
            type(row) is dict and set(row) == set(FIELDS),
            "invalid source record fields",
        )
        for name, value in row.items():
            _integer(value, name)
        _require(
            row["version"] == 2 and row["kind"] in (1, 2, 3, 4, 7),
            "unknown source record version/kind",
        )
        if row["kind"] == 7:
            issues.append(
                f"source observer failure at record {index}, stage={row['status']}"
            )
            continue
        execution = row["execution_id"]
        _require(execution > 0, "missing source execution identity")
        owner = catalog["owner"].get(str(row["owner"]))
        if not owner:
            issues.append(f"unknown source owner at record {index}")
        if row["kind"] == 3:
            _require(
                not row["operation_id"] and all(row[name] == 0 for name in FIELDS[5:]),
                "invalid execution definition fields",
            )
            _require(execution not in executions, "duplicate source execution")
            executions[execution] = {
                "execution_id": execution,
                "owner": owner,
                "owner_id": row["owner"],
                "closed": False,
            }
            continue
        _require(
            execution in executions and not executions[execution]["closed"],
            "missing/closed source execution",
        )
        _require(
            row["owner"] == executions[execution]["owner_id"], "source owner changed"
        )
        if row["kind"] == 4:
            _require(
                not row["operation_id"] and all(row[name] == 0 for name in FIELDS[5:]),
                "invalid execution closure fields",
            )
            if any(begin["execution_id"] == execution for begin in pending.values()):
                issues.append(
                    f"source execution closed during a submission: {execution}"
                )
            executions[execution]["closed"] = True
            continue
        operation = row["operation_id"]
        _require(
            operation > 0 and row["operation_kind"] in OPERATIONS,
            "invalid source operation identity/kind",
        )
        if row["kind"] == 1:
            _require(
                operation not in seen and row["status"] == 0,
                "duplicate source operation or begin status",
            )
            if row["operation_kind"] == 1:
                _require(
                    row["direction"] in DIRECTIONS
                    and row["payload"] > 0
                    and row["payload_instance"] > 0,
                    "missing transfer payload/direction",
                )
            else:
                _require(
                    all(
                        row[name] == 0
                        for name in (
                            "payload",
                            "payload_instance",
                            "dependency",
                            "direction",
                            "bytes",
                        )
                    ),
                    "invalid source fence fields",
                )
            seen.add(operation)
            pending[operation] = row
            continue
        _require(operation in pending, "unmatched source operation end")
        begin = pending.pop(operation)
        _require(
            all(
                row[name] == begin[name]
                for name in FIELDS
                if name not in ("kind", "status")
            ),
            "source operation identity changed",
        )
        role = catalog["role"].get(str(row["role"]))
        site = catalog["site"].get(str(row["site"]))
        payload = catalog["payload"].get(str(row["payload"]))
        if role not in ROLES or not site or not payload:
            issues.append(f"unknown source role/site/payload at operation {operation}")
        call_apis = apis.get(operation, [])
        expected_names = {
            1: {"cudaMemcpy", "cudaMemcpyAsync"},
            2: {"cudaStreamSynchronize"},
            3: {"cudaEventSynchronize"},
        }[row["operation_kind"]]
        runtime = [
            entry
            for entry in call_apis
            if entry["kind"] == 5
            and entry["api_name"]
            and entry["api_name"].split("_", 1)[0] in expected_names
        ]
        if len(runtime) != 1:
            issues.append(
                f"missing/ambiguous runtime API at source operation {operation}"
            )
        if any(
            not entry["start"] or entry["end"] < entry["start"] for entry in runtime
        ):
            issues.append(f"unfinished source API at operation {operation}")
        if row["status"] == (1 << 64) - 1:
            issues.append(f"abandoned source operation {operation}")
        elif any(entry["return_value"] != row["status"] for entry in runtime):
            issues.append(f"source API status disagreement at operation {operation}")
        phase_ids = {
            phase_by_correlation[entry["correlation_id"]]
            for entry in runtime
            if entry["correlation_id"] in phase_by_correlation
        }
        if len(phase_ids) != 1 or not phase_ids <= regions.keys():
            issues.append(f"missing/ambiguous phase at source operation {operation}")
        phase = next(iter(phase_ids)) if len(phase_ids) == 1 else None
        events = work.get(operation, [])
        if any(event["region_id"] != phase for event in events):
            issues.append(f"source/work phase disagreement at operation {operation}")
        transfers = [event for event in events if event["kind"] == "memcpy"]
        if row["operation_kind"] == 1:
            expected = row["bytes"] if row["status"] == 0 else 0
            if (
                any(
                    event["direction"] != DIRECTIONS[row["direction"]]
                    for event in transfers
                )
                or sum(event["bytes"] for event in transfers) != expected
            ):
                issues.append(
                    f"source/actual transfer disagreement at operation {operation}"
                )
            if sum(event["copy_operations"] for event in transfers) != int(
                expected > 0
            ):
                issues.append(
                    f"unqualified copy multiplicity at source operation {operation}"
                )
        elif transfers:
            issues.append(f"unexpected transfer in a source fence: {operation}")
        if row["operation_kind"] in (2, 3):
            expected_sync = 3 if row["operation_kind"] == 2 else 1
            if any(
                event["kind"] == "sync" and event["sync_subtype"] != expected_sync
                for event in events
            ):
                issues.append(
                    f"source/actual synchronization disagreement at operation {operation}"
                )
        if row["dependency"]:
            issues.append(
                f"source dependency has no qualified transform/link evidence: {operation}"
            )
        completed.append(
            {
                "operation_id": operation,
                "execution_id": execution,
                "owner": owner,
                "role": role,
                "site": site,
                "payload": payload,
                "payload_instance": row["payload_instance"],
                "dependency_identity": row["dependency"] or None,
                "region_id": phase,
                "kind": OPERATIONS[row["operation_kind"]],
                "requested_bytes": row["bytes"],
                "status": row["status"],
                "source_call_verified": len(runtime) == 1
                and runtime[0]["start"] > 0
                and runtime[0]["end"] >= runtime[0]["start"]
                and runtime[0]["return_value"] == row["status"]
                and len(phase_ids) == 1
                and phase in regions,
                "events": events,
                "risk": "higher"
                if role in ("iteration", "tile", "lifetime")
                else "explicit_nonhot_role"
                if role in ROLES
                else "unknown",
            }
        )
    if pending or any(not execution["closed"] for execution in executions.values()):
        issues.append("unclosed source operations/executions")
    if (set(apis) | set(work) | set(source_by_correlation.values())) - seen:
        issues.append("CUPTI source correlation without a source operation")
    if not executions:
        issues.append("no source executions observed")
    counters = dict.fromkeys(WORK_COUNTERS, 0)
    by_role = {role: dict(counters) for role in sorted(ROLES)}
    by_owner = {
        owner: dict(counters)
        for owner in sorted(
            {
                execution["owner"]
                for execution in executions.values()
                if execution["owner"]
            }
        )
    }
    by_phase_owner_role: dict[tuple[Any, Any, Any], dict[str, int]] = {}
    for operation in completed:
        for partition, key in (
            (by_role, operation["role"]),
            (by_owner, operation["owner"]),
        ):
            if key not in partition:
                continue
            _add_source_work(partition[key], operation)
        phase_key = (operation["region_id"], operation["owner"], operation["role"])
        _add_source_work(
            by_phase_owner_role.setdefault(phase_key, dict(counters)), operation
        )
    for bucket in (
        *by_role.values(),
        *by_owner.values(),
        *by_phase_owner_role.values(),
    ):
        for name, value in bucket.items():
            _integer(value, name)
    return {
        "status": "INCOMPLETE",
        "source_annotations_intact": not bool(issues),
        "issues": issues,
        "scope": "source-owned HF bucket/graph/resource, device-ledger and post-HF DF compatibility boundary submissions matched to observed CUPTI transfer/sync work; not full residency coverage",
        "executions": list(executions.values()),
        "operations": completed,
        "by_role": by_role,
        "by_owner": by_owner,
        "by_phase_owner_role": [
            {"region_id": phase, "owner": owner, "role": role, "counters": bucket}
            for (phase, owner, role), bucket in sorted(
                by_phase_owner_role.items(),
                key=lambda row: (row[0][0] or 0, row[0][1] or "", row[0][2] or ""),
            )
        ],
        "unannotated_production_activity": unannotated,
        "observed_production_activity_fully_annotated": not bool(unannotated)
        and not bool(issues),
        "payload_dependency_links_complete": False,
        "implicit_wait_coverage_complete": False,
        "graph_wait_coverage_complete": False,
        "zero_round_trip_assertion": None,
    }
