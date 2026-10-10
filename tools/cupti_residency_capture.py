"""Read actual CUPTI 12.9 Update 1 activity, without inventing payload semantics.

This optional collector is not a CUDA allocator or production dependency.
Activity records distinguish executed copies (including graph nodes) from API
construction. Correlation attaches declared phases and observer fences. Even an
intact capture remains INCOMPLETE for the strict residency receipt until source
owners independently establish scientific payload/dependency identities.
"""

from __future__ import annotations

import ctypes
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from tools.audit_replay_allocations import _integer, _require

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

FIELDS = (
    "kind",
    "subtype",
    "bytes",
    "start",
    "end",
    "correlation_id",
    "runtime_correlation_id",
    "external_id",
    "context",
    "stream",
    "device",
    "graph_node",
    "graph",
    "return_value",
    "thread",
    "copy_count",
)
METADATA = (
    "record_count",
    "capacity",
    "dropped_records",
    "buffer_starvations",
    "parse_errors",
    "unknown_records",
    "buffers_pending",
    "cupti_version",
    "stopped",
    "finish_errors",
)
KINDS = {
    1: "memcpy",
    22: "memcpy",
    38: "sync",
    39: "external",
    4: "driver",
    5: "runtime",
    48: "internal_api",
}
DIRECTIONS = {
    1: "h2d",
    2: "d2h",
    3: "h2d",
    4: "d2h",
    5: "d2d",
    6: "d2d",
    7: "d2d",
    8: "d2d",
    9: "h2h",
    10: "d2d",
}
SYNC_TYPES = {1: "event_wait", 2: "event_wait", 3: "stream", 4: "device"}
QUERY_APIS = {
    "cudaEventQuery": "event",
    "cuEventQuery": "event",
    "cudaStreamQuery": "stream",
    "cuStreamQuery": "stream",
}


class CuptiCapture:
    """One fresh-process bounded capture; GPU work must be Slurm-assigned.

    CUPTI activity ownership is process-global. Do not combine this qualification
    process with another CUPTI profiler. Storage remains process-owned after stop
    to keep callbacks safe; a second capture must use another process.
    """

    def __init__(self, path: Path, *, capacity: int = 1 << 18) -> None:
        _require(bool(os.environ.get("SLURM_JOB_ID")), "CUPTI capture requires Slurm")
        _require(
            type(capacity) is int and 1 <= capacity <= 1 << 20,
            "invalid activity capacity",
        )
        self.library = ctypes.CDLL(str(path))
        for name in ("begin", "push", "pop"):
            function = getattr(self.library, f"generativeqc_cupti_{name}_v1")
            function.argtypes, function.restype = [ctypes.c_uint64], ctypes.c_int
        self.library.generativeqc_cupti_stop_v1.argtypes = []
        self.library.generativeqc_cupti_stop_v1.restype = ctypes.c_int
        self.library.generativeqc_cupti_read_v1.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64,
        ]
        self.library.generativeqc_cupti_read_v1.restype = ctypes.c_int
        self.library.generativeqc_cupti_api_name_v1.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        self.library.generativeqc_cupti_api_name_v1.restype = ctypes.c_char_p
        self.stopped = False
        self.capacity = capacity
        self._call("begin", capacity)

    def _call(self, name: str, *arguments: int) -> None:
        status = getattr(self.library, f"generativeqc_cupti_{name}_v1")(*arguments)
        _require(status == 0, f"CUPTI {name} failed: {status}")

    @contextmanager
    def region(self, region_id: int) -> Iterator[None]:
        """Attach a phase on this thread; nested observer IDs remain distinct."""
        _require(not self.stopped, "CUPTI capture has stopped")
        _integer(region_id, "region ID")
        self._call("push", region_id)
        try:
            yield
        finally:
            self._call("pop", region_id)

    def stop(self) -> dict[str, Any]:
        """Read only after an independently established CUDA completion fence."""
        _require(not self.stopped, "CUPTI capture already stopped")
        self._call("stop")
        self.stopped = True
        metadata = (ctypes.c_uint64 * len(METADATA))()
        status = self.library.generativeqc_cupti_read_v1(
            None, 0, metadata, len(metadata)
        )
        _require(status == 0, "CUPTI metadata query failed")
        values = dict(zip(METADATA, metadata, strict=True))
        _require(
            values["record_count"] <= self.capacity,
            "CUPTI record count exceeds capacity",
        )
        storage = (ctypes.c_uint64 * (values["record_count"] * len(FIELDS)))()
        status = self.library.generativeqc_cupti_read_v1(
            storage,
            values["record_count"],
            metadata,
            len(metadata),
        )
        _require(status == 0, "CUPTI record read failed")
        _require(
            values == dict(zip(METADATA, metadata, strict=True)),
            "CUPTI history changed after stop",
        )
        records = []
        for index in range(values["record_count"]):
            offset = index * len(FIELDS)
            row = dict(zip(FIELDS, storage[offset : offset + len(FIELDS)], strict=True))
            row["api_name"] = None
            if row["kind"] in (4, 5, 48):
                name = self.library.generativeqc_cupti_api_name_v1(
                    row["kind"], row["subtype"]
                )
                row["api_name"] = name.decode() if name else None
            records.append(row)
        return {
            "schema": "generativeqc.cupti-activity.v1",
            "metadata": values,
            "records": records,
        }


def summarize(
    raw: dict[str, Any], regions: dict[int, dict[str, str]]
) -> dict[str, Any]:
    """Count actual activity once, not both enclosing runtime and driver calls.

    CUPTI buffers arrive out of order. Resolve external IDs from the complete
    stream before attributing work; never infer an unknown phase from timestamps
    or bytes. Raw correlation/graph IDs are provenance, not scientific payload
    identities and not evidence that any D2H/H2D pair is a host round trip.
    """
    _require(
        type(raw) is dict and set(raw) == {"schema", "metadata", "records"},
        "invalid activity snapshot",
    )
    _require(type(regions) is dict and bool(regions), "missing declared phases")
    names = set()
    for region_id, region in regions.items():
        _integer(region_id, "region ID")
        _require(
            type(region) is dict and set(region) == {"name", "role"},
            "invalid phase declaration",
        )
        _require(
            type(region["name"]) is str
            and bool(region["name"].strip())
            and region["name"] not in names,
            "duplicate/empty phase name",
        )
        _require(
            type(region["role"]) is str
            and region["role"]
            in {
                "prepare",
                "endpoint",
                "replay",
                "geometry_rebuild",
                "publication",
                "observer",
                "iteration",
                "tile",
                "oracle",
                "compatibility",
            },
            "unknown phase role",
        )
        names.add(region["name"])
    _require(
        raw["schema"] == "generativeqc.cupti-activity.v1", "unsupported activity schema"
    )
    metadata = raw["metadata"]
    _require(
        type(metadata) is dict and set(metadata) == set(METADATA),
        "invalid activity metadata",
    )
    _require(type(raw["records"]) is list, "invalid activity record list")
    for name, value in metadata.items():
        _integer(value, name)
    _require(
        metadata["cupti_version"] == 28 and metadata["stopped"] == 1,
        "unqualified/unfinished CUPTI capture",
    )
    _require(
        metadata["record_count"] == len(raw["records"]),
        "activity record count mismatch",
    )
    _require(
        1 <= metadata["capacity"] <= 1 << 20
        and metadata["record_count"] <= metadata["capacity"],
        "invalid activity record bound",
    )
    issues = [
        f"{name}={metadata[name]}"
        for name in (
            "dropped_records",
            "buffer_starvations",
            "parse_errors",
            "unknown_records",
            "buffers_pending",
            "finish_errors",
        )
        if metadata[name]
    ]
    correlations: dict[int, int] = {}
    api_names: dict[int, set[str]] = {}
    for row in raw["records"]:
        _require(
            set(row) == set(FIELDS) | {"api_name"}, "invalid activity record fields"
        )
        for name in FIELDS:
            _integer(row[name], name)
        _require(
            row["api_name"] is None or type(row["api_name"]) is str, "invalid API name"
        )
        _require(row["kind"] in KINDS, "unsupported activity record kind")
        if row["kind"] == 39:
            _require(
                row["subtype"] in (3, 4, 5), "unexpected external correlation domain"
            )
            if row["subtype"] in (4, 5):
                continue
            correlation = row["correlation_id"]
            selected = row["external_id"]
            _require(
                correlation not in correlations
                or correlations[correlation] == selected,
                "conflicting phase correlation",
            )
            correlations[correlation] = selected
        elif row["kind"] in (4, 5, 48) and row["api_name"]:
            api_names.setdefault(row["correlation_id"], set()).add(
                row["api_name"].split("_", 1)[0]
            )
    counters = {
        "transfer_records": 0,
        "copy_operations": 0,
        "h2d_bytes": 0,
        "d2h_bytes": 0,
        "d2d_bytes": 0,
        "h2h_bytes": 0,
        "syncs": 0,
        "event_waits": 0,
        "event_queries": 0,
        "stream_queries": 0,
        "graph_transfer_records": 0,
    }
    totals = {
        key: {"region": dict(value), **counters} for key, value in regions.items()
    }
    unassigned = {"region": {"name": "unassigned", "role": "unknown"}, **counters}
    events = []
    for index, row in enumerate(raw["records"]):
        if row["kind"] not in (1, 22, 38):
            continue
        selected = {
            correlations[key]
            for key in (row["correlation_id"], row["runtime_correlation_id"])
            if key and key in correlations
        }
        _require(len(selected) <= 1, "runtime/driver phase disagreement")
        region_id = next(iter(selected), None)
        bucket = totals.get(region_id, unassigned)
        if bucket is unassigned:
            issues.append(f"unassigned activity {index}")
        if not row["start"] or row["end"] < row["start"]:
            issues.append(f"unfinished/unordered timestamps at activity {index}")
        event = {
            "record_index": index,
            "region_id": region_id,
            "kind": KINDS[row["kind"]],
            "start": row["start"],
            "end": row["end"],
            "correlation_id": row["correlation_id"],
            "runtime_correlation_id": row["runtime_correlation_id"],
            "graph_node": row["graph_node"],
            "graph": row["graph"],
            "context": row["context"],
            "stream": row["stream"],
            "payload_identity": None,
            "dependency_identity": None,
        }
        if row["kind"] == 38:
            _require(row["subtype"] in SYNC_TYPES, "unknown synchronization kind")
            event["sync_type"] = SYNC_TYPES[row["subtype"]]
            event["sync_subtype"] = row["subtype"]
            event["return_value"] = row["return_value"]
            names = api_names.get(row["correlation_id"], set())
            event["api_names"] = sorted(names)
            queries = {QUERY_APIS[name] for name in names if name in QUERY_APIS}
            if queries:
                _require(
                    len(queries) == 1 and all(name in QUERY_APIS for name in names),
                    "query/blocking API correlation conflict",
                )
                query = next(iter(queries))
                _require(
                    row["subtype"] == (1 if query == "event" else 3),
                    "query activity kind mismatch",
                )
                event.update(kind="query", query_type=query, sync_type=None)
                bucket[query + "_queries"] += 1
                if row["return_value"] not in (0, 600):
                    issues.append(f"failed query at activity {index}")
            elif row["subtype"] in (1, 3) and not names:
                event["sync_type"] = None
                issues.append(
                    f"missing synchronization API identity at activity {index}"
                )
            elif row["return_value"]:
                issues.append(f"failed synchronization at activity {index}")
            else:
                bucket[
                    "event_waits" if event["sync_type"] == "event_wait" else "syncs"
                ] += 1
        else:
            _require(
                row["subtype"] in DIRECTIONS and row["bytes"] > 0,
                "unknown/empty transfer kind",
            )
            event["direction"] = DIRECTIONS[row["subtype"]]
            event["bytes"] = row["bytes"]
            event["copy_operations"] = row["copy_count"]
            if not row["copy_count"]:
                issues.append(f"unknown copy multiplicity at activity {index}")
            bucket["transfer_records"] += 1
            bucket["copy_operations"] += row["copy_count"]
            bucket[event["direction"] + "_bytes"] += row["bytes"]
            bucket["graph_transfer_records"] += int(bool(row["graph_node"]))
        events.append(event)
    for bucket in [*totals.values(), unassigned]:
        for name in counters:
            _integer(bucket[name], name)
    return {
        "status": "INCOMPLETE",
        "reason": "activity counters do not establish source-owned roles/payload/dependencies or all graph/implicit waits",
        "scope": "observed CUPTI memcpy/synchronization activity; not a complete residency receipt",
        "role_scope": "public lifecycle intervals; internal publication/iteration roles unqualified",
        "wait_scope": "correlated synchronization activity, excluding nonblocking queries; graph waits not complete",
        "activity_stream_intact": not bool(issues),
        "issues": issues,
        "regions": list(totals.values()),
        "unassigned": unassigned,
        "events": events,
        "payload_links_complete": False,
        "residency_coverage_complete": False,
        "zero_round_trip_assertion": None,
    }
