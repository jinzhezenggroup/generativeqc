"""Bounded source-owned graph definitions and correlated host submissions.

CUDA graph wait nodes need not produce CUPTI synchronization records. Inspect
the actual immutable graph at its production owner, independently of activity
collection. Static nodes and accepted host launches are not execution counts,
especially for device launches, conditionals and opaque host callbacks.
"""

from __future__ import annotations

import ctypes
import os
from typing import TYPE_CHECKING, Any

from tools.audit_replay_allocations import _integer, _require
from tools.cupti_residency_capture import summarize

if TYPE_CHECKING:
    from pathlib import Path

FIELDS = (
    "kind",
    "generation",
    "role",
    "flags",
    "graph",
    "executable",
    "node",
    "node_type",
    "parent_node",
    "launch_id",
    "status",
    "detail",
)
METADATA = (
    "record_count",
    "capacity",
    "dropped_records",
    "errors",
    "node_limit",
    "depth_limit",
    "stopped",
    "outstanding_launches",
)
NODE_TYPES = {
    0: "kernel",
    1: "memcpy",
    2: "memset",
    3: "host_callback",
    4: "child_graph",
    5: "empty",
    6: "event_wait",
    7: "event_record",
    8: "semaphore_signal",
    9: "semaphore_wait",
    10: "allocation",
    11: "free",
    13: "conditional",
}
ROLES = {1: "hf_iteration", 2: "hf_post_eigensolver"}
OBSERVER = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_void_p
)


class GraphInventory:
    """Bind native inspection only on this production submitting thread.

    The callback is a native function pointer, not a Python trampoline. Keep both
    DSOs and its ctypes wrapper alive until detachment; no graph handle escapes
    into Python or survives an owner callback. One fresh process per inventory.
    """

    def __init__(
        self,
        path: Path,
        native: ctypes.CDLL,
        *,
        capacity: int = 1 << 18,
        node_limit: int = 4096,
        depth_limit: int = 16,
        callback_name: str = "generativeqc_cupti_graph_observe_v1",
    ) -> None:
        _require(bool(os.environ.get("SLURM_JOB_ID")), "graph inventory requires Slurm")
        for value, limit, label in (
            (capacity, 1 << 20, "capacity"),
            (node_limit, 4096, "node limit"),
            (depth_limit, 16, "depth limit"),
        ):
            _require(
                type(value) is int and 1 <= value <= limit, f"invalid graph {label}"
            )
        self.library = ctypes.CDLL(str(path))
        self.native = native
        _require(
            all(
                hasattr(native, f"generativeqc_residency_observer_{name}_v1")
                for name in ("bind", "unbind")
            ),
            "native library lacks source residency observer ABI",
        )
        self.callback = OBSERVER((callback_name, self.library))
        self.library.generativeqc_cupti_graph_begin_v1.argtypes = [ctypes.c_uint64] * 3
        self.library.generativeqc_cupti_graph_begin_v1.restype = ctypes.c_int
        self.library.generativeqc_cupti_graph_stop_v1.argtypes = []
        self.library.generativeqc_cupti_graph_stop_v1.restype = ctypes.c_int
        self.library.generativeqc_cupti_graph_read_v1.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64,
        ]
        self.library.generativeqc_cupti_graph_read_v1.restype = ctypes.c_int
        self.native.generativeqc_residency_observer_bind_v1.argtypes = [
            OBSERVER,
            ctypes.c_void_p,
        ]
        self.native.generativeqc_residency_observer_bind_v1.restype = ctypes.c_int
        self.native.generativeqc_residency_observer_unbind_v1.argtypes = [
            OBSERVER,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.native.generativeqc_residency_observer_unbind_v1.restype = ctypes.c_int
        _require(
            self.library.generativeqc_cupti_graph_begin_v1(
                capacity, node_limit, depth_limit
            )
            == 0,
            "graph inventory initialization failed",
        )
        _require(
            self.native.generativeqc_residency_observer_bind_v1(self.callback, None)
            == 0,
            "source graph observer already bound/unavailable",
        )
        self.stopped = False
        self.capacity = capacity

    def stop(self) -> dict[str, Any]:
        """Detach before snapshot, after source owners and CUDA work are closed."""
        _require(not self.stopped, "graph inventory already stopped")
        errors = ctypes.c_uint64()
        _require(
            self.native.generativeqc_residency_observer_unbind_v1(
                self.callback, None, ctypes.byref(errors)
            )
            == 0,
            "source graph observer detach failed",
        )
        self.stopped = True
        _require(
            self.library.generativeqc_cupti_graph_stop_v1() == 0, "graph stop failed"
        )
        metadata = (ctypes.c_uint64 * len(METADATA))()
        _require(
            self.library.generativeqc_cupti_graph_read_v1(
                None, 0, metadata, len(metadata)
            )
            == 0,
            "graph metadata query failed",
        )
        values = dict(zip(METADATA, metadata, strict=True))
        _require(
            values["record_count"] <= self.capacity, "graph record capacity exceeded"
        )
        storage = (ctypes.c_uint64 * (values["record_count"] * len(FIELDS)))()
        _require(
            self.library.generativeqc_cupti_graph_read_v1(
                storage, values["record_count"], metadata, len(metadata)
            )
            == 0,
            "graph record read failed",
        )
        _require(
            values == dict(zip(METADATA, metadata, strict=True)),
            "graph history changed",
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
        return {
            "schema": "generativeqc.cupti-graph-inventory.v1",
            "metadata": values,
            "source_dispatch_errors": errors.value,
            "records": records,
        }


def summarize_graphs(
    raw: dict[str, Any], activity: dict[str, Any], regions: dict[int, dict[str, str]]
) -> dict[str, Any]:
    """Join lifetimes/submissions via source generation and CUSTOM1 IDs.

    Never multiply static wait nodes by submissions and call that an observed
    execution count. Missing definitions, launch API correlation, node types or
    lifecycle events fail closed, independently of CUPTI activity-buffer loss.
    """
    activity_summary = summarize(activity, regions)
    _require(
        type(raw) is dict
        and set(raw) == {"schema", "metadata", "source_dispatch_errors", "records"}
        and raw["schema"] == "generativeqc.cupti-graph-inventory.v1",
        "invalid graph inventory schema",
    )
    metadata = raw["metadata"]
    _require(
        type(metadata) is dict and set(metadata) == set(METADATA),
        "invalid graph metadata",
    )
    for name, value in metadata.items():
        _integer(value, name)
    _integer(raw["source_dispatch_errors"], "source dispatch errors")
    _require(type(raw["records"]) is list, "invalid graph records")
    _require(
        metadata["record_count"] == len(raw["records"]), "graph record count mismatch"
    )
    _require(
        1 <= metadata["capacity"] <= 1 << 20
        and metadata["record_count"] <= metadata["capacity"]
        and 1 <= metadata["node_limit"] <= 4096
        and 1 <= metadata["depth_limit"] <= 16
        and metadata["stopped"] == 1,
        "invalid graph bounds/stop",
    )
    issues = [
        f"{name}={metadata[name]}"
        for name in ("dropped_records", "errors", "outstanding_launches")
        if metadata[name]
    ]
    if raw["source_dispatch_errors"]:
        issues.append(f"source_dispatch_errors={raw['source_dispatch_errors']}")
    if not activity_summary["activity_stream_intact"]:
        issues.append("CUPTI activity stream incomplete")
    phases: dict[int, int] = {}
    launches: dict[int, int] = {}
    for row in activity["records"]:
        if row["kind"] != 39:
            continue
        if row["subtype"] == 5:
            continue
        selected = phases if row["subtype"] == 3 else launches
        correlation, identity = row["correlation_id"], row["external_id"]
        _require(
            correlation not in selected or selected[correlation] == identity,
            "conflicting graph correlation",
        )
        selected[correlation] = identity
    api_by_launch: dict[int, list[dict[str, Any]]] = {}
    for row in activity["records"]:
        if row["kind"] in (4, 5, 48) and row["correlation_id"] in launches:
            api_by_launch.setdefault(launches[row["correlation_id"]], []).append(row)
    graphs: dict[int, dict[str, Any]] = {}
    pending: dict[int, dict[str, Any]] = {}
    submissions = []
    seen_launches: set[int] = set()
    for index, row in enumerate(raw["records"]):
        _require(
            type(row) is dict and set(row) == set(FIELDS), "invalid graph record fields"
        )
        for name, value in row.items():
            _integer(value, name)
        kind, generation = row["kind"], row["generation"]
        _require(kind in (1, 2, 3, 4, 5, 6), "unknown graph record kind")
        if kind == 6:
            issues.append(
                f"source inspection failure at record {index}, stage={row['detail']}"
            )
            continue
        _require(
            generation > 0 and row["graph"] > 0 and row["executable"] > 0,
            "missing graph identity",
        )
        _require(row["role"] in ROLES, "unknown source graph role")
        _require(row["detail"] == 0, "unexpected graph record detail")
        if kind != 2:
            _require(
                not row["node"] and not row["node_type"] and not row["parent_node"],
                "unexpected graph node fields",
            )
        if kind not in (3, 4):
            _require(
                row["launch_id"] == 0 and row["status"] == 0,
                "unexpected graph submission fields",
            )
        if kind == 3:
            _require(row["status"] == 0, "unexpected graph launch begin status")
        if kind == 1:
            _require(generation not in graphs, "duplicate graph generation")
            graphs[generation] = {
                "generation": generation,
                "role": ROLES[row["role"]],
                "flags": row["flags"],
                "graph": row["graph"],
                "executable": row["executable"],
                "nodes": [],
                "destroyed": False,
                "accepted_host_submissions": 0,
            }
            continue
        if generation not in graphs:
            issues.append(f"missing source definition for generation {generation}")
            continue
        graph = graphs[generation]
        _require(not graph["destroyed"], "graph activity after destruction")
        _require(
            row["executable"] == graph["executable"]
            and ROLES[row["role"]] == graph["role"],
            "source graph identity changed",
        )
        if kind == 2:
            _require(row["node"] > 0, "missing graph node identity")
            _require(
                not any(node["node"] == row["node"] for node in graph["nodes"]),
                "duplicate graph node identity",
            )
            if row["node_type"] not in NODE_TYPES:
                issues.append(f"unknown graph node type {row['node_type']}")
            graph["nodes"].append(
                {
                    "node": row["node"],
                    "graph": row["graph"],
                    "type": NODE_TYPES.get(row["node_type"], "unknown"),
                    "parent_node": row["parent_node"],
                }
            )
            if len(graph["nodes"]) > metadata["node_limit"]:
                issues.append(f"node bound exceeded for generation {generation}")
            continue
        _require(row["graph"] == graph["graph"], "source root graph identity changed")
        if kind == 5:
            graph["destroyed"] = True
            if any(begin["generation"] == generation for begin in pending.values()):
                issues.append(f"destroyed during host submission: {generation}")
        elif kind == 3:
            launch = row["launch_id"]
            _require(
                launch > 0 and launch not in seen_launches,
                "duplicate/empty graph launch ID",
            )
            _require(row["flags"] == graph["flags"], "graph launch flags changed")
            seen_launches.add(launch)
            pending[launch] = row
        else:
            launch = row["launch_id"]
            _require(
                launch in pending and pending[launch]["generation"] == generation,
                "unmatched graph launch end",
            )
            _require(
                row["flags"] == pending[launch]["flags"], "graph launch flags changed"
            )
            pending.pop(launch)
            apis = [
                entry
                for entry in api_by_launch.get(launch, [])
                if entry["api_name"]
                and entry["api_name"].split("_", 1)[0]
                in ("cudaGraphLaunch", "cuGraphLaunch")
            ]
            if not apis:
                issues.append(
                    f"missing CUPTI graph launch API for source submission {launch}"
                )
            for api_kind in (4, 5, 48):
                if sum(entry["kind"] == api_kind for entry in apis) > 1:
                    issues.append(
                        f"ambiguous CUPTI launch multiplicity for submission {launch}"
                    )
            if any(
                not entry["start"] or entry["end"] < entry["start"] for entry in apis
            ):
                issues.append(f"unfinished CUPTI launch API for submission {launch}")
            if any(
                bool(entry["return_value"]) != bool(row["status"]) for entry in apis
            ):
                issues.append(
                    f"graph launch status disagreement for submission {launch}"
                )
            phase_ids = {
                phases[entry["correlation_id"]]
                for entry in apis
                if entry["correlation_id"] in phases
            }
            if len(phase_ids) != 1 or not phase_ids <= regions.keys():
                issues.append(f"missing/ambiguous phase for graph submission {launch}")
            phase = next(iter(phase_ids)) if len(phase_ids) == 1 else None
            graph["accepted_host_submissions"] += int(row["status"] == 0)
            submissions.append(
                {
                    "launch_id": launch,
                    "generation": generation,
                    "status": row["status"],
                    "region_id": phase,
                    "api_records": [entry["correlation_id"] for entry in apis],
                }
            )
    if pending:
        issues.append("unfinished source host submissions")
    if set(api_by_launch) - seen_launches:
        issues.append("CUPTI graph correlation without source submission")
    if not graphs:
        issues.append("no source-owned graph definitions observed")
    for graph in graphs.values():
        nodes = {node["node"]: node for node in graph["nodes"]}
        for node in nodes.values():
            parent = node["parent_node"]
            if parent and (
                parent not in nodes or nodes[parent]["type"] != "child_graph"
            ):
                issues.append(f"unmatched child graph ancestry for node {node['node']}")
            if not parent and node["graph"] != graph["graph"]:
                issues.append(f"unmatched root graph for node {node['node']}")
        if not graph["destroyed"]:
            issues.append(f"graph lifetime not closed: {graph['generation']}")
        if graph["flags"] not in (0, 4):
            issues.append(f"unknown graph instantiation flags: {graph['flags']}")
        graph["declared_event_wait_nodes"] = sum(
            node["type"] == "event_wait" for node in nodes.values()
        )
        graph["declared_semaphore_wait_nodes"] = sum(
            node["type"] == "semaphore_wait" for node in nodes.values()
        )
        graph["dynamic_execution_unqualified"] = bool(graph["flags"]) or any(
            node["type"] in ("conditional", "host_callback", "unknown")
            for node in nodes.values()
        )
    return {
        "status": "INCOMPLETE",
        "source_inventory_intact": not bool(issues),
        "issues": issues,
        "scope": "source-owned RHF/UHF graph lifetimes and correlated host submissions, not graph execution counts",
        "graphs": list(graphs.values()),
        "host_submissions": submissions,
        "graph_wait_execution_counts": None,
        "graph_wait_coverage_complete": False,
        "implicit_wait_coverage_complete": False,
        "zero_round_trip_assertion": None,
    }
