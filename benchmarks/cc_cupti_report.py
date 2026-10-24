"""Versioned CUPTI attribution with separate science and endpoint evidence."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from benchmarks.cc_source_census import validate_cc_census
from benchmarks.cc_trace_reader import (
    activity_inventory,
    open_trace,
    require,
    source_inventory,
)

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = "generativeqc.cc-cupti-attribution.v1"
OBSERVER_FENCE_ID = 0x8000000000000001


def integer(value: object, name: str, *, positive: bool = False) -> int:
    """Admit exact JSON uint64 values, never bool or float identities."""
    require(type(value) is int and int(positive) <= value < 2**64, f"invalid {name}")
    return value


def exclusive_ledger(
    start: int, end: int, intervals: list[tuple[int, int, str]]
) -> dict[str, int]:
    """Partition one actual trace-clock window by sets, preserving empty gaps."""
    changes: dict[int, Counter[str]] = {start: Counter(), end: Counter()}
    for left, right, kind in intervals:
        require(
            start <= left <= right <= end,
            "activity lies outside independent caller window",
        )
        if left == right:
            continue
        changes.setdefault(left, Counter())[kind] += 1
        changes.setdefault(right, Counter())[kind] -= 1
    active: Counter[str] = Counter()
    ledger: Counter[str] = Counter()
    previous = start
    for point in sorted(changes):
        label = (
            "+".join(sorted(key for key, count in active.items() if count > 0))
            or "unattributed"
        )
        ledger[label] += point - previous
        active.update(changes[point])
        require(
            all(count >= 0 for count in active.values()),
            "negative activity sweep count",
        )
        previous = point
    require(
        sum(ledger.values()) == end - start and not any(active.values()),
        "ledger tail mismatch",
    )
    return dict(ledger)


def report(
    prefix: Path,
    semantic: dict[str, int],
    geometry: tuple[int, int, int],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Join immutable process-bound records without claiming causality/speedup."""
    require(
        receipt.get("schema") == "generativeqc.cc-cupti-caller-receipt.v1",
        "wrong receipt schema",
    )
    window = receipt["cupti_parent_window"]
    start = integer(window["start_ns"], "caller start", positive=True)
    end = integer(window["end_ns"], "caller end", positive=True)
    require(start < end, "caller window not ordered")
    require(window["clock"] == "cupti28-normalized-default-ns", "wrong caller clock")
    required_true = (
        "allocation_verified",
        "same_host_clock_verified",
        "device_clock_calibration_passed",
        "independent_science_passed",
        "original_work_gates_passed",
        "observer_fence_verified",
        "independent_tail_verified",
    )
    assertions = {key: receipt.get(key) for key in required_true}
    files = {
        kind: open_trace(prefix.with_suffix(f".{kind}.bin"))
        for kind in ("activity", "source", "graph", "legacy")
    }
    require(
        len({trace.identity[:2] for trace in files.values()}) == 4,
        "aliased trace artifacts",
    )
    for kind, trace in files.items():
        require(
            receipt["artifacts"][kind]["sha256"] == trace.sha256,
            "artifact differs from receipt",
        )
    source = source_inventory(files["source"])
    require(
        source["pid"] == integer(receipt["native_pid"], "native pid", positive=True),
        "native PID mismatch",
    )
    require(
        source["pid_starttime"]
        == integer(receipt["native_starttime"], "native start", positive=True),
        "PID reuse/source mismatch",
    )
    census = validate_cc_census(
        source, semantic, geometry, tuple(receipt["allowed_triples_lowerings"])
    )
    activity = activity_inventory(files["activity"])
    custom = integer(receipt["cupti_abi"]["custom0"], "CUSTOM0 kind")
    require(
        receipt["cupti_abi"]["concurrent_kernel"] == 10
        and receipt["cupti_abi"]["memset"] == 2,
        "CUPTI decoder ABI mismatch",
    )
    graph, legacy = files["graph"].metadata, files["legacy"].metadata
    require(
        all(graph[index] == 0 for index in (2, 3, 7)) and graph[6] == 1,
        "graph loss/error/open lifetime",
    )
    require(
        all(legacy[index] == 0 for index in (2, 3, 4)) and legacy[5:8] == (1, 28, 2),
        "legacy source loss/error/tail",
    )
    intervals: list[tuple[int, int, str]] = []
    joined: Counter[int] = Counter()
    unknown: Counter[str] = Counter()
    for row in activity["gpu"] + activity["waits"]:
        kind = (
            "kernel"
            if row[0] == 10
            else "memset"
            if row[0] == 2
            else "wait"
            if row[0] == 38
            else "copy"
        )
        intervals.append((row[3], row[4], kind))
        phase_id = activity["external"].get((custom, row[5]))
        if phase_id in source["phases"]:
            joined[source["phases"][phase_id]["words"][4]] += 1
        elif phase_id == OBSERVER_FENCE_ID and kind == "wait":
            unknown["explicit_observer_region"] += 1
        else:
            unknown["unattributed_activity"] += 1
    for phase in source["phases"].values():
        if phase["words"][4] in (5, 9, 26, 32):
            intervals.append(
                (phase["start"], phase["end"], "host_metadata_preparation")
            )
    root = source["root"]
    require(
        start <= root["start"] <= root["end"] <= end,
        "scientific component outside caller envelope",
    )
    ledger = exclusive_ledger(start, end, intervals)
    return {
        "schema": SCHEMA,
        "campaign_science": "INCOMPLETE_INDEPENDENT_RECEIPT_VALIDATION",
        "component_coverage": census,
        "whole_endpoint_attribution": "INCOMPLETE_INDEPENDENT_PRODUCER_AND_TAIL_VALIDATION",
        "record_inventory": "VALIDATED_OBSERVED_ACTIVITY_LEDGER",
        "unverified_receipt_assertions": assertions,
        "supervised_child_wall": receipt["supervised_child_wall"],
        "cupti_parent_window": window,
        "scientific_source_window": {"start_ns": root["start"], "end_ns": root["end"]},
        "exclusive_ledger_ns": ledger,
        "source_joined_activity_counts": dict(joined),
        "preserved_unknown_activity": dict(unknown),
        "global_nvtx_lossless": "NOT_APPLICABLE_CUPTI_BACKEND; historical result unchanged",
        "implicit_library_wait_coverage": "UNAVAILABLE",
        "queued_completed": activity["queued_completed"],
        "causal_hideable_time": "UNAVAILABLE",
        "performance_profitability": "UNAVAILABLE",
        "note": "Observed overlap is not dependency or speedup evidence. Empty ledger sets are unattributed, never inferred producer/idle time. Distinct clocks/boundaries are not subtracted.",
    }
