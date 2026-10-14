"""Reduce Nsight SQLite into separate exclusive host and device ledgers.

Export a warm capture with ``nsys export --type sqlite ...`` first. This tool
does not add API duration, kernel duration or overlapping parent intervals to
host wall time. Launch-resource metadata is retained as resource bounds, not
as measured occupancy, traffic or stall counters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def exclusive_host_intervals(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Subtract direct children from a properly nested single-thread timeline."""
    totals: Counter[str] = Counter()
    stack: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: (r["start"], -r["end"])):
        if row["end"] < row["start"]:
            raise ValueError("negative NVTX interval")
        while stack and row["start"] >= stack[-1]["end"]:
            stack.pop()
        if stack:
            if row["end"] > stack[-1]["end"]:
                raise ValueError("crossing NVTX host intervals")
            totals[stack[-1]["name"]] -= row["end"] - row["start"]
        totals[row["name"]] += row["end"] - row["start"]
        stack.append(row)
    return {name: value / 1e9 for name, value in sorted(totals.items())}


def exclusive_device_intervals(
    rows: list[dict[str, Any]], start: int, end: int
) -> dict[str, float]:
    """Partition the capture window; concurrent kernels occupy one overlap bin.

    Idle includes copies and host/synchronization gaps: it is deliberately not
    called idle hardware. Per-kernel sums remain a separate explanatory view.
    """
    changes: dict[int, Counter[str]] = defaultdict(Counter)
    changes[start]
    changes[end]
    for row in rows:
        first, last = max(start, row["start"]), min(end, row["end"])
        if first < last:
            changes[first][row["phase"]] += 1
            changes[last][row["phase"]] -= 1
    active: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    previous = start
    for timestamp, delta in sorted(changes.items()):
        count = sum(active.values())
        label = (
            "no_kernel_activity"
            if count == 0
            else "overlapping_kernels"
            if count > 1
            else next(name for name, value in active.items() if value)
        )
        totals[label] += timestamp - previous
        active.update(delta)
        if any(value < 0 for value in active.values()):
            raise ValueError("unbalanced device interval")
        previous = timestamp
    if any(active.values()):
        raise ValueError("unfinished device interval")
    return {name: value / 1e9 for name, value in sorted(totals.items())}


def summarize(path: Path) -> dict[str, Any]:
    """Resolve warm host ranges and attribute kernels by their launch API scope."""
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        names = dict(connection.execute("SELECT id,value FROM StringIds"))
        ranges = [
            {"start": start, "end": end, "name": text or names.get(text_id), "tid": tid}
            for start, end, text, text_id, tid in connection.execute(
                "SELECT start,end,text,textId,globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL"
            )
        ]
        roots = [row for row in ranges if row["name"] == "dft-profile/public_execute"]
        if len(roots) != 1:
            raise ValueError("expected exactly one complete warm public_execute range")
        root = roots[0]
        ranges = [
            row
            for row in ranges
            if row["tid"] == root["tid"]
            and root["start"] <= row["start"] <= row["end"] <= root["end"]
        ]
        if any(row["name"] is None for row in ranges):
            raise ValueError("unresolved NVTX label")
        ranges.sort(key=lambda row: row["end"] - row["start"])
        apis: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for first, tid, correlation in connection.execute(
            "SELECT start,globalTid,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"
        ):
            apis[correlation].append((first, tid))
        kernels: list[dict[str, Any]] = []
        groups: dict[str, dict[str, Any]] = {}
        unmatched = 0
        for row in connection.execute(
            "SELECT start,end,demangledName,correlationId,registersPerThread,"
            "gridX,gridY,gridZ,blockX,blockY,blockZ,staticSharedMemory,"
            "dynamicSharedMemory,localMemoryPerThread FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ):
            first, last, name_id, correlation, registers, *resources = row
            if last <= root["start"] or first >= root["end"]:
                continue
            name = names[name_id]
            candidates = [api for api in apis[correlation] if api[0] <= first]
            api = max(candidates) if candidates else None
            enclosing = (
                [
                    r
                    for r in ranges
                    if r["tid"] == api[1] and r["start"] <= api[0] < r["end"]
                ]
                if api
                else []
            )
            phase = enclosing[0]["name"] if enclosing else "unattributed_launch"
            unmatched += not enclosing
            kernels.append({"start": first, "end": last, "phase": phase})
            group = groups.setdefault(
                name,
                {
                    "name": name,
                    "launches": 0,
                    "total_ns": 0,
                    "phase_ns": Counter(),
                    "resource_bounds": set(),
                },
            )
            group["launches"] += 1
            group["total_ns"] += last - first
            group["phase_ns"][phase] += last - first
            group["resource_bounds"].add((registers, *resources))
    host = exclusive_host_intervals(ranges)
    device = exclusive_device_intervals(kernels, root["start"], root["end"])
    duration = (root["end"] - root["start"]) / 1e9
    for group in groups.values():
        group["kernel_seconds"] = group.pop("total_ns") / 1e9
        group["phase_kernel_seconds"] = {
            name: value / 1e9 for name, value in sorted(group.pop("phase_ns").items())
        }
        group["resource_bounds"] = sorted(group["resource_bounds"])
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return {
        "schema": "generativeqc.dft-profile-summary.v1",
        "diagnostic_only": True,
        "sqlite_sha256": digest.hexdigest(),
        "public_execute_seconds": duration,
        "exclusive_host_seconds": host,
        "host_reconciliation_error_seconds": duration - sum(host.values()),
        "exclusive_device_window_seconds": device,
        "device_reconciliation_error_seconds": duration - sum(device.values()),
        "unattributed_kernel_launches": unmatched,
        "resource_columns": [
            "registers_per_thread",
            "grid_x",
            "grid_y",
            "grid_z",
            "block_x",
            "block_y",
            "block_z",
            "static_shared_bytes",
            "dynamic_shared_bytes",
            "local_bytes_per_thread",
        ],
        "resource_semantics": "launch metadata; not achieved occupancy or memory traffic",
        "kernels": sorted(groups.values(), key=lambda row: -row["kernel_seconds"]),
        "host_intervals": sorted(ranges, key=lambda row: row["start"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.sqlite), indent=2) + "\n")


if __name__ == "__main__":
    main()
