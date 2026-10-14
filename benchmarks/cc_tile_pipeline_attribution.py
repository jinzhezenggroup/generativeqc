"""Fail-closed interval accounting for a complete CC energy diagnostic capture.

All timestamps are integer ns on one trace clock. Host ranges and device
activities are explanatory views, never additive phase timers. The exclusive
ledger partitions wall by *sets* of observed activities; its empty set is
unattributed wall, not a measured producer or device idle stall. This tool does
not implement a CC algorithm or establish causal critical-path savings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

SCHEMA = "generativeqc.cc-tile-attribution.v1"
KINDS = ("producer", "kernel", "copy", "memset", "wait")
IDENTITY = (
    "source_commit",
    "library_sha256",
    "executable_sha256",
    "input_sha256",
    "environment_sha256",
    "device_uuid",
    "run_id",
)


def checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_nsys(database: Path, receipt: dict) -> dict:
    """Read an explicitly bound Nsight CUDA/NVTX SQLite export read-only.

    No table means unavailable, not zero activity. The caller must retain the
    complete capture/export logs and inspect their warnings before asserting
    coverage. CUDA API waits are retained as host API observations; without
    device event dependency records, full wait attribution remains unavailable.
    NVTX producer ranges require exact labels and thread IDs from the capture
    recipe. Never infer producer CPU work from the spaces between GPU kernels.
    """
    bound = receipt["nsys"]
    if checksum(database) != bound.get(
        "sqlite_sha256"
    ) or database.stat().st_size != bound.get("sqlite_size"):
        raise ValueError("SQLite export does not match independent receipt")
    pid = integer(bound.get("cuda_global_pid"), "CUDA global pid")
    device = integer(bound.get("device_id"), "device id")
    tids = bound.get("runtime_global_tids")
    if not isinstance(tids, list) or not tids or len(set(tids)) != len(tids):
        raise ValueError("explicit unique runtime thread selection required")
    for tid in tids:
        integer(tid, "runtime tid")
    # Field contracts are checked before reads; an unfamiliar export fails.
    events = []
    coverage = dict.fromkeys(KINDS, False)
    rows_read = {}
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {
            r[0]
            for r in db.execute("select name from sqlite_master where type='table'")
        }

        def rows(table: str, required: list[str]) -> sqlite3.Cursor | None:
            if table not in tables:
                return None
            columns = {r[1] for r in db.execute(f'pragma table_info("{table}")')}
            if not set(required) <= columns:
                raise ValueError(f"unsupported Nsight columns in {table}")
            # Only literal table names supplied by this adapter reach this helper.
            return db.execute(f'select rowid as attribution_rowid,* from "{table}"')  # noqa: S608

        start, end = receipt["endpoint"]["start_ns"], receipt["endpoint"]["end_ns"]
        signatures = set()

        def add(table: str, row: sqlite3.Row, kind: str, **extra: Any) -> None:
            row = dict(row)
            a, b = row["start"], row["end"]
            signature = (
                table,
                a,
                b,
                row["streamId"] if "streamId" in row else row["globalTid"],
                row.get("correlationId", None),
            )
            if signature in signatures:
                raise ValueError("duplicate Nsight activity")
            signatures.add(signature)
            item = dict(
                id=f"{table}:{row['attribution_rowid']}",
                kind=kind,
                start_ns=a,
                end_ns=b,
                **extra,
            )
            interval(item, start, end)
            events.append(item)

        for table, kind in (
            ("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
            ("CUPTI_ACTIVITY_KIND_MEMCPY", "copy"),
            ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
        ):
            required = [
                "start",
                "end",
                "globalPid",
                "deviceId",
                "streamId",
                "correlationId",
            ]
            if kind == "copy":
                required += ["bytes", "copyKind"]
            records = rows(table, required)
            if records is None:
                continue
            rows_read[table] = 0
            coverage[kind] = True
            for row in records:
                if row["globalPid"] != pid or row["deviceId"] != device:
                    continue
                extra = {"stream": row["streamId"]}
                if kind == "copy":
                    direction = {1: "H2D", 2: "D2H", 8: "D2D"}.get(row["copyKind"])
                    if direction is None:
                        raise ValueError("unsupported Nsight copy kind")
                    extra.update(direction=direction, bytes=row["bytes"])
                add(table, row, kind, **extra)
                rows_read[table] += 1
        strings_rows = rows("StringIds", ["id", "value"])
        strings = (
            {r["id"]: r["value"] for r in strings_rows}
            if strings_rows is not None
            else {}
        )
        records = rows(
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            ["start", "end", "globalTid", "correlationId", "nameId"],
        )
        host_waits = {
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "cudaEventSynchronize",
            "cudaStreamWaitEvent",
        }
        if records is not None:
            for row in records:
                if row["globalTid"] in tids:
                    name = strings.get(row["nameId"])
                    if name is None:
                        raise ValueError("unresolved runtime API name")
                    if name in host_waits:
                        add(
                            "CUPTI_ACTIVITY_KIND_RUNTIME",
                            row,
                            "wait",
                            api=name,
                            clock_role="host_api",
                        )
        producer_labels = bound.get("producer_labels", [])
        records = rows("NVTX_EVENTS", ["start", "end", "globalTid", "text", "textId"])
        if records is not None:
            endpoint_ranges = []
            for row in records:
                label = row["text"] or strings.get(row["textId"])
                if label == bound.get("endpoint_label") and row[
                    "globalTid"
                ] == bound.get("endpoint_global_tid"):
                    endpoint_ranges.append((row["start"], row["end"]))
                if label in producer_labels and row["globalTid"] in tids:
                    add("NVTX_EVENTS", row, "producer", label=label)
            if endpoint_ranges != [(start, end)]:
                raise ValueError(
                    "missing, duplicate or unmatched complete endpoint NVTX range"
                )
            # Explicit recipe assertion is needed to claim *complete* producers.
            coverage["producer"] = (
                bool(producer_labels)
                and bound.get("producer_coverage_complete") is True
            )
        else:
            raise ValueError("complete process endpoint NVTX range unavailable")
    if not any(row["kind"] == "kernel" for row in events):
        raise ValueError("selected CUDA process/device has no executed kernels")
    trace = {
        "schema": SCHEMA,
        "identity": receipt["identity"],
        "endpoint": receipt["endpoint"],
        "capture": receipt["capture"],
        "coverage": coverage,
        "events": events,
    }
    trace["extraction"] = {
        "adapter": "nsys-cuda-nvtx-columns-v1",
        "rows": rows_read,
        "sqlite_sha256": checksum(database),
        "wait_scope": "host API duration only; stream wait enqueue is not device wait time",
    }
    return trace


def integer(value: Any, label: str) -> int:
    """Reject rounded timestamps, bools, missing values and negative counters."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def interval(row: dict, start: int, end: int) -> tuple[int, int]:
    a = integer(row.get("start_ns"), "start_ns")
    b = integer(row.get("end_ns"), "end_ns")
    if not start <= a < b <= end:
        raise ValueError("interval reversed, empty or outside complete endpoint")
    return a, b


def validate_identity(actual: dict, expected: dict) -> None:
    """Compare an independently retained receipt, not self-reported labels."""
    for key in IDENTITY:
        value = actual.get(key)
        if not isinstance(value, str) or not value or value != expected.get(key):
            raise ValueError(f"missing or mismatched identity: {key}")
        if key.endswith("sha256") and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"invalid SHA256: {key}")
        if key == "source_commit" and not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("invalid source commit")


def partition(events: list[dict], start: int, end: int) -> list[dict]:
    """Sweep interval endpoints in O(n log n), retaining cross-stream unions.

    Counts rather than booleans prevent a nested/overlapping range ending from
    clearing an outer range. Half-open ranges touching share no duration.
    """
    changes: dict[int, Counter] = defaultdict(Counter)
    changes[start]
    changes[end]
    for row in events:
        a, b = interval(row, start, end)
        changes[a][row["kind"]] += 1
        changes[b][row["kind"]] -= 1
    active: Counter = Counter()
    previous = start
    result = []
    for tick in sorted(changes):
        if tick > previous:
            kinds = sorted(k for k, count in active.items() if count > 0)
            if result and result[-1]["kinds"] == kinds:
                result[-1]["end_ns"] = tick
            else:
                result.append({"start_ns": previous, "end_ns": tick, "kinds": kinds})
        active.update(changes[tick])
        previous = tick
    return result


def summarize(trace: dict, receipt: dict) -> dict:
    """Accept complete or explicitly incomplete traces; never fill missing data.

    A complete capture requires successful final drain and export with no lost
    records. Missing tail/failure may still yield a partial diagnostic ledger,
    but never an endpoint attribution PASS or a hiding recommendation.
    """
    if trace.get("schema") != SCHEMA or receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported trace/receipt schema")
    validate_identity(trace.get("identity", {}), receipt.get("identity", {}))
    window = trace["endpoint"]
    if window != receipt.get("endpoint"):
        raise ValueError("endpoint boundary differs from independent receipt")
    start = integer(window.get("start_ns"), "endpoint start")
    end = integer(window.get("end_ns"), "endpoint end")
    if end <= start:
        raise ValueError("invalid endpoint window")
    wall = integer(window.get("wall_ns"), "endpoint wall")
    if wall != end - start:
        raise ValueError("endpoint clock/wall does not reconcile")
    coverage = trace.get("coverage", {})
    if set(coverage) != set(KINDS) or any(
        type(v) is not bool for v in coverage.values()
    ):
        raise ValueError("explicit Boolean coverage required for every kind")
    capture = trace.get("capture", {})
    for key in ("complete_tail", "drained", "export_complete", "lossless"):
        if type(capture.get(key)) is not bool:
            raise ValueError(f"explicit capture flag required: {key}")
    if capture.get("outcome") not in ("success", "failure"):
        raise ValueError("explicit success/failure outcome required")
    if (
        not isinstance(capture.get("profiler_version"), str)
        or not capture["profiler_version"]
    ):
        raise ValueError("profiler version required")
    events = trace["events"]
    seen = set()
    transfer = Counter()
    counts = Counter()
    sums = Counter()
    for row in events:
        event_id = row.get("id")
        if not isinstance(event_id, str) or not event_id or event_id in seen:
            raise ValueError("missing or duplicate event identity")
        seen.add(event_id)
        kind = row.get("kind")
        if kind not in KINDS:
            raise ValueError("unsupported event kind")
        a, b = interval(row, start, end)
        counts[kind] += 1
        sums[kind] += b - a
        if kind in ("kernel", "copy", "memset"):
            integer(row.get("stream"), "stream")
        if kind == "copy":
            direction = row.get("direction")
            if direction not in ("H2D", "D2H", "D2D"):
                raise ValueError("unsupported copy direction; do not guess bytes")
            transfer[direction] += integer(row.get("bytes"), "copy bytes")
    segments = partition(events, start, end)
    exclusive: Counter = Counter()
    unions = Counter()
    overlaps = Counter()
    for key in ("kernel+copy", "producer+kernel", "wait+kernel"):
        overlaps[key] = 0
    gpu_gap = producer_gap = wait_gap = 0
    for row in segments:
        duration = row["end_ns"] - row["start_ns"]
        kinds = set(row["kinds"])
        exclusive["+".join(row["kinds"]) or "unattributed"] += duration
        for kind in kinds:
            unions[kind] += duration
        for left, right in (
            ("kernel", "copy"),
            ("producer", "kernel"),
            ("wait", "kernel"),
        ):
            if {left, right} <= kinds:
                overlaps[f"{left}+{right}"] += duration
        if not kinds.intersection(("kernel", "copy", "memset")):
            gpu_gap += duration
            if "producer" in kinds:
                producer_gap += duration
            if "wait" in kinds:
                wait_gap += duration
    assert sum(exclusive.values()) == wall
    issues = []
    if capture["outcome"] != "success":
        issues.append("endpoint failed")
    issues += [
        f"capture {k} false"
        for k in ("complete_tail", "drained", "export_complete", "lossless")
        if not capture[k]
    ]
    issues += [f"{k} coverage unavailable/incomplete" for k in KINDS if not coverage[k]]
    device_complete = all(
        coverage[k] for k in ("kernel", "copy", "memset")
    ) and not any(
        not capture[k]
        for k in ("complete_tail", "drained", "export_complete", "lossless")
    )
    complete = not issues
    return {
        "schema": SCHEMA,
        "status": "PASS" if complete else "INCOMPLETE",
        "identity": trace["identity"],
        "issues": issues,
        "endpoint_wall_ns": wall,
        "observed_exclusive_wall_ns": dict(exclusive),
        "reconciliation_error_ns": sum(exclusive.values()) - wall,
        "union_ns": {k: unions[k] if coverage[k] else None for k in KINDS},
        "duration_sum_ns_explanatory_only": {
            k: sums[k] if coverage[k] else None for k in KINDS
        },
        "counts": {k: counts[k] if coverage[k] else None for k in KINDS},
        "intersection_ns": {
            k: v if all(coverage[c] for c in k.split("+")) else None
            for k, v in overlaps.items()
        },
        "observed_dma_bytes": {
            k: transfer[k] if coverage["copy"] else None for k in ("H2D", "D2H", "D2D")
        },
        "no_recorded_gpu_activity_ns": gpu_gap if device_complete else None,
        "producer_during_gpu_gap_ns": producer_gap
        if device_complete and coverage["producer"]
        else None,
        "wait_during_gpu_gap_ns": wait_gap
        if device_complete and coverage["wait"]
        else None,
        "measured_dram_bytes": None,
        "logical_traffic_bytes": None,
        "critical_path_hideable_ns": None,
        "double_buffer_extra_bytes": None,
        "recommendation": "UNAVAILABLE: timeline intersections do not prove causality, buffer ownership or endpoint benefit",
        "segments": segments,
        "extraction": trace.get("extraction"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path)
    source.add_argument("--nsys-sqlite", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    trace = (
        read_nsys(args.nsys_sqlite, receipt)
        if args.nsys_sqlite
        else json.loads(args.trace.read_text())
    )
    report = summarize(trace, receipt)
    report["inputs"] = {
        str(path): {"sha256": checksum(path), "size": path.stat().st_size}
        for path in (args.nsys_sqlite or args.trace, args.receipt)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
