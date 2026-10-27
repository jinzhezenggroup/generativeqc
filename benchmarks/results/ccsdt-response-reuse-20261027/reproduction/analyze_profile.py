"""Attribute device activities through CUDA API correlation to innermost NVTX.

Host NVTX durations are never treated as GPU time. Each activity belongs to
one leaf, so the joint parent is not added to its children. Work not carrying
our diagnostic ranges is excluded from the response ledger.
"""

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def analyze(path: Path) -> dict:
    """Assign every captured device activity to at most one diagnostic leaf."""
    db = sqlite3.connect(path)
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    ranges = list(
        db.execute(
            "SELECT start,end,text,globalTid FROM NVTX_EVENTS WHERE text LIKE '2516/%' ORDER BY start"
        )
    )
    assert ranges and all(end is not None for start, end, name, tid in ranges)
    by_thread = defaultdict(list)
    labels = {}
    for index, (start, end, name, tid) in enumerate(ranges):
        by_thread[tid].append((start, end, index))
        labels[index] = name
    api = defaultdict(list)
    for start, end, corr, tid in db.execute(
        "SELECT start,end,correlationId,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"
    ):
        api[tid].append((start, end, corr))
    binding = {}
    for tid, calls in api.items():
        region = by_thread[tid]
        cursor = 0
        active = []
        for start, end, corr in calls:
            while cursor < len(region) and region[cursor][0] <= start:
                active = [r for r in active if r[1] > region[cursor][0]]
                active.append(region[cursor])
                cursor += 1
            active = [r for r in active if r[1] >= end]
            if active:
                binding[corr] = active[-1][2]
    totals = defaultdict(Counter)
    per_range = defaultdict(Counter)
    kernel_names = defaultdict(Counter)
    for start, end, corr, name_id in db.execute(
        "SELECT start,end,correlationId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        if corr not in binding:
            continue
        index = binding[corr]
        name = labels[index]
        family = "/".join(name.split("/")[:2])
        per_range[index].update(kernel_count=1, kernel_ns=end - start)
        totals[family].update(kernel_count=1, kernel_ns=end - start)
        kernel_names[family][strings[name_id]] += end - start
    for start, end, corr, count, kind in db.execute(
        "SELECT start,end,correlationId,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"
    ):
        if corr in binding:
            name = labels[binding[corr]]
            family = "/".join(name.split("/")[:2])
            totals[family].update(
                memcpy_count=1, memcpy_ns=end - start, memcpy_bytes=count
            )
            totals[family][f"copy_kind_{kind}_bytes"] += count
    for index, (start, end, name, tid) in enumerate(ranges):
        family = "/".join(name.split("/")[:2])
        totals[family]["host_calls"] += 1
        totals[family]["host_ns"] += end - start
    duplicates = {}
    for prefix in ("2516/panel/", "2516/w/"):
        calls = defaultdict(list)
        for index, (_, _, name, _) in enumerate(ranges):
            if name.startswith(prefix):
                calls[name].append(dict(per_range[index]))
        duplicates[prefix] = {
            "physical_identities": len(calls),
            "calls": sum(len(x) for x in calls.values()),
            "repeated_calls": sum(len(x) - 1 for x in calls.values()),
            "repeated_kernel_ns": sum(
                x["kernel_ns"] for group in calls.values() for x in group[1:]
            ),
            "identities": {
                name: {
                    "calls": len(group),
                    "kernel_ns": sum(x["kernel_ns"] for x in group),
                    "repeated_kernel_ns": sum(x["kernel_ns"] for x in group[1:]),
                }
                for name, group in sorted(calls.items())
            },
        }
    eligible = [
        index
        for index in range(len(ranges) - 1)
        if labels[index] == "2516/resolvent" and labels[index + 1] == "2516/scalar"
    ]
    return {
        "eligible_resolvent": {
            "calls": len(eligible),
            "kernel_ns": sum(per_range[index]["kernel_ns"] for index in eligible),
        },
        "scope": "Diagnostic NVTX-only source overlay; CUPTI kernel times, not clean wall times or achieved savings.",
        "attribution": "CUDA runtime correlationId to innermost thread-matched enclosing NVTX range; exclusive leaf assignment.",
        "families": dict(totals),
        "top_kernels": {
            name: [{"name": k, "duration_ns": v} for k, v in counts.most_common(5)]
            for name, counts in kernel_names.items()
        },
        "forward_recomputation": duplicates,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sqlite", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    data = analyze(args.sqlite)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    for name, row in data["families"].items():
        print(
            name,
            row["host_calls"],
            row.get("kernel_count", 0),
            row.get("kernel_ns", 0) / 1e9,
        )
    for name, row in data["forward_recomputation"].items():
        print(name, {k: v for k, v in row.items() if k != "identities"})
