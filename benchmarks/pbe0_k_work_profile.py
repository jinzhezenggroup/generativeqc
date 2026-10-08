"""Separate class-resolved J/K device time in a complete native PBE0 cold trace.

The native direct-J/K owner enqueues J before K. This reducer fails closed unless
every Fock build contains the same two complete class passes on one CUDA stream.
It includes native dddd and never treats force kernels as value K. Device sums
from an intrusive trace are diagnostics, not clean complete-endpoint timings or
primitive-work counts. Use separate fresh-density invocations for wall timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

GENERATED_CLASS = re.compile(
    r"generated_\w+_([spdf]{4})_shell_class_fock_rhf_(?:work_)?streaming_kernel"
)


def summarize_classes(
    connection: sqlite3.Connection, fock_builds: int
) -> dict[str, Any]:
    """Validate paired class passes before assigning their device time to J/K."""
    if fock_builds <= 0:
        raise ValueError("a converged cold endpoint must report positive Fock builds")
    rows = connection.execute(
        """
        SELECT kernels.start, kernels.end - kernels.start, strings.value,
               kernels.deviceId, kernels.contextId, kernels.streamId
        FROM CUPTI_ACTIVITY_KIND_KERNEL kernels
        JOIN StringIds strings ON strings.id = kernels.demangledName
        WHERE strings.value LIKE '%shell_class_fock_rhf%streaming_kernel%'
           OR strings.value LIKE '%bounded_direct_dddd_streaming_kernel%'
        ORDER BY kernels.start
        """
    )
    events = []
    streams = set()
    for started, duration, name, device, context, stream in rows:
        match = GENERATED_CLASS.search(name)
        if match:
            shell_class = match.group(1)
        elif (
            "bounded_direct_dddd_streaming_kernel" in name
            and "DirectScreeningPurpose)0" in name
        ):
            shell_class = "dddd"
        else:
            raise ValueError(f"unsupported value class trace: {name}")
        if duration < 0:
            raise ValueError("negative kernel duration in trace")
        streams.add((device, context, stream))
        events.append((shell_class, duration, name))
    if not events or len(streams) != 1:
        raise ValueError("expected one nonempty native direct-J/K CUDA stream")
    class_count = len({event[0] for event in events})
    if len(events) != 2 * fock_builds * class_count:
        raise ValueError("incomplete or unmatched J/K class passes")
    inventory = [event[0] for event in events[:class_count]]
    if len(set(inventory)) != class_count:
        raise ValueError("first class pass repeats or omits a class")
    times: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"J": [], "K": []})
    work_classes = set()
    for pass_index in range(2 * fock_builds):
        current = events[pass_index * class_count : (pass_index + 1) * class_count]
        if [event[0] for event in current] != inventory:
            raise ValueError("class launch order changes between J/K passes")
        consumer = "J" if pass_index % 2 == 0 else "K"
        for shell_class, duration, name in current:
            if "_work_streaming_kernel" in name:
                if consumer != "K":
                    raise ValueError("work-bucket kernel found in the J pass")
                work_classes.add(shell_class)
            times[shell_class][consumer].append(duration)
    total = sum(sum(item["K"]) for item in times.values())
    if total == 0:
        raise ValueError("trace contains no positive K device duration")
    classes = [
        {
            "angular_class": shell_class,
            "J_seconds": sum(item["J"]) / 1e9,
            "K_seconds": sum(item["K"]) / 1e9,
            "K_fraction": sum(item["K"]) / total,
            "J_launches": len(item["J"]),
            "K_launches": len(item["K"]),
            "work_buckets": shell_class in work_classes,
        }
        for shell_class, item in times.items()
    ]
    classes.sort(key=lambda item: item["K_seconds"], reverse=True)
    return {
        "schema": "pbe0-cold-k-classes.v1",
        "dispatch_contract": "native direct_jk.cpp: J then K, full-range RKS",
        "timing_kind": "intrusive cumulative device kernel time, not clean wall time",
        "fock_builds": fock_builds,
        "K_seconds": total / 1e9,
        "high_angular_K_fraction": sum(
            item["K_fraction"]
            for item in classes
            if "d" in item["angular_class"] or "f" in item["angular_class"]
        ),
        "psss_psps_K_fraction": sum(
            item["K_fraction"]
            for item in classes
            if item["angular_class"] in ("psss", "psps")
        ),
        "top_three_K_fraction": sum(item["K_fraction"] for item in classes[:3]),
        "primitive_work_counts": None,
        "classes": classes,
    }


def main() -> None:
    """Reduce an Nsight SQLite export with its independently gated cold record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--endpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    endpoint = json.loads(args.endpoint.read_text())
    if endpoint.get("protocol", {}).get("method") != "PBE0/RKS":
        raise ValueError(
            "trace attribution requires the native full-range PBE0/RKS contract"
        )
    if endpoint.get("engine") != "native" or not endpoint.get("converged"):
        raise ValueError("expected a converged native endpoint")
    if endpoint.get("warm_start_used") or not endpoint.get("acceptance", {}).get(
        "gate"
    ):
        raise ValueError(
            "endpoint must be cold and pass the independent numerical gate"
        )
    with sqlite3.connect(
        f"file:{args.trace.resolve()}?mode=ro", uri=True
    ) as connection:
        report = summarize_classes(connection, endpoint["fock_builds"])
    report["endpoint"] = {
        field: endpoint.get(field)
        for field in (
            "protocol",
            "native_build",
            "gpu",
            "slurm_job_id",
            "iterations",
            "fock_builds",
            "acceptance",
        )
    }
    report["endpoint_sha256"] = hashlib.sha256(args.endpoint.read_bytes()).hexdigest()
    with args.trace.open("rb") as trace_file:
        report["trace_sha256"] = hashlib.file_digest(trace_file, "sha256").hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
