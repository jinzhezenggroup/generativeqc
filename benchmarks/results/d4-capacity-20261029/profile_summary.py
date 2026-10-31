"""Summarize scoped explicit CUDA work and live storage from Nsight SQLite."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def summarize(path: Path) -> dict:
    """Count actual activities; exclude static and opaque allocations from peak."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    live = peak = allocations = 0
    for operation, size in connection.execute(
        "SELECT memoryOperationType, bytes FROM CUDA_GPU_MEMORY_USAGE_EVENTS "
        "WHERE memKind=2 ORDER BY start, rowid"
    ):
        if operation == 0:
            allocations += 1
            live += size
        elif operation == 1:
            live -= size
        else:
            raise RuntimeError("unknown memory operation")
        if live < 0:
            raise RuntimeError("incomplete explicit allocation trace")
        peak = max(peak, live)
    result = {
        "explicit_device_peak_bytes": peak,
        "explicit_device_final_live_bytes": live,
        "explicit_device_allocation_count": allocations,
        "static_device_bytes": connection.execute(
            "SELECT COALESCE(SUM(bytes),0) FROM CUDA_GPU_MEMORY_USAGE_EVENTS "
            "WHERE memKind=5 AND memoryOperationType=0"
        ).fetchone()[0],
        "kernels": connection.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0],
        "graph_calls": connection.execute(
            "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s "
            "ON r.nameId=s.id WHERE s.value LIKE '%Graph%'"
        ).fetchone()[0],
        "copies": {
            name: {"count": count, "bytes": size}
            for name, count, size in connection.execute(
                "SELECT e.label, COUNT(*), SUM(m.bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY m "
                "JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind GROUP BY e.label"
            )
        },
    }
    connection.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "scope": "Complete untimed endpoint matrix; explicit dynamic CUDA storage only. Static and opaque provider/driver storage are excluded from the live peak.",
        "baseline": summarize(args.baseline),
        "candidate": summarize(args.candidate),
        "sqlite_sha256": {
            role: hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            for role, path in (
                ("baseline", args.baseline),
                ("candidate", args.candidate),
            )
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(
        report[role]["explicit_device_final_live_bytes"]
        for role in ("baseline", "candidate")
    ):
        raise SystemExit("dynamic CUDA allocations remain live")
    for role in ("baseline", "candidate"):
        if (
            report[role]["kernels"] <= 0
            or report[role]["explicit_device_allocation_count"] <= 0
        ):
            raise SystemExit(f"{role} profile captured no CUDA kernels or allocations")
    if report["baseline"] != report["candidate"]:
        raise SystemExit("actual GPU work or storage changed")


if __name__ == "__main__":
    main()
