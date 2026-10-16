"""Select each original public call without mixing cold and warm profile windows."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

from output_safety import json_output

directory = Path(sys.argv[1])
with json_output(directory / "call-summary.json") as publish:
    from benchmarks.summarize_dft_profile import summarize
    database = directory / "cold-warm.sqlite"
    native = json.loads((directory / "native.json").read_text())
    assert native["diagnostic_only"] and native["stage"] == "cold-and-first-warm-only"
    assert [row["phase"] for row in native["records"]] == ["cold", "warm"]
    digest = hashlib.sha256()
    with database.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    summary = {}
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as original:
        roots = list(original.execute(
            "SELECT NVTX_EVENTS.rowid,start,end FROM NVTX_EVENTS LEFT JOIN StringIds "
            "ON StringIds.id=NVTX_EVENTS.textId "
            "WHERE COALESCE(NULLIF(text,''),value)='dft-profile/public_execute' "
            "AND end IS NOT NULL ORDER BY start"
        ))
        assert len(roots) == 2
        for selected, record in zip(roots, native["records"], strict=True):
            with tempfile.TemporaryDirectory() as temporary:
                isolated = Path(temporary) / "selected.sqlite"
                with sqlite3.connect(isolated) as destination:
                    original.backup(destination)
                    destination.executemany(
                        "DELETE FROM NVTX_EVENTS WHERE rowid=?",
                        [(root[0],) for root in roots if root[0] != selected[0]],
                    )
                result = summarize(isolated)
            assert abs(result["host_reconciliation_error_seconds"]) < 1e-7
            assert abs(result["device_reconciliation_error_seconds"]) < 1e-7
            result.update({
                "original_sqlite_sha256": digest.hexdigest(),
                "original_root_bounds_ns": list(selected[1:]),
                "selection": "Only the other public root marker is removed in a temporary copy; "
                             "all kernel/API events and the selected call's nested markers are unchanged.",
                "complete_endpoint_wall_ledger": record["diagnostic_wall_ledger"],
                "record": {key: record[key] for key in (
                    "phase", "seconds", "prepare_seconds", "complete_seconds", "iterations",
                    "fock_builds", "energy_error", "force_error", "gate", "converged",
                )},
            })
            summary[record["phase"]] = result
    publish(summary)
    for phase, result in summary.items():
        print(phase, json.dumps({
            "host": result["exclusive_host_seconds"],
            "device": result["exclusive_device_window_seconds"],
        }))
