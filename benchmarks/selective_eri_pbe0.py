"""Observe partial ERI reuse in the unchanged complete README PBE0 endpoint.

The solver, moving grid, independent oracle, forces and warm snapshots remain
owned by the official benchmark. This adapter only retains the already emitted
source/work receipts, including refused setup and incumbent handoff actions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

from generativeqc.batch import PreparedBatch

from benchmarks import readme_omol25, readme_pbe0


def main() -> None:
    """Run official cold/warm/moved E+F with observation-only instrumentation."""
    original_execute = PreparedBatch.execute
    original_check = readme_omol25.check_record
    observed: dict[str, Any] = {}

    def execute(owner: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_execute(owner, *args, **kwargs)
        observed["item"] = result.items[0]
        return result

    def check_record(row: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
        item = observed["item"]
        row["precision"] = item.precision
        precision = row["precision"]
        if not precision["complete"] or not precision["operator_inventory_complete"]:
            raise RuntimeError("PBE0 precision inventory is incomplete")
        if not precision["operators"] or any(
            record[dtype] != "fp64" or record["arithmetic_mode"] != "strict"
            for record in precision["operators"]
            for dtype in ("storage", "compute", "accumulation", "reduction")
        ):
            raise RuntimeError("partial source experiment requires strict FP64")
        events = [
            json.loads(line)
            for line in Path(os.environ["GENERATIVEQC_DF_PROGRESS_TRACE"])
            .read_text()
            .splitlines()
        ]
        begin = next(
            index
            for index in range(len(events) - 1, -1, -1)
            if events[index]["event"] == "BEGIN"
            and events[index]["name"] == "batch_execute"
        )
        batch = events[begin]
        events = events[begin:]
        row["source_work"] = {
            event["key"]: event["value"]
            for event in events
            if event["id"] == batch["id"] and event["event"] == "VALUE"
        }
        owners = {
            event["id"]: event["name"]
            for event in events
            if event["event"] == "BEGIN"
            and event["name"]
            in (
                "cuda_ks_resident_values_prepare",
                "cuda_ks_resident_values_completed",
                "cuda_ks_resident_values_skip",
                "cuda_ks_warm_energy_rebase",
            )
        }
        row["source_receipts"] = [
            {"owner": owners[event["id"]], **event}
            for event in events
            if event["id"] in owners and event["event"] in ("VALUE", "END")
        ]
        row["canonical_work_receipts"] = [
            event
            for event in events
            if event["event"] == "VALUE"
            and event["key"]
            in ("phase_canonical_source_values", "phase_direct_values_per_action")
        ]
        row["source_selection"] = {
            name: os.environ.get(name)
            for name in (
                "GENERATIVEQC_KS_RESIDENT_VALUES",
                "GENERATIVEQC_KS_RESIDENT_BUCKET_MASK",
                "GENERATIVEQC_KS_RESIDENT_MAX_BYTES",
            )
        }
        return original_check(row, oracle)

    with (
        patch.object(PreparedBatch, "execute", execute),
        patch.object(readme_omol25, "check_record", check_record),
    ):
        readme_pbe0.main()


if __name__ == "__main__":
    main()
