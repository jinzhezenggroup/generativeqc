"""Offline proof of bounded outcome, never promote the incomplete matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from replay import recover, replay
from restore_actual import CAPSULE, digest, restore
from transport import checked, read

HERE = Path(__file__).resolve().parent


def verify(output: Path, capsule: Path = CAPSULE) -> dict:
    storage = restore(output, capsule)
    plan_path = HERE / "prepared-plan-v4.json"
    checked(
        plan_path, "8d2a61dfdbea39acc2f4852e715b8585a75af8c922702678951c4d0d9178d5ea"
    )
    plan = read(plan_path)
    for name, expected in plan["task_files"].items():
        checked(output / "scripts" / name, expected)
    marker = output / "root-release.json"
    checked(marker, "f24ae323d379f20d7c639075d70d9f73a0a9ab4bf089395895896a485e5d2c7d")
    release = read(marker)
    if (
        release["prepared_plan_sha256"] != digest(plan_path.read_bytes())
        or release["allocation_created_at_raw"] != "1792864826000"
    ):
        raise ValueError("actual creation/release differs from this measured Job")
    before = read(output / "preflight/static.json")
    after = read(output / "output/static-after.json")
    for key in ("whole_original_inputs", "source", "library_sha256", "environment"):
        if before[key] != after[key]:
            raise ValueError("actual source/env changed: " + key)
    if (
        before["whole_original_inputs"]["qualified_input_identity_sha256"]
        != "86d641d1581eea3f7d2fe9d4bdf1da2bf36ebff588f02bfbece5c7623243aca3"
    ):
        raise ValueError("actual qualified runtime identity differs")
    boot = read(output / "preflight/device.json")
    end = read(output / "output/device-after.json")
    for key in (
        "nvidia_smi",
        "hostname",
        "cuda_visible_devices",
        "compute_capability",
        "memory_bytes",
        "visible_device_count",
    ):
        if boot[key] != end[key]:
            raise ValueError("actual device changed: " + key)
    original_pairs = 0
    for directory in (output / "output").glob("*/call-*"):
        if (directory / "observables/return.json").exists():
            primary = recover(directory / "observables")
            enriched = recover(directory)
            if (
                primary["energy"] != enriched["energy"]
                or primary["forces"] != enriched["forces"]
            ):
                raise ValueError("primary E/F differs from metadata representation")
            for core, record in zip(
                primary["components"], enriched["components"], strict=True
            ):
                if (
                    core["energy"] != record["energy"]
                    or core["forces"] != record["forces"]
                ):
                    raise ValueError("primary constituent E/F differs from enrichment")
            original_pairs += 1
    result = replay(plan_path, output / "output")
    counts = {
        status: sum(row["status"] == status for row in result["calls"])
        for status in (
            "NUMERIC_PASS",
            "NOT_RUN",
            "NO_RETURN; inspect original process outcome/streams",
        )
    }
    if (
        counts
        != {
            "NUMERIC_PASS": 16,
            "NOT_RUN": 14,
            "NO_RETURN; inspect original process outcome/streams": 2,
        }
        or result["attempted_calls"] != 18
        or original_pairs != 16
        or result["status"] != "FAIL_OR_INCOMPLETE"
    ):
        raise ValueError("actual incomplete matrix was lost or promoted")
    for process in plan["processes"]:
        outcome = read(
            output / "output" / (process["id"] + "-process") / "process-outcome.json"
        )
        if not outcome.get("cleanup_success") or outcome.get("remaining_owned"):
            raise ValueError("actual scientific process subtree was not clean")
    for process, index in (("conventional-h2-dz-tz", 0), ("archive-water-exact", 12)):
        refused = read(output / "output" / process / "worker-outcome.json")
        if (
            refused["calls_attempted"] != 1
            or refused["calls_returned"] != 0
            or "CUDA RHF provider exceeds retained allocation allowance"
            not in refused["error"]
        ):
            raise ValueError("native refusal was altered")
        if (
            read(output / "output" / process / f"attempt-{index:03d}.json")["call"]
            != plan["calls"][index]
        ):
            raise ValueError("refused request parameters differ")
    stored = read(HERE / "numerical-replay.json")
    return {
        "offline_storage_and_outcome_closure": "PASS",
        "scientific_matrix": result["status"],
        "not_complete_or_parent_closure": True,
        "storage": storage,
        "numerical_replay": result,
        "primary_and_enriched_IEEE_pairs": original_pairs,
        "recomputed_metrics_exactly_match_stored": result == stored,
        "precision_work_peak_fallback": "INCOMPLETE; missing counters never interpreted as zero",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new restored originals directory"
    )
    parser.add_argument(
        "--report", type=Path, required=True, help="new offline report file"
    )
    parser.add_argument("--capsule", type=Path, default=CAPSULE, help="verified restored historical capsule")
    args = parser.parse_args()
    result = verify(args.output, args.capsule)
    with args.report.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("PASS offline storage/outcome closure; scientific matrix FAIL_OR_INCOMPLETE")
