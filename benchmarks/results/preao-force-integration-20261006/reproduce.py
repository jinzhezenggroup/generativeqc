"""Rebuild exact raw records and verify the post-upstream integration cohort."""

import argparse
import hashlib
import json
import lzma
import tempfile
from pathlib import Path

from benchmarks._retention import raw_output_path
from benchmarks.readme_omol25 import check_record
from benchmarks.verify_preao_force import PHASES, load_report, verify_census
from benchmarks.verify_preao_portfolio import (
    MODES,
    WORK_COUNTERS,
    qualify,
    verify_dispatch,
)


def verify(directory: Path) -> dict:
    """Five-round warm qualification at 24; 3/48 are numerical/work smoke pairs."""
    qualification = qualify(directory, [24])
    assert qualification["default_promotion_eligible"]
    cohort = qualification["reports"]["24"]
    smoke = {}
    for atoms in (3, 48):
        oracle = load_report(directory / f"reference-{atoms}.json")
        rows = {}
        for mode in MODES:
            report = load_report(directory / f"smoke-{atoms}-{mode}.json")
            assert report["status"] == "measured" and report["stage"] == "complete"
            assert report["protocol"] == oracle["protocol"]
            assert not report["p0c_intrusive_stage_profile"]
            assert report["scheduler"] == cohort["scheduler"]
            assert (
                report["source_file_sha256"]
                == cohort["source_identity"]["source_file_sha256"]
            )
            assert report["native_build"] == cohort["source_identity"]["native_build"]
            producer = verify_dispatch(report, mode)
            assert tuple(record["phase"] for record in report["records"]) == PHASES
            checked = []
            for record, work in zip(
                report["records"], report["p0c_force_work"], strict=True
            ):
                assert record["converged"] and record["status"] == 0
                assert record["native_ks_diagnostic"]["history"]
                gates = [
                    check_record(record, reference)
                    for reference in oracle["records"]
                    if reference["geometry"] == record["geometry"]
                ]
                assert gates and all(gate["gate"] for gate in gates)
                verify_census(record, work, producer)
                checked.extend(gates)
            rows[mode] = report
            smoke[f"{atoms}/{mode}"] = {
                "producer": producer,
                "max_energy_error": max(gate["energy_error"] for gate in checked),
                "max_force_error": max(gate["force_error"] for gate in checked),
                "complete_seconds": {
                    record["phase"]: record["complete_seconds"]
                    for record in report["records"]
                },
            }
        for baseline, candidate in zip(
            rows[MODES[0]]["p0c_force_work"],
            rows[MODES[1]]["p0c_force_work"],
            strict=True,
        ):
            before, after = (
                baseline["grid_metrics"]["ao_grid_work"],
                candidate["grid_metrics"]["ao_grid_work"],
            )
            assert all(before[name] == after[name] for name in WORK_COUNTERS)
            if atoms == 48:
                assert (
                    baseline["force_active_ao_policy"]
                    == candidate["force_active_ao_policy"]
                )
            else:
                assert candidate["resident_ao_selection"]["work"]["occupancy_declined"]
    return {
        "schema": "generativeqc.preao-post-integration-qualification.v1",
        "warm_qualification": qualification,
        "smoke": smoke,
        "smoke_scope": "One numerical/work pair at 3/48, not a new five-round no-regression performance qualification",
    }


def main() -> None:
    """A single compressed bundle preserves each original UTF-8 JSON byte stream."""
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=raw_output_path, required=True)
    arguments = parser.parse_args()
    bundle = (
        json.loads(lzma.decompress(arguments.bundle.read_bytes()))
        if arguments.bundle.suffix == ".xz"
        else load_report(arguments.bundle)
    )
    assert bundle["schema"] == "generativeqc.preao-integration-bundle.v1"
    with tempfile.TemporaryDirectory(prefix="preao-integration-") as temporary:
        directory = Path(temporary)
        for name, entry in bundle["files"].items():
            assert Path(name).name == name and name.endswith(".json")
            payload = entry["text"].encode("utf-8")
            assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
            (directory / name).write_bytes(payload)
        result = verify(directory)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
