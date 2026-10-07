"""Recheck frozen cold E/F samples without CUDA, from the repository root."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from generativeqc_compiler.common.evidence import block_error

from tools.generativeqc_validation.publication import validate_publication
from tools.generativeqc_validation.record import load_record

ROOT = Path(__file__).resolve().parent
REFERENCE = Path(
    "benchmarks/results/pbe0-composed-baseline-20261005/reference-96.json.gz"
)
SAMPLES = (
    "master-off-1",
    "master-off-2",
    "master-off-3",
    "merged-off-1",
    "merged-off-2",
    "merged-on-1",
    "merged-on-2",
    "merged-gate4-1",
    "fixed-on-0",
    "fixed-off-1",
    "fixed-on-1",
    "fixed-on-2",
    "fixed-off-2",
    "repaired-on-0",
    "repaired-off-1",
    "repaired-on-1",
    "repaired-on-2",
    "repaired-off-2",
)
INTERRUPTED = {
    "merged-gate4-1": "Admission-threshold trial stopped to repair the controller instead.",
    "fixed-off-2": "First-repair campaign stopped after a failed physical closure.",
}
PREFILL = {"master-off-1", "merged-off-1", "fixed-on-0", "repaired-on-0"}
SOURCES = {
    "master": "e50158036b7ec15abaa20ecf505c89c31392f3a341adae5d4f48fb555b3e7307",
    "merged": "afaf381722150902ac46b10cb570998061cf020ef74a6b6414f7125e26be436e",
    "fixed": "19bf682c3dcd8b7d7f4c52bdee66b31905be30693000b08c6489718447c5ce19",
    "repaired": "be6ed903542e00954c5ab8d5b78fd8bc09df5a51028b7cc9ea677ef21aca706e",
}


def require(condition: bool, message: str) -> None:
    """Fail closed when stored identities, science or classifications drift."""
    if not condition:
        raise ValueError(message)


def analyze() -> dict:
    """Retain failures and first fills, but never mix them into paired E/F timing."""
    reference = load_record(REFERENCE)
    oracle = next(row for row in reference["records"] if row["phase"] == "cold")
    reference_hash = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    harness_hash = hashlib.sha256((ROOT / "cold.py.txt").read_bytes()).hexdigest()
    rows, records, errors = [], {}, {}
    for name in SAMPLES:
        record = load_record(ROOT / f"{name}.json.gz")
        records[name] = record
        require(
            record["protocol"] == reference["protocol"], f"{name}: scientific drift"
        )
        require(
            record["reference_sha256"] == reference_hash, f"{name}: reference drift"
        )
        require(record["harness_sha256"] == harness_hash, f"{name}: harness drift")
        require(
            record["native_build"]["probe"]["source_identity"]
            == SOURCES[name.split("-")[0]],
            f"{name}: source drift",
        )
        threshold = record["environment"][
            "GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK_DENSITY_RMS_THRESHOLD"
        ]
        require(
            float(threshold) == (1e-4 if name == "merged-gate4-1" else 0.0),
            f"{name}: admission-threshold drift",
        )
        row = {
            "sample": name,
            "record_status": record["status"],
            "complete_seconds": record.get("complete_seconds"),
            "iterations": record.get("iterations"),
            "fock_builds": record.get("fock_builds"),
            "incremental": record.get("incremental_direct_jk"),
            "acceptance": record.get("acceptance"),
        }
        rows.append(row)
        if name in INTERRUPTED:
            require(
                record["status"] == "running" and record["stage"] == "execute",
                f"{name}: interrupted trial unexpectedly produced a result",
            )
            row.update(
                disposition="interrupted-before-result", reason=INTERRUPTED[name]
            )
            continue
        require(record["status"] == "measured", f"{name}: missing endpoint result")
        work = record["incremental_direct_jk"]
        enabled = record["environment"]["GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK"] == "1"
        require(work["active"] is enabled, f"{name}: wrong incremental route")
        require(
            work["quartet_work_counters_valid"] is False,
            f"{name}: update the unavailable-quartet limitation explicitly",
        )
        if enabled:
            require(
                work["anchor_full_builds"]
                + work["delta_builds"]
                + work["post_scf_full_builds"]
                == record["ks_diagnostic"]["fock_builds"],
                f"{name}: actual build-count conservation failed",
            )
        if name == "fixed-on-1":
            require(
                record["native_status"] == 4
                and not record["converged"]
                and record["forces"] is None
                and record["acceptance"]["gate"] is False,
                "the rejected corrective-closure trial was reinterpreted as a success",
            )
            row.update(disposition="failed-no-forces", reason=record["status_message"])
            continue
        require(
            record["native_status"] == 0
            and record["converged"]
            and record["forces"] is not None
            and not record["warm_start_used"],
            f"{name}: not a successful fresh cold E/F endpoint",
        )
        energy = block_error(
            [record["energy"]], [oracle["energy"]], atol=1e-8, rtol=0.0
        )
        force = block_error(record["forces"], oracle["forces"], atol=1e-7, rtol=0.0)
        require(
            energy["passed"] and force["passed"], f"{name}: independent oracle failed"
        )
        require(
            record["acceptance"]["gate"] is True
            and record["acceptance"]["energy_error"] == energy["max_absolute_error"]
            and record["acceptance"]["force_error"] == force["max_absolute_error"],
            f"{name}: stored errors differ from independent recomputation",
        )
        errors[f"{name}:energy"] = energy
        errors[f"{name}:forces"] = force
        history = record["ks_diagnostic"]["history"]
        require(
            len(history) == record["iterations"] == record["fock_builds"],
            f"{name}: missing complete iteration history",
        )
        require(
            history[-1]["energy_change"] < 1e-12
            and history[-1]["density_change_max"] < 1e-10
            and record["physical_residual_rms"] < 1e-10,
            f"{name}: convergence gate was relaxed",
        )
        force_seconds = sum(sample["seconds"] for sample in record["force_samples"])
        row.update(
            disposition="program-cache-first-fill"
            if name in PREFILL
            else "accepted-cold",
            prepare_seconds=record["prepare_seconds"],
            force_seconds=force_seconds,
            nonforce_seconds=record["complete_seconds"] - force_seconds,
        )
    groups = {}
    for label, names in {
        "primary-repaired-off": ["repaired-off-1", "repaired-off-2"],
        "primary-repaired-on": ["repaired-on-1", "repaired-on-2"],
        "original-incremental-diagnostic": ["merged-on-1", "merged-on-2"],
        "previous-master-cache-populated-diagnostic": ["master-off-2", "master-off-3"],
    }.items():
        values = [records[name]["complete_seconds"] for name in names]
        groups[label] = {
            "samples": names,
            "seconds": values,
            "median_seconds": statistics.median(values),
            "range_seconds": [min(values), max(values)],
            "iterations": [records[name]["iterations"] for name in names],
        }
    paired = [
        records[name]
        for name in (
            "repaired-off-1",
            "repaired-on-1",
            "repaired-on-2",
            "repaired-off-2",
        )
    ]
    require(
        len({row["native_build"]["library_sha256"] for row in paired}) == 1,
        "primary ABBA comparison used different libraries",
    )
    off = groups["primary-repaired-off"]["median_seconds"]
    on = groups["primary-repaired-on"]["median_seconds"]
    original = groups["original-incremental-diagnostic"]["median_seconds"]
    return {
        "samples": rows,
        "groups": groups,
        "independent_accepted_endpoints": len(errors) // 2,
        "block_errors": errors,
        "primary_incremental_speedup": off / on,
        "primary_incremental_time_reduction_percent": 100 * (1 - on / off),
        "observed_repair_speedup_vs_original_incremental": original / on,
        "observed_repair_time_reduction_percent": 100 * (1 - on / original),
        "default_promotion": False,
        "limitations": [
            "Two samples per primary arm; no statistically established performance win.",
            "Only independent cold owners; no warm/moved population qualification.",
            "No valid actual quartet-admission counters or isolated kernel speedup claim.",
            "Native-library compilation and initial stationary program-cache fills are separate.",
            "Monolithic KS native-suite failure also reproduces with the unpatched library.",
        ],
    }


def main() -> None:
    """Authenticate selected bytes, then recompute every retained scientific gate."""
    manifest = json.loads((ROOT / "publication.json").read_text())
    files = {
        entry["path"]: (ROOT / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    actual = analyze()
    require(actual == load_record(ROOT / "summary.json"), "stored summary drift")
    print(
        json.dumps(
            {
                key: value
                for key, value in actual.items()
                if key not in {"samples", "block_errors"}
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
