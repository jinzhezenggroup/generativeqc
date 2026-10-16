"""Recompute complete-endpoint gates and medians from the retained raw records.

Run from any directory. No production calculation, GPU, or reference package is
needed: the independent oracle energies/forces are retained with this evidence.
Profile windows are deliberately separate from clean endpoint measurements.
"""

from __future__ import annotations

import argparse
import json
import lzma
import runpy
from contextlib import ExitStack
import statistics
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
SAFETY = runpy.run_path(str(ROOT / "reproduce/output_safety.py"))
REPRODUCERS = ('output_safety.py', 'run-endpoint.sh', 'run-probe.sh', 'run-guarded.sh', 'run-oracle.sh', 'resident-diagnostic.py', 'selected-j-oracle.py', 'summarize-calls.py')
CASES = (("r2scan-3c", 96), ("r2scan-3c", 48), ("pbe0", 48), ("hf", 48))
PHASES = ["cold", *["warm"] * 5, "moved", *["moved-warm"] * 5]
with lzma.open(ROOT / "evidence.json.xz", "rt") as stream:
    EVIDENCE = json.load(stream)
with np.load(ROOT / "forces.npz", allow_pickle=False) as arrays:
    FORCES = arrays["forces"]
assert FORCES.dtype == np.dtype("<f8") and FORCES.ndim == 1
assert np.isfinite(FORCES).all()


def load(relative: str) -> Any:
    """Resolve pooled metadata and exact standard-NPZ FP64 force slices offline."""
    if relative in EVIDENCE["profiles"]:
        return EVIDENCE["profiles"][relative]
    record = dict(EVIDENCE["records"][relative])
    record["protocol"] = EVIDENCE["protocols"][record.pop("protocol_key")]
    record["records"] = [dict(row) for row in record["records"]]
    for row in record["records"]:
        offset, atoms = row.pop("forces_slice")
        assert atoms == record["protocol"]["atoms"]
        assert 0 <= offset and offset + 3 * atoms <= FORCES.size
        row["forces"] = FORCES[offset : offset + 3 * atoms].reshape(atoms, 3).tolist()
    return record


def qualify(record: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    """Recheck every completed call against the independent geometry oracle."""
    assert record["status"] == "measured", record["status"]
    assert [row["phase"] for row in record["records"]] == PHASES
    assert record["protocol"] == oracle["protocol"]
    references = {
        row["geometry"]: row
        for row in oracle["records"]
        if row["phase"] in ("cold", "moved")
    }
    assert set(references) == {0, 1}
    rows = []
    for row in record["records"]:
        reference = references[row["geometry"]]
        energy_error = abs(row["energy"] - reference["energy"])
        force_error = max(
            abs(actual - expected)
            for force, target in zip(row["forces"], reference["forces"], strict=True)
            for actual, expected in zip(force, target, strict=True)
        )
        assert row["gate"] is True and row["converged"] is True
        assert reference["converged"] is True
        assert energy_error <= 1e-8 and force_error <= 1e-7
        assert abs(energy_error - row["energy_error"]) <= 1e-15
        assert abs(force_error - row["force_error"]) <= 1e-15
        assert type(row["iterations"]) is int and type(row["fock_builds"]) is int
        assert row["iterations"] > 0 and row["fock_builds"] > 0
        rows.append(
            {
                key: row[key]
                for key in (
                    "phase",
                    "geometry",
                    "repeat",
                    "seconds",
                    "prepare_seconds",
                    "complete_seconds",
                    "iterations",
                    "fock_builds",
                    "energy_error",
                    "force_error",
                    "gate",
                    "converged",
                )
            }
        )
    return {
        "complete_seconds": {
            phase: statistics.median(
                row["complete_seconds"] for row in rows if row["phase"] == phase
            )
            for phase in dict.fromkeys(PHASES)
        },
        "maximum_energy_error": max(row["energy_error"] for row in rows),
        "maximum_force_error": max(row["force_error"] for row in rows),
        "library_sha256": record["native_build"]["library_sha256"],
        "scheduler": record["scheduler"],
        "oracle_status": oracle["status"],
        "calls": rows,
    }


def force_work(census: list[dict[str, Any]]) -> dict[str, Any]:
    """Count executed force work; an unavailable counter remains null."""
    assert len(census) == 2
    output = {}
    for phase, record in zip(("cold", "warm"), census, strict=True):
        assert record["valid"] and not record["cuda_error"]
        output[phase] = {}
        for order in (6, 7, 8):
            classes = [
                row for row in record["classes"] if row["angular_order"] == order
            ]
            assert classes
            output[phase][str(order)] = {
                name: None
                if any(row[name] is None for row in classes)
                else sum(row[name] for row in classes)
                for name in (
                    "enumerated",
                    "admitted",
                    "executed_shells",
                    "executed_components",
                    "primitive_pair_products",
                    "bra_preparations",
                    "ket_preparations",
                    "coulomb_preparations",
                    "component_contractions",
                    "published_components",
                )
            }
    return output


def summarize() -> dict[str, Any]:
    """Recompute the unchanged scientific gates and populations."""
    output: dict[str, Any] = {"endpoint_cases": {}, "diagnostic_profiles": {}}
    for method, atoms in CASES:
        name = f"{method}-{atoms}"
        oracle = load(f"raw/reference-{name}.json.xz")
        arms = {
            arm: load(f"raw/{arm}-{name}.json.xz") for arm in ("baseline", "candidate")
        }
        assert arms["baseline"]["protocol"] == arms["candidate"]["protocol"]
        result = {arm: qualify(record, oracle) for arm, record in arms.items()}
        result["candidate_change_percent"] = {
            phase: 100
            * (
                result["candidate"]["complete_seconds"][phase]
                / result["baseline"]["complete_seconds"][phase]
                - 1
            )
            for phase in dict.fromkeys(PHASES)
        }
        output["endpoint_cases"][name] = result
    for arm in ("baseline", "candidate"):
        windows = load(f"profile/{arm}-call-summary.json.xz")
        for window in windows.values():
            assert abs(window["host_reconciliation_error_seconds"]) < 1e-7
            assert abs(window["device_reconciliation_error_seconds"]) < 1e-7
        output["diagnostic_profiles"][arm] = {
            "windows": {
                phase: {
                    key: window[key]
                    for key in (
                        "exclusive_host_seconds",
                        "exclusive_device_window_seconds",
                        "complete_endpoint_wall_ledger",
                        "record",
                        "resource_columns",
                        "resource_semantics",
                        "original_sqlite_sha256",
                    )
                }
                for phase, window in windows.items()
            },
            "force_work": force_work(load(f"profile/{arm}-census.json.xz")),
        }
    return output

def main() -> None:
    """Write checked results to exclusively reserved, fresh destinations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT.parents[2] / ".artifacts/md-j-sparse-support-20261016-summary.json",
    )
    parser.add_argument("--export-reproducers", type=Path)
    arguments = parser.parse_args()
    # Check every destination and read every export source before any write.
    SAFETY["check_outputs"](arguments.output, arguments.export_reproducers)
    sources = {
        name: (ROOT / "reproduce" / name).read_text()
        for name in REPRODUCERS
    } if arguments.export_reproducers is not None else {}
    exported = (
        SAFETY["claim_directory"](arguments.export_reproducers)
        if arguments.export_reproducers is not None else None
    )
    with ExitStack() as stack:
        publish = stack.enter_context(SAFETY["json_output"](arguments.output))
        if exported is not None:
            stack.enter_context(SAFETY["run_receipt"](exported, "export"))
            for name, source in sources.items():
                with SAFETY["claim_file"](exported / name) as stream:
                    stream.write(source)
        output = summarize()
        publish(output)
    for name, result in output["endpoint_cases"].items():
        print(name, result["candidate_change_percent"])


if __name__ == "__main__":
    main()
