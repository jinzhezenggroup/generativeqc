"""Fail closed on missing endpoints, changed scientific work or >5% regression."""

import argparse
import hashlib
import json
import lzma
import math
import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks._native_dft_force_evidence import load as load_evidence
from benchmarks._retention import raw_output_path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
parser.add_argument("--output", type=raw_output_path, required=True)
parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
parser.add_argument("--cohort", help="Named cohort in an exact-FP64 evidence directory")
arguments = parser.parse_args()
retained = None
if (arguments.directory / "bundles.json").is_file():
    _, texts = load_evidence(arguments.directory)
    cohort = json.loads(texts[f"{arguments.cohort or arguments.backend}/cohort.json"])
    if cohort["schema"] != "generativeqc.native-dft-force-cohort.v1":
        raise ValueError("unknown retained endpoint cohort schema")
    retained = {
        "schema": "generativeqc.native-dft-force-observations.v1",
        "backend": cohort["backend"],
        "expected_inputs": cohort["expected_inputs"],
        "blocks": {
            name: json.loads(texts[path]) for name, path in cohort["blocks"].items()
        },
    }
elif arguments.directory.is_file():
    retained = json.loads(lzma.decompress(arguments.directory.read_bytes()))
if retained is not None:
    if retained["schema"] != "generativeqc.native-dft-force-observations.v1":
        raise ValueError("unknown retained endpoint evidence schema")
    if retained["backend"] != arguments.backend:
        raise ValueError("retained endpoint evidence/backend mismatch")
    expected_inputs = retained["expected_inputs"]
else:
    expected_inputs = json.loads(
        (arguments.directory / "expected-inputs.json").read_text()
    )
groups = defaultdict(list)
failures, incomplete, receipts = [], [], []
cases = {
    f"{method}-{spin}-{fitted}"
    for method in (
        "lda",
        "pbe",
        "pbe0",
        "b3lyp",
        *(("pbe50",) if arguments.backend == "cpu" else ()),
        "r2scan",
        "wb97m-v",
        *(("m06-2x", "mn15") if arguments.backend == "cuda" else ()),
        "pbe-d3(bj)",
        "pbe0-d4(bj-eeq-atm)",
        "r2scan-3c",
    )
    for spin in (0, 1)
    for fitted in (0, 1)
}
if arguments.backend == "cuda":
    cases |= {"pbe-water36-df-d", "pbe-water48-df-d"}
cases |= {
    f"libxc-{functional}-{spin}-0"
    for functional in (
        (
            "LDA_C_VWN_4",
            "LDA_C_CHACHIYO",
            "LDA_C_PZ_MOD",
            "GGA_X_APBE",
            "GGA_X_PBE_SOL",
            "GGA_X_RPBE",
            "GGA_X_B86",
            "GGA_C_APBE",
            "MGGA_X_LTA",
            "MGGA_X_GVT4",
            "MGGA_C_TPSS",
        )
        if arguments.backend == "cuda"
        else ()
    )
    for spin in (0, 1)
}
for order in ("ABBA", "BAAB"):
    for block, letter in enumerate(order):
        variant = "baseline" if letter == "A" else "candidate"
        path = arguments.directory / f"paired-{order}-{block}-{variant}.jsonl"
        if retained is not None:
            retained_block = retained["blocks"].get(path.name)
            if retained_block is None:
                incomplete.append(f"missing {path.name}")
                continue
            contents = "".join(
                json.dumps(row) + "\n" for row in retained_block["records"]
            ).encode()
            if hashlib.sha256(contents).hexdigest() != retained_block["sha256"]:
                failures.append({"file": path.name, "retained_bytes_mismatch": True})
        elif not path.is_file():
            incomplete.append(f"missing {path.name}")
            continue
        else:
            contents = path.read_bytes()
        records = [json.loads(line) for line in contents.splitlines()]
        header = records.pop(0)["metadata"]
        if any(
            header.get(field) != value
            for field, value in expected_inputs[variant].items()
        ):
            failures.append({"file": path.name, "input_identity_mismatch": header})
        if (
            header["variant"] != variant
            or header["samples"] != 11
            or header["warmups"] != 3
            or header["tracing"]
        ):
            failures.append({"file": path.name, "invalid_metadata": header})
        if set(header["cases"]) != cases:
            incomplete.append(f"incomplete declared matrix in {path.name}")
        seen = defaultdict(list)
        for record in records:
            if "error" in record:
                failures.append({"file": path.name, **record})
                continue
            if (
                record["case"] not in cases
                or record["executed_backend"]
                != ("cuda" if arguments.backend == "cuda" else "cpu_reference")
                or not record["native"]
            ):
                failures.append({"file": path.name, "invalid_execution": record})
                continue
            if not math.isfinite(record["seconds"]) or record["seconds"] <= 0:
                failures.append({"file": path.name, "invalid_time": record})
            expected_atoms = (
                36
                if record["case"] == "pbe-water36-df-d"
                else 48
                if record["case"] == "pbe-water48-df-d"
                else 4
                if record["case"].startswith("libxc-")
                else 3
            )
            if (
                not math.isfinite(record["energy"])
                or len(record["forces"]) != expected_atoms
                or any(
                    len(atom) != 3 or any(not math.isfinite(value) for value in atom)
                    for atom in record["forces"]
                )
            ):
                failures.append({"file": path.name, "invalid_observable": record})
                continue
            seen[record["case"], record["phase"]].append(record["sample"])
            groups[order, variant, record["case"], record["phase"]].append(record)
        for case in cases:
            for phase in ("cold", "warm", "moved"):
                if sorted(seen[case, phase]) != list(range(11)):
                    incomplete.append(f"{path.name}: incomplete {case}/{phase}")
        receipts.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(contents).hexdigest(),
                "metadata": header,
            }
        )

comparisons = []
work_fields = (
    "iterations",
    "fock_builds",
    "warm_start_used",
    "grid_points",
    "tile_points",
    "coulomb_exchange_work",
    "nonlocal_work",
)
for order in ("ABBA", "BAAB"):
    for case in sorted(cases):
        for phase in ("cold", "warm", "moved"):
            baseline = groups[order, "baseline", case, phase]
            candidate = groups[order, "candidate", case, phase]
            if len(baseline) != 22 or len(candidate) != 22:
                continue
            baseline_median = statistics.median(
                record["seconds"] for record in baseline
            )
            candidate_median = statistics.median(
                record["seconds"] for record in candidate
            )
            energy_error = force_error = 0.0
            work_mismatches = []
            for left in baseline:
                for right in candidate:
                    if phase == "moved" and left["sample"] % 2 != right["sample"] % 2:
                        continue
                    energy_error = max(
                        energy_error, abs(left["energy"] - right["energy"])
                    )
                    force_error = max(
                        force_error,
                        max(
                            abs(first - second)
                            for first_atom, second_atom in zip(
                                left["forces"], right["forces"], strict=True
                            )
                            for first, second in zip(
                                first_atom, second_atom, strict=True
                            )
                        ),
                    )
                    # Work depends on the complete warm/moved input history,
                    # not just geometry parity. Compare identical sample ages;
                    # observables above still compare every matching geometry.
                    changed = {
                        field: [left[field], right[field]]
                        for field in work_fields
                        if left["sample"] == right["sample"]
                        and left[field] != right[field]
                    }
                    if changed:
                        work_mismatches.append(changed)
            comparison = {
                "order": order,
                "case": case,
                "phase": phase,
                "baseline_median_seconds": baseline_median,
                "candidate_median_seconds": candidate_median,
                "ratio": candidate_median / baseline_median,
                "maximum_energy_difference": energy_error,
                "maximum_force_difference": force_error,
                "work_mismatches": len(work_mismatches),
                "work_mismatch_examples": work_mismatches[:2],
            }
            comparisons.append(comparison)
            if (
                comparison["ratio"] > 1.05
                or energy_error > 2e-8
                or force_error > 3e-7
                or work_mismatches
            ):
                failures.append(comparison)
result = {
    "status": "INCOMPLETE" if incomplete else "FAIL" if failures else "PASS",
    "gate": {
        "maximum_ratio": 1.05,
        "maximum_energy_difference": 2e-8,
        "maximum_force_difference": 3e-7,
    },
    "incomplete": incomplete,
    "failures": failures,
    "receipts": receipts,
    "comparisons": comparisons,
}
arguments.output.write_text(json.dumps(result, indent=2) + "\n")
print(
    result["status"],
    "comparisons",
    len(comparisons),
    "failures",
    len(failures),
    "incomplete",
    len(incomplete),
)
raise SystemExit(result["status"] != "PASS")
