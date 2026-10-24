"""Recover original bytes and independently replay the complete cache-state study."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from capsule import pretty, restore

from benchmarks._retention import raw_output_path
from benchmarks.runtime_cache_state import CASES, qualify_campaign


def records(raw: Path, cohort: str) -> list[tuple[dict, dict]]:
    campaign = json.loads((raw / cohort / "campaign.json").read_bytes())
    return [
        (row, json.loads((raw / cohort / row["folder"] / "endpoint.json").read_bytes()))
        for row in campaign["results"]
    ]


def work(record: dict) -> dict:
    sample = record["endpoint"]["sample"]
    if record["case"] != "cbs-c2":
        return {
            "iterations_total": sample["iterations"],
            "reported_fock_builds": sample["fock_builds"],
            "nonlocal_work": (sample.get("ks_diagnostic") or {}).get("nonlocal_work"),
            "history": (sample.get("ks_diagnostic") or {}).get("history"),
            "initial_guess": sample.get("initial_guess"),
            "incremental_direct_jk": sample.get("incremental_direct_jk"),
        }
    return {
        "execution_groups": sample["execution_groups"],
        "convergence": sample["convergence"],
        "components": [
            {
                "method": component["method"],
                "roles": component["roles"],
                "basis_identity": component["basis_identity"],
                "iterations": component["result"]["iterations"],
                "correlation": component["result"].get("correlation"),
                "cc_performance": component["result"].get("cc_performance"),
            }
            for component in sample["components"]
        ],
    }


def summarize(raw: Path) -> tuple[list[dict], str, list[dict]]:
    measured = [
        (row, record)
        for row, record in records(raw, "formal")
        if row["state"] != "prime"
    ]
    summary, work_records = [], []
    lines = [
        "# Fresh-process Standard cache-state observations",
        "",
        "Three processes per row; E+F and process-wall medians/ranges are separate. The observed Python driver counter is not internal NVCC passes. Native NVRTC and embedded-AOT selection counts remain UNAVAILABLE.",
        "",
        "| Case | State | E+F median (range), s | Process wall median (range), s | Observed driver calls | Driver wall per process, s |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for case in CASES:
        for state in ("empty", "primed"):
            population = [
                (row, record)
                for row, record in measured
                if row["case"] == case and row["state"] == state
            ]
            if len(population) != 3:
                raise ValueError("incomplete cache-state timing population")
            energy_force = [
                record["endpoint"]["sample"][
                    "complete_seconds" if case == "cbs-c2" else "seconds"
                ]
                for row, record in population
            ]
            process = [
                row["process"]["launch_to_exit_wall_seconds"]
                for row, record in population
            ]
            calls = [
                record["observation"]["cached_driver_invocations"]
                for row, record in population
            ]
            driver = [
                record["observation"]["cached_driver_wall_seconds"]
                for row, record in population
            ]
            summary.append(
                {
                    "case": case,
                    "state": state,
                    "energy_force_seconds": energy_force,
                    "process_seconds": process,
                    "observed_driver_calls": calls,
                    "observed_driver_wall_seconds": driver,
                    "pids": [record["pid"] for row, record in population],
                }
            )
            lines.append(
                f"| {case} | {state} | {statistics.median(energy_force):.6f} ({min(energy_force):.6f}–{max(energy_force):.6f}) | "
                f"{statistics.median(process):.6f} ({min(process):.6f}–{max(process):.6f}) | {calls} | "
                + ", ".join(f"{value:.6f}" for value in driver)
                + " |"
            )
            work_records.extend(
                {"case": case, "state": state, "pid": record["pid"], **work(record)}
                for row, record in population
            )
    return summary, "\n".join(lines) + "\n", work_records


def envelope(raw: Path, qualifications: dict) -> dict:
    import numpy as np
    from generativeqc_compiler.common.evidence import (
        block_error,
        new_evidence,
        outcome,
        validate_evidence,
    )

    from benchmarks.readme_defaults import COMMON_GATES
    from tools.render_readme_cbs import constituents

    campaign = json.loads((raw / "formal/campaign.json").read_bytes())
    config = campaign["config"]
    all_records = {cohort: records(raw, cohort) for cohort in ("pilot", "formal")}
    first = all_records["formal"][0][1]
    evidence = new_evidence(
        tier="endpoint",
        subject="Five Standard complete E+F fresh-process cache-state populations; static geometries only",
        inputs_hash=campaign["config_sha256"],
    )
    evidence.update(
        revision=config["production_source_revision"],
        device=first["accelerator"],
        toolchain=first["environment"]["toolchain"],
        backend_selected="cuda",
        hardware=outcome(
            "pass", "actual Slurm allocation and one unchanged physical GPU UUID"
        ),
    )
    evidence["settings"] = {
        "policy": "standard",
        "device": "cuda",
        "fast_compile": False,
        "physical_gpu_uuid": qualifications["formal"]["physical_gpu_uuid"],
        "installation": config["installation_layout"],
        "scope": "static five-case sample qualification, not method-domain/default promotion",
    }
    evidence["hashes"]["source"] = config["native_source_identity"]
    evidence["hash_reasons"] = {
        "equation": "Heterogeneous PBE0/WB97M-V/CBS endpoints; exact methods, basis records and geometry are retained per case.",
        "ir": "No single IR describes all five complete endpoints; full frozen production source/SDK identities retained.",
        "schedule": "Public Standard defaults, fixed cache-state interleave and complete semantic work retained; no schedule promotion.",
    }
    for name in ("representation", "source", "compilation", "numerical", "endpoint"):
        evidence["stages"][name] = outcome(
            "pass",
            "source-matched raw records, byte recovery and independent replay",
            scope="sampled static endpoints",
        )
    evidence["performance"] = outcome(
        "not-run",
        "Descriptive cache-state timing; no default/performance promotion or universal gain claimed",
    )
    evidence["compilation"] = {
        "seconds": None,
        "reason": "Scoped original Python compiler-driver spans are retained per process; native NVRTC/internal passes and total compiler CPU time are UNAVAILABLE",
    }
    evidence["memory"] = {
        "allocated_bytes": None,
        "peak_bytes": None,
        "reason": "not measured; 32 GiB CBS numerical capacity is not measured VRAM",
    }
    for cohort, pairs in all_records.items():
        for (row, record), check in zip(
            pairs, qualifications[cohort]["checks"], strict=True
        ):
            prefix = f"{cohort}/{row['folder']}"
            numerical = check["numerical"]
            if row["case"] == "cbs-c2":
                for name, errors in numerical["checks"]["component_errors"].items():
                    evidence["block_errors"][prefix + "/" + name + "/energy"] = (
                        block_error([errors["energy"]], [0], atol=5e-8, rtol=0)
                    )
                    evidence["block_errors"][prefix + "/" + name + "/directional"] = (
                        block_error(
                            errors["directional_derivatives"], [0, 0], atol=5e-6, rtol=0
                        )
                    )
                for name, sample in constituents(
                    record["endpoint"]["sample"], record["endpoint"]["protocol"]
                ).items():
                    evidence["block_errors"][prefix + "/" + name + "/translation"] = (
                        block_error(
                            np.sum(sample["forces"], axis=0),
                            np.zeros(3),
                            atol=1e-6,
                            rtol=0,
                        )
                    )
            else:
                checks = [
                    ("observable", numerical["accuracy"], 1),
                    ("cpu-gpu-spot", numerical["spot"], 0.25),
                    *(
                        (f"grid-stability-{index}", value, 0.25)
                        for index, value in enumerate(numerical["stability"])
                    ),
                ]
                for name, errors, fraction in checks:
                    for metric, tolerance in COMMON_GATES.items():
                        evidence["block_errors"][prefix + "/" + name + "/" + metric] = (
                            block_error(
                                [errors[metric]], [0], atol=fraction * tolerance, rtol=0
                            )
                        )
            if row["state"] != "prime":
                sample = record["endpoint"]["sample"]
                evidence["timings"].append(
                    {
                        "selection": "baseline"
                        if row["state"] == "primed"
                        else "candidate",
                        "workload": "energy-plus-force",
                        "seconds": sample[
                            "complete_seconds" if row["case"] == "cbs-c2" else "seconds"
                        ],
                        "inputs_hash": campaign["config_sha256"],
                        "case": row["case"],
                        "cache_state": row["state"],
                        "cohort": cohort,
                    }
                )
    if not all(row["passed"] for row in evidence["block_errors"].values()):
        raise ValueError("quantitative publication envelope contains a failing gate")
    validate_evidence(evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    publication = bundle / "publication.json"
    if publication.exists():
        from tools.generativeqc_validation.publication import validate_publication
        from tools.generativeqc_validation.retention import safe_relative

        manifest = json.loads(publication.read_bytes())
        files = {}
        for entry in manifest["files"]:
            name = safe_relative(entry["path"])
            path = bundle / name
            if (
                ".env" in path.parts
                or path.name.startswith(".env.")
                or not path.resolve().is_relative_to(bundle)
            ):
                raise ValueError("unsafe publication input")
            files[name] = path.read_bytes()
        validate_publication(manifest, files)
    output = raw_output_path(args.output.resolve())
    output.mkdir(parents=True, exist_ok=False)
    recovery = restore(bundle, output / "raw")
    qualifications = {
        cohort: qualify_campaign(output / "raw" / cohort)
        for cohort in ("pilot", "formal")
    }
    if any(value["status"] != "PASS" for value in qualifications.values()):
        raise ValueError("independent replay did not pass every population")
    summary, table, work_records = summarize(output / "raw")
    (output / "recovery.json").write_bytes(pretty(recovery))
    (output / "qualification.json").write_bytes(pretty(qualifications))
    (output / "timings.json").write_bytes(pretty(summary))
    (output / "timings.md").write_bytes(table.encode())
    (output / "work.json").write_bytes(pretty(work_records))
    (output / "validation.json").write_bytes(
        pretty(envelope(output / "raw", qualifications))
    )
    print(
        json.dumps(
            {
                "recovery": recovery,
                "pilot": qualifications["pilot"]["status"],
                "formal": qualifications["formal"]["status"],
                "formal_measured_processes": 30,
                "sdk": "one frozen production source/binary",
                "gpu": qualifications["formal"]["physical_gpu_uuid"],
            }
        )
    )


if __name__ == "__main__":
    main()
