"""Audit the two BH76 rounds and list what blocks each incomplete barrier."""

import argparse
import csv
import hashlib
import json
import lzma
from collections import Counter
from pathlib import Path
from statistics import median

from plan_round2 import build_plan
from replay import (
    barrier,
    failure_reason,
    load_journal,
    metric,
    sha,
    summarize,
    verify_native,
    verify_provenance,
)
from run_gpu import RECIPES, ROOT, SOURCE_REVISION
from run_round2 import ANGULAR, ENDPOINT_SECONDS, NUMERIC_BYTES, SCALAR_WORK

EIGENSOLVE = "df_eigensolve_input_refusal"

KNOWN = {
    "timeout": "protocol_limit",
    "correlation_work_policy": "resource_limit",
    "numeric_memory_refusal": "resource_limit",
    "scf_nonconvergence": "scf_convergence",
    "ccsd_nonconvergence": "ccsd_convergence",
    ANGULAR: "development_required",
    EIGENSOLVE: "development_required",
}


def verify_round2(raw: Path, plans: dict[str, Path]) -> dict[str, list[Path]]:
    identity = json.loads((raw / "build-identity.json").read_text())
    assert identity["source_revision"] == SOURCE_REVISION
    assert identity["build_driver_sha256"] == sha(ROOT / "build_production_qz.sh")
    with lzma.open(raw / "build-CMakeCache.txt.xz", "rt") as file:
        cache = file.read().splitlines()
    assert {
        "CMAKE_BUILD_TYPE:STRING=Release",
        "GENERATIVEQC_CUDA_FAST_COMPILE:BOOL=OFF",
        "GENERATIVEQC_ENABLE_AOT_SHELLS:BOOL=ON",
        "GENERATIVEQC_AOT_PROFILE:STRING=portable_cuda",
        "GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT:BOOL=ON",
        "GENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA:BOOL=ON",
        "GENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT:BOOL=OFF",
    } <= set(cache)
    hashes = json.loads((raw.parent / "acquisition-sha256.json").read_text())
    assert identity["aot_audit_sha256"] == hashes["build-aot-audit.json"]["acquired"]
    assert (
        identity["cmake_cache_sha256"] == hashes["build-CMakeCache.txt.xz"]["acquired"]
    )
    assert all(
        sha(raw / name) == record["published"] for name, record in hashes.items()
    )
    driver_name = "acquisition-round2-driver.py.xz"
    driver = lzma.decompress((raw / driver_name).read_bytes())
    driver_sha = hashlib.sha256(driver).hexdigest()
    assert driver_sha == hashes[driver_name]["acquired"]
    audit = json.loads((raw / "build-aot-audit.json").read_text())["primary"]
    assert audit["architecture"] == "sm_90" and len(audit["artifacts"]) == 12
    assert all(
        "cubin" in item["code_kinds"] and not item["driver_ptx_jit_required"]
        for item in audit["artifacts"]
    )
    journals: dict[str, list[Path]] = {}
    for journal in sorted(raw.glob("rank-*.jsonl.xz")):
        rank_id = journal.name.removesuffix(".jsonl.xz")
        receipt = json.loads((raw / f"{rank_id}-provenance.json").read_text())
        assert receipt["library_build_revision"] == SOURCE_REVISION
        assert receipt["source_revision"] == SOURCE_REVISION
        assert receipt["inputs_sha256"] == sha(ROOT / "inputs.json")
        assert receipt["driver_sha256"] == sha(ROOT / "run_gpu.py")
        assert receipt["round2_driver_sha256"] == driver_sha
        job_name = receipt["environment"]["GQC_QZ_JOB_NAME"]
        assert receipt["plan_sha256"] == sha(plans[job_name])
        journals.setdefault(job_name, []).append(journal)
        assert receipt["native_source_identity"] == identity["native_source_identity"]
        assert (
            receipt["reported_library_source_identity"]
            == identity["native_source_identity"]
        )
        assert receipt["library_sha256"] == identity["library_sha256"]
        assert receipt["max_scalar_work"] == SCALAR_WORK
        assert receipt["numeric_memory_bytes"] == NUMERIC_BYTES
        assert receipt["endpoint_limit_seconds"] == ENDPOINT_SECONDS
        assert receipt["build_variant"] == "paper_h100_production"
    return journals


def row_diagnoses(row: dict) -> list[str]:
    if row["status"] == "capability_gap":
        assert ANGULAR in row["gaps"] and not row["native_attempted"]
        if EIGENSOLVE in row["gaps"]:
            # The acquired driver copied this diagnosis from round 1.
            assert row["round1_diagnosis"] == EIGENSOLVE
        return [ANGULAR]
    if row["status"] in ("started", "planned"):
        return ["not_finished"]
    reason = failure_reason(row)
    if reason == "other_error" and any(
        marker in row["error"].lower()
        for marker in ("out of memory", "cudaerrormemoryallocation", "std::bad_alloc")
    ):
        reason = "numeric_memory_refusal"
    return [reason]


def coverage(
    data: dict, endpoints: dict[tuple[str, str], dict]
) -> tuple[list[dict], dict]:
    energies = {recipe: {} for recipe in RECIPES}
    for (system, recipe), row in endpoints.items():
        if row["status"] == "ok":
            energies[recipe][system] = verify_native(row)
    rows = []
    impact = {}
    for recipe in RECIPES:
        impact[recipe] = {}
        for entry in data["barriers"]:
            systems = [entry["transition"], *entry["side"]]
            value = barrier(energies[recipe], systems)
            missing = [system for system in systems if system not in energies[recipe]]
            row = {
                "recipe": recipe,
                "set": entry["set"],
                "index": entry["index"],
                "direction": entry["direction"],
                "reaction": entry["reaction"],
                "reference_kcal_mol": entry["ref1_kcal_mol"],
                "barrier_kcal_mol": value,
                "error": None if value is None else value - entry["ref1_kcal_mol"],
                "missing_endpoints": {},
            }
            for system in missing:
                source = endpoints.get((system, recipe))
                diagnoses = row_diagnoses(source) if source else ["not_in_round"]
                row["missing_endpoints"][system] = diagnoses
                for diagnosis in set(diagnoses) & {ANGULAR, EIGENSOLVE}:
                    impact[recipe].setdefault(diagnosis, {}).setdefault(
                        entry["set"], set()
                    ).add(entry["index"])
            rows.append(row)

    completed = {
        recipe: {
            (r["set"], r["index"])
            for r in rows
            if r["recipe"] == recipe and r["error"] is not None
        }
        for recipe in RECIPES
    }
    common = set.intersection(*completed.values())
    summary = {}
    for recipe in RECIPES:
        summary[recipe] = {}
        for group in ("HTBH38", "NHTBH38", "BH76"):
            selected = [
                r for r in rows if r["recipe"] == recipe and group in (r["set"], "BH76")
            ]
            matched = [r for r in selected if (r["set"], r["index"]) in common]
            systems = [
                key for key in data["systems"] if group in (key.split("/")[0], "BH76")
            ]
            diagnostics = [
                row_diagnoses(endpoints[key, recipe])
                for key in systems
                if (key, recipe) in endpoints and key not in energies[recipe]
            ]
            times = [
                endpoints[key, recipe]["wall_seconds"]
                for key in systems
                if key in energies[recipe]
            ]
            development_blocked = {
                (r["set"], r["index"])
                for r in selected
                if any(
                    set(reasons) & {ANGULAR, EIGENSOLVE}
                    for reasons in r["missing_endpoints"].values()
                )
            }
            summary[recipe][group] = {
                "successful_endpoints": sum(key in energies[recipe] for key in systems),
                "expected_endpoints": len(systems),
                "max_barriers_with_known_capabilities": len(selected)
                - len(development_blocked),
                "recorded_endpoints": sum(
                    (key, recipe) in endpoints for key in systems
                ),
                "successful_wall_seconds": {
                    "min": min(times),
                    "median": median(times),
                    "max": max(times),
                    "sum": sum(times),
                }
                if times
                else None,
                "failure_classes": dict(
                    sorted(
                        Counter(
                            category
                            for diagnoses in diagnostics
                            for category in {
                                KNOWN.get(reason, "unresolved") for reason in diagnoses
                            }
                        ).items()
                    )
                ),
                "diagnoses": dict(
                    sorted(
                        Counter(
                            reason for diagnoses in diagnostics for reason in diagnoses
                        ).items()
                    )
                ),
                "vs_ref1": metric(selected, "error", len(selected)),
                "vs_ref1_on_common_coverage": metric(matched, "error", len(matched)),
            }
    return rows, {
        "by_recipe": summary,
        "common_barrier_indices": sorted(common),
        "capability_gap_barrier_indices": {
            recipe: {
                reason: {
                    dataset: sorted(indices)
                    for dataset, indices in sorted(by_set.items())
                }
                for reason, by_set in sorted(reasons.items())
            }
            for recipe, reasons in impact.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "round2/raw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((ROOT / "inputs.json").read_text())
    prior_raw = ROOT / "raw"
    verify_provenance(prior_raw, data)
    prior_paths = sorted(prior_raw.glob("gpu-*.jsonl.xz"))
    first = load_journal(prior_paths, "recipe")
    plan_path = ROOT / "round2/plan.json"
    plan = json.loads(plan_path.read_text())
    assert plan["inputs_sha256"] == sha(ROOT / "inputs.json")
    assert plan["round1_journal_sha256"] == {p.name: sha(p) for p in prior_paths}
    assert plan["points"] == build_plan(data, first)
    requested = {(row["system"], row["recipe"]): row for row in plan["points"]}
    plans = {"gqc-3168-r2-campaign": plan_path}
    plans.update(
        {
            path.stem: path
            for path in sorted((plan_path.parent / "plans").glob("*.json"))
        }
    )
    journals = verify_round2(args.raw, plans)
    retries = {}
    for name, path in plans.items():
        job_plan = json.loads(path.read_text())
        assert job_plan["inputs_sha256"] == plan["inputs_sha256"]
        assert job_plan["round1_journal_sha256"] == plan["round1_journal_sha256"]
        points = {(row["system"], row["recipe"]): row for row in job_plan["points"]}
        assert all(row == requested[key] for key, row in points.items())
        if name != "gqc-3168-r2-campaign":
            assert not any(
                retries.get(key, {}).get("status") not in (None, "started")
                for key in points
            )
        attempt = load_journal(journals.get(name, []), "recipe")
        assert attempt.keys() <= points.keys()
        retries.update(attempt)
    for key, row in retries.items():
        assert row["round1_diagnosis"] == requested[key]["round1_diagnosis"]
        assert row["round1_error"] == requested[key]["round1_error"]

    complete = {
        **first,
        **{key: {**task, "status": "planned"} for key, task in requested.items()},
        **retries,
    }
    stages = {
        "round1": coverage(data, first),
        "round2_only": coverage(data, retries),
        "cumulative": coverage(data, complete),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {phase: counts for phase, (_, counts) in stages.items()}
    cpu = load_journal(sorted(prior_raw.glob("cpu-*.jsonl.xz")), "cardinal")
    _, anchors = summarize(data, complete, cpu)
    summary["cumulative_vs_retained_cpu_anchors"] = {
        recipe: anchors[recipe]["BH76"]["gpu_vs_canonical"] for recipe in RECIPES
    }
    summary["recoveries_by_round1_diagnosis"] = {
        recipe: dict(
            sorted(
                Counter(
                    requested[key]["round1_diagnosis"]
                    for key, row in retries.items()
                    if key[1] == recipe and row["status"] == "ok"
                ).items()
            )
        )
        for recipe in RECIPES
    }
    summary["former_timeouts"] = {}
    for recipe in RECIPES:
        timed_out = [
            key
            for key in first
            if key[1] == recipe and first[key]["status"] == "timeout"
        ]
        recovered = [
            key for key in timed_out if retries.get(key, {}).get("status") == "ok"
        ]
        summary["former_timeouts"][recipe] = {
            "round1_count": len(timed_out),
            "round2_successes": len(recovered),
            "round2_successes_below_round1_process_limit": sum(
                retries[key]["process_wall_seconds"] < first[key]["limit_seconds"]
                for key in recovered
            ),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = []
    for phase, (barriers, _) in stages.items():
        for entry in barriers:
            rows.append(
                {
                    "phase": phase,
                    **entry,
                    "missing_endpoints": json.dumps(
                        entry["missing_endpoints"], sort_keys=True
                    ),
                }
            )
    with (args.output / "barriers.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    endpoints = [
        {
            "recipe": recipe,
            "system": key,
            "round1_status": first[key, recipe]["status"],
            "round1_diagnosis": ""
            if first[key, recipe]["status"] == "ok"
            else failure_reason(first[key, recipe]),
            "round2_status": retries[key, recipe]["status"]
            if (key, recipe) in retries
            else "",
            "round2_diagnoses": " ".join(row_diagnoses(retries[key, recipe]))
            if (key, recipe) in retries and retries[key, recipe]["status"] != "ok"
            else "",
            "round2_native_attempted": retries.get((key, recipe), {}).get(
                "native_attempted", ""
            ),
            "energy_round": "1"
            if first[key, recipe]["status"] == "ok"
            else "2"
            if retries.get((key, recipe), {}).get("status") == "ok"
            else "",
        }
        for key, recipe in sorted(first)
    ]
    with (args.output / "endpoints.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(endpoints[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(endpoints)


if __name__ == "__main__":
    main()
