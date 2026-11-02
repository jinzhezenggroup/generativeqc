"""Replay metric scheduling evidence, independent gates and complete timings."""

from __future__ import annotations

import hashlib
import json
import lzma
import runpy
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from benchmarks._exact_fp64_evidence import load, require
from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parents[3]
ANCHOR = runpy.run_path(
    str(ROOT / "benchmarks/results/range-anchor-default-20261101/report.py")
)
PRIOR = runpy.run_path(
    str(ROOT / "benchmarks/results/deferred-range-anchor-20261101/report.py")
)


def timing(records: list[dict]) -> dict:
    """Keep every repeat, first warm cost and dispersion in the result."""
    samples = [record["samples"] for record in records]
    values = {
        "cold": [rows[0]["seconds"] for rows in samples],
        "first_warm": [rows[1]["seconds"] for rows in samples],
        "warm": [row["seconds"] for rows in samples for row in rows[1:]],
        "lifecycle": [sum(row["seconds"] for row in rows) for rows in samples],
    }
    return {key: ANCHOR["statistics"](items) for key, items in values.items()}


def main() -> None:
    """Authenticate the retained bytes, qualify all arms, then compare time."""
    bundle, files = load(Path(__file__).parent)

    def read(name: str) -> dict:
        return json.loads(files[name])

    def completed(folder: str) -> None:
        require(files[folder + "/exit-code.txt"].strip() == "0", folder)

    candidate, parent = read("source.json"), read("pilot/source.json")
    build = read("build-provenance.json")
    require(build["source_identity"] == candidate["source_identity"], "build source")
    require(
        build["native_library_sha256"] == candidate["binaries"]["libgenerativeqc.so"],
        "build library",
    )
    references = {}
    for name, digest in bundle["references"].items():
        raw = (ROOT / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, name)
        references[name] = json.loads(lzma.decompress(raw))
    oracles = references["benchmarks/results/readme-refresh-20261026/oracles.json.xz"]
    reuse = references[
        "benchmarks/results/independent-force-tiles-20261031/receipts.json.xz"
    ]

    def owner(folder: str, arm: str = "candidate", proof: dict = candidate) -> dict:
        identity = read(folder + "/" + arm + "-identity.json")
        for key in ("source_identity", "source_archive_sha256", "binaries"):
            require(identity[key] == proof[key], "owner identity: " + key)
        return identity

    def qualify_record(record: dict, device: str) -> dict:
        driver = build["inherited_receipt_revision"]
        require(
            record["source_revision"] == driver["value"], "inherited driver revision"
        )
        require(record["runner_sha256"] == driver["driver_sha256"], "primary driver")
        if record["arm"] == "standard":
            for sample in record["samples"]:
                require(
                    sample["precision"]["requested_mode"] == "fp64"
                    and sample["precision"]["mixed_stage_fock_builds"] == 0,
                    "strict FP64 endpoint",
                )
        gate = qualify(
            [record],
            oracles,
            reference_reuse=reuse[device]["analysis"]["reference_reuse"],
        )[0]
        require(gate["status"] == "qualified", str(gate))
        return gate

    def paired(left: dict, right: dict) -> list[dict]:
        PRIOR["same_task"](left, right)
        errors = []
        for a, b in zip(left["samples"], right["samples"], strict=True):
            require(
                (a["iterations"], a["fock_builds"])
                == (b["iterations"], b["fock_builds"]),
                "paired physical work",
            )
            errors.append(PRIOR["errors"](a, b))
        return errors

    summaries, pilot_summaries, gates = [], [], {}
    exclusions = []
    census = set()
    for cohort in bundle["cohorts"]:
        folder, device = cohort["folder"], cohort["reference_device"]
        completed(folder)
        identities = {
            "candidate": owner(folder),
            "parent": owner(folder, "parent", parent),
        }
        continuation = cohort.get("continuation")
        require(
            bool(continuation) == folder.endswith("compare-wb97m-v-15148"),
            "resource continuation is required for the interrupted cohort",
        )
        if continuation:
            completed(continuation)
            require(
                files[continuation + "/driver-sha256.txt"].split()[0]
                == hashlib.sha256(files["run-tail-v3.sh"].encode()).hexdigest(),
                "continuation runner",
            )
            replacement_identity = owner(continuation, "parent", parent)
            interrupted = read(continuation + "/allocation-interruption.json")
            controller = interrupted["controller"]
            require(
                "JobId=15148 " in controller and "JobState=TIMEOUT " in controller,
                "original allocation timed out",
            )
            ended = controller.split("EndTime=", 1)[1].split()[0]
            written = datetime.fromtimestamp(
                interrupted["mtime_unix"], ZoneInfo("Asia/Shanghai")
            ).replace(tzinfo=None)
            require(
                written.isoformat() == interrupted["mtime_local"],
                "original receipt timestamp",
            )
            require(written > datetime.fromisoformat(ended), "documented late receipt")
            completion = read(continuation + "/completion.json")
            control = completion["controller"]
            require(
                "JobState=COMPLETED " in control and "TimeLimit=00:20:00 " in control,
                "finite successful continuation allocation",
            )
            job = replacement_identity["scheduler"]["SLURM_JOB_ID"]
            require(
                "JobId=" + job + " " in control and job != "15148",
                "actual new allocation",
            )
            start = datetime.fromisoformat(control.split("StartTime=", 1)[1].split()[0])
            end = datetime.fromisoformat(control.split("EndTime=", 1)[1].split()[0])
            write_time = datetime.fromtimestamp(
                completion["mtime_unix"], ZoneInfo("Asia/Shanghai")
            ).replace(tzinfo=None)
            # Slurm's controller timestamp has whole-second resolution.
            require(
                start <= write_time < end + timedelta(seconds=1),
                "replacement receipt inside allocation",
            )
        arms = {name: [] for name in ("parent", "candidate", "gpu4")}
        order = (
            "parent",
            "candidate",
            "gpu4",
            "parent",
            "candidate",
            "gpu4",
            "gpu4",
            "candidate",
            "parent",
        )
        for slot, arm in enumerate(order):
            name = folder + f"/slot-{slot}-{arm}.json"
            record = read(name)
            identity = identities.get(arm)
            if continuation and slot == 8:
                require(
                    arm == "parent"
                    and interrupted["excluded_population"] == "slot-8-parent.json",
                    "only the resource-invalid final parent is replaced",
                )
                require(
                    hashlib.sha256(files[name].encode()).hexdigest()
                    == interrupted["sha256"],
                    "unchanged original complete receipt",
                )
                ANCHOR["bind"](record, identity)
                gates[name] = qualify_record(record, device)
                exclusions.append(
                    {
                        "path": name,
                        "reason": interrupted["reason"],
                        "timings": timing([record]),
                        "scientific_gate": gates[name],
                        "paired_errors": paired(
                            record, read(folder + "/slot-7-candidate.json")
                        ),
                    }
                )
                original = record
                name = continuation + "/slot-8-parent.json"
                record = read(name)
                require(
                    hashlib.sha256(files[name].encode()).hexdigest()
                    == completion["sha256"],
                    "replacement complete receipt",
                )
                PRIOR["same_task"](original, record)
                identity = replacement_identity
            require(
                record["arm"] == ("gpu4pyscf" if arm == "gpu4" else "standard"),
                "endpoint arm",
            )
            if arm != "gpu4":
                ANCHOR["bind"](record, identity)
            gates[name] = qualify_record(record, device)
            require(
                [row["phase"] for row in record["samples"]]
                == ["cold"] + ["warm"] * (1 if slot < 3 else 3),
                "complete conditioning/measured lifecycle",
            )
            if slot >= 3:
                arms[arm].append(record)
        key = (device, record["method"])
        require(key not in census, "duplicate method/device cohort")
        census.add(key)
        paired_errors = []
        for left, right, reference in zip(*arms.values(), strict=True):
            paired_errors.extend(paired(left, right))
            for field in (
                "method",
                "mode",
                "atoms",
                "geometry_bohr",
                "orbital_basis",
                "basis_representation",
                "charge",
                "spin",
                "scheduler",
                "thread_controls",
            ):
                require(right[field] == reference[field], "reference task: " + field)
        times = {arm: timing(records) for arm, records in arms.items()}
        changes = {
            base: {
                phase: 100
                * (times["candidate"][phase]["median"] / values["median"] - 1)
                for phase, values in times[base].items()
            }
            for base in ("parent", "gpu4")
        }
        summaries.append(
            {
                "device": device,
                "method": record["method"],
                "folder": folder,
                "continuation": continuation,
                "timings": times,
                "candidate_vs_percent": changes,
                "paired_errors": paired_errors,
            }
        )
    methods = {"pbe", "pbe0", "b3lyp", "r2scan", "r2scan-3c", "m06-2x", "wb97m-v"}

    for cohort in bundle["pilot"]:
        folder, device = cohort["folder"], cohort["reference_device"]
        completed(folder)
        identity = owner(folder, proof=parent)
        arms = {name: [] for name in ("auto", "auxiliary")}
        for slot, arm in enumerate(
            ("auto", "auxiliary", "auto", "auxiliary", "auxiliary", "auto")
        ):
            name = folder + f"/slot-{slot}-{arm}.json"
            record = read(name)
            ANCHOR["bind"](record, identity)
            control = read(folder + f"/slot-{slot}-control.json")
            require(control["metric_mapping"] == arm, "pilot mapping")
            require(
                control["source_identity"] == parent["source_identity"], "pilot source"
            )
            gates[name] = qualify_record(record, device)
            require(
                [row["phase"] for row in record["samples"]]
                == ["cold"] + ["warm"] * (1 if slot < 2 else 3),
                "pilot lifecycle",
            )
            if slot >= 2:
                arms[arm].append(record)
        errors = [
            error for a, b in zip(*arms.values(), strict=True) for error in paired(a, b)
        ]
        pilot_summaries.append(
            {
                "device": device,
                "timings": {arm: timing(records) for arm, records in arms.items()},
                "paired_errors": errors,
            }
        )

    checks = bundle["checks"]
    completed(checks)
    owner(checks)
    require("16 passed" in files[checks + "/packed.log"], "mapping test census")
    require("4 passed" in files[checks + "/g.log"], "derivative test census")
    independent = read(checks + "/independent/report.json")
    require(
        independent["library_hash"] == candidate["binaries"]["libgenerativeqc.so"],
        "independent native library",
    )
    require(
        independent["probe_hash"]
        == candidate["binaries"]["generativeqc_df_value_probe"],
        "independent probe",
    )
    require(
        independent["passed"] and len(independent["runs"]) == 48,
        "complete independent case census",
    )
    for row in independent["runs"]:
        require(row["passed"] and row["ranks_match"], row["name"])
        require(
            all(error["passed"] for error in row["errors"].values()),
            "independent tensor gate",
        )
    lifecycle = bundle["lifecycle"]
    completed(lifecycle)
    owner(lifecycle)
    require(
        "6 passed" in files[lifecycle + "/tests.log"], "independent lifecycle census"
    )

    sanitizer = bundle["memcheck"]
    completed(sanitizer)
    owner(sanitizer)
    for fixture in (
        "cartesian",
        "spherical",
        "spherical-cartesian",
        "cartesian-spherical",
        "long",
        "dependent",
    ):
        log = files[sanitizer + "/" + fixture + "-memcheck.log"]
        require("ERROR SUMMARY: 0 errors" in log, "native memcheck")
        require("LEAK SUMMARY: 0 bytes leaked in 0 allocations" in log, "native leaks")
        require(
            all(
                error["passed"]
                for error in read(sanitizer + "/" + fixture + "-gates.json").values()
            ),
            "sanitized independent tensors",
        )

    large = bundle["large"]
    completed(large)
    large_records = {}
    for arm, proof in (("parent", parent), ("candidate", candidate)):
        identity = owner(large, arm, proof)
        record = read(large + "/large-" + arm + ".json")
        require(
            record["status"] == "complete" and record["atoms"] == 108,
            "larger complete endpoint",
        )
        require(
            record["native_build"]["source_identity"] == proof["source_identity"],
            "larger source",
        )
        require(
            record["native_build"]["library_sha256"]
            == proof["binaries"]["libgenerativeqc.so"],
            "larger library",
        )
        for key, value in identity["scheduler"].items():
            require(record["scheduler"][key] == value, "larger allocation")
        require(
            [row["phase"] for row in record["samples"]]
            == [
                "cold",
                "retained",
                "evicted-fallback",
                "second",
                "moved",
                "moved-warm",
            ],
            "larger lifecycle",
        )
        large_records[arm] = record
    left, right = large_records.values()
    for key in (
        "method",
        "atoms",
        "geometry_bohr",
        "moved_positions_bohr",
        "scheduler",
        "density_fitting_memory_budget_bytes",
    ):
        require(left[key] == right[key], "larger task: " + key)
    large_rows = []
    for a, b in zip(left["samples"], right["samples"], strict=True):
        for row in (a, b):
            PRIOR["work"](row)
        for key in ("phase", "iterations", "fock_builds", "evaluated_positions_bohr"):
            require(a[key] == b[key], "larger work: " + key)
        large_rows.append(
            {
                "phase": a["phase"],
                "parent_seconds": a["seconds"],
                "candidate_seconds": b["seconds"],
                "errors": PRIOR["errors"](a, b),
            }
        )

    profile = bundle["profile"]
    completed(profile)
    for arm, proof in (("parent", parent), ("candidate", candidate)):
        identity = owner(profile, arm, proof)
        record = read(profile + "/" + arm + ".json")
        ANCHOR["bind"](record, identity)
        gates[profile + "/" + arm + ".json"] = qualify_record(record, "5090")
    work = read(profile + "/metric-work-summary.json")
    a, b = (work["arms"][arm] for arm in ("parent", "candidate"))
    require(
        a["source_identity"] == parent["source_identity"]
        and b["source_identity"] == candidate["source_identity"],
        "profile source",
    )
    require(
        a["complete_work_receipts"] == b["complete_work_receipts"],
        "unchanged executed work",
    )
    require(
        a["derived_output_elements"] == b["derived_output_elements"],
        "unchanged metric domain",
    )
    require(
        a["launched_threads"] == 32 * b["launched_threads"], "measured metric placement"
    )
    require(
        census
        == {(device, method) for device in ("5090", "pro6000") for method in methods},
        "complete seven-method two-device census",
    )
    require(len(exclusions) == 1, "retain the one resource-unqualified complete owner")
    print(
        json.dumps(
            {
                "source": candidate,
                "cohorts": summaries,
                "pilot": pilot_summaries,
                "independent_gates": gates,
                "resource_excluded_populations": exclusions,
                "large_paired_only": large_rows,
                "diagnostic_metric_work": work,
                "pending_a100": bundle["pending_a100"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
