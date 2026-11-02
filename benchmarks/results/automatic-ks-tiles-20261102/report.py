"""Replay exact default-tile evidence; incomplete allocations never rank."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import re
from datetime import datetime
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

from benchmarks._exact_fp64_evidence import load, require
from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parents[3]


def timestamp(value: str) -> float:
    """Interpret controller wall time using the cluster's recorded time zone."""
    return (
        datetime.fromisoformat(value)
        .replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        .timestamp()
    )


def lsf_lifetime(text: str, job: str) -> tuple[float, float]:
    """Accept actual terminal bjobs or bhist proofs, including finite limits."""
    require(f"Job <{job}>" in text and ("Done successfully." in text or "Exited" in text), "LSF terminal")
    require(
        re.search(r"RUNLIMIT\s+\d+(?:\.\d+)? min", text) is not None, "finite LSF time"
    )
    stamps = re.findall(
        r"(\w{3} \w{3}\s+\d+ \d\d:\d\d:\d\d): (Started|Running with execution home|Done successfully\.|Exited[^\n]*)",
        text,
    )
    times = {}
    for stamp, event in stamps:
        value = datetime.strptime(
            "2026 " + " ".join(stamp.split()), "%Y %a %b %d %H:%M:%S"
        )
        times["begin" if event in ("Started", "Running with execution home") else "end"] = value.replace(
            tzinfo=ZoneInfo("Asia/Shanghai")
        ).timestamp()
    require(set(times) == {"begin", "end"}, "LSF owner boundaries")
    return times["begin"], times["end"] + 1


def replay(folder: Path) -> dict:
    """Authenticate all retained populations before any complete timing ratio."""
    bundle, files = load(folder)
    root = ROOT if (ROOT / "benchmarks").exists() else Path.cwd()
    for name, expected in bundle["references"].items():
        require(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,
            "reference " + name,
        )
    oracles = json.loads(
        lzma.decompress(
            (
                root / "benchmarks/results/readme-refresh-20261026/oracles.json.xz"
            ).read_bytes()
        )
    )
    references = json.loads(
        lzma.decompress(
            (
                root
                / "benchmarks/results/independent-force-tiles-20261031/receipts.json.xz"
            ).read_bytes()
        )
    )
    groups = {}
    for name, text in files.items():
        path = Path(name)
        if (
            not path.name.startswith(("slot-", "conditioning-"))
            or path.suffix != ".json"
            or path.name.endswith("-owner.json")
        ):
            continue
        record = json.loads(text)
        if record.get("schema") != "generativeqc.readme-defaults.v2":
            raise ValueError("unexpected endpoint schema " + name)
        prefix = path.parent.as_posix()
        if name.startswith(("n1/", "final/n1/")):
            device = "5090"
        elif name.startswith(("n2/", "final/n2/")):
            device = "pro6000"
        else:
            device = "a100"
        gate = qualify(
            [record],
            oracles,
            reference_reuse=references[device]["analysis"]["reference_reuse"],
        )[0]
        group = groups.setdefault(
            prefix,
            {"device": device, "method": record["method"], "gates": [], "records": {}},
        )
        group["gates"].append({"path": name, "qualification": gate})
        if record.get("native_build"):
            identity_path = prefix + ("/identity.json" if device == "a100" else "/candidate-identity.json")
            proof = json.loads(files[identity_path])
            require(record["native_build"]["source_identity"] == proof["source_identity"], "allocation native source")
            require(record["native_build"]["library_sha256"] == proof["binaries"]["libgenerativeqc.so"], "allocation native binary")

        owner = json.loads(
            files[prefix + "/" + path.name.replace(".json", "-owner.json")]
        )
        complete = record["status"] == "complete" and owner.get("returncode") == 0
        driver = "default-driver.py" if "requested_scf_tile" in owner else "tile-driver.py"
        require(record["runner_sha256"] == hashlib.sha256(files[prefix + "/" + driver].encode()).hexdigest(), "executed driver bytes")
        if "driver_sha256" in owner:
            require(owner["driver_sha256"] == record["runner_sha256"], "owner driver identity")

        if complete:
            require(gate["status"] == "qualified", name + ": independent gates")
            require(
                hashlib.sha256(text.encode()).hexdigest() == owner["receipt_sha256"],
                "owner receipt checksum",
            )
            require(
                all(
                    record["scheduler"].get(k) == v
                    for k, v in owner["scheduler"].items()
                    if k != "SLURM_CPUS_PER_TASK"
                ),
                "owner visibility",
            )
            if device == "a100":
                job = owner["scheduler"]["LSB_JOBID"]
                key = (
                    "a100-pilot/lsf-" + job + ".txt"
                    if name.startswith("a100-pilot/")
                    else ("support/" if name.startswith("final/") else "scf-support/") + "lsf-" + job + ".txt"
                )
                controller = files.get(key)
                if controller is None and job == "224408":
                    controller = files["scf-support/a100-pilot-submit.log"].split(
                        "Job <224442>"
                    )[0]
                require(controller is not None, "missing LSF controller")
                begin, end = lsf_lifetime(controller, job)
                group["allocation_complete"] = "Done successfully." in controller
            else:
                job = owner["scheduler"]["SLURM_JOB_ID"]
                host = "n1" if device == "5090" else "n2"
                controller = files[("support/" if name.startswith("final/") else "scf-support/") + host + "-" + job + "-controller.txt"]
                require("JobId=" + job in controller, "Slurm identity")
                require(
                    re.search(r"TimeLimit=\d\d:\d\d:\d\d", controller) is not None,
                    "finite Slurm time",
                )
                require(
                    any(
                        "JobState=" + state in controller
                        for state in ("COMPLETED", "TIMEOUT", "FAILED", "CANCELLED")
                    ),
                    "terminal Slurm proof",
                )
                group["allocation_complete"] = "JobState=COMPLETED" in controller
                begin = timestamp(re.search(r"StartTime=(\S+)", controller)[1])
                end = timestamp(re.search(r"EndTime=(\S+)", controller)[1]) + 1
            require(
                begin
                <= owner["start_epoch"]
                <= owner["receipt_mtime"]
                <= owner["end_epoch"]
                < end,
                "owner lifetime",
            )
        group["records"][path.name] = {
            "record": record,
            "owner": owner,
            "complete": complete,
        }
    report = {
        "candidate": bundle["candidate"],
        "groups": [],
        "limitations": [
            "Only measured capabilities/workloads are qualified; no all-GPU or all-functional cold-speedup claim is inferred."
        ],
    }
    for prefix, group in sorted(groups.items()):
        device = group["device"]
        exit_text = files.get(prefix + "/exit-code.txt")
        complete = (
            exit_text is not None
            and group.get("allocation_complete", False)
            and exit_text.strip() == "0"
            and all(r["complete"] for r in group["records"].values())
        )
        observations = {}
        work = {}
        for name, row in group["records"].items():
            if not name.startswith("slot-") or not row["complete"]:
                continue
            record, owner = row["record"], row["owner"]
            if record["arm"] == "gpu4pyscf":
                variant = "gpu4pyscf"
            elif "requested_scf_tile" in owner:
                variant = owner["requested_scf_tile"]
            else:
                variant = str(owner["scf_tile_points"])
            target = observations.setdefault(variant, {"cold": [], "warm": []})
            work.setdefault(variant, [])
            for sample in record["samples"]:
                target[sample["phase"]].append(sample["seconds"])
                work[variant].append(
                    {
                        "phase": sample["phase"],
                        "iterations": sample["iterations"],
                        "fock_builds": sample.get("fock_builds"),
                        "effective_potential_builds": sample.get("effective_potential_builds"),
                        "settings": sample["settings"],
                        "precision": sample.get("precision"),
                        "ks_diagnostic": sample.get("ks_diagnostic"),
                    }
                )
                if record["arm"] != "gpu4pyscf":
                    require(
                        sample["precision"]["requested_mode"] == "fp64"
                        and sample["precision"]["mixed_stage_fock_builds"] == 0,
                        "strict FP64",
                    )
                    if prefix.startswith(("scf/a100-bench-", "final/")):
                        require(
                            record["native_build"]["source_identity"]
                            == bundle["candidate" if prefix.startswith("final/") else "scf_candidate"]["source_identity"],
                            "candidate native source",
                        )
                        expected = (
                            (512 if record["method"] == "wb97m-v" else 1024)
                            if variant == "auto" and device == "a100"
                            else 256
                        )
                        require(
                            sample["settings"]["scf_tile_points"] == expected,
                            "default resolved tile",
                        )
                        require(
                            sample["settings"]["force_tile_points"] is None,
                            "actual force default",
                        )
                        if prefix.startswith("final/"):
                            require(sample["settings"]["resolved_force_tile_points"] == (1024 if variant == "auto" else 256), "actual force schedule")

        changes = None
        if complete:
            require(
                all(
                    len(v["cold"]) == 2 and len(v["warm"]) == 6
                    for v in observations.values()
                ),
                "complete measurement census",
            )
            changes = {
                variant: {
                    phase: 100
                    * (median(v[phase]) / median(observations["256"][phase]) - 1)
                    for phase in ("cold", "warm")
                }
                for variant, v in observations.items()
            }
        result = {k: v for k, v in group.items() if k != "records"}
        result.update(
            folder=prefix,
            complete=complete,
            observations=observations,
            work=work,
            percent_vs_explicit256=changes,
        )
        if complete and "auto" in observations and "gpu4pyscf" in observations:
            result["auto_vs_gpu4_percent"] = {
                phase: 100
                * (
                    median(observations["auto"][phase])
                    / median(observations["gpu4pyscf"][phase])
                    - 1
                )
                for phase in ("cold", "warm")
            }
        report["groups"].append(result)
    report["validation"] = validation(files, bundle)
    return report


def validation(files: dict[str, str], bundle: dict) -> list[dict]:
    """Recompute larger/moved oracle stability and complete resource lifetimes."""
    import numpy as np
    from benchmarks.readme_defaults import COMMON_GATES
    from benchmarks.readme_defaults_accuracy import errors

    results = []
    for name, text in files.items():
        if not name.endswith(("/lifecycle.json", "/resource.json")):
            continue
        record = json.loads(text)
        prefix = name.rsplit("/", 1)[0]
        device = "5090" if "/n1/" in name else "pro6000" if "/n2/" in name else "a100"
        if record["status"] != "complete" or files[prefix + "/exit-code.txt"].strip() != "0":
            results.append({"path": name, "device": device, "method": record.get("method"), "complete": False, "status": record["status"], "retained_samples": len(record["samples"]), "failure": files.get(prefix + "/driver.log", "")[-1600:]})
            continue
        if device == "a100":
            job = record["scheduler"]["LSB_JOBID"]
            key = ("support/" if name.startswith("final/") else "scf-support/") + "lsf-" + job + ".txt"
            lsf_lifetime(files[key], job)
            require("Done successfully." in files[key], "validation scheduler success")
        else:
            host = "n1" if device == "5090" else "n2"
            job = record["scheduler"]["SLURM_JOB_ID"]
            controller = files["support/" + host + "-" + job + "-controller.txt"]
            require("JobState=COMPLETED" in controller and "TimeLimit=" in controller, "validation scheduler success")
        proof = json.loads(files[prefix + ("/identity.json" if device == "a100" else "/candidate-identity.json")])
        require(record["native_build"]["source_identity"] == proof["source_identity"] and record["native_build"]["library_sha256"] == proof["binaries"]["libgenerativeqc.so"], "validation native identity")
        if name.startswith("final/"):
            require(proof["source_identity"] == bundle["candidate"]["source_identity"], "final validation source")
        checked = []
        if name.endswith("/lifecycle.json"):
            require(len(record["samples"]) == 8 and len(record["references"]) == 4, "lifecycle census")
            for phase in ("original", "moved"):
                refs = {r["level"]: r["result"] for r in record["references"] if r["phase"] == phase}
                require(set(refs) == ({5, 6} if name.startswith("final/") else {4, 5}) and all(r["converged"] for r in refs.values()), "independent tighter reference")
                lo, hi = sorted(refs)
                stability = errors(refs[lo]["energy"], refs[lo]["forces"], refs[hi], 120)
                stable = all(stability[k] <= COMMON_GATES[k] / 4 for k in COMMON_GATES)
                if name.startswith("final/"):
                    require(stable, "tighter grid stability")
                checked.append({"phase": phase, "grid_stability": stability, "stable": stable})
                for sample in record["samples"]:
                    if ("moved" if sample["phase"].startswith("moved") else "original") != phase:
                        continue
                    error = errors(sample["energy"], sample["forces"], refs[hi], 120)
                    require(error["passed"], "larger/moved independent observable gate")
                    expected = 1024 if sample["variant"] == "auto" and device == "a100" else 256
                    require(sample["settings"]["scf_tile_points"] == expected, "larger/moved SCF schedule")
                    if name.startswith("final/"):
                        require(sample["settings"]["resolved_force_tile_points"] == (1024 if sample["variant"] == "auto" else 256), "larger/moved force schedule")
                    checked.append({"variant": sample["variant"], "phase": sample["phase"], "errors": error})
        else:
            require(record["explicit_refusal"] and len(record["samples"]) == 7, "force resource census")
            reference = record["samples"][0]
            root = ROOT if (ROOT / "benchmarks").exists() else Path.cwd()
            original = json.loads(lzma.decompress((root / "benchmarks/results/readme-refresh-20261026/oracles.json.xz").read_bytes()))
            oracle = next(r for r in original if r["method"] == "pbe" and r["atoms"] == 96 and r["backend"] == "cuda")
            require(record["geometry_bohr"] == oracle["geometry_bohr"], "resource oracle geometry")
            independent = oracle["samples"][-1]
            require(record["capacities"]["explicit256"][6] == 256, "conservative capacity")
            for sample in record["samples"]:
                error = errors(sample["energy"], sample["forces"], independent, 96)
                require(error["passed"], "resource independent observable gate")
                checked.append({"label": sample["label"], "errors": error})
                require(abs(sample["energy"]-reference["energy"]) <= 2e-8, "resource energy parity")
                require(float(np.max(np.abs(np.asarray(sample["forces"])-reference["forces"]))) <= 2e-7, "resource force parity")
                require(sample["settings"]["resolved_force_tile_points"] == (1024 if sample["label"] == "auto-before-limit" else 256), "bounded force schedule")
            require("DFT density-fitting resource plans" in record["numeric_plan_rejection"], "public DF numeric-promise refusal")
            require(record["samples"][-1]["settings"]["scf_tile_points"] == 256 and record["constrained_provider"]["maximum_bytes"] == 4 << 30, "constrained native provider shape")
        if name.endswith("/resource.json"):
            log = files[prefix + "/options-resources.log"]
            require(re.search(r"\b\d+ passed\b", log) is not None and "FAILED" not in log and "ERROR" not in log, "public parameter/resource suite")
        results.append({"path": name, "device": device, "method": record.get("method"), "complete": True, "checks": checked})
    return results


def require_complete(report: dict, devices: tuple[str, ...] = ("a100", "5090")) -> None:
    """Require the requested device matrix; cancelled devices do not rank."""
    methods = {"pbe", "pbe0", "b3lyp", "r2scan", "r2scan-3c", "m06-2x", "wb97m-v"}
    groups = [g for g in report["groups"] if g["folder"].startswith("final/")]
    for device in devices:
        rows = [g for g in groups if g["device"] == device]
        require(len(rows) == len(methods) and {g["method"] for g in rows} == methods and all(g["complete"] for g in rows), "complete device matrix")
        require(all(set(g["observations"]) == {"auto", "256", "gpu4pyscf"} for g in rows), "three actual measured arms")
        require(all(g["percent_vs_explicit256"]["auto"][phase] <= 2 for g in rows for phase in ("cold", "warm")), "default nonregression gate")
        checks = [v for v in report["validation"] if v["device"] == device and v["path"].startswith("final/") and v["complete"]]
        require(len(checks) == 3 and {v["method"] for v in checks if v["path"].endswith("/lifecycle.json")} == {"pbe", "r2scan-3c"}, "larger/moved/resource census")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--devices", nargs="+", choices=("a100", "5090", "pro6000"), default=("a100", "5090"))
    args = parser.parse_args()
    report = replay(Path(__file__).resolve().parent)
    if args.require_complete:
        require_complete(report, tuple(args.devices))
    print(json.dumps(report, indent=2))
