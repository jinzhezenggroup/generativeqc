"""Authenticate intrusive derivative work observations; no endpoint speed claim.

Run from the repository root with PYTHONPATH=python:.; this verifier needs no GPU.
Counters before AO screening are upper bounds, while angular timings come from
thirteen separate traversals and cannot partition the production kernel time.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re

from tools.generativeqc_validation.publication import validate_publication

ROOT = Path(__file__).resolve().parent
COUNTERS = ("shell_quartets", "tiles", "ao_quartet_upper_bound", "primitive_quartet_upper_bound")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def raw(name: str) -> bytes:
    if name.endswith(".log") or name.endswith("/job.txt"):
        return json.loads((ROOT / "receipts.json").read_text())[name].encode()
    data = (ROOT / name).read_bytes()
    return gzip.decompress(data) if name.endswith(".gz") else data


def read(name: str):
    return json.loads(raw(name))


def angular_orders() -> list[int]:
    """Invert the documented triangular pair/quartet class numbering."""
    pairs = [(high, low) for high in range(4) for low in range(high + 1)]
    return [sum(pairs[high]) + sum(pairs[low])
            for high in range(len(pairs)) for low in range(high + 1)]


def check_driver_receipts(cohort: str, binary: str, driver: str, launcher: str) -> None:
    """Bind retained source and inputs to hashes recorded inside the GPU job."""
    entries = dict(line.split(maxsplit=1)[::-1]
                   for line in raw(f"{cohort}/driver.sha256").decode().splitlines())
    names = [driver, launcher] + [f"input-{atoms}-{geometry}.txt"
                                 for atoms in (48, 96) for geometry in (0, 1)]
    for name in names:
        path = f"drivers/{name}" + (".gz" if name.startswith("input-") else ".txt")
        if cohort == "failed" and name == driver:
            path = "failed/derivative-census.cpp.txt"
        require(hashlib.sha256(raw(path)).hexdigest() == entries[f".artifacts/{name}"],
                f"{cohort}: driver/input receipt mismatch: {name}")
    if cohort != "failed":
        binaries = read("driver-binaries.json")
        require(entries[f".artifacts/{binary}"] == next(
            value for path, value in binaries.items() if path.endswith(f"/{binary}")),
            "driver binary identity mismatch")


def analyze_run(cohort: str, atoms: int, source: str) -> dict:
    suffix = "jsonl" if cohort == "census" else "txt"
    # Nsight interleaves progress messages with the JSONL application stdout.
    rows = [json.loads(line) for line in raw(f"{cohort}/work-{atoms}.{suffix}.gz").splitlines()
            if line.startswith(b"{")]
    require(all(row["kind"] in ("identity", "iteration", "result", "derivative_admissions",
                                "intrusive_angular_replay") for row in rows), "unknown record")
    result = {}
    for geometry in (0, 1):
        local = [row for row in rows if row["geometry"] == geometry]
        expected = ["identity"] + ["iteration"] * sum(row["kind"] == "iteration" for row in local)
        expected += ["result", "derivative_admissions"]
        if cohort == "angular" and geometry == 0:
            expected += ["intrusive_angular_replay"]
        require([row["kind"] for row in local] == expected, "record inventory/order changed")
        identity = local[0]
        final = next(row for row in local if row["kind"] == "result")
        admissions = next(row for row in local if row["kind"] == "derivative_admissions")
        require(identity["source_identity"] == source and identity["atoms"] == atoms
                and identity["aos"] == atoms * 8 and identity["job"] == ("5857" if cohort == "census" else "5859"),
                "source/input/job mismatch")
        require(identity["present_mask"] == identity["covered_mask"]
                and identity["observer_device_bytes"] == 880, "incomplete value observer")
        iterations = [row for row in local if row["kind"] == "iteration"]
        require([row["iteration"] for row in iterations] == list(range(1, len(iterations) + 1)),
                "missing observed iteration")
        for row in iterations:
            require(row["replay"] == 0 and not row["failed"]
                    and row["build"] == "strict-full-density", "invalid iteration")
            for field in ("j_admitted", "k_admitted"):
                require(len(row[field]) == 55 and sum(row[field]) > 0
                        and all(type(x) is int and x >= 0 for x in row[field]), "invalid value counts")
        require(final["converged"] and final["fock_builds"] == len(iterations)
                and final["xc_evaluations"] == len(iterations)
                and final["final_residual_audits"] > 0, "incomplete SCF attempt")
        require(math.isfinite(final["energy"]) and math.isfinite(final["energy_change"])
                and abs(final["energy_change"]) <= 1e-12
                and 0 <= final["physical_residual_rms"] <= 1e-10, "SCF residual gate")
        require(0 < final["ao_point_square_sum"] < final["ao_dense_point_square_sum"],
                "local AO work missing")
        require(admissions["atoms"] == atoms and admissions["shared_recurrence"]
                and admissions["source_channels"] == ["J", "K"]
                and admissions["observer_device_bytes"] == 1760
                and 0 <= admissions["max_production_replay_error"] < 1e-9,
                "derivative observer gate")
        require([c["class"] for c in admissions["classes"]] == list(range(55)), "class inventory")
        orders = [{key: 0 for key in COUNTERS} for _ in range(13)]
        for order, row in zip(angular_orders(), admissions["classes"], strict=True):
            require(all(type(row[key]) is int and row[key] >= 0 for key in COUNTERS), "invalid counts")
            require(row["shell_quartets"] <= row["tiles"] <= row["ao_quartet_upper_bound"]
                    <= row["primitive_quartet_upper_bound"], "inconsistent admission upper bounds")
            for key in COUNTERS:
                orders[order][key] += row[key]
        require(all(not any(order.values()) for order in orders[9:]), "unexpected higher-l work")
        result[str(geometry)] = {"fock_builds": final["fock_builds"], "physical_residual_rms": final["physical_residual_rms"],
                                "max_replay_error": admissions["max_production_replay_error"], "orders": orders}
    if cohort == "angular":
        replay = [row for row in rows if row["kind"] == "intrusive_angular_replay"]
        require(len(replay) == 1 and replay[0]["repeated_traversal"]
                and replay[0]["shipping_schedule"] is False
                and 0 <= replay[0]["max_production_replay_error"] < 1e-9, "angular replay gate")
        result["angular_max_replay_error"] = replay[0]["max_production_replay_error"]
    return result


def angular_times(atoms: int) -> list[float]:
    """Cross-check CSV totals against kernel events exported from raw SQLite.

    Profiler binaries stay in ignored local storage under the evidence policy;
    their exact hashes and the retained event projection remain reviewable.
    """
    rows = list(csv.DictReader(io.StringIO(raw(f"angular/statistics-{atoms}_cuda_gpu_kern_sum.csv").decode())))
    require(len(rows) == 13, "expected thirteen separate angular launches")
    totals = {}
    for row in rows:
        match = re.search(r"DirectScreeningPurpose\)1, \(bool\)1, \(int\)(\d+), \(int\)4>", row["Name"])
        require(match is not None and int(row["Instances"]) == 1, "unexpected profiled kernel")
        order = int(match[1])
        require(order not in totals and int(row["Total Time (ns)"]) > 0, "duplicate/empty angular pass")
        totals[order] = int(row["Total Time (ns)"])
    require(sorted(totals) == list(range(13)), "missing angular pass")
    events = read(f"angular/kernel-events-{atoms}.json")
    require(sorted(row["name"] for row in events) == sorted(row["Name"] for row in rows),
            "CSV kernel names differ from raw event projection")
    # Bind each exact integer-nanosecond duration to its kernel name. Sorting
    # preserves duplicate-event multiplicity while allowing export row reordering.
    raw_pairs = sorted((row["name"], row["end_ns"] - row["start_ns"]) for row in events)
    csv_pairs = sorted((row["Name"], int(row["Total Time (ns)"])) for row in rows)
    require(raw_pairs == csv_pairs, "CSV kernel name/duration pairs differ from raw profiler export")
    return [totals[order] / 1e9 for order in range(13)]


def analyze() -> dict:
    identities = read("identities.json")
    source = identities["control"]["source"]
    patch = hashlib.sha256(raw("control-source.patch.gz")).hexdigest()
    reconstruction = read("reconstruction.json")
    require(patch == identities["patches"]["control-source.patch.gz"]
            and reconstruction["patch_sha256"] == patch and reconstruction["matched"]
            and reconstruction["source"] == source
            and reconstruction["reconstruction_base"] == identities["base"], "reconstruction mismatch")
    require(raw("build/source-identity.txt").decode().strip() == source, "build source mismatch")
    require(identities["control"]["library"] in raw("build/binaries.sha256").decode(), "library mismatch")
    for cohort, job in (("census", 5857), ("angular", 5859), ("failed", 5856)):
        require(raw(f"{cohort}/job.exit").decode().strip() == ("1" if cohort == "failed" else "0"), "job status changed")
        require(f"JobId={job} " in raw(f"{cohort}/job.txt").decode()
                and "gres/gpu:5090:1" in raw(f"{cohort}/job.txt").decode(), "scheduler receipt mismatch")
        require(f"SOURCE_MATCH {identities['control']['revision']} {source}" in raw(f"{cohort}/source-check.txt").decode(), "loaded/checkout identity mismatch")
        require(all(line.endswith(": OK") for line in raw(f"{cohort}/binary-check.txt").decode().splitlines()), "binary verification failed")
        angular = cohort == "angular"
        check_driver_receipts(cohort, "derivative-orders" if angular else "derivative-census",
                              "derivative-orders.cpp" if angular else "derivative-census.cpp",
                              "profile-derivative-orders-n1.sh" if angular else "census-derivative-n1.sh")
    require(b"missing resident provider binding" in raw("failed/work-48.log"), "failed diagnostic omitted")
    return {"endpoint_qualification": False, "production_time_partition": False,
            "census": {str(atoms): analyze_run("census", atoms, source) for atoms in (48, 96)},
            "angular": {str(atoms): {"work": analyze_run("angular", atoms, source),
                                      "thirteen_pass_kernel_seconds": angular_times(atoms)} for atoms in (48, 96)}}


def verify() -> dict:
    manifest = read("publication.json")
    validate_publication(manifest, {row["path"]: (ROOT / row["path"]).read_bytes() for row in manifest["files"]})
    result = analyze()
    require(result == read("summary.json"), "derived summary mismatch")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
