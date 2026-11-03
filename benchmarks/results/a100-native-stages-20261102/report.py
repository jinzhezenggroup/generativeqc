"""Replay immutable A100 native-stage evidence on a CPU, including failures.

This reader authenticates the original texts, compares full force vectors and
recomputes the emitted independent RSH finite differences. It never treats a
loader failure as a sanitizer pass or promotes a historical source to HEAD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from benchmarks._exact_fp64_evidence import load, require
from benchmarks.readme_defaults_accuracy import errors

SOURCES = {
    2902: "4b8ab33e04c4347589fdaa232c6eb3a3e8a079c5385d827f0a39dea564a01f51",
    2900: "5aa7bf5569c69f173232442e9cf7ba52be80fcb3381b618a8254215e0ae62ffc",
}
DENSE_SHA = "d61a3e8ba0a8f8a8c08ee4c4b30c8e3e397e67863b8d75e7ba974e90cf0f85a4"
NATIVE = (
    "generativeqc_cuda_fock_composition_tests",
    "generativeqc_native_stationary_cuda_binding_tests",
)


def distance(a: list, b: list) -> float:
    """Require complete finite vectors before comparing their largest error."""
    require(len(a) == len(b) and bool(a), "vector shape")
    require(all(math.isfinite(x) for x in [*a, *b]), "nonfinite vector")
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


def rsh_stencils(text: str) -> dict:
    """Replay both CPU steps and all three signed derivative channels."""
    cases = []
    for line in text.splitlines():
        if line.startswith("RSH_CASE "):
            cases.append({"case": json.loads(line[9:]), "stencils": [], "vectors": {}})
        elif line.startswith("RSH_CPU_STENCIL "):
            cases[-1]["stencils"].append(json.loads(line[16:]))
        elif line.startswith("RSH_VECTOR "):
            name, raw = line[11:].split(" ", 1)
            cases[-1]["vectors"][name] = json.loads(raw)
    require(len(cases) == 10, "missing RSH cases")
    worst = 0.0
    count = 0
    for case in cases:
        vectors = case["vectors"]
        require(len(vectors["density-alpha"]) == case["case"]["nbf"] ** 2, "density")
        if case["case"]["spin"] == "UKS":
            require(
                distance(vectors["density-alpha"], vectors["density-beta"]) > 0,
                "distinct spin densities",
            )
        channels = {}
        for row in case["stencils"]:
            e = row["energies"]
            require(len(e) == 4 and row["step"] > 0, "stencil shape")
            value = (e[0] - 8 * e[1] + 8 * e[2] - e[3]) / (12 * row["step"])
            require(abs(value - row["derivative"]) < 2e-12, "recorded derivative")
            channels.setdefault((row["range"], row["coordinate"]), []).append(
                (row["step"], value)
            )
            count += 1
        require(len(channels) == 12, "six coordinates for each exchange channel")
        for (kind, coordinate), values in channels.items():
            require(len(values) == 2, "two independent steps")
            values.sort()
            require(abs(values[0][1] - values[1][1]) < 2e-9, "step stability")
            index = (1 if kind == "short" else 2) * 6 + coordinate
            require(
                abs(vectors["cpu"][index] - values[0][1]) < 2e-12, "CPU stencil binding"
            )
        for route in ("host", "resident", "borrowed"):
            worst = max(worst, distance(vectors["cpu"], vectors[route]))
    require(count == 240 and worst < 2e-9, "independent RSH derivative gate")
    drains = [
        json.loads(x.removeprefix("RSH_FAILURE_DRAIN "))
        for x in text.splitlines()
        if x.startswith("RSH_FAILURE_DRAIN ")
    ]
    require(
        {x["spin"] for x in drains if x["pending_witness"]} == {"RKS", "UKS"},
        "pending-work failure drain",
    )
    return {
        "cases": len(cases),
        "stencils": count,
        "maximum_error": worst,
        "passed": True,
    }


def report(folder: Path) -> dict:
    """Return source-bound acceptance and every completed latency population."""
    files = {}
    archives = []
    omitted = []
    for part in (
        [folder]
        if (folder / "sha256.txt").exists()
        else sorted(p.parent for p in folder.glob("*/sha256.txt"))
    ):
        snapshot = part / "historical-snapshot.json"
        bundle, recovered = load(
            part,
            historical_manifest=(
                snapshot
                if snapshot.exists() and part.resolve().is_relative_to(ROOT)
                else None
            ),
        )
        archives.extend(bundle["input_archives"])
        omitted.extend(bundle.get("intrusive_stream_inventory", []))
        for name, text in recovered.items():
            require(
                name not in files or files[name] == text, "conflicting shard receipts"
            )
            files[name] = text
    require(bool(archives), "missing evidence shards")
    dense_path = ROOT / "benchmarks/results/readme-repaired-20261021/oracles.json.xz"
    require(
        hashlib.sha256(dense_path.read_bytes()).hexdigest() == DENSE_SHA,
        "dense oracle identity",
    )
    dense = {
        (x["method"], x["atoms"]): x
        for x in json.loads(lzma.decompress(dense_path.read_bytes()))
        if x["backend"] == "cuda"
    }
    output = {
        "scope": "A100 incremental stages; final majority and independent review remain separate",
        "input_archives": archives,
        "sources": SOURCES,
        "intrusive_stream_inventory": omitted,
        "runs": [],
        "matrix": [],
        "fresh_oracles": [],
    }
    for prefix in sorted(
        {
            p.split("/")[0]
            for p in files
            if p.startswith(("build-", "checks-", "bench-"))
        }
    ):
        exit_path = prefix + "/exit-code.txt"
        if exit_path not in files:
            continue
        row = {
            "run": prefix,
            "exit_code": int(files[exit_path]),
            "normal": {},
            "memcheck": {},
        }
        for kind in ("production", "fixture"):
            path = f"{prefix}/{kind}-identity.json"
            if path in files:
                proof = json.loads(files[path])
                if "source_identity" not in proof:
                    row["fixture_overlay"] = proof
                    continue
                require(proof["source_identity"] in SOURCES.values(), "measured source")
                require(
                    proof["build_tests"] == (kind == "fixture"),
                    "production/fixture separation",
                )
                row[kind] = proof
        for name in NATIVE:
            for mode in ("normal", "memcheck"):
                path = f"{prefix}/{name}-{mode}.exit"
                if path not in files:
                    continue
                status = int(files[path])
                log = files.get(f"{prefix}/{name}-{mode}.log", "")
                receipt = {"exit_code": status, "test_pass": "PASS" in log}
                if mode == "memcheck":
                    sanitizer = files.get(f"{prefix}/{name}-memcheck.txt", "")
                    receipt["qualified"] = (
                        status == 0
                        and "--target-processes all" in files[prefix + "/runner.sh"]
                        and "PASS" in log
                        and "ERROR SUMMARY: 0 errors" in sanitizer
                        and "before first instrumented API" not in sanitizer
                    )
                row[mode][name] = receipt
        identity_path = prefix + "/identity.json"
        if identity_path in files:
            identity = json.loads(files[identity_path])
            require(identity["source_identity"] in SOURCES.values(), "molecular source")
            row["molecular_identity"] = identity
        for name in ("tests", "tests-production", "tests-fixture"):
            xml = files.get(f"{prefix}/{name}.xml")
            if not xml:
                continue
            suites = ET.fromstring(xml)
            suites = list(suites) if suites.tag == "testsuites" else [suites]
            row.setdefault("test_suites", {})[name] = {
                k: sum(int(x.attrib.get(k, 0)) for x in suites)
                for k in ("tests", "failures", "errors", "skipped")
            }
        runtime_path = prefix + "/target-runtime-identity.json"
        if runtime_path in files:
            runtime = json.loads(files[runtime_path])
            for name, proof in runtime.items():
                require(
                    proof["original"]["sha256"] == row["fixture"]["binaries"][name],
                    "frozen native executable",
                )
                require(
                    proof["original"]["scientific_sections"]
                    == proof["runtime_copy"]["scientific_sections"],
                    "unchanged executable CUDA/scientific sections",
                )
            row["runtime_loader_proof"] = runtime
        for mode in ("normal", "memcheck"):
            text = files.get(f"{prefix}/{NATIVE[0]}-{mode}.log", "")
            if "RSH_CASE " in text:
                row.setdefault("rsh_stencils", {})[mode] = rsh_stencils(text)
        output["runs"].append(row)
        oracle = files.get(prefix + "/oracle.json")
        if oracle:
            fresh = json.loads(oracle)
            original = json.loads(files[prefix + "/old-reference.json"])
            require(
                fresh["geometry_bohr"] == original["geometry_bohr"],
                "fresh oracle geometry",
            )
            require(
                len(fresh["samples"]) == len(original["samples"]) == 2, "oracle levels"
            )
            comparisons = [
                errors(x["energy"], x["forces"], y, fresh["atoms"])
                for x, y in zip(fresh["samples"], original["samples"], strict=True)
            ]
            comparisons.append(
                errors(
                    fresh["samples"][0]["energy"],
                    fresh["samples"][0]["forces"],
                    fresh["samples"][1],
                    fresh["atoms"],
                )
            )
            quarter = all(
                x["energy_abs_per_atom_hartree"] <= 2.5e-6
                and x["force_max_abs_hartree_per_bohr"] <= 2.5e-5
                and x["force_rms_hartree_per_bohr"] <= 7.5e-6
                for x in comparisons
            )
            output["fresh_oracles"].append(
                {
                    "run": prefix,
                    "method": fresh["method"],
                    "atoms": fresh["atoms"],
                    "comparisons": comparisons,
                    "quarter_gates_pass": quarter,
                }
            )
    for job in (224654, 224655, *range(224683, 224697)):
        prefix = f"bench-{job}/"
        if prefix + "campaign.json" not in files:
            continue
        campaign = json.loads(files[prefix + "campaign.json"])
        processes = {}
        for path, raw in files.items():
            if (
                path.startswith(prefix)
                and "-pair" in path
                and path.endswith(".json")
                and not path.endswith(".plan.json")
            ):
                d = json.loads(raw)
                if d.get("schema") != "generativeqc.readme-defaults.v2":
                    continue
                pair = int(path.split("-pair")[1].split("-")[0])
                require(
                    d["source_identity"] == SOURCES[int(campaign["pr"])],
                    "endpoint source",
                )
                require(
                    d["status"] == "complete" and len(d["samples"]) == 4,
                    "complete cold plus three warm",
                )
                require(
                    d["geometry_bohr"]
                    == dense[d["method"], d["atoms"]]["geometry_bohr"],
                    "dense geometry",
                )
                if d["arm"] == "standard":
                    require(not d["build_identity"]["build_tests"], "timed fixture")
                    require(
                        d["native_build"]["library_sha256"]
                        == d["build_identity"]["binaries"]["libgenerativeqc.so"],
                        "timed core hash",
                    )
                processes[d["atoms"], d["mode"], pair, d["arm"]] = d
        for atoms in (48, 96):
            for mode in ("direct", "df"):
                pairs = []
                for pair in range(3):
                    native = processes[atoms, mode, pair, "standard"]
                    reference = processes[atoms, mode, pair, "gpu4pyscf"]
                    cross = [
                        errors(x["energy"], x["forces"], y, atoms)
                        for x in native["samples"]
                        for y in reference["samples"]
                    ]
                    oracle = dense[campaign["method"], atoms]["samples"][-1]
                    gates = [
                        errors(s["energy"], s["forces"], oracle, atoms)
                        for d in (native, reference)
                        for s in d["samples"]
                    ]
                    timings = {
                        d["arm"]: [s["seconds"] for s in d["samples"]]
                        for d in (native, reference)
                    }
                    ratio = statistics.median(
                        timings["gpu4pyscf"][1:]
                    ) / statistics.median(timings["standard"][1:])
                    pairs.append(
                        {
                            "pair": pair,
                            "seconds": timings,
                            "ratio": ratio,
                            "paired_pass": all(x["passed"] for x in cross),
                            "dense_pass": all(x["passed"] for x in gates),
                            "dense_errors": gates,
                            "iterations": {
                                d["arm"]: [s["iterations"] for s in d["samples"]]
                                for d in (native, reference)
                            },
                        }
                    )
                output["matrix"].append(
                    {
                        "pr": campaign["pr"],
                        "method": campaign["method"],
                        "atoms": atoms,
                        "mode": mode,
                        "pairs": pairs,
                        "median_ratio": statistics.median(x["ratio"] for x in pairs),
                        "qualified": all(
                            x["paired_pass"] and x["dense_pass"] for x in pairs
                        ),
                    }
                )
    output["rsh_independent"] = rsh_stencils(
        files["checks-224637/generativeqc_cuda_fock_composition_tests-normal.log"]
    )
    output["sanitizer_qualified"] = {
        pr: any(
            r.get("fixture", {}).get("source_identity") == source
            and all(
                r["memcheck"].get(name, {}).get("qualified", False) for name in NATIVE
            )
            for r in output["runs"]
        )
        for pr, source in SOURCES.items()
    }
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report(args.folder)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
