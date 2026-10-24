"""Recompute errors from lossless raw endpoint returns and committed oracles."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import lzma
import math
from pathlib import Path, PurePosixPath
import re
import subprocess

# Integrity and numerical gates below use assertions, which optimization removes.
if not __debug__:
    raise RuntimeError(
        "CPU evidence verification requires non-optimized Python; "
        "unset PYTHONOPTIMIZE and omit -O/-OO"
    )

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def dot(a, b):
    return sum(x * y for row, other in zip(a, b, strict=True)
               for x, y in zip(row, other, strict=True))


def errors(value, reference, direction):
    delta = [x-y for row, other in zip(value["forces"], reference["forces"], strict=True)
             for x, y in zip(row, other, strict=True)]
    numbers = [value["energy"], reference["energy"], *delta]
    assert all(math.isfinite(x) for x in numbers)
    e = abs(value["energy"] - reference["energy"])
    f = max(map(abs, delta))
    projection = abs(dot(value["forces"], direction)-dot(reference["forces"], direction))
    return {"energy_abs_Eh": e, "force_max_Eh_Bohr": f,
            "force_rms_Eh_Bohr": math.sqrt(sum(x*x for x in delta)/len(delta)),
            "direction_projection_error_Eh_Bohr": projection,
            "pass": e <= 1e-9 and f <= 2e-6}


def load_raw(path):
    bundle = json.loads(lzma.decompress(path.read_bytes()))
    assert set(bundle["files"]) == set(bundle["file_sha256"])
    for name, content in bundle["files"].items():
        assert sha(content.encode()) == bundle["file_sha256"][name], name
    for name, row in bundle.get("file_metadata", {}).items():
        data = bundle["files"][name].encode()
        assert len(data) == row["bytes"] and sha(data) == row["sha256"]
        assert type(row["mode"]) is int and 0 <= row["mode"] <= 0o777
    return bundle


def git_source_identity(revision):
    """Recompute the official source inventory from committed LF blobs."""
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    manifest = "cmake/GenerativeQCSourceIdentity.json"
    specification = json.loads(subprocess.check_output(
        ["git", "show", f"{revision}:{manifest}"], cwd=ROOT))
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", revision], cwd=ROOT, text=True).splitlines()
    chosen = set(specification["files"]) | {manifest}
    for group in specification["recursive_groups"]:
        prefix = group["root"] + "/"
        chosen.update(path for path in paths if path.startswith(prefix)
                      and any(fnmatch.fnmatch(path[len(prefix):], pattern)
                              for pattern in group["patterns"]))
    chosen = sorted(chosen)
    data = subprocess.run(["git", "cat-file", "--batch"], cwd=ROOT,
                          input="".join(f"{revision}:{p}\n" for p in chosen).encode(),
                          capture_output=True, check=True).stdout
    offset = 0
    records = []
    for path in chosen:
        end = data.index(b"\n", offset)
        header = data[offset:end].split()
        assert header[1] == b"blob"
        size = int(header[2])
        offset = end + 1
        records.append(f"{path}:{sha(data[offset:offset+size])}\n")
        offset += size + 1
    assert offset == len(data)
    return sha("".join(records).encode())


def replay(bundle, cases=("h2-dz-tz", "water-algebra")):
    files = bundle["files"]
    identity = json.loads(files["identity.json"])
    source = identity["source_commit"]
    for name, expected in identity["inputs"].items():
        content = subprocess.check_output(["git", "show", f"{source}:{name}"], cwd=ROOT)
        assert sha(content) == expected, name
    for name, expected in bundle["file_sha256"].items():
        assert sha(files[name].encode()) == expected, name
    calls = [json.loads(line) for line in files["native/endpoints.jsonl"].splitlines()]
    reports = json.loads(files["native/reports.json"])
    assert not any(r["outcome"] == "failed" for r in reports)
    assert any(r["outcome"] == "passed" for r in reports)
    outcome = json.loads(files["outcome.json"])
    if "fixture_exit" in outcome:
        assert outcome["fixture_exit"] == 0 and "exception" not in outcome
    else:
        assert outcome["native_exit"] == 0 and outcome["unit_exit"] == 0
    rows = {}
    for case in cases:
        path = f"tests/reference_data/cbs/{case}.json"
        ref = json.loads(subprocess.check_output(["git", "show", f"{source}:{path}"], cwd=ROOT))
        matching = [c for c in calls if c["atoms"] == ref["atoms"] and "result" in c
                    and c["result"]["forces"] is not None]
        assert matching, case
        base = matching[0]["result"]
        assert base["converged"] and base["executed_backend"] == "cpu_reference"
        table = {"composite": errors(base, ref["reference"], ref["direction"])}
        for component in base["components"]:
            name = component["roles"][0]
            table[name] = errors(component["result"], ref["reference"]["components"][name], ref["direction"])
        assert len(table) == 6 and all(r["pass"] for r in table.values())
        terms = {name: abs(base[name]-ref["reference"][name])
                 for name in ("hf_energy", "mp2_correlation_energy", "delta_ccsdt_energy")}
        assert all(math.isfinite(value) and value <= 1e-9 for value in terms.values())
        for component in base["components"]:
            if component["method"] != "rhf":
                assert component["result"]["correlation"]["force_provenance_flags"] & 7 == 7
            if component["method"] == "ccsd(t)":
                oracle = ref["reference"]["components"][component["roles"][0]]
                assert abs(component["result"]["correlation"]["ccsd_t_triples_energy"]-oracle["triples_energy"]) <= 1e-9
        assert all(abs(sum(row[axis] for row in base["forces"])) <= 2e-7 for axis in range(3))
        analytic = -dot(base["forces"], ref["direction"])
        independent = [{"step_bohr": p["step_bohr"],
                        "error_Eh_Bohr": abs(analytic-p["derivative"])}
                       for p in ref["finite_difference"]]
        reconverged = []
        for step in (3e-4, 1e-4):
            energies = []
            displaced_results = []
            for sign in (-1, 1):
                displaced = [[n, [x+sign*step*d for x, d in zip(pos, direction, strict=True)]]
                             for (n, pos), direction in zip(ref["atoms"], ref["direction"], strict=True)]
                matches = [c for c in calls if c["atoms"] == displaced and "result" in c]
                assert matches, (case, step, sign)
                energies.append(matches[0]["result"]["energy"])
                displaced_results.append(matches[0]["result"])
            finite = (energies[1]-energies[0])/(2*step)
            component_directions = {}
            for component in base["components"]:
                name = component["roles"][0]
                pair = [next(c["result"]["energy"] for c in value["components"]
                             if c["roles"][0] == name) for value in displaced_results]
                derivative = (pair[1]-pair[0])/(2*step)
                native_analytic = -dot(component["result"]["forces"], ref["direction"])
                oracle_analytic = -dot(ref["reference"]["components"][name]["forces"], ref["direction"])
                assert all(math.isfinite(x) for x in (*pair, derivative, native_analytic, oracle_analytic))
                component_directions[name] = {
                    "energies_Eh": pair,
                    "native_FD_derivative_Eh_Bohr": derivative,
                    "native_analytic_error_Eh_Bohr": abs(derivative-native_analytic),
                    "independent_PySCF_force_projection_error_Eh_Bohr": abs(derivative-oracle_analytic),
                }
            reconverged.append({"step_bohr": step, "energies_Eh": energies,
                                "derivative_Eh_Bohr": finite,
                                "error_Eh_Bohr": abs(finite-analytic),
                                "components": component_directions})
        assert all(p["error_Eh_Bohr"] < 2e-6 for p in independent+reconverged)
        rows[case] = {"errors": table, "algebra_energy_abs_Eh": terms,
                      "independent_PySCF_directional": independent,
                      "native_reconverged_directional": reconverged}
    return {"status": "PASS", "scope": list(cases),
            "remaining_cases": [c for c in ("h2-dz-tz", "water-algebra") if c not in cases],
            "model": "CPU conventional exact FP64 all-electron",
            "DF_correlation_only": "UNAVAILABLE", "full_RI_JK": "UNAVAILABLE",
            "CUDA": "UNAVAILABLE", "cases": rows}


def verify_directory(directory):
    """Require the complete measured CPU slice, without executing native code."""
    build = load_raw(directory / "raw-build.json.xz")
    identity = json.loads(build["files"]["identity.json"])
    assert git_source_identity(identity["source_commit"]) == identity["native_source_identity"]
    assert "build 0\n" in build["files"]["controller.txt"]
    assert "sccache 0.16.0" in build["files"]["cache-version.txt"]
    old_outcome = json.loads(build["files"]["outcome.json"])
    preflight = json.loads(build["files"]["capacity-preflight.json"])
    assert old_outcome.get("exception")
    assert (preflight["partial_lossless_bytes"]
            + preflight["reserved_driver_replay_readme_raw_growth_bytes"]
            > preflight["available_bytes"])
    observed = set()
    cases = {}
    for case, filename in (("water-algebra", "raw-water.json.xz"),
                           ("h2-dz-tz", "raw-h2.json.xz")):
        phase = load_raw(directory / filename)
        meta = json.loads(phase["files"]["frozen-inputs.json"])
        assert meta["source"] == identity["source_commit"]
        assert meta["library_sha256"] == identity["library_sha256"]
        assert sha(phase["files"]["input/observe.py"].encode()) == meta["observer_sha256"]
        assert sha(phase["files"]["input/run.py"].encode()) == meta["runner_sha256"]
        cases.update(replay(phase, (case,))["cases"])
        reports = json.loads(phase["files"]["native/reports.json"])
        passed = {r["nodeid"] for r in reports if r["when"] == "call" and r["outcome"] == "passed"}
        assert len(passed) == 4 and not observed.intersection(passed)
        observed.update(passed)
    support = load_raw(directory / "raw-support.json.xz")
    outcome = json.loads(support["files"]["outcome.json"])
    assert all(outcome[name] == 0 for name in
               ("fixture_exit", "unit_exit", "policy_tests_exit", "collection_exit"))
    reports = json.loads(support["files"]["native/reports.json"])
    assert not any(r["outcome"] == "failed" for r in reports)
    passed = {r["nodeid"] for r in reports if r["when"] == "call" and r["outcome"] == "passed"}
    assert not observed.intersection(passed)
    observed.update(passed)
    expected = {line.strip() for line in support["files"]["native-collection.txt"].splitlines()
                if line.startswith("tests/python/test_ccsdt_cbs_native.py::") and "cuda" not in line}
    assert expected and observed == expected and len(observed) == 13
    unit = re.search(r"(\d+) passed", support["files"]["unit-tests.txt"])
    assert unit
    return {"status": "PASS", "scope": "frozen-source conventional exact native CPU reference slice",
            "source_commit": identity["source_commit"],
            "native_source_identity": identity["native_source_identity"],
            "library_sha256": identity["library_sha256"],
            "native_CPU_tests_passed": len(observed), "unit_tests_passed": int(unit[1]),
            "DF_correlation_only": "UNAVAILABLE", "full_RI_JK": "UNAVAILABLE",
            "CUDA": "UNAVAILABLE", "cases": cases}


def verify_publication(directory):
    manifest = json.loads((directory / "publication.json").read_bytes())
    for row in manifest["files"]:
        path = PurePosixPath(row["path"])
        assert not path.is_absolute() and ".." not in path.parts
        data = (directory / str(path)).read_bytes()
        assert len(data) == row["bytes"] and sha(data) == row["sha256"], path
    result = verify_directory(directory)
    assert result == json.loads((directory / "summary.json").read_bytes())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", nargs="?", type=Path)
    parser.add_argument("--case", choices=("all", "water-algebra", "h2-dz-tz"), default="all")
    parser.add_argument("--verify-publication", action="store_true")
    args = parser.parse_args()
    if args.bundle is None:
        result = verify_publication(HERE) if args.verify_publication else verify_directory(HERE)
    else:
        cases = ("h2-dz-tz", "water-algebra") if args.case == "all" else (args.case,)
        result = replay(load_raw(args.bundle), cases)
    print(json.dumps(result, indent=2))
