"""Replay current-source A100 endpoints and preserve native/molecular outcomes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from benchmarks._exact_fp64_evidence import load, require
from benchmarks.readme_defaults_accuracy import errors

SOURCE = "a53c7618fcd59b88693a693a08a36f7ad3bfa51969f6564b46561e95e04505fa"


def report(folder: Path) -> dict:
    """Authenticate complete production pairs and recompute observable gates.

    The frozen receipt's scientific identity owns this qualification. A later
    publication revision does not transfer the measurements to different code.
    Native fixture results and molecular production tests remain distinct.
    """
    bundle, files = load(folder)
    exits = {
        name: int(text)
        for name, text in files.items()
        if name.endswith(("/exit-code.txt", ".exit"))
    }
    pairs = []
    for name in sorted(files):
        if not name.endswith("/standard.json"):
            continue
        prefix = name.removesuffix("standard.json")
        native, reference = (
            json.loads(files[prefix + arm + ".json"])
            for arm in ("standard", "gpu4pyscf")
        )
        proof = json.loads(files[prefix + "identity.json"])
        require(
            proof["source_identity"] == SOURCE and not proof["build_tests"],
            "production source and build contract",
        )
        require(
            native["native_build"]["source_identity"] == SOURCE
            and native["native_build"]["library_sha256"]
            == proof["binaries"]["libgenerativeqc.so"],
            "measured production core",
        )
        require(
            native["status"] == reference["status"] == "complete"
            and len(native["samples"]) == len(reference["samples"]) == 4,
            "complete cold plus three warm calls",
        )
        require(
            native["geometry_bohr"] == reference["geometry_bohr"]
            and native["auxiliary_input_sha256"] == reference["auxiliary_input_sha256"],
            "matched geometry and auxiliary input",
        )
        comparisons = [
            errors(x["energy"], x["forces"], y, native["atoms"])
            for x in native["samples"]
            for y in reference["samples"]
        ]
        cold = native["samples"][0]
        times = {
            arm: [sample["seconds"] for sample in data["samples"]]
            for arm, data in (("native", native), ("reference", reference))
        }
        pairs.append(
            {
                "run": prefix.rstrip("/"),
                "seconds": times,
                "warm_reference_over_native": statistics.median(times["reference"][1:])
                / statistics.median(times["native"][1:]),
                "cold_native_over_reference": times["native"][0]
                / times["reference"][0],
                "iterations": {
                    arm: [sample["iterations"] for sample in data["samples"]]
                    for arm, data in (("native", native), ("reference", reference))
                },
                "native_initial_guess": cold["initial_guess"],
                "native_nonlocal_work": cold["ks_diagnostic"]["nonlocal_work"],
                "paired_errors": comparisons,
                "paired_pass": all(row["passed"] for row in comparisons),
            }
        )
    require(len(pairs) == 3, "three fresh process pairs")
    molecular = []
    latest_cases = {}
    recovered_cases = []
    for name, text in sorted(files.items()):
        if not name.endswith("/tests.xml"):
            continue
        document = ET.fromstring(text)  # noqa: S314 - authenticated local JUnit receipt
        suites = list(document.iter("testsuite"))
        counts = {
            key: sum(int(suite.get(key, "0")) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        prefix = name.removesuffix("tests.xml")
        for kind in ("production", "fixture"):
            proof_path = prefix + kind + "-identity.json"
            if proof_path not in files:
                require(kind == "fixture", "missing molecular production proof")
                continue
            proof = json.loads(files[proof_path])
            require(
                proof["source_identity"] == SOURCE
                and proof["build_tests"] == (kind == "fixture"),
                "molecular/fixture source identity",
            )
        molecular.append({"run": prefix.rstrip("/"), **counts})
        # A package-only retry may supersede an identical test ID on the same
        # scientific source. Preserve both original outcomes and the recovery
        # mapping; never turn an entire failed run into a success retrospectively.
        for case in document.iter("testcase"):
            key = (case.get("classname"), case.get("name"))
            status = (
                "failed"
                if case.find("failure") is not None or case.find("error") is not None
                else "skipped"
                if case.find("skipped") is not None
                else "passed"
            )
            previous = latest_cases.get(key)
            if previous and previous["status"] == "failed" and status == "passed":
                recovered_cases.append(
                    {"test": list(key), "original": previous["run"], "retry": prefix}
                )
            latest_cases[key] = {"run": prefix, "status": status}
    require(
        len(molecular) == 3, "original molecular population and both D4 package retries"
    )
    native_controls = {
        name: value
        for name, value in exits.items()
        if name.endswith(
            ("/pairs.exit", "/range.exit", "/factor.exit", "/semilocal.exit")
        )
    }
    require(len(native_controls) == 4, "four complete native controls")
    final_counts = {
        status: sum(row["status"] == status for row in latest_cases.values())
        for status in ("passed", "failed", "skipped")
    }
    return {
        "scope": "A100 matched fitted-LR source only; no default promotion or all-GPU claim",
        "source_identity": SOURCE,
        "input_archives": bundle["input_archives"],
        "pairs": pairs,
        "median_warm_reference_over_native": statistics.median(
            row["warm_reference_over_native"] for row in pairs
        ),
        "all_paired_gates_pass": all(row["paired_pass"] for row in pairs),
        "molecular": molecular,
        "latest_test_outcomes": final_counts,
        "recovered_cases": recovered_cases,
        "original_exit_codes": exits,
        "all_native_and_molecular_controls_pass": all(
            value == 0 for value in native_controls.values()
        )
        and final_counts["failed"] == 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(report(Path(__file__).resolve().parent), indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
