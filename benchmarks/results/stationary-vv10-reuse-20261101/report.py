"""Replay exact native stationary VV10 qualification before ranking endpoints."""

from __future__ import annotations

import hashlib
import json
import lzma
import runpy
from pathlib import Path

from benchmarks._exact_fp64_evidence import load, require
from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parents[3]
PRIOR = runpy.run_path(
    str(ROOT / "benchmarks/results/deferred-range-anchor-20261101/report.py")
)
ANCHOR = runpy.run_path(
    str(ROOT / "benchmarks/results/range-anchor-default-20261101/report.py")
)


def main() -> None:
    """Keep source, scientific gates, every warm call and larger controls explicit."""
    bundle, files = load(Path(__file__).parent)
    proof = json.loads(files["source.json"])

    def check_owner(folder: str, expected: dict = proof) -> dict:
        identity = json.loads(files[folder + "/candidate-identity.json"])
        require(
            identity["source_identity"] == expected["source_identity"], "GPU source"
        )
        require(identity["binaries"] == expected["binaries"], "GPU binaries")
        require(
            identity["source_archive_sha256"] == expected["source_archive_sha256"],
            "GPU source archive",
        )
        return identity

    references = {}
    for name, digest in bundle["references"].items():
        raw = (ROOT / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, name)
        references[name] = json.loads(lzma.decompress(raw))
    oracles = references["benchmarks/results/readme-refresh-20261026/oracles.json.xz"]
    reuse = references[
        "benchmarks/results/independent-force-tiles-20261031/receipts.json.xz"
    ]

    def oracle_error(sample: dict, expected: dict) -> None:
        import numpy as np

        require(expected["converged"], "independent oracle convergence")
        actual, reference = np.asarray(sample["forces"]), np.asarray(expected["forces"])
        require(actual.shape == reference.shape, "oracle force dimensions")
        error = actual - reference
        require(
            np.isfinite(error).all() and np.max(np.abs(error)) < 2e-7, "oracle force"
        )
        require(abs(sample["energy"] - expected["energy"]) < 2e-8, "oracle energy")

    for folder in bundle["gpu_checks"]:
        require(files[folder + "/exit-code.txt"].strip() == "0", folder)
        require("7 passed" in files[folder + "/tests.log"], "complete GPU case census")
        check_owner(folder)
    sanitizer = bundle["memcheck"]
    require(files[sanitizer + "/exit-code.txt"].strip() == "0", "memcheck exit")
    check_owner(sanitizer)
    oracle = json.loads(files[sanitizer + "/oracle.json"])
    for control in (
        "enabled",
        "disabled",
        "default-small",
        "minimum",
        "schedule",
        "failure",
        "moved",
    ):
        log = files[sanitizer + "/memcheck-" + control + ".log"]
        require("ERROR SUMMARY: 0 errors" in log, "sanitizer errors")
        require(
            "LEAK SUMMARY: 0 bytes leaked in 0 allocations" in log, "sanitizer leaks"
        )
        require(
            "PASS " + control in files[sanitizer + "/native-" + control + ".log"],
            "native gates",
        )
        samples = json.loads(files[sanitizer + "/native-" + control + "/results.json"])
        for i, sample in enumerate(samples):
            expected = oracle["moved" if control == "moved" and i >= 2 else "original"]
            oracle_error(sample, expected)
    for folder in bundle["batch"]:
        require(files[folder + "/exit-code.txt"].strip() == "0", "batch exit")
        check_owner(folder)
        oracle_batch = json.loads(files[folder + "/oracle.json"])
        batch = json.loads(files[folder + "/batch-results.json"])
        require(len(batch) == 7, "batch lifecycle")
        for phase in batch:
            for sample, label in zip(phase["items"], phase["labels"], strict=True):
                expected = oracle_batch[label]
                oracle_error(sample, expected)
    # Keep deployment/test-harness failures visible, with their original archives.
    for folder, reason in bundle["negative_evidence"].items():
        require(
            files[folder + "/exit-code.txt"].strip() == "1", "negative evidence exit"
        )
        require(
            reason
            in files[folder + ("/batch.log" if "/batch-" in folder else "/tests.log")],
            "failure diagnosis",
        )
    capacity = bundle["capacity"]
    require(files[capacity + "/exit-code.txt"].strip() == "0", "capacity exit")
    check_owner(capacity)
    parent_capacity = json.loads(files[capacity + "/parent-capacity.json"])
    candidate_capacity = json.loads(files[capacity + "/candidate-capacity.json"])
    require(parent_capacity == candidate_capacity, "unchanged public capacity tiers")
    gates, summaries = {}, []
    for cohort in bundle["cohorts"]:
        folder = cohort["folder"]
        require(files[folder + "/exit-code.txt"].strip() == "0", folder)
        arms = {"parent": [], "candidate": []}
        identities = {
            arm: json.loads(files[folder + "/" + arm + "-identity.json"])
            for arm in arms
        }
        require(
            identities["candidate"]["source_identity"] == proof["source_identity"],
            "candidate source",
        )
        require(
            identities["candidate"]["binaries"] == proof["binaries"],
            "candidate binaries",
        )
        require(
            identities["parent"]["source_identity"] == bundle["parent_source_identity"],
            "parent source",
        )
        for slot, arm in enumerate(
            ("parent", "parent", "candidate", "candidate", "parent")
        ):
            name = f"{folder}/slot-{slot}-{arm}.json"
            record = json.loads(files[name])
            ANCHOR["bind"](record, identities[arm])
            require(
                identities[arm]["controls"]["lazy_range_anchor"] is None,
                "anchor default",
            )
            gate = qualify(
                [record],
                oracles,
                reference_reuse=reuse[cohort["reference_device"]]["analysis"][
                    "reference_reuse"
                ],
            )[0]
            require(gate["status"] == "qualified", str(gate))
            gates[name] = gate
            if slot:
                require(
                    [s["phase"] for s in record["samples"]]
                    == ["cold", "warm", "warm", "warm"],
                    "complete lifecycle",
                )
                arms[arm].append(record)
        errors = []
        for left, right in zip(arms["parent"], arms["candidate"], strict=True):
            PRIOR["same_task"](left, right)
            for a, b in zip(left["samples"], right["samples"], strict=True):
                require(
                    (a["iterations"], a["fock_builds"])
                    == (b["iterations"], b["fock_builds"]),
                    "changed physical work",
                )
                errors.append(PRIOR["errors"](a, b))
        timings, observations = {}, {}
        for arm, records in arms.items():
            calls = [r["samples"] for r in records]
            values = {
                "cold": [c[0]["seconds"] for c in calls],
                "first_warm": [c[1]["seconds"] for c in calls],
                "pooled_warm": [s["seconds"] for c in calls for s in c[1:]],
                "lifecycle": [sum(s["seconds"] for s in c) for c in calls],
            }
            timings[arm] = {
                phase: ANCHOR["statistics"](v) for phase, v in values.items()
            }
            observations[arm] = [
                [
                    {k: s[k] for k in ("phase", "seconds", "iterations", "fock_builds")}
                    for s in c
                ]
                for c in calls
            ]
        summaries.append(
            {
                **cohort,
                "timings": timings,
                "observations": observations,
                "candidate_vs_parent_percent": {
                    phase: 100
                    * (timings["candidate"][phase]["median"] / item["median"] - 1)
                    for phase, item in timings["parent"].items()
                },
                "paired_errors": errors,
            }
        )
    trace = bundle["trace"]
    require(files[trace + "/exit-code.txt"].strip() == "0", "trace exit")
    trace_identity = check_owner(trace, proof)
    traced = json.loads(files[trace + "/trace.json"])
    ANCHOR["bind"](traced, trace_identity)
    gate = qualify(
        [traced], oracles, reference_reuse=reuse["5090"]["analysis"]["reference_reuse"]
    )[0]
    require(gate["status"] == "qualified", str(gate))
    work = [row["counters"] for row in json.loads(files[trace + "/force-work.json"])]
    require(len(work) == 3, "traced complete force count")
    require([r["nonlocal_pair_sweeps"] for r in work] == [1, 0, 0], "actual pair work")
    require(
        [r["nonlocal_pair_cache_hits"] for r in work] == [0, 1, 1],
        "actual exact-key hits",
    )
    # Diagnostic trace times never enter clean ABBA ranking populations.
    large = []
    for folder in bundle["large"]:
        require(files[folder + "/exit-code.txt"].strip() == "0", folder)
        controls = []
        for arm in ("parent", "candidate"):
            record = json.loads(files[f"{folder}/large-{arm}.json"])
            identity = json.loads(files[f"{folder}/{arm}-identity.json"])
            ANCHOR["bind"](record, identity)
            require(
                identity["source_identity"]
                == (
                    proof["source_identity"]
                    if arm == "candidate"
                    else bundle["parent_source_identity"]
                ),
                "larger source",
            )
            require(
                [s["phase"] for s in record["samples"]]
                == [
                    "cold",
                    "retained",
                    "evicted-fallback",
                    "second",
                    "moved",
                    "moved-warm",
                ],
                "larger phases",
            )
            controls.append(record)
        for key in (
            "geometry_bohr",
            "moved_positions_bohr",
            "density_fitting_memory_budget_bytes",
            "method",
            "atoms",
            "scheduler",
        ):
            require(controls[0][key] == controls[1][key], "larger task: " + key)
        errors = []
        for a, b in zip(controls[0]["samples"], controls[1]["samples"], strict=True):
            require(
                (a["iterations"], a["fock_builds"])
                == (b["iterations"], b["fock_builds"]),
                "larger work",
            )
            errors.append(PRIOR["errors"](a, b, 1e-9))
        large.append(
            {
                "folder": folder,
                "paired_errors": errors,
                "observations": {
                    arm: [
                        {
                            k: s[k]
                            for k in ("phase", "seconds", "iterations", "fock_builds")
                        }
                        for s in r["samples"]
                    ]
                    for arm, r in zip(("parent", "candidate"), controls, strict=True)
                },
            }
        )
    print(
        json.dumps(
            {
                "source_identity": proof["source_identity"],
                "gates": gates,
                "summaries": summaries,
                "large": large,
                "trace_gate": gate,
                "trace_work": work,
                "negative_evidence": bundle["negative_evidence"],
                "pending_a100": bundle["pending_a100"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
