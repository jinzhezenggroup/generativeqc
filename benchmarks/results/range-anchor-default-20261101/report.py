"""Replay source-matched anchor controls, including every first warm endpoint."""

from __future__ import annotations

import hashlib
import json
import lzma
import runpy
from pathlib import Path
from statistics import median

from benchmarks._exact_fp64_evidence import load, require
from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parents[3]
PRIOR = runpy.run_path(
    str(ROOT / "benchmarks/results/deferred-range-anchor-20261101/report.py")
)


def statistics(values: list[float]) -> dict:
    """Keep all calls and dispersion; never hide a first-warm cost in pooling."""
    center = median(values)
    return {
        "values": values,
        "median": center,
        "min": min(values),
        "max": max(values),
        "mad": median(abs(v - center) for v in values),
    }


def bind(record: dict, identity: dict) -> None:
    """Bind complete scientific results to the actual binary and allocation."""
    require(record["status"] == "complete", "incomplete population")
    require(
        record["native_build"]["source_identity"] == identity["source_identity"],
        "source mismatch",
    )
    require(
        record["native_build"]["library_sha256"]
        == identity["binaries"]["libgenerativeqc.so"],
        "library mismatch",
    )
    for key, value in identity["scheduler"].items():
        actual = (
            record.get("thread_controls", record["scheduler"])
            if key == "SLURM_CPUS_PER_TASK"
            else record["scheduler"]
        )
        require(actual.get(key) == value, "allocation mismatch: " + key)
    for sample in record["samples"]:
        PRIOR["work"](sample)
        settings = sample["settings"]
        require(
            settings["scf_tile_points"] == 256
            and settings["force_tile_points"] == 1024,
            "public tile drift",
        )


def main() -> None:
    """Authenticate the full evidence, apply independent gates, then rank arms."""
    bundle, files = load(Path(__file__).parent)
    proof = json.loads(files["source.json"])
    for folder in bundle["gpu_checks"]:
        require(files[folder + "/exit-code.txt"].strip() == "0", folder)
        require("66 passed" in files[folder + "/tests.log"], "GPU case census")
        identity = json.loads(files[folder + "/identity.json"])
        require(
            identity["source_identity"] == proof["source_identity"], "GPU test source"
        )
        require(identity["binaries"] == proof["binaries"], "GPU test binaries")
    sanitizer = bundle["native_memcheck"]
    require(files[sanitizer + "/exit-code.txt"].strip() == "0", "native memcheck exit")
    require("5 passed" in files[sanitizer + "/tests.log"], "sanitizer case census")
    log = files[sanitizer + "/memcheck.log"]
    require("ERROR SUMMARY: 0 errors" in log, "native sanitizer errors")
    require("LEAK SUMMARY: 0 bytes leaked in 0 allocations" in log, "native leaks")
    identity = json.loads(files[sanitizer + "/identity.json"])
    require(identity["source_identity"] == proof["source_identity"], "sanitizer source")
    require(identity["binaries"] == proof["binaries"], "sanitizer binaries")
    oracle_proof = json.loads(files["oracle-proof.json"])
    require(
        hashlib.sha256(files["oracle.json"].encode()).hexdigest()
        == oracle_proof["reference_sha256"],
        "sanitizer independent oracle",
    )
    require(
        files["oracle.json"] == files["n5/oracle-memory-1957/reference-0.json"],
        "sanitizer oracle producer",
    )
    require(
        oracle_proof["producer_identity"]
        == json.loads(files["n5/oracle-memory-1957/identity.json"]),
        "independent oracle allocation",
    )
    require(
        oracle_proof["fixture_sha256"]
        == identity["test_sha256"]["test_lazy_range_warm_anchor_cuda.py"],
        "independent oracle fixture",
    )
    references = {}
    for name, digest in bundle["references"].items():
        raw = (ROOT / name).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, "reference: " + name)
        references[name] = json.loads(lzma.decompress(raw))
    oracles = references["benchmarks/results/readme-refresh-20261026/oracles.json.xz"]
    reuse = references[
        "benchmarks/results/independent-force-tiles-20261031/receipts.json.xz"
    ]
    gates, summaries = {}, []
    for cohort in bundle["cohorts"]:
        folder = cohort["folder"]
        require(files[folder + "/exit-code.txt"].strip() == "0", folder)
        identity = json.loads(files[folder + "/identity.json"])
        require(identity["source_identity"] == proof["source_identity"], folder)
        arms = {"eager": [], "default": []}
        for stem in (
            "conditioning-eager",
            "slot-1-eager",
            "slot-2-default",
            "slot-3-default",
            "slot-4-eager",
        ):
            name = folder + "/" + stem + ".json"
            record = json.loads(files[name])
            bind(record, identity)
            control = json.loads(files[folder + "/" + stem + "-control.json"])
            arm = "default" if stem.endswith("default") else "eager"
            require(
                control["lazy_range_anchor"] == (None if arm == "default" else "0"),
                "anchor environment mismatch",
            )
            require(
                control["source_identity"] == proof["source_identity"], "control source"
            )
            require(
                control["runner_sha256"]
                == hashlib.sha256(files["run.sh"].encode()).hexdigest(),
                "control runner",
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
            if stem.startswith("slot-"):
                require(
                    [s["phase"] for s in record["samples"]]
                    == ["cold", "warm", "warm", "warm"],
                    "incomplete cold/three-warm lifecycle",
                )
                arms[arm].append(record)
        paired = []
        for left, right in zip(arms["eager"], arms["default"], strict=True):
            PRIOR["same_task"](left, right)
            for a, b in zip(left["samples"], right["samples"], strict=True):
                require(
                    a["iterations"] == b["iterations"], "changed physical iterations"
                )
                paired.append(PRIOR["errors"](a, b))
                if a["phase"] == "warm":
                    require(
                        a["fock_builds"] == b["fock_builds"] == 1, "warm work transfer"
                    )
        timings, observations = {}, {}
        for arm, records in arms.items():
            calls = [r["samples"] for r in records]
            values = {
                "cold": [c[0]["seconds"] for c in calls],
                "first_warm": [c[1]["seconds"] for c in calls],
                "pooled_warm": [s["seconds"] for c in calls for s in c[1:]],
                "lifecycle": [sum(s["seconds"] for s in c) for c in calls],
            }
            timings[arm] = {phase: statistics(v) for phase, v in values.items()}
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
                "default_vs_eager_percent": {
                    phase: 100
                    * (timings["default"][phase]["median"] / item["median"] - 1)
                    for phase, item in timings["eager"].items()
                },
                "paired_errors": paired,
            }
        )
    larger = []
    for folder in bundle["large"]:
        require(files[folder + "/exit-code.txt"].strip() == "0", folder)
        identity = json.loads(files[folder + "/identity.json"])
        require(identity["source_identity"] == proof["source_identity"], folder)
        controls = [
            json.loads(files[f"{folder}/large-{arm}.json"])
            for arm in ("eager", "default")
        ]
        for record in controls:
            bind(record, identity)
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
                "larger phase census",
            )
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
            require(a["iterations"] == b["iterations"], "larger physical work")
            errors.append(PRIOR["errors"](a, b, 1e-9))
        larger.append(
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
                    for arm, r in zip(("eager", "default"), controls, strict=True)
                },
            }
        )
    print(
        json.dumps(
            {
                "source_identity": proof["source_identity"],
                "gates": gates,
                "summaries": summaries,
                "large": larger,
                "checks": bundle["checks"],
                "pending_a100": bundle["pending_a100"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
