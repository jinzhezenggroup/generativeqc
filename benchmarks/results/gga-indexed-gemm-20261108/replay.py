"""Replay source-bound GGA endpoints, independent accuracy and default gates."""

from __future__ import annotations

import hashlib
import itertools
import json
import lzma
import math
import statistics
import sys
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from benchmarks._exact_fp64_evidence import load
from benchmarks.readme_defaults import COMMON_GATES
from benchmarks.readme_defaults_qualification import (
    STABILITY_FRACTION,
    _comparison,
    _geometry,
    _oracle,
    _reference_contract,
    identity,
)
from benchmarks.readme_defaults_report import checked_sample


def require(condition: bool, detail: str) -> None:
    """Preserve scientific and provenance checks under optimized Python."""
    if not condition:
        raise ValueError(detail)


def load_evidence(directory: Path = DIRECTORY) -> dict:
    """Authenticate exact original receipt bytes and all FP64 words."""
    bundle, files = load(directory)
    return {**bundle["experiment"], "files": files}


def solve_work(sample: dict) -> dict:
    """Report semantic work, not launches or implementation-spelling counts."""
    guess = sample["initial_guess"]
    return {
        "iterations": sample["iterations"],
        "fock_builds": sample["fock_builds"],
        "grid_points": sample["ks_diagnostic"]["grid_points"],
        "jk": sample["ks_diagnostic"]["coulomb_exchange_work"],
        "initial_guess": {
            key: value for key, value in guess.items() if key != "preparation_seconds"
        },
        "precision": sample["precision"],
    }


def independent_accuracy(record: dict, oracles: list[dict], legacy: dict) -> dict:
    """Reuse unchanged dense/CPU observations, never historical timing or density.

    Experimental source labels include an explicit uncommitted-patch suffix.
    Validate their source/ELF separately; do not rewrite them to manufacture a
    40-character commit accepted by the historical campaign reducer.
    """
    method = record["method"]
    indexed = {
        (oracle["atoms"], oracle["backend"]): oracle
        for oracle in oracles
        if oracle["method"] == method
    }
    target, cpu, gpu = (indexed[key] for key in ((96, "cuda"), (3, "cpu"), (3, "cuda")))
    for oracle in (target, cpu, gpu):
        _oracle(oracle)
        require(oracle["status"] == "complete", "incomplete independent oracle")
        require(
            legacy[identity(oracle)] == _reference_contract(record, oracle),
            "changed independent producer/input contract",
        )
    require(_geometry(target) == _geometry(record), "changed independent geometry")
    require(_geometry(cpu) == _geometry(gpu), "changed CPU/GPU spot geometry")
    require(bool(record["environment"]["packages"]), "missing measured packages")
    stability = [
        _comparison(*oracle["samples"], oracle["atoms"], fraction=STABILITY_FRACTION)
        for oracle in (target, gpu)
    ]
    cpu_sample = cpu["samples"][-1]
    spots = [
        sample
        for sample in gpu["samples"]
        if sample["level"] == cpu_sample["level"]
        and sample.get("nonlocal_grid") == cpu_sample.get("nonlocal_grid")
    ]
    require(len(spots) == 1, "unmatched independent spot quadrature")
    spot = _comparison(cpu_sample, spots[0], 3, fraction=STABILITY_FRACTION)
    checks = [
        _comparison(sample, target["samples"][-1], 96) for sample in record["samples"]
    ]
    return {
        "qualified": spot["passed"]
        and all(check["passed"] for check in stability + checks),
        "dense_reference_sha256": identity(target),
        "cpu_spot_sha256": identity(cpu),
        "gpu_spot_sha256": identity(gpu),
        "grid_stability": stability,
        "cpu_gpu_spot": spot,
        "endpoints": checks,
        "measurement_packages": record["environment"]["packages"],
        "reference_packages": target["environment"]["packages"],
    }


def replay(directory: Path = DIRECTORY, *, evidence: dict | None = None) -> dict:
    """Keep early losses, every warm and the final default image distinct."""
    evidence = load_evidence(directory) if evidence is None else evidence
    require(evidence["schema"] == "generativeqc.gga-indexed-gemm.v1", "wrong campaign")
    files = evidence["files"]
    require(set(files) == set(evidence["sha256"]), "missing original receipt")
    for name, text in files.items():
        require(
            hashlib.sha256(text.encode()).hexdigest() == evidence["sha256"][name],
            "changed receipt: " + name,
        )
    legacy_root = ROOT / "benchmarks/results/readme-refresh-20261026"
    for name, expected in evidence["legacy_sha256"].items():
        require(
            hashlib.sha256((legacy_root / name).read_bytes()).hexdigest() == expected,
            "changed independent evidence",
        )
    oracles = json.loads(
        lzma.decompress((legacy_root / "oracles.json.xz").read_bytes())
    )
    legacy = json.loads((legacy_root / "reference-reuse.json").read_text())["entries"]
    for name, image in evidence["images"].items():
        prefix = name + "/"
        source = json.loads(files[prefix + "source-receipt.json"])
        require(
            source["source_identity"] == image["source_identity"],
            "source/image mismatch",
        )
        require(
            files[prefix + "build/exit-code.txt"].strip() == "0",
            "failed normal Release build",
        )
        require(
            json.loads(files[prefix + "build/identity.json"]) == image,
            "wrong compiled image",
        )
        require(
            "sccache 0.16.0" in files[prefix + "build/cache-version.txt"],
            "unverified compiler cache",
        )
        require(
            "4c73bcf8a409a15362355d290695e7866c3a2b28753a42a1d61c91aba4ffe83a"
            in files[prefix + "build/restored-runtime.sha256"],
            "borrowed runtime not restored",
        )
        require(
            hashlib.sha256(files[prefix + "candidate.patch"].encode()).hexdigest()
            == source["patch_sha256"],
            "wrong frozen source patch",
        )
        delta = json.loads(files[prefix + "runtime-delta-receipt.json"])
        require(
            delta["target_sha256"] == image["library_sha256"]
            and delta["local_full_reconstruction_verified"],
            "unverified image reconstruction",
        )
    require(
        set(evidence["images"]) == {"experimental", "default"},
        "missing distinct final image",
    )
    require(
        json.loads(files["default/source-receipt.json"])["changed_native_paths"]
        == ["src/dft/cuda_ks.cpp"],
        "HF prototype entered default closure",
    )
    require(
        "OWNED_GGA_TRANSFER_STOPPED" in files["campaign/cleanup.log"],
        "transfer listener not retired",
    )
    expected_arms = {
        "screen-a100": {"off", "on", "reference"},
        "screen-5090": {"off", "on"},
        "cohort-a100": {
            "forward-off",
            "forward-on",
            "reverse-on",
            "reverse-off",
            "reference",
        },
        "final-a100": {"off", "default", "reference"},
        "final-5090": {"off", "default"},
    }
    require(
        set(evidence["groups"]) == set(expected_arms), "missing qualification cohort"
    )
    previous = {}
    for arm in ("standard", "gpu4pyscf"):
        for record in json.loads(
            lzma.decompress((legacy_root / (arm + ".json.xz")).read_bytes())
        ):
            previous[record["method"], record["mode"], record["atoms"], arm] = record
    report = {"groups": {}, "populations": 0, "endpoints": 0, "qualified": True}
    for name, group in evidence["groups"].items():
        expected_cases = {
            (method, mode)
            for method in (
                ("pbe0", "b3lyp")
                if name.startswith("screen")
                else ("pbe", "pbe0", "b3lyp")
            )
            for mode in (("df",) if name.startswith("screen") else ("direct", "df"))
        }
        require(
            len(group["cases"]) == len(expected_cases)
            and {(case["method"], case["mode"]) for case in group["cases"]}
            == expected_cases,
            "missing measured case",
        )
        prefix = group["prefix"] + "/"
        image = evidence["images"][group["image"]]
        executed = json.loads(files[prefix + "identity.json"])
        require(
            all(executed[key] == value for key, value in image.items()),
            "wrong executed image",
        )
        scheduler = executed["scheduler"]
        job_key = "LSB_JOBID" if group["device"] == "A100" else "SLURM_JOB_ID"
        require(
            str(scheduler[job_key]) == str(group["job"])
            and bool(scheduler["CUDA_VISIBLE_DEVICES"]),
            "unallocated execution",
        )
        require(files[prefix + "exit-code.txt"].strip() == "0", "failed allocation")
        rows = []
        for case in group["cases"]:
            require(set(case["arms"]) == expected_arms[name], "missing endpoint arm")
            expected_pairs = (
                [["forward-off", "forward-on"], ["reverse-off", "reverse-on"]]
                if name == "cohort-a100"
                else [["off", "default" if name.startswith("final") else "on"]]
            )
            require(case["pairs"] == expected_pairs, "missing paired timing order")
            require(
                group["promotion_gate"] == (not name.startswith("screen")),
                "disabled promotion gate",
            )
            arms = {}
            qualifications = {}
            for arm, basename in case["arms"].items():
                record = json.loads(files[prefix + basename + ".json"])
                require(
                    record["source_revision"] == group["revision"],
                    "relabelled source revision",
                )
                require(
                    record["runner_sha256"] == evidence["runner_sha256"],
                    "changed endpoint runner",
                )
                require(record["scheduler"] == scheduler, "wrong population allocation")
                require(
                    group["device_token"] in record["accelerator"]["name"],
                    "wrong device",
                )
                require(
                    record["method"] == case["method"]
                    and record["mode"] == case["mode"]
                    and record["atoms"] == 96,
                    "changed case",
                )
                require(
                    record["status"] == "complete" and len(record["samples"]) == 4,
                    "incomplete population",
                )
                protocol = record["protocol"]
                require(
                    protocol["common_observable_gates"] == COMMON_GATES
                    and not protocol["grid_overrides"],
                    "weakened scientific gates",
                )
                require(
                    protocol["cold"] == "constructor+prepare+SCF+forces+host-export"
                    and protocol["warm"]
                    == "same-owner; frozen engine-local cold density",
                    "changed endpoint contract",
                )
                require(
                    record["arm"]
                    == ("gpu4pyscf" if arm == "reference" else "standard"),
                    "wrong engine",
                )
                prior = previous[record["method"], record["mode"], 96, record["arm"]]
                for key in (
                    "geometry_bohr",
                    "orbital_basis",
                    "basis_representation",
                    "charge",
                    "spin",
                    "auxiliary_input_sha256",
                ):
                    require(
                        record.get(key) == prior.get(key),
                        "changed scientific input: " + key,
                    )
                require(
                    record["mode"] != "df"
                    or bool(record.get("auxiliary_input_sha256")),
                    "missing fitted-basis identity",
                )
                for index, sample in enumerate(record["samples"]):
                    checked_sample(sample, record)
                    require(
                        sample["converged"]
                        and sample["phase"] == ("cold" if index == 0 else "warm"),
                        "invalid converged endpoint",
                    )
                    require(
                        math.isfinite(sample["seconds"]) and sample["seconds"] > 0,
                        "invalid timing",
                    )
                    if arm != "reference":
                        settings = sample["settings"]
                        require(
                            settings["energy_tolerance"] == 1e-9
                            and settings["density_tolerance"] == 1e-5
                            and settings["screening_tolerance"] == 1e-12,
                            "changed native convergence defaults",
                        )
                        require(
                            settings["force_grid_response"]
                            and settings["xc_grid_points"] == 1078336,
                            "changed quadrature or moving-grid force response",
                        )
                        require(
                            all(
                                record["native_build"][key] == value
                                for key, value in image.items()
                            ),
                            "wrong native image",
                        )
                        require(
                            sample["precision"]["effective_bits"] == 64
                            and sample["precision"]["mixed_stage_fock_builds"] == 0,
                            "non-FP64 work",
                        )
                if arm != "reference" and group.get("environments"):
                    control = dict(
                        line.split("=", 1)
                        for line in files[prefix + basename + ".env"].splitlines()
                    )
                    expected = (
                        "0"
                        if arm.endswith("off")
                        else "1"
                        if arm.endswith("on")
                        else None
                    )
                    require(
                        control.get("GENERATIVEQC_CUDA_XC_GGA_GEMM") == expected,
                        "not the advertised default/rollback arm",
                    )
                    require(
                        not any(
                            key in control
                            for key in (
                                "GENERATIVEQC_INCREMENTAL_DIRECT_JK",
                                "GENERATIVEQC_HF_PREPARED_FOCK",
                                "GENERATIVEQC_HF_PREPARED_DELTA",
                            )
                        ),
                        "unrelated experimental control",
                    )
                qualifications[arm] = independent_accuracy(record, oracles, legacy)
                report["qualified"] &= qualifications[arm]["qualified"]
                arms[arm] = record
                report["populations"] += 1
                report["endpoints"] += 4
            pair_errors = {}
            for first, second in itertools.combinations(arms, 2):
                checks = [
                    _comparison(left, right, 96)
                    for left, right in itertools.product(
                        arms[first]["samples"], arms[second]["samples"]
                    )
                ]
                require(all(check["passed"] for check in checks), "paired E/F failure")
                pair_errors[first + "/" + second] = {
                    key: max(check[key] for check in checks) for key in COMMON_GATES
                }
            ratios = []
            for before, after in case["pairs"]:
                first, second = arms[before], arms[after]
                for left, right in zip(
                    first["samples"], second["samples"], strict=True
                ):
                    require(
                        left["settings"] == right["settings"]
                        and solve_work(left) == solve_work(right),
                        "changed defaults or semantic work",
                    )
                values = [
                    right["seconds"] / left["seconds"]
                    for left, right in zip(
                        first["samples"], second["samples"], strict=True
                    )
                ]
                if group.get("promotion_gate"):
                    cold_limit = 0.98 if group["device"] == "A100" else 1.03
                    require(
                        values[0] < cold_limit and max(values[1:]) <= 1.03,
                        "failed complete-endpoint promotion gate",
                    )
                ratios.append(
                    {"pair": [before, after], "candidate_over_baseline": values}
                )
            rows.append(
                {
                    "method": case["method"],
                    "mode": case["mode"],
                    "timing": {
                        arm: {
                            "cold": row["samples"][0]["seconds"],
                            "warm": statistics.median(
                                sample["seconds"] for sample in row["samples"][1:]
                            ),
                        }
                        for arm, row in arms.items()
                    },
                    "ratios": ratios,
                    "paired_errors": pair_errors,
                    "independent_accuracy": qualifications,
                    "work": {
                        arm: [solve_work(sample) for sample in row["samples"]]
                        for arm, row in arms.items()
                        if arm != "reference"
                    },
                }
            )
        report["groups"][name] = rows
    require(report["qualified"], "independent dense/CPU qualification failed")
    return report


if __name__ == "__main__":
    result = replay()
    print(
        json.dumps(
            {key: result[key] for key in ("qualified", "populations", "endpoints")},
            indent=2,
        )
    )
