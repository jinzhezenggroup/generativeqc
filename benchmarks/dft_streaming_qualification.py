"""Replay scoped native DFT cold qualification against independent oracles."""

from __future__ import annotations

import csv
import json
import lzma
from pathlib import Path
from statistics import mean

import numpy as np

from benchmarks.readme_defaults_accuracy import errors
from benchmarks.readme_defaults_qualification import (
    STABILITY_FRACTION,
    _comparison,
    _geometry,
    _oracle,
    _reference_contract,
    identity,
)
from benchmarks.readme_defaults_report import reduce_record


def qualify_native(directory: Path, card: str) -> dict:
    """Authenticate the complete ABBA, work, numerical and dispatch evidence."""
    if card not in ("a100", "5090"):
        raise ValueError("Unknown qualification card")
    root = Path(__file__).resolve().parents[1]
    legacy_root = root / "benchmarks/results/readme-refresh-20261026"
    oracles = json.loads(
        lzma.decompress((legacy_root / "oracles.json.xz").read_bytes())
    )
    legacy = json.loads((legacy_root / "reference-reuse.json").read_text())["entries"]

    def require(condition: bool, detail: str) -> None:
        if not condition:
            raise ValueError(detail)

    def scientific(record: dict) -> dict:
        """Use the unchanged dense/CPU contract at the actual measured atom count."""
        indexed = {
            (item["atoms"], item["backend"]): item
            for item in oracles
            if item["method"] == record["method"]
        }
        target, cpu, gpu = (
            indexed[key] for key in ((record["atoms"], "cuda"), (3, "cpu"), (3, "cuda"))
        )
        for oracle in (target, cpu, gpu):
            _oracle(oracle)
            require(oracle["status"] == "complete", "Incomplete independent reference")
            require(
                legacy[identity(oracle)] == _reference_contract(record, oracle),
                "Changed independent producer/input contract",
            )
        require(_geometry(target) == _geometry(record), "Changed independent geometry")
        require(_geometry(cpu) == _geometry(gpu), "Changed CPU/GPU spot geometry")
        stability = [
            _comparison(
                *oracle["samples"], oracle["atoms"], fraction=STABILITY_FRACTION
            )
            for oracle in (target, gpu)
        ]
        cpu_sample = cpu["samples"][-1]
        matches = [
            sample
            for sample in gpu["samples"]
            if sample["level"] == cpu_sample["level"]
            and sample.get("nonlocal_grid") == cpu_sample.get("nonlocal_grid")
        ]
        require(len(matches) == 1, "Unmatched independent spot quadrature")
        spot = _comparison(cpu_sample, matches[0], 3, fraction=STABILITY_FRACTION)
        endpoints = [
            _comparison(sample, target["samples"][-1], record["atoms"])
            for sample in record["samples"]
        ]
        require(
            spot["passed"] and all(item["passed"] for item in stability + endpoints),
            "Independent numerical gate failed",
        )
        return {
            "dense_sha256": identity(target),
            "cpu_sha256": identity(cpu),
            "gpu_spot_sha256": identity(gpu),
            "grid_stability": stability,
            "cpu_gpu_spot": spot,
            "endpoints": endpoints,
            "measurement_packages": record["environment"]["packages"],
            "reference_packages": target["environment"]["packages"],
        }

    require(
        (directory / "exit-code.txt").read_text().strip() == "0",
        "Failed GPU allocation",
    )
    expected = {
        "control": (
            "334aea84a9517b2023e36b819a51950c781045be2ea8e5657a416b5d8fe91b91",
            "26429e92a70a293c3ef15040beae7721a63cb21ca47e274e37427f3160656b26",
        ),
        "candidate": (
            "7fbcd2d2e35fac47b64517f70bba10e6af2d65c3e74112801a6c159a5a7bc51c",
            "9a37512dc5f96f8a5d0ca78cae46c33323cb8a693620ff3bc93958f33c82534f",
        ),
    }
    report = {
        "card": card,
        "default_promotion_qualified": False,
        "all_readme_leadership_established": False,
        "methods": {},
    }
    for method in ("pbe0", "b3lyp"):
        records = {}
        qualifications = {}
        for ordinal, arm in enumerate(
            ("control", "candidate", "candidate", "control"), 1
        ):
            name = f"{method}-{ordinal}-{arm}"
            record = json.loads((directory / (name + ".json")).read_text())
            image = json.loads((directory / ("identity-" + name + ".json")).read_text())
            require(
                (image["source_identity"], image["library_sha256"]) == expected[arm],
                "Wrong normal ELF",
            )
            require(
                "streaming_trial" not in record,
                "Interposer is not normal native qualification",
            )
            require(
                record["status"] == "complete" and len(record["samples"]) == 4,
                "Incomplete population",
            )
            reduce_record(record)
            qualifications[name] = scientific(record)
            records[name] = record
        warm_ratios, cold, parity = [], {"control": [], "candidate": []}, []
        for name, record in records.items():
            arm = name.rsplit("-", 1)[1]
            cold[arm].append(record["samples"][0]["seconds"])
        baseline = records[method + "-1-control"]
        for name, record in records.items():
            for current, reference in zip(
                record["samples"], baseline["samples"], strict=True
            ):
                require(
                    current["iterations"] == reference["iterations"], "Changed SCF work"
                )
                require(
                    current["fock_builds"] == reference["fock_builds"],
                    "Changed Fock work",
                )
                require(
                    current["settings"] == reference["settings"], "Changed settings"
                )
                for key, value in reference["initial_guess"].items():
                    if key != "preparation_seconds":
                        require(
                            current["initial_guess"][key] == value,
                            "Changed initial-guess work",
                        )
                for key in ("grid_points", "tile_points", "ao_order", "scf_domain"):
                    require(
                        current["ks_diagnostic"][key]
                        == reference["ks_diagnostic"][key],
                        "Changed grid work",
                    )
                energy = abs(current["energy"] - reference["energy"])
                force = float(
                    np.max(
                        np.abs(
                            np.asarray(current["forces"])
                            - np.asarray(reference["forces"])
                        )
                    )
                )
                require(
                    energy < 1e-8 and force < 1e-7, "Strict native E/F parity failed"
                )
                parity.append({"energy_abs": energy, "force_max_abs": force})
        for ordinal, other in ((2, 1), (3, 4)):
            candidate = records[f"{method}-{ordinal}-candidate"]["samples"]
            control = records[f"{method}-{other}-control"]["samples"]
            warm_ratios.extend(
                current["seconds"] / baseline["seconds"]
                for current, baseline in zip(candidate[1:], control[1:], strict=True)
            )
        reference_path = directory / (method + "-reference.json")
        reference = (
            json.loads(reference_path.read_text()) if reference_path.exists() else None
        )
        if reference:
            reduce_record(reference)
            qualifications["current_gpu4pyscf"] = scientific(reference)
            for record in records.values():
                for sample in record["samples"]:
                    require(
                        all(
                            errors(sample["energy"], sample["forces"], oracle, 96)[
                                "passed"
                            ]
                            for oracle in reference["samples"]
                        ),
                        "Current GPU4PySCF comparison failed",
                    )
        report["methods"][method] = {
            "cold_samples": cold,
            "cold_means": {arm: mean(values) for arm, values in cold.items()},
            "cold_ratio": mean(cold["candidate"]) / mean(cold["control"]),
            "paired_warm_ratios": warm_ratios,
            "paired_warm_gate": max(warm_ratios) <= 1.03,
            "independent_qualifications": qualifications,
            "maximum_native_parity": {
                key: max(item[key] for item in parity) for key in parity[0]
            },
            "reference_cold": reference["samples"][0]["seconds"] if reference else None,
            "work": {
                name: [
                    {
                        key: item[key]
                        for key in (
                            "iterations",
                            "fock_builds",
                            "initial_guess",
                            "incremental_direct_jk",
                            "ks_diagnostic",
                        )
                    }
                    for item in record["samples"]
                ]
                for name, record in records.items()
            },
        }
    small = {
        arm: json.loads((directory / ("pbe0-48-" + arm + ".json")).read_text())
        for arm in ("control", "candidate")
    }
    report["independent_48"] = {
        arm: scientific(record) for arm, record in small.items()
    }
    for current, baseline in zip(
        small["candidate"]["samples"], small["control"]["samples"], strict=True
    ):
        reduce_record(small["candidate"])
        require(
            abs(current["energy"] - baseline["energy"]) < 1e-8,
            "48-atom native energy failed",
        )
        require(
            np.max(
                np.abs(np.asarray(current["forces"]) - np.asarray(baseline["forces"]))
            )
            < 1e-7,
            "48-atom native forces failed",
        )
    proof = (directory / "default-cold-kernels.tsv").read_text().splitlines()
    header = dict(field.split("=", 1) for field in proof[0].removeprefix("# ").split())
    require(
        set(header)
        == {"version", "dropped", "starved", "errors", "unknown", "flush", "disable"}
        and int(header["version"]) in (27, 28)
        and all(int(value) == 0 for key, value in header.items() if key != "version"),
        "Incomplete cold-only collector",
    )
    rows = list(csv.DictReader(proof[1:], delimiter="\t"))
    shared = [item for item in rows if "shared_rys_xy_streaming" in item["name"]]
    if card == "a100":
        require(
            len(shared) == 6 and all(int(item["count"]) == 9 for item in shared),
            "Automatic shared-XY execution was not proved",
        )
    else:
        require(not shared, "5090 default unexpectedly uses shared-XY")
    report["default_kernel_census"] = shared
    report["collector"] = header
    report["unacquired_counts"] = ["quartet_evaluations", "recurrence_evaluations"]
    return report
