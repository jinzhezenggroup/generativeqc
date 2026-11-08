"""Qualify radius-adjusted Becke lowering without rewriting measured populations."""

from __future__ import annotations

import csv
import json
import lzma
import math
import re
from functools import partial
from pathlib import Path
from statistics import mean, median

import numpy as np

from benchmarks.dft_streaming_qualification import qualify_independent_record
from benchmarks.readme_defaults_accuracy import errors
from benchmarks.readme_defaults_report import reduce_record


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def _paired_work_and_science(candidate: dict, control: dict) -> dict:
    """Compare physical outputs and semantic work, never normalize iteration counts."""
    parity = []
    for current, baseline in zip(candidate["samples"], control["samples"], strict=True):
        for key in (
            "phase",
            "repeat",
            "iterations",
            "fock_builds",
            "settings",
            "precision",
        ):
            _require(current[key] == baseline[key], f"Changed {key}")
        for key, value in baseline["initial_guess"].items():
            if key != "preparation_seconds":
                _require(
                    current["initial_guess"][key] == value, "Changed initial-guess work"
                )
        for key in ("grid_points", "tile_points", "ao_order", "scf_domain"):
            _require(
                current["ks_diagnostic"][key] == baseline["ks_diagnostic"][key],
                "Changed grid work",
            )
        energy = abs(current["energy"] - baseline["energy"])
        force = float(
            np.max(
                np.abs(np.asarray(current["forces"]) - np.asarray(baseline["forces"]))
            )
        )
        _require(
            energy < 1e-8 and force < 1e-7, "Strict native energy/force parity failed"
        )
        parity.append({"energy_abs": energy, "force_max_abs": force})
    return {key: max(row[key] for row in parity) for key in parity[0]}


def _cold_census(path: Path, *, scaled: bool, record: dict) -> dict:
    """Authenticate actual first-gradient launches, not a modeled saving or counter."""
    lines = path.read_text().splitlines()
    header = dict(field.split("=", 1) for field in lines[0].removeprefix("# ").split())
    _require(
        set(header)
        == {"version", "dropped", "starved", "errors", "unknown", "flush", "disable"}
        and int(header["version"]) in (27, 28)
        and all(int(value) == 0 for key, value in header.items() if key != "version"),
        "Incomplete cold collector",
    )
    rows = list(csv.DictReader(lines[1:], delimiter="\t"))
    _require(bool(rows), "Empty cold collector")
    for row in rows:
        _require(
            all(int(row[key]) > 0 for key in ("count", "total_ns", "max_ns")),
            "Invalid cold collector row",
        )
    diagnostic = record["samples"][0]["ks_diagnostic"]
    batches = math.ceil(diagnostic["grid_points"] / diagnostic["tile_points"])
    selected = [
        row
        for row in rows
        if any(
            marker in row["name"]
            for marker in (
                "phased_becke_pairILb0ELb0ELb0ELb1",
                "phased_becke_pairILb1ELb0ELb0ELb1",
                "phased_becke_atomILi1ELb0ELb1",
                "phased_becke_normalize_cooperativeILb1",
            )
        )
    ]
    _require(
        (len(selected) == 4 and all(int(row["count"]) == batches for row in selected))
        if scaled
        else not selected,
        "Actual scaled Becke execution disagrees with selected arm",
    )
    becke = [
        row
        for row in rows
        if any(
            marker in row["name"]
            for marker in (
                "phased_becke_pair",
                "phased_becke_atom",
                "phased_becke_normalize",
            )
        )
    ]
    _require(
        len(becke) == 7 and all(int(row["count"]) == batches for row in becke),
        "Changed cold Becke phase work",
    )
    return {
        "collector": header,
        "force_batches": batches,
        "scaled_launches": selected,
        "all_becke_rows": becke,
        "becke_kernel_seconds": sum(int(row["total_ns"]) for row in becke) / 1e9,
        "kernel_seconds_are_not_endpoint_savings": True,
    }


def qualify_becke(directory: Path, card: str, *, image: dict[str, str]) -> dict:
    """Gate balanced complete endpoints, all own-density warms and actual dispatch.

    Both arms use one normal full ELF. The incumbent explicitly disables the
    representation, while the candidate leaves its existing selector unset.
    Single-pair r2SCAN-3c/nonlocal diagnostics remain separate from ABBA claims.
    """
    _require(card in ("a100", "5090"), "Unknown qualification card")
    performance_failures = []

    def performance(condition: bool, detail: str) -> None:
        if not condition:
            performance_failures.append(detail)

    allocation_exit = int((directory / "exit-code.txt").read_text().strip())
    performance(
        allocation_exit == 0, "Allocation failed; incomplete diagnostics cannot qualify"
    )
    _require(
        re.search(r"\b36 passed in", (directory / "native-owner-tests.log").read_text())
        is not None,
        "Missing actual native-owner gates",
    )
    oracle_root = Path(__file__).resolve().parent / "results/readme-refresh-20261026"
    scientific = partial(
        qualify_independent_record,
        oracles=json.loads(
            lzma.decompress((oracle_root / "oracles.json.xz").read_bytes())
        ),
        reference_contracts=json.loads(
            (oracle_root / "reference-reuse.json").read_text()
        )["entries"],
    )
    expected = {}
    for method in ("b3lyp", "pbe0"):
        for ordinal, arm in enumerate(
            ("control", "candidate", "candidate", "control"), 1
        ):
            expected[f"{method}-{ordinal}-{arm}"] = (method, 96, "standard")
        for ordinal in (1, 2):
            expected[f"{method}-reference-{ordinal}"] = (method, 96, "gpu4pyscf")
        for arm in ("control", "candidate"):
            expected[f"{method}-{arm}-profile"] = (method, 96, "standard")
    for arm in ("control", "candidate"):
        expected[f"pbe0-48-{arm}"] = ("pbe0", 48, "standard")
        for method in ("r2scan-3c", "wb97m-v"):
            expected[f"{method}-single-{arm}"] = (method, 96, "standard")
    records, qualifications = {}, {}
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("identity-"):
            continue
        record = json.loads(path.read_text())
        _require(path.stem in expected, "Unexpected endpoint population")
        _require(
            (record["method"], record["atoms"], record["arm"]) == expected[path.stem]
            and record["mode"] == "direct",
            "Wrong endpoint method, geometry size, arm or mode",
        )
        _require(
            record["status"] == "complete" and len(record["samples"]) == 4,
            "Incomplete endpoint population",
        )
        reduce_record(record)
        qualifications[path.stem] = scientific(record)
        records[path.stem] = record
        if record["arm"] == "standard":
            _require(
                all(
                    record["native_build"][key] == image[key]
                    for key in ("source_identity", "library_sha256")
                )
                and "streaming_trial" not in record,
                "Wrong profile or endpoint production ELF",
            )
        if record["arm"] == "standard" and not path.stem.endswith("-profile"):
            observed = json.loads(path.with_name("identity-" + path.name).read_text())
            _require(
                all(
                    observed[key] == image[key]
                    for key in ("source_identity", "library_sha256")
                ),
                "Wrong normal ELF",
            )
            controls = dict(
                line.split("=", 1)
                for line in path.with_suffix(".env").read_text().splitlines()
            )
            _require(
                controls.get("GENERATIVEQC_STATIONARY_BECKE_SCALED_PRODUCTS") == "0"
                if path.stem.endswith("-control")
                else "GENERATIVEQC_STATIONARY_BECKE_SCALED_PRODUCTS" not in controls,
                "Candidate is not automatic or incumbent selector changed",
            )
    missing = set(expected) - set(records)
    _require(
        not missing
        or (
            allocation_exit != 0 and all(name.endswith("-profile") for name in missing)
        ),
        "Missing endpoint population",
    )
    report = {
        "card": card,
        "methods": {},
        "single_pairs": {},
        "profiles": {},
        "independent": qualifications,
        "allocation_exit": allocation_exit,
        "missing_diagnostics": sorted(missing),
    }
    for method in ("b3lyp", "pbe0"):
        names = [
            f"{method}-{ordinal}-{arm}"
            for ordinal, arm in enumerate(
                ("control", "candidate", "candidate", "control"), 1
            )
        ]
        population = {name: records[name] for name in names}
        baseline = population[names[0]]
        parity = [
            _paired_work_and_science(record, baseline) for record in population.values()
        ]
        cold = {
            arm: [
                record["samples"][0]["seconds"]
                for name, record in population.items()
                if name.endswith("-" + arm)
            ]
            for arm in ("control", "candidate")
        }
        cold_ratio = mean(cold["candidate"]) / mean(cold["control"])
        relative_mads = [
            median(abs(value - median(values)) for value in values) / median(values)
            for values in cold.values()
        ]
        noise_gate = max(0.02, 2 * sum(relative_mads))
        performance(
            cold_ratio <= (1 - noise_gate if card == "a100" else 1.03),
            "Cold/noise or 5090 regression gate failed",
        )
        warm_ratios = [
            current["seconds"] / control["seconds"]
            for candidate_name, control_name in (
                (names[1], names[0]),
                (names[2], names[3]),
            )
            for current, control in zip(
                population[candidate_name]["samples"][1:],
                population[control_name]["samples"][1:],
                strict=True,
            )
        ]
        performance(max(warm_ratios) <= 1.03, f"{method}: paired warm regression")
        references = [records[f"{method}-reference-{ordinal}"] for ordinal in (1, 2)]
        for reference in references:
            for record in population.values():
                for sample in record["samples"]:
                    _require(
                        all(
                            errors(sample["energy"], sample["forces"], oracle, 96)[
                                "passed"
                            ]
                            for oracle in reference["samples"]
                        ),
                        "Fresh GPU4PySCF science gate failed",
                    )
        reference_colds = [record["samples"][0]["seconds"] for record in references]
        report["methods"][method] = {
            "cold_samples": cold,
            "cold_means": {arm: mean(values) for arm, values in cold.items()},
            "cold_ratio": cold_ratio,
            "noise_gate": noise_gate,
            "paired_warm_ratios": warm_ratios,
            "maximum_native_parity": {
                key: max(row[key] for row in parity) for key in parity[0]
            },
            "reference_cold_samples": reference_colds,
            "reference_cold_mean": mean(reference_colds),
            "reference_anomaly_cause": "unmeasured; timing alone does not establish a cause",
        }
        for arm in ("control", "candidate"):
            name = f"{method}-{arm}"
            if name + "-profile" not in records:
                continue
            profile = records[name + "-profile"]
            _paired_work_and_science(profile, baseline)
            report["profiles"][name] = _cold_census(
                directory / (name + "-cold-kernels.tsv"),
                scaled=arm == "candidate",
                record=profile,
            )
    for method in ("r2scan-3c", "wb97m-v", "pbe0-48"):
        stem = method if method == "pbe0-48" else method + "-single"
        control, candidate = (
            records[f"{stem}-{arm}"] for arm in ("control", "candidate")
        )
        parity = _paired_work_and_science(candidate, control)
        cold_ratio = (
            candidate["samples"][0]["seconds"] / control["samples"][0]["seconds"]
        )
        warms = [
            current["seconds"] / baseline["seconds"]
            for current, baseline in zip(
                candidate["samples"][1:], control["samples"][1:], strict=True
            )
        ]
        performance(
            cold_ratio <= 1.03 and max(warms) <= 1.03,
            "Single-pair nonregression failed",
        )
        report["single_pairs"][method] = {
            "cold_ratio": cold_ratio,
            "paired_warm_ratios": warms,
            "native_parity": parity,
            "balanced_performance_qualification": False,
        }
    if card == "5090":
        for tool in ("memcheck", "racecheck", "synccheck"):
            text = (directory / f"sanitizer-{tool}.log").read_text()
            _require(
                re.search(r"(?:ERROR|RACECHECK) SUMMARY: 0 (?:errors|hazards)", text)
                is not None,
                "Missing passing actual sanitizer",
            )
    report["performance_failures"] = performance_failures
    report["default_promotion_qualified"] = not performance_failures
    report["all_readme_leadership_established"] = False
    return report
