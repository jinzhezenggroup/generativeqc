"""Replay a rejected frozen-history pilot without converting it into promotion evidence."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

DIRECTORY = Path(__file__).resolve().parent
REFERENCE_PATH = (
    "benchmarks/results/direct-range-warm-anchor-20261105/publication.json.xz"
)
REFERENCE_SHA256 = "6659c9962ca035cc6c0b5411cf54c6f1f64409a922eda2d00aff29726b65880f"
REFERENCE_GATES = (1e-5, 1e-4, 3e-5)
PAIRED_GATES = (1e-9, 1e-7, 3e-8)
PROTECTION_RATIO = 1.03


def require(condition: Any, message: str) -> None:
    """Keep scientific and provenance checks active under ``python -O``."""
    if not condition:
        raise ValueError(message)


def original_file(population: dict[str, Any], name: str) -> str:
    payload = population["files"][name]
    text = payload["text"]
    require(hashlib.sha256(text.encode()).hexdigest() == payload["sha256"], name)
    return text


def errors(sample: dict[str, Any], target: dict[str, Any]) -> tuple[float, ...]:
    """Use every exported Cartesian force, with energy reported per atom."""
    differences = [
        actual - expected
        for row, other in zip(sample["forces"], target["forces"], strict=True)
        for actual, expected in zip(row, other, strict=True)
    ]
    require(len(sample["forces"]) == len(target["forces"]) == 96, "force atoms")
    require(len(differences) == 288, "Cartesian force population")
    observed = (
        abs(sample["energy"] - target["energy"]) / 96,
        max(map(abs, differences)),
        math.sqrt(math.fsum(value * value for value in differences) / 288),
    )
    require(all(map(math.isfinite, observed)), "nonfinite scientific output")
    return observed


def load() -> dict[str, Any]:
    cohort = json.loads((DIRECTORY / "cohort.json").read_text())
    payload = (DIRECTORY / "evidence.json.xz").read_bytes()
    require(
        hashlib.sha256(payload).hexdigest() == cohort["evidence_sha256"],
        "bundle checksum",
    )
    return json.loads(lzma.decompress(payload))


def replay(evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate both rejected first pairs; no historical reference latency is ranked."""
    if evidence is None:
        evidence = load()
    require(evidence["schema"] == "generativeqc.vv10-history-cold.v1", "schema")
    require(evidence["decision"] == "reject-default", "not promotion evidence")
    require(
        evidence["gates"]
        == {
            "reference": list(REFERENCE_GATES),
            "paired": list(PAIRED_GATES),
            "protection_ratio": PROTECTION_RATIO,
        },
        "unchanged acceptance gates",
    )
    require(
        set(evidence["devices"]) == {"A100", "RTX5090"}, "both actual devices required"
    )
    require(evidence["reference"]["path"] == REFERENCE_PATH, "reference input")
    require(evidence["reference"]["sha256"] == REFERENCE_SHA256, "reference identity")
    reference_bytes = (DIRECTORY.parents[2] / REFERENCE_PATH).read_bytes()
    require(
        hashlib.sha256(reference_bytes).hexdigest() == REFERENCE_SHA256,
        "reference bytes",
    )
    reference = json.loads(lzma.decompress(reference_bytes))
    reference_record = reference["records"]["A100/reference"]
    require(reference_record["status"] == "complete", "reference endpoint")
    require(reference_record["arm"] == "gpu4pyscf", "independent oracle")
    targets = reference_record["samples"]
    require(
        len(targets) == 2 and all(row["converged"] for row in targets),
        "reference samples",
    )
    image = evidence["image"]
    require(
        all(re.fullmatch(r"[0-9a-f]{64}", value) for value in image.values()),
        "image identity",
    )
    build = evidence["build"]
    require(original_file(build, "exit-code.txt").strip() == "0", "normal build failed")
    require(json.loads(original_file(build, "identity.json")) == image, "built image")
    require(
        "4c73bcf8a409a15362355d290695e7866c3a2b28753a42a1d61c91aba4ffe83a"
        in original_file(build, "restored-runtime.sha256"),
        "borrowed runtime restoration",
    )
    require(
        "OWNED_TRANSFER_STOPPED" in original_file(build, "transfer-cleanup.log"),
        "owned transfer cleanup",
    )
    summary: dict[str, Any] = {
        "decision": "reject-default",
        "source_image": image,
        "fresh_gpu4pyscf_ranking": False,
        "independent_comparisons": 0,
        "worst_reference_errors": [0.0] * 3,
        "worst_paired_errors": [0.0] * 3,
        "unmeasured": [
            "active compact points",
            "force pair sweeps",
            "exclusive rebase/Gram time",
        ],
        "devices": {},
    }
    for device, population in evidence["devices"].items():
        identity = json.loads(original_file(population, "identity.json"))
        require(all(identity[key] == image[key] for key in image), "loaded image")
        scheduler = identity["scheduler"]
        require(scheduler["CUDA_VISIBLE_DEVICES"], "assigned visibility")
        require(
            scheduler["LSB_JOBID"] if device == "A100" else scheduler["SLURM_JOB_ID"],
            "allocation",
        )
        if device == "A100":
            require(not scheduler["SLURM_JOB_ID"], "do not fabricate Slurm on LSF")
        fixture = original_file(population, "linked-fixture.log")
        require(
            "VV10_REBASE_LINKED_HELPERS_PASS cases=7 shutdown=0" in fixture,
            "linked helpers",
        )
        if device == "RTX5090":
            require(
                "ERROR SUMMARY: 0 errors" in original_file(population, "memcheck.log"),
                "memcheck",
            )
            require(
                "0 hazards displayed (0 errors, 0 warnings)"
                in original_file(population, "racecheck.log"),
                "racecheck",
            )
        records = {
            arm: json.loads(original_file(population, f"forward-{arm}.json"))
            for arm in ("baseline", "candidate")
        }
        baseline = records["baseline"]
        paired_passed = True
        result: dict[str, Any] = {"scheduler": scheduler}
        for arm, record in records.items():
            require(record["status"] == "complete", "incomplete endpoint")
            require(
                record["geometry_bohr"] == reference["cohort"]["geometry_bohr"],
                "geometry",
            )
            require(
                record["method"] == "wb97m-v" and record["mode"] == "direct", "method"
            )
            require(
                record["atoms"] == 96 and record["orbital_basis"] == "def2-SVP", "basis"
            )
            require(record["basis_representation"] == "spherical", "AO representation")
            require(record["charge"] == record["spin"] == 0, "electronic state")
            require(
                record["source_revision"] == image["source_identity"], "runner source"
            )
            require(
                all(record["native_build"][key] == image[key] for key in image),
                "native image",
            )
            require(record["scheduler"] == scheduler, "scheduler identity")
            require(
                ("A100" if device == "A100" else "5090")
                in record["accelerator"]["name"],
                "card",
            )
            require(not record["protocol"]["grid_overrides"], "grid overrides")
            control = record["history_transport_experiment"]
            require(
                control["arm"] == arm and control["clean"],
                "intrusive or mislabeled arm",
            )
            require(control["request"] == str(int(arm == "candidate")), "selector")
            require(len(record["samples"]) == 4, "retain first cold and three warms")
            for index, sample in enumerate(record["samples"]):
                require(
                    sample["converged"] and not sample["warm_start_fallback"],
                    "convergence",
                )
                require(sample["phase"] == ("cold" if index == 0 else "warm"), "phase")
                require(sample["warm_start_used"] == (index != 0), "warm provenance")
                require(
                    math.isfinite(sample["seconds"]) and sample["seconds"] > 0,
                    "endpoint time",
                )
                precision = sample["precision"]
                require(
                    precision["complete"] and precision["operator_inventory_complete"],
                    "precision inventory",
                )
                require(
                    precision["effective_bits"] == 64
                    and precision["mixed_stage_fock_builds"] == 0,
                    "FP64",
                )
                settings = sample["settings"]
                require(
                    settings == baseline["samples"][index]["settings"],
                    "public settings",
                )
                require(
                    settings["energy_tolerance"] == 1e-9
                    and settings["density_tolerance"] == 1e-5
                    and settings["screening_tolerance"] == 1e-12,
                    "physical stopping gates",
                )
                require(settings["force_grid_response"], "grid response")
                for target in targets:
                    observed = errors(sample, target)
                    require(
                        all(
                            value < gate
                            for value, gate in zip(
                                observed, REFERENCE_GATES, strict=True
                            )
                        ),
                        "independent E/F",
                    )
                    summary["independent_comparisons"] += 1
                    summary["worst_reference_errors"] = [
                        max(old, value)
                        for old, value in zip(
                            summary["worst_reference_errors"], observed, strict=True
                        )
                    ]
                observed = errors(sample, baseline["samples"][index])
                paired_passed &= all(
                    value < gate
                    for value, gate in zip(observed, PAIRED_GATES, strict=True)
                )
                summary["worst_paired_errors"] = [
                    max(old, value)
                    for old, value in zip(
                        summary["worst_paired_errors"], observed, strict=True
                    )
                ]
            log = original_file(population, f"forward-{arm}.log")
            receipts = [
                json.loads(match)
                for match in re.findall(
                    r"GENERATIVEQC_VV10_REBASE_HISTORY (\{[^\n]+\})", log
                )
            ]
            if arm == "candidate":
                require(
                    len(receipts) == 4 and receipts[0]["admission"] == 2,
                    "cold bridge census",
                )
                require(
                    receipts[0]["successful_rebases"] > 0
                    and receipts[0]["refusals"] == 0,
                    "actual transport",
                )
                require(
                    0 < receipts[0]["allocated_bytes"] <= 256 << 20,
                    "bounded optional lease",
                )
                require(
                    all(
                        row["admission"] == 3
                        and row["density_writes"] == 0
                        and row["successful_rebases"] == 0
                        for row in receipts[1:]
                    ),
                    "warm isolation",
                )
            cold = record["samples"][0]
            scopes = Counter(cold["ks_diagnostic"]["nonlocal_history_scopes"])
            require(sum(scopes.values()) == cold["fock_builds"], "phase work census")
            result[arm] = {
                "cold_seconds": cold["seconds"],
                "warm_median_seconds": statistics.median(
                    row["seconds"] for row in record["samples"][1:]
                ),
                "cold_focks": cold["fock_builds"],
                "phase_focks": dict(scopes),
                "nonlocal_work": cold["ks_diagnostic"]["nonlocal_work"],
                "bridge_receipts": receipts,
            }
        require(not paired_passed, "this pilot failed the tighter paired gate")
        require(
            original_file(population, "exit-code.txt").strip() == "2",
            "retain gate-failure exit",
        )
        require(
            "paired scientific gate refused" in original_file(population, "screen.log"),
            "gate refusal",
        )
        result["cold_ratio"] = (
            result["candidate"]["cold_seconds"] / result["baseline"]["cold_seconds"]
        )
        require(
            result["cold_ratio"] > PROTECTION_RATIO, "rejected cold protection gate"
        )
        result["warm_ratios"] = [
            after["seconds"] / before["seconds"]
            for before, after in zip(
                baseline["samples"][1:],
                records["candidate"]["samples"][1:],
                strict=True,
            )
        ]
        result["independent_EF_passed"] = True
        result["paired_EF_passed"] = False
        result["cold_protection_passed"] = False
        result["warm_protection_passed"] = (
            max(result["warm_ratios"]) <= PROTECTION_RATIO
        )
        summary["devices"][device] = result
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
