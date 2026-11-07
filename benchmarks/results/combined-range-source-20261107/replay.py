"""Replay source-bound complete endpoints and independent numerical gates offline."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    """Keep acceptance gates active when the reader runs with Python -O."""
    if not condition:
        raise ValueError(message)


def errors(sample: dict, target: dict) -> tuple[float, float, float]:
    """Compare every exported force component, rejecting incomplete endpoints."""
    for record in (sample, target):
        require(len(record["forces"]) == 96, "incomplete force atoms")
        require(
            all(len(row) == 3 for row in record["forces"]),
            "incomplete force components",
        )
        require(math.isfinite(record["energy"]), "nonfinite energy")
        require(
            all(math.isfinite(value) for row in record["forces"] for value in row),
            "nonfinite force",
        )
    differences = [
        actual - expected
        for row, other in zip(sample["forces"], target["forces"], strict=True)
        for actual, expected in zip(row, other, strict=True)
    ]
    return (
        abs(sample["energy"] - target["energy"]) / 96,
        max(map(abs, differences)),
        math.sqrt(math.fsum(value * value for value in differences) / len(differences)),
    )


def replay(directory: Path = DIRECTORY) -> dict[str, Any]:
    """Bind pilot/final populations to their own images; never relabel a pilot."""
    evidence = json.loads(
        lzma.decompress((directory / "evidence.json.xz").read_bytes())
    )
    require(
        evidence["schema"] == "generativeqc.combined-range-source.v1", "wrong schema"
    )
    require(
        set(evidence["cohorts"]) == {"pilot", "final"}, "missing source-bound cohort"
    )
    reference_bytes = (ROOT / evidence["reference"]["path"]).read_bytes()
    require(
        hashlib.sha256(reference_bytes).hexdigest() == evidence["reference"]["sha256"],
        "reference identity mismatch",
    )
    reference = json.loads(lzma.decompress(reference_bytes))
    targets = reference["records"]["A100/reference"]["samples"]
    summary: dict[str, Any] = {"fresh_reference_latency_ranking": False, "cohorts": {}}
    for name, cohort in evidence["cohorts"].items():
        image = cohort["image"]
        require(
            cohort["source_receipt"]["source_identity"] == image["source_identity"],
            "source receipt mismatch",
        )
        require(
            cohort["build"]["identity"] == image and cohort["build"]["exit_code"] == 0,
            "unqualified build",
        )
        require(
            hashlib.sha256(cohort["native_patch"].encode()).hexdigest()
            == cohort["native_patch_sha256"],
            "native patch mismatch",
        )
        require(cohort["owned_transfer_stopped"], "owned transfer not retired")
        devices = {}
        for device, population in cohort["devices"].items():
            require(population["exit_code"] == 0, "incomplete GPU allocation")
            require(
                "combined range oracle checks 4" in population["oracle_log"],
                "missing independent composite source gate",
            )
            require(
                any(
                    entry.get("key") == "radial_terms_per_exchange"
                    and entry.get("value") == 2
                    for entry in population["oracle_semantic_values"]
                ),
                "missing actual two-radial composite dispatch",
            )
            if device == "RTX5090":
                require(
                    population["oracle_tests"] == (4 if name == "final" else 2),
                    "incorrect oracle test population",
                )
                require(population["oracle_skipped"] == 0, "skipped source gate")
                require(
                    "ERROR SUMMARY: 0 errors" in population["sanitizer"],
                    "memcheck failed",
                )
            records = population["records"]
            candidate_arm = "default" if name == "final" else "combined"
            baseline, candidate = records["baseline"], records[candidate_arm]
            for arm, record in records.items():
                require(record["status"] == "complete", "incomplete endpoint")
                require(
                    record["source_revision"] == image["source_identity"],
                    "wrong image source",
                )
                for key, value in image.items():
                    require(
                        record["native_build"][key]
                        == value
                        == population["identities"][arm][key],
                        "endpoint image identity mismatch",
                    )
                require(
                    record["scheduler"] == population["identities"][arm]["scheduler"],
                    "scheduler identity mismatch",
                )
                require(
                    bool(record["scheduler"]["CUDA_VISIBLE_DEVICES"]),
                    "lost scheduler visibility",
                )
                require(
                    ("A100" if device == "A100" else "5090")
                    in record["accelerator"]["name"],
                    "wrong physical GPU",
                )
                require(
                    record["method"] == "wb97m-v"
                    and record["mode"] == "direct"
                    and record["atoms"] == 96
                    and record["basis_representation"] == "spherical",
                    "changed scientific problem",
                )
                require(
                    record["geometry_bohr"] == reference["cohort"]["geometry_bohr"],
                    "wrong geometry",
                )
                require(
                    record["protocol"]["cold"]
                    == "constructor+prepare+SCF+forces+host-export",
                    "incomplete cold scope",
                )
                experiment = record["source_schedule_experiment"]
                selector = (
                    "0" if arm == "baseline" else None if name == "final" else "1"
                )
                require(
                    experiment["combined_range_source"] == selector
                    and experiment["clean"],
                    "not a clean selected endpoint",
                )
                require(
                    len(record["samples"]) == 2, "missing cold/first-warm population"
                )
            samples = []
            for index, (old, new, target) in enumerate(
                zip(baseline["samples"], candidate["samples"], targets, strict=True)
            ):
                phase = "cold" if index == 0 else "warm"
                require(old["phase"] == new["phase"] == phase, "wrong endpoint phase")
                require(old["converged"] and new["converged"], "unconverged endpoint")
                require(
                    old["iterations"] == new["iterations"] == (16 if index == 0 else 1),
                    "changed SCF work",
                )
                require(
                    old["fock_builds"] == new["fock_builds"] == old["iterations"],
                    "changed Fock work",
                )
                require(old["settings"] == new["settings"], "changed grids/convergence")
                require(
                    old["ks_diagnostic"]["nonlocal_work"]
                    == new["ks_diagnostic"]["nonlocal_work"],
                    "changed VV10 work",
                )
                for sample in (old, new):
                    require(
                        sample["precision"]["requested_mode"] == "fp64"
                        and sample["precision"]["effective_bits"] == 64
                        and sample["precision"]["conversion_count"] == 0,
                        "changed arithmetic",
                    )
                    work = sample["ks_diagnostic"]["coulomb_exchange_work"]
                    require(
                        work["complete"]
                        and work["coulomb_applications"] == sample["fock_builds"],
                        "incomplete J work",
                    )
                    reference_errors = errors(sample, target)
                    require(
                        reference_errors[0] < 1e-8
                        and reference_errors[1] < 1e-4
                        and reference_errors[2] < 3e-5,
                        "independent E/F acceptance failed",
                    )
                paired = errors(new, old)
                require(
                    paired[0] < 1e-10 and paired[1] < 2e-9,
                    "native paired E/F gate failed",
                )
                old_k = old["ks_diagnostic"]["coulomb_exchange_work"][
                    "exchange_applications"
                ]
                new_k = new["ks_diagnostic"]["coulomb_exchange_work"][
                    "exchange_applications"
                ]
                require(
                    old_k == 2 * old["fock_builds"]
                    and new_k == (1 if device == "A100" else 2) * new["fock_builds"],
                    "weighted contraction/fallback work mismatch",
                )
                require(
                    math.isfinite(new["seconds"])
                    and old["seconds"] > 0
                    and new["seconds"] > 0,
                    "invalid timing",
                )
                ratio = new["seconds"] / old["seconds"]
                require(
                    ratio < (0.98 if device == "A100" and phase == "cold" else 1.03),
                    "complete endpoint performance gate failed",
                )
                samples.append(
                    {
                        "phase": phase,
                        "baseline_seconds": old["seconds"],
                        "candidate_seconds": new["seconds"],
                        "ratio": ratio,
                        "paired_energy_error_per_atom": paired[0],
                        "paired_max_force_error": paired[1],
                        "baseline_k_actions": old_k,
                        "candidate_k_actions": new_k,
                    }
                )
            devices[device] = {
                "scheduler_job": population["scheduler_job"],
                "samples": samples,
            }
        summary["cohorts"][name] = {"image": image, "devices": devices}
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))
