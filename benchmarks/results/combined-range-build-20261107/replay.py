"""Recheck complete composite-build endpoints and independent E/F acceptance."""

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
    """Retain scientific and provenance gates under optimized Python."""
    if not condition:
        raise ValueError(message)


def errors(sample: dict, target: dict) -> tuple[float, float, float]:
    """Compare the complete finite energy and all exported force components."""
    for record in (sample, target):
        require(len(record["forces"]) == 96, "incomplete force atoms")
        require(all(len(row) == 3 for row in record["forces"]), "incomplete force axes")
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


def source_values(sample: dict) -> int:
    """Count completed resident outputs, not both represented radial terms."""
    return sum(
        operator["count"]
        for operator in sample["precision"]["operators"]
        if operator["name"] == "coulomb_recurrence"
    )


def replay(
    directory: Path = DIRECTORY, evidence_name: str = "evidence.json.xz"
) -> dict[str, Any]:
    """Qualify each same-image paired population without reusing old latency."""
    evidence = json.loads(lzma.decompress((directory / evidence_name).read_bytes()))
    require(
        evidence["schema"] == "generativeqc.combined-range-build.v1", "wrong schema"
    )
    image = evidence["image"]
    require(
        evidence["source_receipt"]["source_identity"] == image["source_identity"],
        "wrong frozen source",
    )
    require(
        evidence["build"]["identity"] == image and evidence["build"]["exit_code"] == 0,
        "unqualified native build",
    )
    require(
        hashlib.sha256(evidence["native_patch"].encode()).hexdigest()
        == evidence["native_patch_sha256"],
        "wrong reconstruction patch",
    )
    if sync := evidence.get("master_sync"):
        previous_bytes = (ROOT / sync["previous_evidence"]).read_bytes()
        require(
            hashlib.sha256(previous_bytes).hexdigest() == sync["previous_sha256"],
            "wrong pre-master evidence",
        )
        previous = json.loads(lzma.decompress(previous_bytes))
        require(
            sync["integration_commit"] == evidence["source_receipt"]["base"],
            "wrong master integration source",
        )
        require(
            previous["image"]["source_identity"] != image["source_identity"]
            and previous["image"]["library_sha256"] != image["library_sha256"],
            "master synchronization relabeled an old image",
        )
    require(evidence["owned_transfer_stopped"], "owned transfer remains active")
    reference_bytes = (ROOT / evidence["reference"]["path"]).read_bytes()
    require(
        hashlib.sha256(reference_bytes).hexdigest() == evidence["reference"]["sha256"],
        "wrong independent reference",
    )
    reference = json.loads(lzma.decompress(reference_bytes))
    incumbent_bytes = (ROOT / evidence["incumbent"]["path"]).read_bytes()
    require(
        hashlib.sha256(incumbent_bytes).hexdigest() == evidence["incumbent"]["sha256"],
        "wrong retained incumbent",
    )
    incumbent = json.loads(lzma.decompress(incumbent_bytes))["cohorts"]["final"]
    targets = reference["records"]["A100/reference"]["samples"]
    require(
        set(evidence["devices"]) == {"A100", "RTX5090"}, "missing GPU qualification"
    )
    summary: dict[str, Any] = {
        "image": image,
        "fresh_reference_latency_ranking": False,
        "devices": {},
    }
    for device, population in evidence["devices"].items():
        require(population["exit_code"] == 0, "incomplete scheduler allocation")
        require(
            "combined build oracle checks 4" in population["oracle_log"],
            "missing independent signed composite gate",
        )
        require(
            "combined build refusal checks 4" in population["oracle_log"],
            "missing usable raw fallback gate",
        )
        for key, value in (("combined_source_build", 1), ("source_radial_terms", 2)):
            require(
                any(
                    entry.get("key") == key and entry.get("value") == value
                    for entry in population["oracle_semantic_values"]
                ),
                "missing direct composite dispatch",
            )
        if device == "RTX5090":
            require(
                population["oracle_tests"] == 6 and population["oracle_skipped"] == 0,
                "incomplete displaced/f-shell oracle population",
            )
            require(
                "ERROR SUMMARY: 0 errors" in population["sanitizer"], "memcheck failed"
            )
        baseline, candidate = (
            population["records"][arm] for arm in ("baseline", "default")
        )
        previous_default = incumbent["devices"][device]["records"]["default"]
        for arm, record in population["records"].items():
            require(record["status"] == "complete", "incomplete endpoint")
            require(
                record["source_revision"] == image["source_identity"],
                "wrong source image",
            )
            require(
                record["native_build"]["source_identity"] == image["source_identity"]
                and record["native_build"]["library_sha256"] == image["library_sha256"],
                "wrong loaded library",
            )
            for key, value in image.items():
                require(
                    population["identities"][arm][key] == value, "wrong image receipt"
                )
            scheduler = record["scheduler"]
            require(
                scheduler == population["identities"][arm]["scheduler"]
                and bool(scheduler["CUDA_VISIBLE_DEVICES"]),
                "lost scheduler visibility",
            )
            job_key = "LSB_JOBID" if device == "A100" else "SLURM_JOB_ID"
            require(
                str(scheduler[job_key]) == str(population["scheduler_job"]),
                "wrong GPU job",
            )
            require(
                ("A100" if device == "A100" else "5090")
                in record["accelerator"]["name"],
                "wrong accelerator",
            )
            require(
                record["method"] == "wb97m-v"
                and record["mode"] == "direct"
                and record["atoms"] == 96
                and record["basis_representation"] == "spherical",
                "changed scientific problem",
            )
            for key in ("geometry_bohr", "orbital_basis", "charge", "spin"):
                require(
                    record[key] == reference["cohort"][key],
                    "changed independent scientific inputs",
                )
            require(
                record["protocol"] == previous_default["protocol"],
                "changed cold/warm endpoint protocol",
            )
            require(
                record["protocol"]["cold"]
                == "constructor+prepare+SCF+forces+host-export",
                "incomplete cold scope",
            )
            experiment = record["source_schedule_experiment"]
            require(
                experiment["combined_range_source"] is None
                and experiment["combined_range_build"]
                == ("0" if arm == "baseline" else None)
                and experiment["clean"],
                "not a clean incumbent/default comparison",
            )
            require(len(record["samples"]) == 2, "missing first own-density warm")
        samples = []
        for index, (old, new, target) in enumerate(
            zip(baseline["samples"], candidate["samples"], targets, strict=True)
        ):
            phase = "cold" if index == 0 else "warm"
            require(old["phase"] == new["phase"] == phase, "wrong endpoint phase")
            require(old["converged"] and new["converged"], "unconverged endpoint")
            require(
                old["iterations"] == new["iterations"] == (16 if index == 0 else 1),
                "changed SCF iterations",
            )
            require(
                old["fock_builds"] == new["fock_builds"] == old["iterations"],
                "changed Fock work",
            )
            require(
                old["settings"]
                == new["settings"]
                == previous_default["samples"][index]["settings"],
                "changed grids/convergence",
            )
            require(
                old["ks_diagnostic"]["nonlocal_work"]
                == new["ks_diagnostic"]["nonlocal_work"],
                "changed VV10 work",
            )
            nonlocal_work = new["ks_diagnostic"]["nonlocal_work"]
            require(
                nonlocal_work["provisional_focks"] == (11 if index == 0 else 0)
                and nonlocal_work["target_focks"] == (5 if index == 0 else 1)
                and nonlocal_work["pair_sweeps"] == (4 if index == 0 else 0),
                "wrong retained stage/sweep population",
            )
            require(
                old["initial_guess"]["preliminary_fock_builds"]
                == new["initial_guess"]["preliminary_fock_builds"],
                "changed staged solve",
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
                paired[0] < 1e-10 and paired[1] < 2e-9, "paired native E/F gate failed"
            )
            old_k = old["ks_diagnostic"]["coulomb_exchange_work"][
                "exchange_applications"
            ]
            new_k = new["ks_diagnostic"]["coulomb_exchange_work"][
                "exchange_applications"
            ]
            require(
                old_k == (1 if device == "A100" else 2) * old["fock_builds"]
                and new_k == new["fock_builds"],
                "composite K work mismatch",
            )
            completed, previous = source_values(new), source_values(old)
            if phase == "cold":
                require(
                    completed == 1873221716
                    and previous == (2 if device == "A100" else 1) * completed,
                    "duplicated or incomplete resident construction",
                )
            else:
                require(
                    completed == previous == 0, "warm call rebuilt resident sources"
                )
            require(
                math.isfinite(new["seconds"])
                and math.isfinite(old["seconds"])
                and min(new["seconds"], old["seconds"]) > 0,
                "invalid endpoint timing",
            )
            ratio = new["seconds"] / old["seconds"]
            require(
                ratio < (0.98 if device == "A100" and phase == "cold" else 1.03),
                "complete endpoint performance gate failed",
            )
            # A slower rollback kernel must not manufacture a same-image win.
            # This is a historical guard, not a fresh cross-image paired cohort.
            prior_seconds = previous_default["samples"][index]["seconds"]
            require(
                new["seconds"] < 1.03 * prior_seconds,
                "retained default non-regression gate failed",
            )
            samples.append(
                {
                    "phase": phase,
                    "baseline_seconds": old["seconds"],
                    "default_seconds": new["seconds"],
                    "ratio": ratio,
                    "retained_prior_default_seconds": prior_seconds,
                    "paired_energy_error_per_atom": paired[0],
                    "paired_max_force_error": paired[1],
                    "baseline_k_actions": old_k,
                    "default_k_actions": new_k,
                    "baseline_source_outputs": previous,
                    "default_source_outputs": completed,
                }
            )
        summary["devices"][device] = {
            "scheduler_job": population["scheduler_job"],
            "samples": samples,
        }
    return summary


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "master_synced": replay(),
                "pre_master": replay(evidence_name="pre-master-evidence.json.xz"),
            },
            indent=2,
        )
    )
