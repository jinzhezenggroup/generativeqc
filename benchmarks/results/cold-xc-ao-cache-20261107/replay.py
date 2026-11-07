"""Replay the rejected AO-cache experiment, including incomplete GPU calls."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
from pathlib import Path
import statistics
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parent
REFERENCE = "benchmarks/results/direct-range-warm-anchor-20261105/publication.json.xz"
REFERENCE_SHA256 = "6659c9962ca035cc6c0b5411cf54c6f1f64409a922eda2d00aff29726b65880f"
INCUMBENT = "benchmarks/results/combined-range-build-20261107/evidence.json.xz"


def require(condition: bool, message: str) -> None:
    """Keep all acceptance gates effective under optimized Python."""
    if not condition:
        raise ValueError(message)


def errors(sample: dict, target: dict) -> tuple[float, float, float]:
    """Check complete finite exports before calculating independent E/F errors."""
    for record in (sample, target):
        require(math.isfinite(record["energy"]), "nonfinite energy")
        require(len(record["forces"]) == 96, "incomplete force atoms")
        require(all(len(row) == 3 for row in record["forces"]), "incomplete force axes")
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


def retained(path: str, digest: str) -> dict:
    """Authenticate historical numerical inputs, never reuse their timings."""
    payload = (ROOT / path).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == digest, "wrong retained evidence")
    return json.loads(lzma.decompress(payload))


def replay(directory: Path = DIRECTORY) -> dict[str, Any]:
    """Retain complete A100 science without promoting the failed 5090 candidate."""
    evidence = json.loads(lzma.decompress((directory / "evidence.json.xz").read_bytes()))
    require(evidence["schema"] == "generativeqc.cold-xc-ao-cache-rejection.v1", "wrong schema")
    require(evidence["decision"] == "rejected-not-promoted", "unqualified promotion")
    experiment = evidence["experiment"]
    image = experiment["image"]
    for snapshot in (experiment, evidence["handoff_compile_only"]):
        require(snapshot["source_receipt"]["source_identity"] == snapshot["image"]["source_identity"], "wrong source")
        require(snapshot["build_exit_code"] == 0, "build failed")
        require(bool(snapshot["overlays"]) and set(snapshot["overlays"]) <= set(snapshot["source_receipt"]["overlays"]), "missing reconstruction overlay")
        for name, text in snapshot["overlays"].items():
            require(hashlib.sha256(text.encode()).hexdigest() == snapshot["source_receipt"]["overlays"][name], "wrong frozen overlay")
        require(hashlib.sha256(snapshot["patch"].encode()).hexdigest() == snapshot["patch_sha256"], "wrong reconstruction patch")
    require(
        evidence["handoff_compile_only"]["image"]["source_identity"] != image["source_identity"]
        and evidence["handoff_compile_only"]["image"]["library_sha256"] != image["library_sha256"]
        and evidence["handoff_gpu_jobs"] == [],
        "compile-only repair relabelled as GPU evidence",
    )
    require(any("OWNED_TRANSFER_STOPPED" in text for text in evidence["a100_transport"].values()), "owned transfer remains active")
    require(
        evidence["reference"] == {"path": REFERENCE, "sha256": REFERENCE_SHA256},
        "changed independent reference",
    )
    reference = retained(REFERENCE, REFERENCE_SHA256)
    require(evidence["incumbent"]["path"] == INCUMBENT, "changed settings authority")
    incumbent = retained(INCUMBENT, evidence["incumbent"]["sha256"])
    require(set(evidence["devices"]) == {"A100", "RTX5090"}, "missing device")
    summary: dict[str, Any] = {
        "image": image,
        "decision": evidence["decision"],
        "fresh_gpu4pyscf_ranking": False,
        "devices": {},
    }
    for device, population in evidence["devices"].items():
        require(population["exit_code"] == (0 if device == "A100" else 1), "changed scheduler outcome")
        require(
            "Independent AO cache E/V, changed density, nonlocal pullback and retirement passed"
            in population["oracle_log"],
            "missing independent AO gate",
        )
        if device == "RTX5090":
            require("ERROR SUMMARY: 0 errors" in population["memcheck"], "memcheck failed")
        require(set(population["records"]) == {"baseline", "default"}, "missing paired arm")
        prior = incumbent["devices"][device]["records"]["default"]
        for arm, record in population["records"].items():
            if device == "RTX5090" and arm == "default":
                require(record["status"] == "failed" and record["samples"] == [], "incomplete OOM call relabelled")
                require("out of memory" in population["logs"][arm], "missing retained OOM")
                require(all(population["identities"][arm][key] == value for key,value in image.items()), "wrong failed image")
                require(record["source_revision"] == image["source_identity"] and record["scheduler"] == population["identities"][arm]["scheduler"], "wrong failed source/scheduler")
                require(record["protocol"] == prior["protocol"], "changed failed endpoint scope")
                for key in ("geometry_bohr", "orbital_basis", "charge", "spin"):
                    require(record[key] == reference["cohort"][key], "changed failed scientific inputs")
                continue
            require(record["status"] == "complete", "incomplete endpoint")
            require(record["source_revision"] == image["source_identity"], "wrong source revision")
            require(
                all(record["native_build"][key] == value for key, value in image.items()),
                "wrong loaded ELF",
            )
            require(
                all(population["identities"][arm][key] == value for key, value in image.items()),
                "wrong identity receipt",
            )
            scheduler = record["scheduler"]
            job_key = "LSB_JOBID" if device == "A100" else "SLURM_JOB_ID"
            require(
                scheduler == population["identities"][arm]["scheduler"]
                and str(scheduler[job_key]) == str(population["scheduler_job"])
                and bool(scheduler["CUDA_VISIBLE_DEVICES"]),
                "lost scheduler ownership/visibility",
            )
            require(
                ("A100" if device == "A100" else "5090") in record["accelerator"]["name"],
                "wrong accelerator",
            )
            require(
                record["method"] == "wb97m-v" and record["mode"] == "direct"
                and record["atoms"] == 96 and record["basis_representation"] == "spherical",
                "changed scientific task",
            )
            for key in ("geometry_bohr", "orbital_basis", "charge", "spin"):
                require(record[key] == reference["cohort"][key], "changed independent inputs")
            require(record["protocol"] == prior["protocol"], "changed endpoint/warm protocol")
            require(
                record["protocol"]["cold"] == "constructor+prepare+SCF+forces+host-export"
                and record["protocol"]["warm"] == "same-owner; frozen engine-local cold density",
                "incomplete endpoint scope",
            )
            experiment = record["geometry_reuse_experiment"]
            require(
                experiment["arm"] == arm and experiment["clean"]
                and experiment["ao_cache_control"] == ("0" if arm == "baseline" else None)
                and experiment["combined_range_source"] is None
                and experiment["combined_range_build"] is None,
                "unclean or weakened incumbent comparison",
            )
            require(record["warm_repeats"] == 3 and len(record["samples"]) == 4, "missing warm samples")
        baseline, candidate = (population["records"][arm] for arm in ("baseline", "default"))
        if device == "RTX5090":
            require(all(math.isfinite(sample["seconds"]) and sample["seconds"] > 0 for sample in baseline["samples"]), "invalid completed baseline timing")
            for index, sample in enumerate(baseline["samples"]):
                cold = index == 0
                target_index = 0 if cold else 1
                require(sample["phase"] == ("cold" if cold else "warm") and sample["repeat"] == max(index-1,0), "wrong baseline phase")
                require(sample["converged"] and sample["iterations"] == sample["fock_builds"] == (16 if cold else 1), "changed baseline Focks")
                require(sample["settings"] == prior["samples"][target_index]["settings"], "changed baseline settings")
                precision = sample["precision"]
                require(precision["requested_mode"] == "fp64" and precision["effective_bits"] == 64 and precision["conversion_count"] == 0, "changed baseline arithmetic")
                actions = sample["ks_diagnostic"]["coulomb_exchange_work"]
                require(actions["complete"] and actions["coulomb_applications"] == actions["exchange_applications"] == sample["fock_builds"], "changed baseline J/K work")
                work = sample["ks_diagnostic"]["nonlocal_work"]
                require(work["provisional_focks"] == (11 if cold else 0) and work["target_focks"] == (5 if cold else 1) and work["pair_sweeps"] == (4 if cold else 0), "changed baseline VV10 work")
                outputs = sum(operator["count"] for operator in precision["operators"] if operator["name"] == "coulomb_recurrence")
                require(outputs == (1873221716 if cold else 0), "changed baseline source work")
                energy,maximum,rms = errors(sample, reference["records"]["A100/reference"]["samples"][target_index])
                require(energy < 1e-8 and maximum < 1e-4 and rms < 3e-5, "baseline independent E/F failed")
            diagnostic = evidence["rtx5090_diagnostic"]
            require(diagnostic["exit_code"] == 1 and diagnostic["record"]["samples"] == [] and "out of memory" in diagnostic["log"], "failed force diagnostic relabelled")
            require(all(diagnostic["identity"][key] == value for key,value in image.items()), "wrong diagnostic image")
            require(any(entry["operation"] == "stationary_force" for entry in diagnostic["components"]), "missing force-lifetime diagnosis")
            summary["devices"][device] = {
                "scheduler_job": population["scheduler_job"],
                "baseline_cold_seconds": baseline["samples"][0]["seconds"],
                "baseline_warm_seconds": [sample["seconds"] for sample in baseline["samples"][1:]],
                "candidate_endpoint_complete": False,
                "candidate_seconds": None,
                "outcome": "out-of-memory; not performance-qualified",
            }
            continue
        samples = []
        for index, (old, new) in enumerate(zip(baseline["samples"], candidate["samples"], strict=True)):
            cold = index == 0
            target_index = 0 if cold else 1
            prior_sample = prior["samples"][target_index]
            target = reference["records"]["A100/reference"]["samples"][target_index]
            require(old["phase"] == new["phase"] == ("cold" if cold else "warm"), "wrong phase")
            require(old["repeat"] == new["repeat"] == max(index - 1, 0), "wrong repeat")
            require(old["settings"] == new["settings"] == prior_sample["settings"], "changed settings")
            require(
                old["ks_diagnostic"]["nonlocal_work"] == new["ks_diagnostic"]["nonlocal_work"],
                "changed nonlocal work",
            )
            work = new["ks_diagnostic"]["nonlocal_work"]
            require(
                work["provisional_focks"] == (11 if cold else 0)
                and work["target_focks"] == (5 if cold else 1)
                and work["pair_sweeps"] == (4 if cold else 0),
                "changed stage/sweep work",
            )
            for sample in (old, new):
                require(sample["converged"], "unconverged endpoint")
                require(sample["iterations"] == sample["fock_builds"] == (16 if cold else 1), "changed Focks")
                require(sample["warm_start_used"] == (not cold), "lost own-density warm start")
                require(
                    sample["initial_guess"]["preliminary_fock_builds"]
                    == prior_sample["initial_guess"]["preliminary_fock_builds"],
                    "changed preliminary solve",
                )
                precision = sample["precision"]
                require(
                    precision["requested_mode"] == "fp64" and precision["effective_bits"] == 64
                    and precision["conversion_count"] == 0,
                    "changed arithmetic",
                )
                actions = sample["ks_diagnostic"]["coulomb_exchange_work"]
                require(
                    actions["complete"]
                    and actions["coulomb_applications"] == actions["exchange_applications"] == sample["fock_builds"],
                    "changed J/K work",
                )
                outputs = sum(operator["count"] for operator in precision["operators"] if operator["name"] == "coulomb_recurrence")
                require(outputs == (1873221716 if cold else 0), "changed source construction")
                require(math.isfinite(sample["seconds"]) and sample["seconds"] > 0, "invalid timing")
                energy, maximum, rms = errors(sample, target)
                require(energy < 1e-8 and maximum < 1e-4 and rms < 3e-5, "independent E/F gate failed")
            energy, maximum, _ = errors(new, old)
            require(energy < 1e-10 and maximum < 2e-9, "paired E/F gate failed")
            samples.append({
                "phase": new["phase"], "repeat": new["repeat"],
                "baseline_seconds": old["seconds"], "default_seconds": new["seconds"],
                "ratio": new["seconds"] / old["seconds"],
                "paired_energy_error_per_atom": energy, "paired_force_max": maximum,
            })
        cold_ratio = samples[0]["ratio"]
        require(0.98 < cold_ratio < 1.03, "changed modest A100 observation")
        old_warm = [sample["seconds"] for sample in baseline["samples"][1:]]
        new_warm = [sample["seconds"] for sample in candidate["samples"][1:]]
        warm_ratio = statistics.median(new_warm) / statistics.median(old_warm)
        require(warm_ratio < 1.03 and all(sample["ratio"] < 1.05 for sample in samples[1:]), "warm regression")
        require(candidate["samples"][0]["seconds"] < 1.03 * prior["samples"][0]["seconds"], "historical cold regression")
        summary["devices"][device] = {
            "scheduler_job": population["scheduler_job"], "samples": samples,
            "cold_ratio": cold_ratio, "warm_median_ratio": warm_ratio,
            "baseline_warm_median": statistics.median(old_warm),
            "default_warm_median": statistics.median(new_warm),
            "default_warm_min": min(new_warm), "default_warm_max": max(new_warm),
        }
    diagnostic = evidence["devices"]["A100"]["diagnostic"]
    require(diagnostic["record"]["status"] == "complete" and not diagnostic["record"]["geometry_reuse_experiment"]["clean"], "intrusive diagnostic relabelled as clean")
    require(all(diagnostic["record"]["native_build"][key] == value for key,value in image.items()), "wrong admission image")
    admission = {entry["key"]: entry["value"] for entry in diagnostic["progress"] if entry.get("name") == "cuda_xc_ao_cache_prepare" and entry.get("event") == "VALUE"}
    require(0 < admission["xc_ao_cache_bytes"] <= 8 << 30 and admission["xc_ao_cache_tiles"] > 0, "no bounded cache admission")
    summary["a100_geometry_bytes"] = admission["xc_ao_cache_bytes"]
    summary["handoff_repair_gpu_qualified"] = False
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))
