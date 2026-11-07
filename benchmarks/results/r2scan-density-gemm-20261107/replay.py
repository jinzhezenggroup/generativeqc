"""Replay source-bound complete endpoints, not a kernel-only speed claim."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parents[2]))
from benchmarks._exact_fp64_evidence import load as load_exact


def require(condition: bool, detail: str) -> None:
    """Keep scientific and provenance gates active under Python -O."""
    if not condition:
        raise ValueError(detail)


def errors(sample: dict, reference: dict) -> list[float]:
    """Compare all 96 atomic forces without silently truncating either vector."""
    for record in (sample, reference):
        require(len(record["forces"]) == 96, "incomplete atomic forces")
        require(all(len(row) == 3 for row in record["forces"]), "incomplete Cartesian forces")
        require(math.isfinite(record["energy"]), "nonfinite energy")
        require(all(math.isfinite(value) for row in record["forces"] for value in row), "nonfinite force")
    differences = [actual - expected
                   for row, target in zip(sample["forces"], reference["forces"], strict=True)
                   for actual, expected in zip(row, target, strict=True)]
    return [abs(sample["energy"] - reference["energy"]),
            max(map(abs, differences)),
            math.sqrt(math.fsum(value * value for value in differences) / 288)]


def load_evidence(directory: Path = DIRECTORY) -> dict[str, Any]:
    """Authenticate and reconstruct exact original FP64 endpoint receipt bytes."""
    bundle, files = load_exact(directory)
    evidence = bundle["experiment"]
    evidence["files"] = files
    return evidence


def replay(directory: Path = DIRECTORY) -> dict[str, Any]:
    """Recompute the frozen predeclared gates and expose every retained cold."""
    evidence = load_evidence(directory)
    require(evidence["schema"] == "generativeqc.r2scan-density-gemm.v1", "wrong schema")
    files = evidence["files"]
    require(set(files) == set(evidence["sha256"]), "missing raw receipt checksum")
    for name, text in files.items():
        require(hashlib.sha256(text.encode()).hexdigest() == evidence["sha256"][name], "corrupt raw receipt: " + name)

    def text(name: str) -> str:
        return files[name]

    def record(name: str) -> dict:
        return json.loads(text(name))

    image = evidence["image"]
    require(record("source-receipt.json")["source_identity"] == image["source_identity"], "source/image mismatch")
    require(text("build/exit-code.txt").strip() == "0", "failed build")
    require(record("build/identity.json") == image, "wrong final build")
    require("sccache 0.16.0" in text("build/cache-version.txt"), "unverified compiler cache")
    require(evidence["restored_library_sha256"] in text("build/restored-runtime.sha256"), "borrowed image not restored")
    require("OWNED_TRANSFER_STOPPED" in text("cleanup.log"), "owned transfer not retired")
    require(hashlib.sha256(text("source.patch").encode()).hexdigest() == evidence["source_patch_sha256"], "wrong reconstruction patch")
    reconstruction = record("campaign/source-reconstruction.json")
    require(reconstruction["one_off_reconstruction_verified"] and reconstruction["reconstructed_source_identity"] == image["source_identity"] and reconstruction["patch_sha256"] == evidence["source_patch_sha256"], "unverified source reconstruction")
    require(set(evidence["devices"]) == {"A100", "RTX5090"}, "missing hardware protection")
    qualified = True
    summary: dict[str, Any] = {"image": image, "devices": {}, "scope": "96-atom static default-grid complete E/F"}
    common_inputs = None
    for device, population in evidence["devices"].items():
        prefix = population["prefix"] + "/"
        identity = record(prefix + "identity.json")
        require(all(identity[key] == value for key, value in image.items()), "wrong executed source/image")
        scheduler = identity["scheduler"]
        job_key = "LSB_JOBID" if device == "A100" else "SLURM_JOB_ID"
        require(str(scheduler[job_key]) == str(population["job"]) and bool(scheduler["CUDA_VISIBLE_DEVICES"]), "lost scheduler provenance")
        require("Qualified indexed PBE/r2SCAN RKS/UKS full-AO and bounded-fallback gates passed" in text(prefix + "independent-density.log"), "missing independent scalar oracle")
        if device == "RTX5090":
            require("ERROR SUMMARY: 0 errors" in text(prefix + "memcheck.log"), "memcheck failed")
            require("0 errors, 0 warnings" in text(prefix + "racecheck.log"), "racecheck failed")
        reference_prefix = population.get("reference_prefix", population["prefix"]) + "/"
        reference_identity = record(reference_prefix + "identity.json")
        require(all(reference_identity[key] == value for key, value in image.items()), "wrong reference-campaign image")
        if device == "A100":
            failed = record(prefix + "reference.json")
            require(failed["status"] == "failed" and failed["samples"] == [], "failed original reference relabelled")
            require("cuda_fp16.h" in text(prefix + "reference.log"), "missing original reference failure")
            require(text(prefix + "exit-code.txt").strip() == "1", "lost original scheduler failure")
            require(reference_prefix != prefix and str(reference_identity["scheduler"][job_key]) == str(population["reference_job"]), "relabelled reference repair allocation")
            require("NVRTC header root:" in text(reference_prefix + "reference-environment-repair.txt"), "missing environment-only repair")
        require(text(reference_prefix + "exit-code.txt").strip() == "0", "incomplete final qualification")
        reference = record(reference_prefix + "reference.json")
        require(reference["status"] == "complete" and len(reference["samples"]) == 4, "incomplete independent reference")
        require(reference["scheduler"] == reference_identity["scheduler"], "wrong independent reference allocation")
        require(all(sample["converged"] and math.isfinite(sample["seconds"]) and sample["seconds"] > 0 for sample in reference["samples"]), "invalid reference endpoint")
        results = {"reference_cold": reference["samples"][0]["seconds"], "reference_same_allocation": reference_prefix == prefix, "pairs": {}}
        for order in ("forward", "reverse"):
            names = [prefix + order + "-" + arm + ".json" for arm in ("baseline", "candidate")]
            if not all(name in files for name in names):
                require(order == "reverse", "missing first cold pair")
                qualified = False
                continue
            before, after = map(record, names)
            ratios = []
            pair_errors = [0.0, 0.0, 0.0]
            independent_errors = [0.0, 0.0, 0.0]
            for arm, endpoint in zip(("baseline", "candidate"), (before, after), strict=True):
                require(endpoint["status"] == "complete" and len(endpoint["samples"]) == 4, "incomplete native population")
                require(endpoint["source_revision"] == image["source_identity"] and all(endpoint["native_build"][key] == value for key, value in image.items()), "relabelled executable")
                require(endpoint["scheduler"] == scheduler, "wrong endpoint allocation")
                require(endpoint["method"] == "r2scan-3c" and endpoint["mode"] == "direct" and endpoint["atoms"] == 96, "changed workload")
                require(("A100" if device == "A100" else "5090") in endpoint["accelerator"]["name"], "wrong measured GPU")
                experiment = endpoint["density_gemm_experiment"]
                require(experiment == {"arm": arm, "control": "0" if arm == "baseline" else None, "clean": True}, "not the default-versus-rollback clean experiment")
                inputs = {key: endpoint[key] for key in ("geometry_bohr", "orbital_basis", "basis_representation", "charge", "spin", "protocol")}
                require(inputs["protocol"]["cold"] == "constructor+prepare+SCF+forces+host-export" and inputs["protocol"]["warm"] == "same-owner; frozen engine-local cold density", "changed cold/warm ownership contract")
                if common_inputs is None:
                    common_inputs = inputs
                require(inputs == common_inputs, "changed scientific inputs or complete endpoint protocol")
                for key in ("geometry_bohr", "orbital_basis", "charge", "spin"):
                    require(endpoint[key] == reference[key], "mismatched independent scientific task")
                for index, (sample, target) in enumerate(zip(endpoint["samples"], reference["samples"], strict=True)):
                    require(sample["phase"] == ("cold" if index == 0 else "warm"), "lost first cold")
                    require(sample["converged"] and math.isfinite(sample["seconds"]) and sample["seconds"] > 0, "invalid complete endpoint")
                    require(sample["settings"]["energy_tolerance"] == 1e-9 and sample["settings"]["density_tolerance"] == 1e-5 and sample["settings"]["screening_tolerance"] == 1e-12, "weakened convergence")
                    require(sample["settings"]["force_grid_response"], "missing moving-grid force response")
                    require(sample["settings"]["xc_grid_points"] == 1078336, "changed sampled quadrature")
                    precision = sample["precision"]
                    require(precision["requested_mode"] == "fp64" and precision["effective_bits"] == 64 and precision["mixed_stage_fock_builds"] == 0 and precision["execution_retries"] == 0, "not strict FP64 execution")
                    require(all(operator[key] == "fp64" for operator in precision["operators"] for key in ("storage", "compute", "accumulation", "reduction")), "lower precision operator")
                    observed = errors(sample, target)
                    observed[0] /= 96
                    require(all(value < gate for value, gate in zip(observed, (1e-5, 1e-4, 3e-5), strict=True)), "independent E/F gate")
                    independent_errors = [max(old, value) for old, value in zip(independent_errors, observed, strict=True)]
            for initial, result in zip(before["samples"], after["samples"], strict=True):
                require(initial["settings"] == result["settings"] and initial["precision"] == result["precision"], "changed numerical policy")
                for key in ("fock_builds", "iterations"):
                    require(initial[key] == result[key], "different solver work")
                for key in ("coulomb_exchange_work", "nonlocal_work", "grid_points", "tile_points"):
                    require(initial["ks_diagnostic"][key] == result["ks_diagnostic"][key], "different semantic work")
                observed = errors(result, initial)
                qualified &= all(value < gate for value, gate in zip(observed, (1e-8, 1e-7, 3e-8), strict=True))
                pair_errors = [max(old, value) for old, value in zip(pair_errors, observed, strict=True)]
                ratios.append(result["seconds"] / initial["seconds"])
            qualified &= ratios[0] < (0.98 if device == "A100" else 1.03) and max(ratios[1:]) <= 1.03
            results["pairs"][order] = {"cold_seconds": [before["samples"][0]["seconds"], after["samples"][0]["seconds"]],
                                       "warm_medians": [statistics.median(item["seconds"] for item in endpoint["samples"][1:]) for endpoint in (before, after)],
                                       "ratios": ratios, "paired_errors": pair_errors,
                                       "independent_errors": independent_errors,
                                       "cold_focks": [endpoint["samples"][0]["fock_builds"] for endpoint in (before, after)]}
        if device == "RTX5090":
            require(text(prefix + "exit-code.txt").strip() == "0", "unexpected scheduler outcome")
        summary["devices"][device] = results
    trace_prefix = evidence["devices"]["A100"]["reference_prefix"] + "/"
    traces = {}
    for arm in ("baseline", "candidate"):
        profile = record(trace_prefix + "profile-" + arm + ".json")
        require(profile["density_gemm_experiment"] == {"arm": arm, "control": "0" if arm == "baseline" else None, "clean": False}, "intrusive attribution relabelled as clean")
        require(profile["source_revision"] == image["source_identity"] and profile["status"] == "complete", "wrong attribution source")
        attribution = record(trace_prefix + arm + "-components.summary.json")
        require(attribution["schema"] == "generativeqc.named-owner-attribution.v1", "wrong attribution aggregation")
        require(attribution["raw"] == evidence["local_raw_inventory"][trace_prefix + arm + "-components.jsonl"], "lost original trace identity")
        rows = attribution["rows"]
        require(all(row["valid"] and not row["dropped_regions"] and not row["dropped_tiles"] for row in rows), "incomplete attribution")
        require(all(math.isfinite(region["gpu_ms"]) and region["gpu_ms"] >= 0 for row in rows for region in row["regions"]), "invalid CUDA-event attribution")
        xc = [row for row in rows if row["operation"] == "cuda_ks_xc_shared"]
        cold_focks = profile["samples"][0]["fock_builds"]
        require(len(xc) == sum(sample["fock_builds"] for sample in profile["samples"]), "incomplete XC owner attribution")
        cold = xc[:cold_focks]
        require(all(row["counters"]["xc_density_provider_allowance"] == (0 if arm == "baseline" else 96 << 20) for row in cold), "provider was not actually selected")
        require(all(row["counters"]["xc_density_matrix_bytes"] == (0 if arm == "baseline" else 9469952) for row in cold), "wrong explicit matrix storage")
        traces[arm] = cold
    work_keys = ("xc_projection_tiles", "xc_active_point_ao", "xc_point_ao_visits", "xc_point_ao_square_sum", "xc_matrix_projection_fma_pairs")
    for before, after in zip(traces["baseline"], traces["candidate"], strict=True):
        require(all(before["counters"][key] == after["counters"][key] for key in work_keys), "different attributed logical matrix work")
    summary["attribution_matrix_projection_ms"] = {
        arm: math.fsum(region["gpu_ms"] for row in rows for region in row["regions"] if region["name"] == "xc_matrix_projection")
        for arm, rows in traces.items()
    }
    require(evidence["decision"] == ("default-qualified" if qualified else "rejected"), "unqualified default promotion")
    summary["decision"] = evidence["decision"]
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
