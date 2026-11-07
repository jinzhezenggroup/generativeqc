"""Replay immutable dual-GPU HF promotion and rejected DFT evidence, without CUDA."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
from pathlib import Path
from typing import Any

from benchmarks.readme_defaults_report import reduce_record

DIRECTORY = Path(__file__).resolve().parent
METHODS = {"hf", "r2scan-3c", "pbe0", "wb97m-v"}


def require(condition: bool, message: str) -> None:
    """Keep integrity and scientific gates active under python -O."""
    if not condition:
        raise ValueError(message)


def errors(sample: dict, target: dict) -> tuple[float, float]:
    """Reject partial/nonfinite host exports before comparing observables."""
    for item in (sample, target):
        require(math.isfinite(item["energy"]), "nonfinite energy")
        require(len(item["forces"]) == 96, "missing force atoms")
        require(all(len(row) == 3 for row in item["forces"]), "missing force axes")
        require(
            all(math.isfinite(value) for row in item["forces"] for value in row),
            "nonfinite force",
        )
    maximum = max(
        abs(actual - expected)
        for row, other in zip(sample["forces"], target["forces"], strict=True)
        for actual, expected in zip(row, other, strict=True)
    )
    return abs(sample["energy"] - target["energy"]), maximum


def load_payload(directory: Path, specification: dict) -> dict:
    """Bind compressed raw records to an explicitly named card and image cohort."""
    path = directory / specification["file"]
    require(path.name == specification["file"], "unsafe evidence path")
    raw = path.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == specification["sha256"],
        "wrong receipt bytes",
    )
    payload = json.loads(lzma.decompress(raw))
    require(payload["exit_code"] == 0, "incomplete GPU qualification")
    require("6 passed" in payload["hf_log"], "missing independent HF/lifetime gates")
    require(
        "16 passed" in payload["libcint_log"], "missing independent signed J/K gates"
    )
    require(
        "numerical fallbacks and raw AD orbits PASS" in payload["raw_ad_log"],
        "missing finite/mask/exchange fallbacks",
    )
    if specification["device"] == "RTX5090":
        for tool, marker in (
            ("memcheck", "ERROR SUMMARY: 0 errors"),
            ("racecheck", "0 hazards displayed (0 errors, 0 warnings)"),
        ):
            require(marker in payload["sanitizers"][tool], "failed CUDA sanitizer")
    return payload


def check_pair(
    pair: dict, method: str, specification: dict, images: dict, target: dict
) -> dict:
    """Validate every endpoint; do not filter inaccurate or slower repeats."""
    require(set(pair) == {"baseline", "candidate"}, "missing comparison arm")
    reductions = {}
    for arm, record in pair.items():
        image = images[arm]
        for key in ("source_identity", "library_sha256"):
            require(
                record["native_build"][key] == image[key], "wrong source/ELF identity"
            )
        require(
            record["source_revision"] == image["source_identity"], "relabeled source"
        )
        scheduler = "LSB_JOBID" if specification["device"] == "A100" else "SLURM_JOB_ID"
        require(
            record["scheduler"][scheduler] == specification["job_id"], "wrong scheduler"
        )
        require(
            bool(record["scheduler"]["CUDA_VISIBLE_DEVICES"]), "missing GPU visibility"
        )
        name = record["accelerator"]["name"].replace(" ", "")
        require(specification["device"] in name, "wrong physical GPU")
        require(
            record["method"] == method
            and record["atoms"] == 96
            and record["mode"] == "direct"
            and record["arm"] == "standard",
            "wrong physical workload",
        )
        require(record["status"] == "complete", "failed endpoint")
        require(
            [(sample["phase"], sample["repeat"]) for sample in record["samples"]]
            == [("cold", 0), ("warm", 0), ("warm", 1), ("warm", 2)],
            "incomplete cold/warm population",
        )
        require(
            record["protocol"]["cold"] == "constructor+prepare+SCF+forces+host-export",
            "incomplete cold boundary",
        )
        require(
            record["protocol"]["warm"]
            == "same-owner; frozen engine-local cold density",
            "changed warm ownership",
        )
        for sample in record["samples"]:
            settings, work = sample["settings"], sample["precision"]
            require(
                (
                    settings["energy_tolerance"],
                    settings["density_tolerance"],
                    settings["screening_tolerance"],
                )
                == (1e-9, 1e-5, 1e-12),
                "changed public numerical controls",
            )
            require(
                work["effective_bits"] == 64 and work["mixed_stage_fock_builds"] == 0,
                "unqualified precision",
            )
            require(work["operator_work_counters_valid"], "invalid work census")
            if method != "hf":
                require(
                    work["complete"] and work["operator_inventory_complete"],
                    "incomplete DFT operator inventory",
                )
            require(
                sample["fock_builds"]
                == work["strict_stage_fock_builds"] + work["post_scf_fock_builds"],
                "omitted complete Fock work",
            )
            require(
                work["strict_stage_fock_builds"] == sample["iterations"],
                "omitted strict iterations",
            )
            energy, force = errors(sample, target if method == "hf" else sample)
            if method == "hf":
                require(
                    energy < 1e-8 and force < 1e-7, "independent HF observable failure"
                )
        reductions[arm] = reduce_record(record)
    baseline, candidate = pair["baseline"], pair["candidate"]
    for key in (
        "geometry_bohr",
        "orbital_basis",
        "basis_representation",
        "charge",
        "spin",
        "protocol",
        "runner_sha256",
    ):
        require(
            baseline[key] == candidate[key], "mismatched physical/protocol identity"
        )
    for first, last in zip(baseline["samples"], candidate["samples"], strict=True):
        require(first["settings"] == last["settings"], "mismatched executed settings")
        for key in ("complete", "operator_inventory_complete"):
            require(
                first["precision"][key] == last["precision"][key],
                "changed precision-inventory coverage",
            )
        if method != "hf":
            energy, force = errors(first, last)
            require(energy < 1e-8 and force < 1e-7, "changed DFT observables")
            require(
                (first["iterations"], first["fock_builds"])
                == (last["iterations"], last["fock_builds"]),
                "changed DFT SCF work",
            )
            for key in ("coulomb_exchange_work", "nonlocal_work"):
                require(
                    first["ks_diagnostic"][key] == last["ks_diagnostic"][key],
                    "changed semantic DFT operator work",
                )
    ratios = {
        phase: reductions["candidate"][phase]["median"]
        / reductions["baseline"][phase]["median"]
        for phase in ("cold", "warm")
    }
    require(
        all(value <= 1.03 for value in ratios.values()), "protected endpoint regression"
    )
    return {
        "ratios": ratios,
        "populations": reductions,
        "operator_inventory_complete": {
            arm: [
                sample["precision"]["operator_inventory_complete"]
                for sample in record["samples"]
            ]
            for arm, record in pair.items()
        },
    }


def replay(directory: Path = DIRECTORY) -> dict[str, Any]:
    """Separate final default qualification from the rejected combined-image trial."""
    cohort = json.loads((directory / "cohort.json").read_text())
    require(cohort["schema"] == "generativeqc.hf-cold-defaults.v1", "wrong cohort")
    require(cohort["decision"] == "scoped-hf-default", "unqualified promotion")
    require(cohort["build"]["exit_code"] == 0, "failed native build")
    require(
        cohort["build"]["identity"] == cohort["images"]["accepted"], "wrong final build"
    )
    require(cohort["owned_transfers_stopped"], "owned transfer still active")
    require(
        set(cohort["transfer_stop_receipts"]) == {"accepted", "rejected"}
        and all(
            "OWNED_TRANSFER_STOPPED" in receipt
            for receipt in cohort["transfer_stop_receipts"].values()
        ),
        "missing actual transfer cleanup receipt",
    )
    require(set(cohort["history"]) == {"A100", "RTX5090"}, "missing adverse history")
    for specification in cohort["history"].values():
        path = directory / specification["file"]
        require(path.name == specification["file"], "unsafe history path")
        require(
            hashlib.sha256(path.read_bytes()).hexdigest() == specification["sha256"],
            "wrong historical receipt bytes",
        )
    require(
        len({image["source_identity"] for image in cohort["images"].values()}) == 3,
        "rejected image relabeled as final",
    )
    experiment = cohort["rejected_dft"]
    require(
        experiment["decision"] == "rejected-no-endpoint-benefit", "DFT trial promoted"
    )
    require(
        hashlib.sha256(experiment["patch"].encode()).hexdigest()
        == experiment["patch_sha256"],
        "wrong rejected patch",
    )
    result: dict[str, Any] = {
        "images": cohort["images"],
        "cohorts": {},
        "statistical_cold_ranking": False,
    }
    for stage in ("accepted", "rejected"):
        devices = cohort["cohorts"][stage]
        require(set(devices) == {"A100", "RTX5090"}, "missing protected device")
        report = {}
        for device, specification in devices.items():
            require(specification["device"] == device, "mismatched card label")
            payload = load_payload(directory, specification)
            require(set(payload["records"]) == METHODS, "missing complete method")
            methods = {}
            for method, pair in payload["records"].items():
                methods[method] = check_pair(
                    pair,
                    method,
                    specification,
                    {
                        "baseline": cohort["images"]["baseline"],
                        "candidate": cohort["images"][stage],
                    },
                    payload["hf_targets"][0],
                )
            require(methods["hf"]["ratios"]["cold"] < 1, "missing HF cold benefit")
            if stage == "rejected":
                require(
                    methods["r2scan-3c"]["ratios"]["cold"] > 0.98,
                    "DFT rejection no longer matches the observed flat endpoint",
                )
            report[device] = methods
        result["cohorts"][stage] = report
    return result


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
