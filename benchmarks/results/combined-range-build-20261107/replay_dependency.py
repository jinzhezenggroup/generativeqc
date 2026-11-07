"""Validate late dependency integration without inventing another paired timing."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import runpy
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]
OWNERS = runpy.run_path(str(DIRECTORY / "replay.py"))
require, errors, source_values = (
    OWNERS[name] for name in ("require", "errors", "source_values")
)


def replay(directory: Path = DIRECTORY) -> dict:
    """Gate the actual integrated default against independent E/F and prior work."""
    evidence = json.loads(
        lzma.decompress((directory / "dependency-evidence.json.xz").read_bytes())
    )
    require(
        evidence["schema"] == "generativeqc.combined-range-build.dependency.v1",
        "wrong schema",
    )
    paired_bytes = (DIRECTORY / "evidence.json.xz").read_bytes()
    require(
        hashlib.sha256(paired_bytes).hexdigest() == evidence["paired_evidence_sha256"],
        "wrong paired performance cohort",
    )
    paired = json.loads(lzma.decompress(paired_bytes))
    reference_bytes = (ROOT / paired["reference"]["path"]).read_bytes()
    require(
        hashlib.sha256(reference_bytes).hexdigest() == paired["reference"]["sha256"],
        "wrong independent reference",
    )
    targets = json.loads(lzma.decompress(reference_bytes))["records"]["A100/reference"][
        "samples"
    ]
    image = evidence["image"]
    require(
        image == evidence["build"]["identity"] and evidence["build"]["exit_code"] == 0,
        "incomplete integrated build",
    )
    require(
        image["source_identity"] == evidence["source_receipt"]["source_identity"]
        and image["source_identity"] != paired["image"]["source_identity"],
        "dependency synchronization relabeled an old image",
    )
    require(evidence["owned_transfer_stopped"], "owned transfer still active")
    require(
        set(evidence["devices"]) == {"A100", "RTX5090"}, "missing integrated GPU gate"
    )
    summary = {"image": image, "fresh_paired_performance_claim": False, "devices": {}}
    for device, population in evidence["devices"].items():
        record = population["record"]
        previous = paired["devices"][device]["records"]["default"]
        require(
            ("A100" if device == "A100" else "5090") in record["accelerator"]["name"],
            "wrong integrated accelerator",
        )
        require(
            population["exit_code"] == 0 and record["status"] == "complete",
            "incomplete integrated endpoint",
        )
        require(
            record["source_revision"] == image["source_identity"],
            "wrong integrated source",
        )
        for key, value in image.items():
            require(
                record["native_build"][key] == population["identity"][key] == value,
                "wrong integrated executable",
            )
        scheduler = record["scheduler"]
        job_key = "LSB_JOBID" if device == "A100" else "SLURM_JOB_ID"
        require(
            scheduler == population["identity"]["scheduler"]
            and bool(scheduler["CUDA_VISIBLE_DEVICES"])
            and str(scheduler[job_key]) == str(population["scheduler_job"]),
            "wrong integrated scheduler allocation",
        )
        for key in (
            "method",
            "mode",
            "atoms",
            "basis_representation",
            "orbital_basis",
            "charge",
            "spin",
            "geometry_bohr",
            "protocol",
        ):
            require(
                record[key] == previous[key], "changed integrated scientific endpoint"
            )
        selectors = record["source_schedule_experiment"]
        require(
            selectors["combined_range_source"] is None
            and selectors["combined_range_build"] is None
            and selectors["clean"],
            "integrated default is opt-in or contaminated",
        )
        require(
            "combined build oracle checks 4" in population["oracle_log"]
            and "combined build refusal checks 4" in population["oracle_log"],
            "missing integrated composite/refusal oracle",
        )
        if device == "RTX5090":
            require(
                population["oracle_tests"] == 6
                and population["oracle_skipped"] == 0
                and "ERROR SUMMARY: 0 errors" in population["sanitizer"],
                "incomplete integrated displaced/f-shell/compact qualification",
            )
        require(len(record["samples"]) == 2, "missing integrated cold or first warm")
        samples = []
        for index, (sample, old, target) in enumerate(
            zip(record["samples"], previous["samples"], targets, strict=True)
        ):
            require(
                sample["phase"] == old["phase"] and sample["converged"],
                "wrong phase or convergence",
            )
            require(
                sample["iterations"]
                == sample["fock_builds"]
                == (16 if index == 0 else 1),
                "changed integrated Fock work",
            )
            require(
                sample["settings"] == old["settings"]
                and sample["initial_guess"]["preliminary_fock_builds"]
                == old["initial_guess"]["preliminary_fock_builds"],
                "changed integrated solve",
            )
            require(
                sample["ks_diagnostic"]["nonlocal_work"]
                == old["ks_diagnostic"]["nonlocal_work"],
                "changed integrated VV10 work",
            )
            precision = sample["precision"]
            require(
                precision["requested_mode"] == "fp64"
                and precision["effective_bits"] == 64
                and precision["conversion_count"] == 0,
                "changed integrated arithmetic",
            )
            work = sample["ks_diagnostic"]["coulomb_exchange_work"]
            require(
                work["complete"]
                and work["coulomb_applications"]
                == work["exchange_applications"]
                == sample["fock_builds"],
                "changed integrated J/K work",
            )
            require(
                source_values(sample) == (1873221716 if index == 0 else 0),
                "changed integrated source work",
            )
            discrepancy, parity = errors(sample, target), errors(sample, old)
            require(
                discrepancy[0] < 1e-8
                and discrepancy[1] < 1e-4
                and discrepancy[2] < 3e-5,
                "integrated independent E/F failed",
            )
            require(
                parity[0] < 1e-10 and parity[1] < 2e-9, "integrated native E/F changed"
            )
            require(
                math.isfinite(sample["seconds"])
                and 0 < sample["seconds"] < 1.03 * old["seconds"],
                "integrated historical non-regression failed",
            )
            samples.append(
                {
                    "phase": sample["phase"],
                    "seconds": sample["seconds"],
                    "historical_default_seconds": old["seconds"],
                    "native_energy_error_per_atom": parity[0],
                    "native_max_force_error": parity[1],
                }
            )
        summary["devices"][device] = {
            "scheduler_job": population["scheduler_job"],
            "samples": samples,
        }
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))
