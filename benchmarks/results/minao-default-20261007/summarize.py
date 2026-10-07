"""Summarize all interleaved cold samples, including losing trajectories.

The two native arms share one binary and AO convention. A separate GPU4PySCF
oracle supplies energy/force gates; saved native densities detect any different
stationary solution without importing one arm's density into the other.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    """Require the full sample matrix before publishing median comparisons."""
    root = Path(sys.argv[1])
    report = {"schema": "minao-cold-profit.v1", "cases": {}}
    identities = set()
    for case in ("48", "96", "peroxide"):
        arms = {}
        for guess in ("core", "minao"):
            samples = [
                json.loads((root / f"{guess}{repeat}-{case}.json").read_text())
                for repeat in range(3)
            ]
            for sample in samples:
                if sample["status"] != "measured" or not sample["acceptance"]["gate"]:
                    raise RuntimeError(f"failed endpoint: {guess} {case}")
                if sample["warm_start_used"]:
                    raise RuntimeError(
                        f"cold sample reused a retained density: {guess} {case}"
                    )
                identities.add(sample["native_build"]["library_sha256"])
            times = [sample["complete_seconds"] for sample in samples]
            arms[guess] = {
                "complete_seconds": times,
                "median_seconds": statistics.median(times),
                "prepare_seconds": [sample["prepare_seconds"] for sample in samples],
                "iterations": [sample["iterations"] for sample in samples],
                "fock_builds": [sample["fock_builds"] for sample in samples],
                "xc_evaluations": [
                    sample["ao_work"]["xc_evaluations"] for sample in samples
                ],
                "physical_residual_rms": [
                    sample["physical_residual_rms"] for sample in samples
                ],
                "density_rms": [sample["density_rms"] for sample in samples],
                "initial_guess": [
                    sample["initial_guess_diagnostic"] for sample in samples
                ],
                "maximum_reference_energy_error": max(
                    sample["acceptance"]["energy_error"] for sample in samples
                ),
                "maximum_reference_force_error": max(
                    sample["acceptance"]["force_error"] for sample in samples
                ),
            }
            if guess == "minao":
                for sample in samples:
                    diagnostic = sample["initial_guess_diagnostic"]
                    if diagnostic["preliminary_fock_builds"] != 0:
                        raise RuntimeError(
                            "MINAO unexpectedly built a preliminary Fock"
                        )
                    if not diagnostic["work_counters_complete"]:
                        raise RuntimeError("incomplete target work census")
        differences = []
        for repeat in range(3):
            core = np.load(root / f"core{repeat}-{case}.density.npy")
            minao = np.load(root / f"minao{repeat}-{case}.density.npy")
            if (
                core.shape != minao.shape
                or not np.isfinite(core).all()
                or not np.isfinite(minao).all()
            ):
                raise RuntimeError("invalid paired final densities")
            differences.append(float(np.max(np.abs(core - minao))))
        arms["relative_wall_reduction"] = (
            1 - arms["minao"]["median_seconds"] / arms["core"]["median_seconds"]
        )
        arms["speedup"] = (
            arms["core"]["median_seconds"] / arms["minao"]["median_seconds"]
        )
        arms["paired_maximum_density_errors"] = differences
        report["cases"][case] = arms
    if len(identities) != 1:
        raise RuntimeError("native arms did not use the same binary")
    report["library_sha256"] = identities.pop()
    warm = json.loads((root / "minao2-peroxide.json").read_text())["warm_probe"]
    report["warm_probe"] = warm
    report["warm_skips_seed"] = (
        warm["warm_start_used"]
        and warm["acceptance"]["gate"]
        and warm["initial_guess_diagnostic"]["outcome"] == "existing_density"
        and warm["initial_guess_diagnostic"]["preliminary_fock_builds"] == 0
        and warm["initial_guess_diagnostic"]["preparation_seconds"] == 0
    )
    (root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for case, arms in report["cases"].items():
        print(
            f"{case}: core={arms['core']['median_seconds']:.6f}s "
            f"minao={arms['minao']['median_seconds']:.6f}s "
            f"reduction={arms['relative_wall_reduction']:.2%} "
            f"Fock {arms['core']['fock_builds']} -> {arms['minao']['fock_builds']}"
        )


if __name__ == "__main__":
    main()
