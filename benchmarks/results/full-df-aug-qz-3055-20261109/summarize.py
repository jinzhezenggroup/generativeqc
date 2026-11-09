"""Recheck retained independent diatomic E/F evidence and summarize endpoints.

For two atoms, rotational invariance fixes each force to the bond direction.
An independent energy slope along a direction with nonzero bond projection
therefore determines every Cartesian force component, not just one projection.
This supplements the executable molecular tests without rerunning an oracle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CASES = (
    "aug-cc-pvqz-df-ccsd(t)-1",
    "aug-cc-pvqz-mp2--1",
    "augmented-tq-cbs-delta-d",
    "augmented-tq-cbs-delta-t",
)


def independent_forces(atoms: list, direction: list, slope: float) -> np.ndarray:
    """Recover the complete diatomic force from an independent energy slope."""
    bond = np.array(atoms[1][1]) - np.array(atoms[0][1])
    bond /= np.linalg.norm(bond)
    shifts = np.asarray(direction)
    projection = (shifts[1] - shifts[0]) @ bond
    assert abs(projection) > 0.1, "ill-conditioned diatomic force reconstruction"
    radial_derivative = slope / projection
    return np.array([radial_derivative * bond, -radial_derivative * bond])


def summarize(case: str) -> dict:
    """Verify both displacement steps and retain absolute error metrics."""
    report = json.loads((ROOT / (case + ".json")).read_text())
    cbs = "result" in report
    if cbs:
        result = report["result"]
        components = [component["result"] for component in result["components"]]
        coefficients = np.array(
            [component["coefficient"] for component in result["components"]]
        )
        expected = np.array(report["oracle_component_energies"])
        actual = np.array([component["energy"] for component in components])
        energy_error = max(
            float(np.max(np.abs(actual - expected))),
            abs(result["energy"] - coefficients @ expected),
        )
        assert energy_error < 1e-8
        endpoint_seconds = result["endpoint_seconds"]
    else:
        components = list(report["components"].values())
        energy_error = max(
            abs(component["energy"] - report["oracle_energies"][method])
            for method, component in report["components"].items()
        )
        assert energy_error < 5e-9
        endpoint_seconds = report["native_complete_endpoint_seconds"]

    comparisons = []
    for difference in report["finite_differences"]:
        if cbs:
            slopes = difference["component_slopes"]
            native = [component["forces"] for component in components]
            pairs = list(zip(native, slopes, strict=True))
            pairs.append((result["forces"], coefficients @ slopes))
        else:
            pairs = [
                (
                    report["components"][difference["method"]]["forces"],
                    difference["oracle_slope"],
                )
            ]
        for force, slope in pairs:
            independent = independent_forces(
                report["atoms_bohr"], difference["direction"], slope
            )
            maximum_error = float(np.max(np.abs(np.asarray(force) - independent)))
            assert maximum_error < 3e-6
            comparisons.append(
                {
                    "step_bohr": difference["step_bohr"],
                    "maximum_cartesian_force_error": maximum_error,
                    "independent_forces": independent.tolist(),
                }
            )
    assert {entry["step_bohr"] for entry in comparisons} == {1e-4, 3e-5}
    memory = report["memory_observation"]
    assert memory["samples"] > 0
    assert memory["sampled_device_process_peak_bytes"] > 0
    return {
        "maximum_component_or_assembled_energy_error": float(energy_error),
        "complete_endpoint_seconds": endpoint_seconds,
        "memory_observation": memory,
        "force_comparisons": comparisons,
    }


if __name__ == "__main__":
    summary = {case: summarize(case) for case in CASES}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for case, result in summary.items():
        force_error = max(
            value["maximum_cartesian_force_error"]
            for value in result["force_comparisons"]
        )
        print(case, result["complete_endpoint_seconds"], force_error)
