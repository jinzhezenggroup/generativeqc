"""Replay installed force-test observations without expanding catalog admission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks._exact_fp64_evidence import require
from benchmarks._native_dft_force_evidence import load
from benchmarks._retention import raw_output_path


def bounded(value: Any, gate: float, detail: str) -> None:
    """Check every nonnegative error and reject empty or nonfinite inputs."""
    array = np.asarray(value, dtype=float)
    require(
        array.size > 0
        and bool(np.all(np.isfinite(array)))
        and bool(np.all(array >= 0))
        and bool(np.all(array <= gate)),
        detail,
    )


def validate_automatic(record: dict, index: dict) -> int:
    """Keep one-phase vectors and directional lifecycles in their actual scopes."""
    require(
        record["schema"] == "generativeqc.native-automatic-cuda-forces.v1"
        and record["slurm_job_id"] == index["slurm_job_id"]
        and record["native_library_sha256"] == index["installed_library_sha256"],
        "automatic observation identity mismatch",
    )
    context = record["context"]
    require(
        context["precision"] == "fp64"
        and context["integrals"] == "direct"
        and context["ecp"] is False
        and context["basis"] == "sto-3g"
        and context["grid"] == [24, 8, 16]
        and context["partition_iterations"] == 3
        and context["tile_points"] == 37
        and type(context["full_vector"]) is bool
        and context["reference"] == "PySCF-2.14.0/Libxc-7.0.0",
        "automatic observation context mismatch",
    )
    # The water probe deliberately includes a two-channel singlet. Spin-block
    # count is an execution layout, not the molecule's spin multiplicity.
    expected_charge, expected_multiplicity = (
        (record["spins"] - 1, record["spins"]) if context["fixture"] == "h4" else (0, 1)
    )
    require(
        type(record["spins"]) is int
        and record["spins"] in (1, 2)
        and record["charge"] == expected_charge
        and record["multiplicity"] == expected_multiplicity,
        "automatic observation spin mismatch",
    )
    atom_count = {"h4": 4, "water": 3}[context["fixture"]]
    rows = record["rows"]
    phases = [row["phase"] for row in rows]
    require(
        phases in (["cold"], ["cold", "warm", "moved"]),
        "incomplete automatic observation lifecycle",
    )
    if len(rows) == 3:
        require(
            rows[0]["coordinates"] == rows[1]["coordinates"]
            and rows[0]["coordinates"] != rows[2]["coordinates"],
            "automatic observation did not move geometry",
        )
    for row in rows:
        require(
            row["converged"] is True and row["executed_backend"] == "cuda",
            "automatic observation did not execute converged CUDA forces",
        )
        bounded(abs(row["energy"] - row["reference_energy"]), 3e-9, "SCF energy")
        bounded(
            abs(row["energy"] - row["fixed_density_energy"]),
            3e-10,
            "fixed-density energy",
        )
        for field, gate in (
            ("density_error", 2e-7),
            ("fock_error", 3e-9),
            ("directional_error", 5e-8),
        ):
            bounded(row[field], gate, field)
        forces = np.asarray(row["forces"], dtype=float)
        require(forces.shape == (atom_count, 3), "automatic force shape")
        bounded(np.abs(forces.sum(axis=0)), 2e-8, "automatic translation")
        analytic = np.asarray(row["analytic_directions"], dtype=float)
        require(analytic.ndim == 1 and analytic.size > 0, "missing force directions")
        if context["full_vector"]:
            require(analytic.size == forces.size, "incomplete Cartesian force oracle")
            bounded(np.abs(analytic - forces.ravel()), 3e-5, "Cartesian analytic force")
        for field in ("native_fd", "independent_fd"):
            values = np.asarray(row[field], dtype=float)
            require(values.shape == analytic.shape, "finite-difference coverage")
            bounded(np.abs(values - analytic), 3e-5, field)
        artifact = row["artifact"]
        require(
            artifact["schema"] == "generativeqc.native-automatic-artifact.v2"
            and artifact["identifier"] == record["name"]
            and artifact["spin_blocks"] == record["spins"]
            and artifact["force_completed"] is True
            and artifact["source_count"] == 7
            and artifact["force_schedule"] == "shared-staged"
            and artifact["fallback_reason"] == "none",
            "incomplete native force ownership",
        )
        owners = [
            artifact[field]
            for field in (
                "point_artifact",
                "force_artifact",
                "point_executable_artifact",
                "force_executable_artifact",
            )
        ]
        require(
            all(
                owner["sha256"] == index["installed_library_sha256"]
                and owner["bytes"] == owners[0]["bytes"] > 0
                for owner in owners
            ),
            "native callback owner mismatch",
        )
        require(
            np.isfinite(row["complete_endpoint_seconds"])
            and row["complete_endpoint_seconds"] >= row["endpoint_seconds"] > 0,
            "invalid complete endpoint duration",
        )
    return len(rows)


def validate_r2scan(name: str, record: dict, index: dict) -> None:
    """Replay the original correction, independent-oracle and resource gates."""
    require(record["allocation"] == index["slurm_job_id"], "r2SCAN allocation")
    if name.startswith("composition-"):
        require(record["dispersion_backend"] == "cuda", "r2SCAN correction backend")
        bounded(record["energy_composition_error"], 2e-9, "correction energy")
        bounded(record["force_composition_max_error"], 2e-7, "correction force")
    elif name == "water-total-force-directional-fd.json":
        bounded(record["step_convergence"], 5e-6, "r2SCAN FD step convergence")
        bounded(record["analytic_error"], 5e-6, "r2SCAN directional force")
    elif name == "water-spd-resource-work.json":
        require(
            record["full_range_derivative_route"] == "prepared-native-stationary"
            and record["full_range_ao_task_domain_elided"] is True
            and record["primitive_record_page_budget"] == 1
            and record["primitive_page_peak_records"] == 0
            and record["primitive_pages"] == 0
            and bool(record["stationary_native_integral_sources"])
            and record["additional_device_peak_bound"]
            <= record["additional_device_budget"]
            and record["rejected_device_budget_bytes"] == 1,
            "r2SCAN bounded native resource ownership",
        )
    elif name == "ragged-force-changed-geometry.json":
        require(
            len(record["energy_errors"])
            == len(record["force_max_errors"])
            == len(record["atom_counts"])
            == 2,
            "incomplete ragged force observations",
        )
        bounded(record["energy_errors"], 2e-9, "ragged energy")
        bounded(record["force_max_errors"], 2e-7, "ragged force")
    elif name == "charged-ragged-failure-isolation.json":
        require(
            record["first_statuses"] == [0, 0]
            and len(record["isolated_statuses"]) == 2
            and record["isolated_statuses"][0] != 0
            and record["isolated_statuses"][1] == 0,
            "ragged force failure isolation",
        )
        bounded(record["surviving_energy_error"], 2e-9, "isolated energy")
        bounded(record["surviving_force_max_error"], 2e-7, "isolated force")
    elif name.endswith("-production-grid-independent-oracle.json"):
        for field, gate in (
            ("energy_error", 2e-9),
            ("force_max_error", 1e-7),
            ("d4_energy_error", 2e-9),
            ("gcp_energy_error", 2e-9),
        ):
            bounded(record[field], gate, field)
    elif name == "canonical-3atom-r2scan3c-df-independent-force.json":
        for field, gate in (
            ("max_energy_error", 1e-8),
            ("max_force_error", 1e-7),
            ("translation_error", 2e-8),
            ("directional_fd_error", 5e-6),
        ):
            bounded(record[field], gate, field)
        for entries in record["errors"].values():
            require(bool(entries), "missing independent DF reference rows")
            for entry in entries:
                bounded(entry["energy"], 1e-8, "independent DF energy")
                bounded(entry["force"], 1e-7, "independent DF force")
    else:
        raise ValueError("unknown retained r2SCAN scientific observation: " + name)


def replay(directory: Path) -> dict:
    """Authenticate every original receipt and replay only its measured scope."""
    documents = load(directory)[1]
    index = json.loads(documents["scientific/index.json"])
    require(
        index["schema"] == "generativeqc.r1-installed-force-observations.v1",
        "scientific index schema",
    )
    automatic, r2scan, row_count, functionals = [], [], 0, set()
    for name, metadata in index["files"].items():
        text = documents[name]
        require(
            len(text.encode()) == metadata["bytes"]
            and hashlib.sha256(text.encode()).hexdigest() == metadata["sha256"],
            "original scientific receipt bytes changed",
        )
        record = json.loads(text)
        if name.startswith("scientific/automatic-forces/"):
            row_count += validate_automatic(record, index)
            automatic.append(name)
            functionals.add(record["name"])
        elif name.startswith("scientific/r2scan3c-forces/"):
            validate_r2scan(Path(name).name, record, index)
            r2scan.append(name)
        else:
            raise ValueError("unknown scientific observation family")
    require(
        len(automatic) == 34 and len(r2scan) == 11 and len(functionals) == 11,
        "incomplete installed scientific observation inventory",
    )
    return {
        "status": "PASS",
        "automatic_records": len(automatic),
        "automatic_rows": row_count,
        "automatic_functionals": len(functionals),
        "r2scan3c_records": len(r2scan),
        "installed_library_sha256": index["installed_library_sha256"],
        "slurm_job_id": index["slurm_job_id"],
        "scope": "Original installed scientific test gates, not endpoint-catalog promotion or paired performance qualification.",
    }


def main() -> None:
    """Keep replay offline and write its summary outside retained result inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=raw_output_path, required=True)
    arguments = parser.parse_args()
    result = replay(arguments.directory)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
