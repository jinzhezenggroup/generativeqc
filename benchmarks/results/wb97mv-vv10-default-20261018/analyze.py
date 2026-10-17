"""Reduce source-bound default qualification; read references only after execution."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read(path: Path) -> dict[str, Any]:
    """Read an unmodified endpoint, provenance or independent reference receipt."""
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    """Bind a reduction to the bytes actually consumed, including density exports."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_gate(record: dict[str, Any], atoms: int) -> bool:
    """Reject provisional endpoints, tolerance changes, retries and warm densities."""
    history = record["ks_diagnostic"]["history"]
    work = record["ks_diagnostic"]["nonlocal_work"]
    precision = record["precision"]
    forces = np.asarray(record["forces"])
    return bool(
        record["succeeded"] and record["converged"]
        and not record["warm_start_used"] and not record["warm_start_fallback"]
        and record["finite_forces"] and forces.shape == (atoms, 3)
        and np.isfinite(forces).all() and math.isfinite(record["energy"])
        and record["energy_tolerance"] == 1e-12
        and record["density_tolerance"] == 1e-10
        and record["screening_tolerance"] == 1e-12
        and record["grid"] == [48, 16, 32] and record["density_fitting"] == "none"
        and precision["requested_mode"] == "fp64" and precision["effective_bits"] == 64
        and precision["mixed_stage_fock_builds"] == 0
        and precision["fallback_count"] == 0 and precision["execution_retries"] == 0
        and abs(record["energy_change"]) < 1e-12 and record["density_rms"] < 1e-10
        and record["physical_residual_rms"] < 1e-10
        and record["ks_diagnostic"]["physical_residual_max"] < 1e-10
        and record["ks_diagnostic"]["density_change_max"] < 1e-10
        and record["initial_guess"]["target_attempts"] == 1
        and len(history) >= 2
        and all(row["operator_scope"] == "full-target" for row in history[-2:])
        and len(history) == record["iterations"] <= 100
        and record["fock_builds"] >= record["iterations"]
        and work["provisional_focks"] + work["target_focks"] == record["iterations"]
    )


def main() -> None:
    """Require all declared endpoints and acceptance gates before publishing evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root, historical = arguments.root, arguments.historical
    job = root / f"job{arguments.job_id}"
    assert (root / "srun-exit-code.txt").read_text().strip() == "0"
    authentication = read(root / "build-receipts/source-authentication.json")
    source = root / "source"
    specification = importlib.util.spec_from_file_location(
        "default_source_identity", source / "tools/generate_build_identity.py"
    )
    assert specification is not None and specification.loader is not None
    identity_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(identity_module)
    identity = identity_module._source_identity(source, identity_module._inventory(
        source, source / "cmake/GenerativeQCSourceIdentity.json"
    ))
    assert identity == authentication["native_identity"]
    master_source = historical / "source-master"
    inventory = {}
    for flavor, tree in (("master", master_source), ("default", source)):
        inventory[flavor] = {
            path.as_posix(): digest(tree / path) for path in identity_module._inventory(
                tree, tree / "cmake/GenerativeQCSourceIdentity.json"
            )
        }
    changed_inputs = sorted(path for path in inventory["master"].keys() | inventory["default"].keys()
                            if inventory["master"].get(path) != inventory["default"].get(path))
    assert changed_inputs == ["src/dft/application_precision.hpp"]
    assert (root / "restore-exit-code.txt").read_text().strip() == "0"
    expected_library = (root / "build-receipts/library.sha256").read_text().split()[0]
    assert digest(root / "library/libgenerativeqc.so") == expected_library
    assert (job / "library-after.sha256").read_text().split()[0] == expected_library
    assert (job / "visibility.txt").read_text().strip()
    assert "passed" in (job / "pytest.log").read_text()
    assert (root / "expanded-tests-exit-code.txt").read_text().strip() == "0"
    for line in (job / "input-hashes.txt").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        local = root / "library/libgenerativeqc.so" if name.endswith(".so") else root / Path(name).name
        assert digest(local) == expected
    reference = read(historical / "reference12.json")
    oracle = next(row for row in reference["records"] if row["phase"] == "cold")
    output = {
        "schema": "generativeqc.wb97mv-frozen-inner-default.v1",
        "base_revision": authentication["base_revision"],
        "native_source_identity": authentication["native_identity"],
        "native_input_inventory_size": len(inventory["default"]),
        "native_input_changes_from_master": changed_inputs,
        "library_sha256": expected_library,
        "slurm_job_id": arguments.job_id,
        "production_defaults_changed": True,
        "experiment_selector_present": False,
        "reference12_sha256": digest(historical / "reference12.json"),
        "reference_binding": reference["historical_binding"],
        "water96_independent_reference_available": False,
        "timing_scope": "Default-source validation observations, not a new matched speedup campaign.",
        "records": [],
        "receipt_hashes": {},
        "validation": {
            "local_focused_tests": "82 passed, 8 GPU tests deferred to Slurm",
            "device_schedule_tests": (job / "pytest.log").read_text().strip(),
            "expanded_tests": (root / "expanded-tests.log").read_text().strip(),
            "expanded_tests_source": "n1 actual default library, Slurm job 10075",
        },
    }
    for atoms, baseline_name in ((12, "screen-job1120/baseline/water12.json"),
                                  (96, "endpoint-baseline-job1121/water96.json")):
        path = job / f"water{atoms}/result.json"
        record = read(path)
        baseline_path = historical / baseline_name
        baseline = read(baseline_path)
        assert physical_gate(record, atoms) and physical_gate(baseline, atoms)
        assert record["production_defaults_changed"]
        assert record["experimental_proposal_policy"] is None
        assert record["native_source_identity"] == authentication["native_identity"]
        assert record["native_build"]["probe"]["source_identity"] == authentication["native_identity"]
        assert record["library_sha256"] == expected_library
        for key in ("basis_identity", "basis_sha256", "coordinates_bohr", "grid",
                    "density_fitting", "energy_tolerance", "density_tolerance", "screening_tolerance"):
            assert record[key] == baseline[key], key
        density_path = path.with_suffix(".density.npy")
        baseline_density = baseline_path.with_suffix(".density.npy")
        assert digest(density_path) == record["final_density"]["sha256"]
        assert digest(baseline_density) == baseline["final_density"]["sha256"]
        density = np.load(density_path)
        assert np.isfinite(density).all()
        row = {key: record[key] for key in (
            "iterations", "fock_builds", "energy", "energy_change", "density_rms",
            "physical_residual_rms", "prepare_seconds", "execute_seconds", "cold_endpoint_seconds",
            "initial_guess", "native_build",
        )}
        row.update(
            atoms=atoms, physical_gate=True, raw_path=str(path.relative_to(root)),
            raw_sha256=digest(path), density_sha256=digest(density_path),
            baseline_raw_path=baseline_name, baseline_raw_sha256=digest(baseline_path),
            baseline_density_sha256=digest(baseline_density),
            nonlocal_work=record["ks_diagnostic"]["nonlocal_work"],
            additional_endpoint_fock_builds=record["fock_builds"] - record["iterations"],
            paired_energy_error=abs(record["energy"] - baseline["energy"]),
            paired_force_error=float(np.max(np.abs(
                np.asarray(record["forces"]) - np.asarray(baseline["forces"])))),
            paired_density_error=float(np.max(np.abs(density - np.load(baseline_density)))),
            phases=[], oracle_gate=None,
        )
        row["paired_gate"] = (row["paired_energy_error"] <= 1e-8
                              and row["paired_force_error"] <= 1e-7
                              and row["paired_density_error"] <= 1e-7)
        assert row["paired_gate"]
        for entry in record["ks_diagnostic"]["history"]:
            scope = entry["operator_scope"]
            if not row["phases"] or row["phases"][-1]["scope"] != scope:
                row["phases"].append({"scope": scope, "steps": 0})
            row["phases"][-1]["steps"] += 1
        if atoms == 12:
            assert record["basis_identity"] == reference["protocol"]["basis_identity"]
            assert record["coordinates_bohr"] == [
                atom[1] for atom in reference["protocol"]["geometries_bohr"][0]
            ]
            row["oracle_energy_error"] = abs(record["energy"] - oracle["energy"])
            row["oracle_force_error"] = float(np.max(np.abs(
                np.asarray(record["forces"]) - np.asarray(oracle["forces"]))))
            row["oracle_gate"] = row["oracle_energy_error"] <= 1e-8 and row["oracle_force_error"] <= 1e-7
            assert row["oracle_gate"]
        output["records"].append(row)
    for receipt in sorted((root / "build-receipts").glob("*")) + sorted(job.glob("*")):
        if receipt.is_file():
            output["receipt_hashes"][str(receipt.relative_to(root))] = digest(receipt)
    for receipt in sorted(root.glob("expanded-tests*")):
        output["receipt_hashes"][receipt.name] = digest(receipt)
    output["receipt_hashes"]["restore-exit-code.txt"] = digest(root / "restore-exit-code.txt")
    arguments.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    for row in output["records"]:
        print(json.dumps({key: row.get(key) for key in (
            "atoms", "iterations", "fock_builds", "nonlocal_work", "physical_gate",
            "paired_gate", "oracle_gate", "paired_energy_error", "paired_force_error",
            "paired_density_error", "cold_endpoint_seconds",
        )}))


if __name__ == "__main__":
    main()
