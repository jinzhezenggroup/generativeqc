"""Supplemental scientific evidence must not complete a failed timing series."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from test_readme_dft_benchmark import completed_sample

from benchmarks.readme_accuracy_oracle import SCHEMA as ORACLE_SCHEMA
from benchmarks.readme_accuracy_oracle import merge_geometry_oracles
from benchmarks.readme_dft import SCHEMA
from tools.render_readme_dft import collect_point


def receipt() -> dict[str, Any]:
    """Small independent geometry receipt with the same endpoint contract."""
    return {
        "schema": ORACLE_SCHEMA,
        "status": "complete",
        "geometry": 1,
        "protocol": {
            "method": "WB97M-V/RKS",
            "atoms": 3,
            "repeats": 5,
            "force_return": "host_array_in_timed_endpoint",
            "reference_fock_policy": "full-density-rebuild",
            "energy_tolerance": 1e-12,
            "density_tolerance": 1e-10,
            "reference_gradient_tolerance": 1e-10,
            "max_iterations": 100,
        },
        "basis_sha256": "b" * 64,
        "initial_density": "GPU4PySCF default atomic guess; no native density, orbitals, energy or force input",
        "converged": True,
        "energy": -1.0,
        "forces": [[0.0, 0.0, 0.0]] * 3,
        "convergence": {
            "converged": True,
            "iterations": 19,
            "final_residuals": {
                "energy_change_hartree": 1e-13,
                "density_rms": 1e-12,
                "orbital_gradient_norm": 1e-11,
            },
        },
        "source_file_sha256": {"reference.py": "c" * 64},
        "driver_sha256": "d" * 64,
        "scheduler": {"SLURM_JOB_ID": "independent-reference-receipt"},
        "diagnostic_seconds": 1180.0,
    }


@pytest.mark.parametrize(
    "mutation",
    ["protocol", "basis", "force_shape", "energy", "residual", "seed", "duplicate"],
)
def test_invalid_scientific_oracle_cannot_qualify_a_native_sample(
    mutation: str,
) -> None:
    oracle = receipt()
    scientific = deepcopy(oracle["protocol"])
    baselines = {}
    if mutation == "protocol":
        oracle["protocol"]["density_tolerance"] = 1e-4
    elif mutation == "basis":
        oracle["basis_sha256"] = "wrong"
    elif mutation == "force_shape":
        oracle["forces"] = [[0.0, 0.0, 0.0]]
    elif mutation == "energy":
        oracle["energy"] = float("nan")
    elif mutation == "residual":
        oracle["convergence"]["final_residuals"]["orbital_gradient_norm"] = 1e-5
    elif mutation == "seed":
        oracle["initial_density"] = "native density"
    else:
        baselines[1] = {}
    with pytest.raises(ValueError, match="geometry accuracy oracle"):
        merge_geometry_oracles(baselines, [oracle], scientific, basis_sha256="b" * 64)


def test_geometry_oracle_qualifies_only_native_accuracy_and_is_hash_bound(
    tmp_path: Path,
) -> None:
    oracle = receipt()
    directory = tmp_path / "3"
    directory.mkdir()
    reference = {
        "schema": SCHEMA,
        "protocol": oracle["protocol"],
        "status": "running",
        "records": [dict(completed_sample(), energy=-1.0)],
    }
    reference_path = directory / "reference.json"
    reference_path.write_text(json.dumps(reference))
    (directory / "reference.outcome").write_text(json.dumps({"exit_code": 124}))
    oracle_path = directory / "accuracy-reference-moved.json"
    oracle_path.write_text(json.dumps(oracle))
    rows = [
        dict(completed_sample(phase, geometry), energy=-1.0, repeat=repeat)
        for phase, geometry, count in (
            ("cold", 0, 1),
            ("warm", 0, 5),
            ("moved", 1, 1),
            ("moved-warm", 1, 5),
        )
        for repeat in range(count)
    ]
    native = {
        "schema": SCHEMA,
        "protocol": oracle["protocol"],
        "status": "measured",
        "records": rows,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "basis_file_sha256": oracle["basis_sha256"],
        "accuracy_reference_files": [
            {
                "file": oracle_path.name,
                "sha256": hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
            }
        ],
    }
    (directory / "native.json").write_text(json.dumps(native))
    (directory / "native.outcome").write_text(json.dumps({"exit_code": 0}))
    point = collect_point(tmp_path, 3)
    assert point["engines"]["native"]["status"] == "measured"
    assert point["engines"]["reference"]["status"] == "timeout"
    assert len(point["engines"]["reference"]["records"]) == 1
    assert point["engines"]["reference"]["phase_status"]["moved"] == "incomplete"
    assert point["independent_accuracy_receipts"] == [oracle]
    baseline = merge_geometry_oracles(
        {}, [oracle], oracle["protocol"], basis_sha256="b" * 64
    )[1]
    assert "complete_seconds" not in baseline
    oracle_path.write_text(json.dumps(dict(oracle, diagnostic_seconds=1.0)))
    with pytest.raises(ValueError, match="different accuracy-only reference"):
        collect_point(tmp_path, 3)
