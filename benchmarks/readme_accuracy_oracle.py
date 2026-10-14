"""Accept separate reconverged geometry oracles without inventing timing samples.

A reference population may time out after its cold geometry. A fresh independent
solve can still validate native forces at the moved geometry. These receipts
remain accuracy evidence only; they never complete the reference timing series.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA = "generativeqc.readme-independent-geometry-oracle.v1"


def read_geometry_oracle(
    path: Path, *, directory: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    """Parse and hash one retained byte snapshot, rejecting directory escapes."""
    resolved = path.resolve()
    if resolved.parent != directory.resolve():
        raise ValueError(
            "accuracy references must be retained beside the native output"
        )
    payload = resolved.read_bytes()
    return json.loads(payload), {
        "file": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def merge_geometry_oracles(
    baselines: dict[int, dict[str, Any]],
    receipts: list[dict[str, Any]],
    scientific: dict[str, Any],
    *,
    basis_sha256: str | None,
) -> dict[int, dict[str, Any]]:
    """Validate receipt identity, independent convergence and full E/F arrays.

    Existing timing-population baselines cannot be replaced. The exact protocol
    includes geometry, Hamiltonian, grid, density policy and stopping tolerances.
    A receipt contributes no cold/warm timing, initialization cost or speedup.
    """
    merged = dict(baselines)
    for receipt in receipts:
        try:
            geometry = receipt["geometry"]
            convergence = receipt["convergence"]
            residuals = convergence["final_residuals"]
            forces = np.asarray(receipt["forces"], dtype=float)
            valid = (
                receipt["schema"] == SCHEMA
                and receipt["status"] == "complete"
                and receipt["protocol"] == scientific
                and scientific["method"] == "WB97M-V/RKS"
                and scientific["reference_fock_policy"] == "full-density-rebuild"
                and type(geometry) is int
                and geometry in (0, 1)
                and geometry not in merged
                and _sha256(basis_sha256)
                and receipt["basis_sha256"] == basis_sha256
                and receipt["initial_density"]
                == "GPU4PySCF default atomic guess; no native density, orbitals, energy or force input"
                and receipt["converged"] is True
                and convergence["converged"] is True
                and type(convergence["iterations"]) is int
                and 0 < convergence["iterations"] <= scientific["max_iterations"]
                and forces.shape == (scientific["atoms"], 3)
                and np.isfinite(forces).all()
                and math.isfinite(float(receipt["energy"]))
                and abs(residuals["energy_change_hartree"])
                <= scientific["energy_tolerance"]
                and 0 <= residuals["density_rms"] <= scientific["density_tolerance"]
                and 0
                <= residuals["orbital_gradient_norm"]
                <= scientific["reference_gradient_tolerance"]
                and isinstance(receipt["source_file_sha256"], dict)
                and bool(receipt["source_file_sha256"])
                and all(
                    isinstance(path, str) and bool(path) and _sha256(sha)
                    for path, sha in receipt["source_file_sha256"].items()
                )
                and _sha256(receipt["driver_sha256"])
                and bool(receipt["scheduler"]["SLURM_JOB_ID"])
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ValueError(
                "invalid or conflicting independent geometry accuracy oracle"
            )
        merged[geometry] = {
            "geometry": geometry,
            "energy": float(receipt["energy"]),
            "forces": forces.tolist(),
            "converged": True,
            "status": 0,
            "purpose": "accuracy-only independent geometry; no timing sample",
        }
    return merged
