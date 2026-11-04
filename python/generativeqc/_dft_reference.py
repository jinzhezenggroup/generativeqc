"""Explicit, model-bound reuse of independently produced DFT observations."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from .profiles import canonical_hash, file_hash

SCHEMA = "generativeqc.dft-reference.v1"


def model(workload: dict) -> dict:
    """Resolve scientific inputs without SCF, gradients, or a CUDA context.

    Hardware and native scheduling are deliberately absent: oracle observations
    are reusable scientific evidence, never transferable performance evidence.
    Both reference Gaussian contents and the native catalog invalidate reuse.
    """
    from pyscf import gto

    from ._dft_autotune import DFT_FUNCTIONALS

    molecule = gto.M(
        atom=workload["atoms"],
        basis=workload["basis"],
        unit="Bohr",
        charge=workload["charge"],
        spin=workload["multiplicity"] - 1,
        cart=workload["representation"] == "cartesian",
        verbose=0,
    )
    return {
        "atoms_bohr": copy.deepcopy(workload["atoms"]),
        "basis": workload["basis"],
        "reference_gaussian_hash": canonical_hash(
            {"basis": molecule._basis, "ecp": molecule._ecp, "pseudo": molecule._pseudo}
        ),
        "native_basis_catalog_hash": file_hash(
            Path(__file__).parent / "data/basis_pack.json"
        ),
        "method": workload["method"],
        "functional": DFT_FUNCTIONALS[workload["method"]],
        "charge": workload["charge"],
        "multiplicity": workload["multiplicity"],
        "representation": workload["representation"],
        "batch": workload["batch"],
        "grid": {
            "level": workload["grid_level"],
            "radial": "treutler_ahlrichs",
            "angular_pruning": "nwchem_prune",
            "radii_adjust": "treutler_atomic_radii_adjust",
            "becke": "original_becke",
        },
        "small_rho_cutoff": 0.0,
        "force_grid_response": True,
    }


def load(path: Path, expected_sha256: str, workload: dict) -> dict:
    """Refuse corrupt, mismatched, incomplete or unconverged cached references.

    The caller explicitly pins the entire document, including original producer
    identity and numerical settings. No historical timing becomes a native
    endpoint sample. Every retained independent observation remains a gate.
    """
    from ._dft_autotune import numerical_gate

    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("independent reference file SHA256 differs")
    document = json.loads(payload)
    expected = model(workload)
    if (
        not isinstance(document, dict)
        or document.get("schema") != SCHEMA
        or document.get("model") != expected
    ):
        raise ValueError("independent reference scientific model differs")
    producer = document.get("producer", {})
    if (
        not isinstance(producer, dict)
        or producer.get("engine") not in ("pyscf", "gpu4pyscf")
        or not isinstance(producer.get("versions"), dict)
        or not producer["versions"]
        or not isinstance(producer.get("source_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", producer["source_sha256"])
    ):
        raise ValueError("independent reference requires original producer provenance")
    observations = document.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("independent reference observations are missing")
    for observation in observations:
        if not isinstance(observation, dict):
            raise TypeError("independent reference observation must be an object")
        settings = observation.get("settings", {})
        if (
            not isinstance(settings, dict)
            or observation.get("converged") is not True
            or not observation.get("independent_reference")
            or "observations" in observation
            or settings.get("force_grid_response") is not True
            or settings.get("small_rho_cutoff") != expected["small_rho_cutoff"]
            or not isinstance(settings.get("xc_grid"), dict)
            or any(
                settings["xc_grid"].get(key) != value
                for key, value in expected["grid"].items()
            )
        ):
            raise ValueError(
                "independent reference lacks converged model-matched observations"
            )
        try:
            numerical_gate([observation], observation)
        except (KeyError, TypeError) as error:
            raise ValueError(
                "independent reference energy/force data is incomplete"
            ) from error
        if len(observation["energies"]) != workload["batch"] or any(
            len(force) != len(workload["atoms"]) for force in observation["forces"]
        ):
            raise ValueError("independent reference geometry/batch layout differs")
    return {
        "independent_reference": "reused independent DFT energy and analytic gradient with grid response",
        "observations": observations,
        "provenance": {
            "file": str(path.resolve()),
            "sha256": expected_sha256,
            "producer": producer,
        },
    }
