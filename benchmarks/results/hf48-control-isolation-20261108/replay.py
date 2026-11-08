"""Recheck one-public-control diagnostics without GPU, NumPy, or an oracle run."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    """Keep evidence checks effective under optimized Python as well."""
    if not condition:
        raise ValueError(message)


def load(directory: Path = ROOT) -> tuple[dict[str, Any], dict[str, str]]:
    """Authenticate the lossless raw-text capsule before parsing observations."""
    cohort = json.loads((directory / "cohort.json").read_text())
    payload = (directory / "observations.json.xz").read_bytes()
    require(
        hashlib.sha256(payload).hexdigest() == cohort["capsule_sha256"],
        "changed observation capsule",
    )
    raw = json.loads(lzma.decompress(payload))
    require(set(raw) == set(cohort["members"]), "changed capsule members")
    for name, digest in cohort["members"].items():
        require(
            hashlib.sha256(raw[name].encode()).hexdigest() == digest,
            "changed observation: " + name,
        )
    return cohort, raw


def observable_error(actual: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Validate full force shape and finiteness before the unchanged E/F gates."""
    require(actual["converged"] and target["converged"], "unconverged endpoint")
    actual_forces, target_forces = actual["forces"], target["forces"]
    require(len(actual_forces) == len(target_forces) == 48, "force atom count")
    differences = []
    for actual_row, target_row in zip(actual_forces, target_forces, strict=True):
        require(len(actual_row) == len(target_row) == 3, "force component count")
        for actual_value, target_value in zip(actual_row, target_row, strict=True):
            require(
                math.isfinite(actual_value) and math.isfinite(target_value),
                "nonfinite force",
            )
            differences.append(abs(actual_value - target_value))
    require(
        math.isfinite(actual["energy"]) and math.isfinite(target["energy"]),
        "nonfinite energy",
    )
    energy = abs(actual["energy"] - target["energy"])
    maximum_force = max(differences)
    return {
        "energy": energy,
        "maximum_force": maximum_force,
        "passed": energy < 1e-8 and maximum_force < 1e-7,
    }


def replay(directory: Path = ROOT) -> dict[str, Any]:
    """Separate convergence-family evidence from the coupled internal gate."""
    cohort, raw = load(directory)
    reference = json.loads(raw["oracle-16815/oracle.json"])
    require(reference["direct_scf"] is False, "reference is not full Fock")
    require(reference["actual_orbital_gradient_norm"] < 1e-10, "reference stationarity")
    summary: dict[str, Any] = {
        "purpose": cohort["purpose"],
        "measurement_revision": cohort["measurement_revision"],
        "current_head_qualification": False,
        "default_promotion_qualified": False,
        "cards": {},
    }
    for card, job in cohort["jobs"].items():
        prefix = f"{card}-{job}"
        require(raw[prefix + "/exit-code.txt"].strip() == "0", "native job failure")
        conditions = {}
        for condition, expected in cohort["conditions"].items():
            record = json.loads(raw[prefix + "/" + condition + ".json"])
            require(
                record["geometry_bohr"] == reference["geometry_bohr"],
                "geometry mismatch",
            )
            require(
                record["explicit_controls"] == expected["explicit_controls"],
                "public controls",
            )
            require(
                record["effective_controls"] == expected["effective_controls"],
                "executed controls",
            )
            require(record["xy_control"] is None, "XY is not automatic")
            require(
                record["native_build"]["source_identity"] == cohort["source_identity"],
                "source identity",
            )
            require(
                record["native_build"]["library_sha256"] == cohort["library_sha256"],
                "ELF identity",
            )
            samples = record["samples"]
            require(len(samples) == 4, "incomplete cold/warm population")
            errors = []
            for ordinal, sample in enumerate(samples):
                require(
                    sample["phase"] == ("cold" if ordinal == 0 else "warm"),
                    "phase order",
                )
                require(
                    sample["iterations"]
                    == (expected["cold_iterations"] if ordinal == 0 else 1),
                    "target work",
                )
                require(sample["fock_builds"] == sample["iterations"], "Fock work")
                require(
                    math.isfinite(sample["seconds"]) and sample["seconds"] > 0,
                    "endpoint time",
                )
                error = observable_error(sample, reference)
                require(
                    error["passed"] is expected["independent_pass"],
                    "changed scientific outcome",
                )
                errors.append(error)
            conditions[condition] = {
                "cold_seconds": samples[0]["seconds"],
                "cold_iterations": samples[0]["iterations"],
                "warm_seconds": [sample["seconds"] for sample in samples[1:]],
                "independent_errors": errors,
            }
        summary["cards"][card] = conditions
    return summary


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
