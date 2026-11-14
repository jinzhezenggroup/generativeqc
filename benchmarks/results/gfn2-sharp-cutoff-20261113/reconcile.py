"""Recheck frozen GPU observations against explicitly matched tblite references.

This is an opt-in historical-evidence reducer, not a current-device benchmark.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GATE = 5e-7


def load(name: str) -> dict:
    return json.loads(lzma.decompress((ROOT / name).read_bytes()))


def reconcile() -> dict:
    """Preserve input identity and unchanged energy/force acceptance gates."""
    provenance = json.loads((ROOT / "provenance.json").read_text())
    for name, identity in provenance["archive_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != identity:
            raise ValueError("archived evidence identity mismatch: " + name)
    reference = load("matched-oracle.json.xz")
    if reference["xtb_config"] != {"smooth_cutoff": 0.0}:
        raise ValueError("oracle does not select the native sharp-cutoff model")
    inputs = lzma.decompress((ROOT / "inputs.json.xz").read_bytes())
    identity = hashlib.sha256(inputs).hexdigest()
    if reference["input_sha256"] != identity:
        raise ValueError("oracle input identity mismatch")
    references = {(row["case"], row["geometry"]): row for row in reference["rows"]}
    observations = []
    libraries = {}
    for name in (
        "native-energy",
        "native-force",
        "xtbloom-force",
        "native-fixtures",
        "replay-native-energy",
        "replay-native-force",
    ):
        report = load(name + ".json.xz")
        if report["status"] != "measured" or report["input_sha256"] != identity:
            raise ValueError("incomplete or input-mismatched native receipt")
        libraries[name] = report["library_sha256"]
        for row in report["rows"]:
            expected = references[row["case"], row["geometry"]]
            if (
                row["geometry_sha256"] != expected["geometry_sha256"]
                or row["converged"] is not True
                or row["executed_backend"] != "cuda"
            ):
                raise ValueError("geometry or execution contract mismatch")
            energy_error = abs(row["energy"] - expected["energy"])
            if not math.isfinite(energy_error):
                raise ValueError("nonfinite energy in scientific evidence")
            force_error = None
            if row["forces"] is not None:
                actual = [value for atom in row["forces"] for value in atom]
                wanted = [value for atom in expected["forces"] for value in atom]
                if not all(math.isfinite(value) for value in (*actual, *wanted)):
                    raise ValueError("nonfinite force in scientific evidence")
                force_error = max(
                    abs(value - target)
                    for value, target in zip(actual, wanted, strict=True)
                )
            observations.append(
                {
                    "report": name,
                    "case": row["case"],
                    "geometry": row["geometry"],
                    "phase": row["phase"],
                    "repeat": row["repeat"],
                    "energy_error": energy_error,
                    "force_error": force_error,
                    "pass": energy_error < GATE
                    and (force_error is None or force_error < GATE),
                }
            )
    historical = [
        row for row in observations if not row["report"].startswith("replay-")
    ]
    replay = [row for row in observations if row["report"].startswith("replay-")]
    large = [row for row in historical if row["case"] == "water-128"]
    if len(historical) != 136 or len(large) != 36 or len(replay) != 72:
        raise ValueError("incomplete historical campaign")
    return {
        "scope": "historical observations and fresh replay of the same frozen binary",
        "production_commit": "dc68eb0b2b496de18fd765bda577d836487b2883",
        "xtb_config": reference["xtb_config"],
        "gate": GATE,
        "input_sha256": identity,
        "library_sha256": libraries,
        "observations": len(observations),
        "passing": sum(row["pass"] for row in observations),
        "historical_observations": len(historical),
        "replay_observations": len(replay),
        "replay_passing": sum(row["pass"] for row in replay),
        "large_observations": len(large),
        "large_passing": sum(row["pass"] for row in large),
        "maximum_energy_error": max(row["energy_error"] for row in observations),
        "maximum_force_error": max(row["force_error"] or 0.0 for row in observations),
        "rows": observations,
    }


if __name__ == "__main__":
    result = reconcile()
    print(json.dumps(result, indent=2))
    raise SystemExit(result["passing"] != result["observations"])
