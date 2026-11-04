"""Replay retained independent and paired native E+F gates without a GPU.

This checks scientific observations and their committed input identities, not
the presence of a local 965-MB library or an inference from successful build logs.
It does not rerun native calculations or qualify a different source revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import subprocess
from pathlib import Path


def force_errors(actual: list, expected: list) -> tuple[float, float]:
    """Check the fixture shape and recompute max/RMS from every retained value."""
    if len(actual) != 3 or len(expected) != 3:
        raise ValueError("expected a three-atom fixture")
    differences = []
    for force, reference in zip(actual, expected, strict=True):
        if len(force) != 3 or len(reference) != 3:
            raise ValueError("expected xyz forces")
        for component, target in zip(force, reference, strict=True):
            if not math.isfinite(component) or not math.isfinite(target):
                raise ValueError("nonfinite retained force")
            differences.append(component - target)
    return max(map(abs, differences)), math.sqrt(
        sum(value**2 for value in differences) / len(differences)
    )


def main() -> None:
    """Reject incomplete pair sets, altered scientific gates and stale inputs."""
    capsule = Path(__file__).resolve().parents[1]
    root = capsule.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt", type=Path, default=capsule / "native-receipt.json.xz"
    )
    path = parser.parse_args().receipt
    payload = path.read_bytes()
    receipt = json.loads(lzma.decompress(payload) if path.suffix == ".xz" else payload)
    if receipt["gates"] != {"energy_eh": 2e-9, "force_eh_per_bohr": 1e-7}:
        raise ValueError("native acceptance gates changed")
    source = receipt["source_commit"]
    for name, expected in receipt["source_files_sha256"].items():
        content = subprocess.check_output(
            ["git", "show", f"{source}:{name}"], cwd=root, timeout=30
        )
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"historical source input changed: {name}")
    references = {
        (row["spin"], row["geometry"]): row for row in receipt["independent_oracles"]
    }
    observations = receipt["observations"]
    expected_keys = {
        (spin, repeat, selector, phase)
        for spin in ("rks", "uks")
        for repeat in range(10)
        for selector in ("0", "1")
        for phase in ("cold", "warm", "moved", "moved_warm")
    }
    measured = {}
    maximum_energy_error = maximum_force_error = 0.0
    for row in observations:
        key = row["spin"], row["repeat"], row["selector"], row["phase"]
        if key in measured:
            raise ValueError(f"duplicate endpoint observation: {key}")
        reference = references[row["spin"], int(row["phase"].startswith("moved"))]
        energy_error = abs(row["energy"] - reference["energy"])
        maximum, _ = force_errors(row["forces"], reference["forces"])
        if not math.isfinite(energy_error) or energy_error > 2e-9 or maximum > 1e-7:
            raise ValueError(f"independent native oracle gate failed: {key}")
        work = row["coulomb_exchange_work"]
        if (
            not work["complete"]
            or work["exchange_applications"] != 0
            or work["coulomb_applications"] != row["fock_builds"]
        ):
            raise ValueError(f"incomplete semilocal operator work: {key}")
        measured[key] = row
        maximum_energy_error = max(maximum_energy_error, energy_error)
        maximum_force_error = max(maximum_force_error, maximum)
    if set(measured) != expected_keys:
        raise ValueError("the complete interleaved endpoint matrix was not retained")
    for (spin, repeat, selector, phase), row in measured.items():
        other = measured[spin, repeat, "1" if selector == "0" else "0", phase]
        maximum, _ = force_errors(row["forces"], other["forces"])
        if (
            abs(row["energy"] - other["energy"]) > 2e-9
            or maximum > 1e-7
            or row["iterations"] != other["iterations"]
            or row["fock_builds"] != other["fock_builds"]
        ):
            raise ValueError("paired endpoint scientific/work gate failed")
    print(
        f"{source}: {len(measured)} native observations passed; "
        f"max energy error={maximum_energy_error:.6g} Eh; "
        f"max force error={maximum_force_error:.6g} Eh/Bohr"
    )


if __name__ == "__main__":
    main()
