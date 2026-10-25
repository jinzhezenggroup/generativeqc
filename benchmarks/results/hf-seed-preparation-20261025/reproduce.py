"""Replay every retained complete endpoint and independent numerical gate."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name: str) -> object:
    return json.loads(gzip.decompress((ROOT / name).read_bytes()))


def check(sample: dict, reference: dict, atoms: int) -> dict:
    assert sample["converged"]
    energy = abs(sample["energy"] - reference["energy"]) / atoms
    assert len(sample["forces"]) == len(reference["forces"]) == atoms
    delta = [
        a - b
        for x, y in zip(sample["forces"], reference["forces"], strict=True)
        for a, b in zip(x, y, strict=True)
    ]
    maximum = max(map(abs, delta))
    rms = math.sqrt(sum(x * x for x in delta) / len(delta))
    assert energy < 1e-5 and maximum < 1e-4 and rms < 3e-5
    return {"energy_per_atom": energy, "force_max": maximum, "force_rms": rms}


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    records = read("moved.json.gz")
    references = {
        (x["atoms"], x["geometry"]): x["sample"]
        for x in records
        if x["kind"] == "reference"
    }
    moved = [x for x in records if x["kind"] == "native"]
    assert len(moved) == 8
    checks = [
        check(x, references[x["atoms"], x["geometry"]], x["atoms"]) for x in moved
    ]
    paired = read("paired.json.gz")
    populations = [x for x in paired.values() if isinstance(x, dict) and "samples" in x]
    assert len(populations) == 12
    summary = []
    for atoms in (48, 96):
        for variant in ("baseline", "candidate"):
            selected = [
                x
                for name, x in paired.items()
                if name.endswith(f"hf-{atoms}-{variant}.json")
            ]
            assert len(selected) == 3
            cold, warm, seed = [], [], []
            for population in selected:
                assert population["status"] == "complete"
                assert len(population["samples"]) == 3
                for sample in population["samples"]:
                    checks.append(check(sample, references[atoms, "original"], atoms))
                    (cold if sample["phase"] == "cold" else warm).append(
                        sample["seconds"]
                    )
                    if sample["phase"] == "cold":
                        assert sample["initial_guess"]["outcome"] == "used"
                        assert sample["iterations"] == (10 if atoms == 48 else 11)
                        seed.append(sample["initial_guess"]["preparation_seconds"])
            summary.append(
                {
                    "atoms": atoms,
                    "variant": variant,
                    "cold": statistics.median(cold),
                    "warm": statistics.median(warm),
                    "seed": statistics.median(seed),
                }
            )
    print(
        json.dumps(
            {
                "complete_energy_force_checks": len(checks),
                "results": summary,
                "maximum_force_error": max(x["force_max"] for x in checks),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
