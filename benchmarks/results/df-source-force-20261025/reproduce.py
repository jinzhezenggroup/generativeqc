"""Replay retained DF endpoint gates and paired timing populations without a GPU."""

from __future__ import annotations

import gzip
import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = (("pbe", 48), ("pbe", 96), ("r2scan-3c", 96))


def read(name: str) -> object:
    """Read lossless XZ numerical receipts; gzip remains for frozen scripts."""
    decompress = lzma.decompress if name.endswith(".xz") else gzip.decompress
    return json.loads(decompress((ROOT / name).read_bytes()))


def check(sample: dict, reference: dict, atoms: int) -> dict:
    assert sample["converged"]
    energy = abs(sample["energy"] - reference["energy"]) / atoms
    assert len(sample["forces"]) == len(reference["forces"]) == atoms
    delta = [
        a - b
        for x, y in zip(sample["forces"], reference["forces"], strict=True)
        for a, b in zip(x, y, strict=True)
    ]
    assert all(math.isfinite(value) for value in delta)
    maximum = max(map(abs, delta))
    rms = math.sqrt(sum(x * x for x in delta) / len(delta))
    assert energy < 1e-5 and maximum < 1e-4 and rms < 3e-5
    return {"energy_per_atom": energy, "force_max": maximum, "force_rms": rms}


def populations(records: dict) -> list[dict]:
    return [
        item
        for item in records.values()
        if isinstance(item, dict) and "samples" in item
    ]


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    provenance = json.loads((ROOT / "provenance.json").read_text())
    assert (
        hashlib.sha256(
            gzip.decompress((ROOT / "measured-change.patch.gz").read_bytes())
        ).hexdigest()
        == provenance["measured_change_sha256"]
    )
    moved = read("moved.json.xz")
    references = {
        (item["method"], item["atoms"], item["geometry"]): item["sample"]
        for item in moved
        if item["kind"] == "reference"
    }
    native = [item for item in moved if item["kind"] == "native"]
    assert len(native) == 12
    checks = [
        check(
            item,
            references[item["method"], item["atoms"], item["geometry"]],
            item["atoms"],
        )
        for item in native
    ]
    paired = read("paired.json.xz")
    assert len(populations(paired)) == 27
    assert all(item["returncode"] == 0 for item in paired["commands.json"])
    ablations = read("ablations.json.xz")
    retained = populations(paired) + [
        item for records in ablations.values() for item in populations(records)
    ]
    assert len(retained) == 39
    for population in retained:
        assert population["status"] == "complete"
        assert len(population["samples"]) == 3
        method, atoms = population["method"], population["atoms"]
        for sample in population["samples"]:
            checks.append(check(sample, references[method, atoms, "original"], atoms))
    summaries = []
    for method, atoms in CASES:
        for repeat in range(3):
            control = paired[f"{repeat}-{method}-{atoms}-control.json"]
            candidate = paired[f"{repeat}-{method}-{atoms}-candidate.json"]
            for left, right in zip(
                control["samples"], candidate["samples"], strict=True
            ):
                assert left["settings"] == right["settings"]
                assert left["iterations"] == right["iterations"]
                assert left["fock_builds"] == right["fock_builds"]
        for variant in ("control", "candidate", "gpu4pyscf"):
            selected = [
                paired[f"{r}-{method}-{atoms}-{variant}.json"] for r in range(3)
            ]
            summary = {"method": method, "atoms": atoms, "variant": variant}
            for phase in ("cold", "warm"):
                values = [
                    sample["seconds"]
                    for item in selected
                    for sample in item["samples"]
                    if sample["phase"] == phase
                ]
                assert len(values) == (3 if phase == "cold" else 6)
                summary[phase] = {
                    "median": statistics.median(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
            summaries.append(summary)
    print(
        json.dumps(
            {
                "complete_energy_force_checks": len(checks),
                "maximum_force_error": max(item["force_max"] for item in checks),
                "results": summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
