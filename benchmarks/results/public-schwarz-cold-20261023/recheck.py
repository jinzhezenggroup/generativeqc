"""Requalify retained historical endpoints without compiling or using a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path

import numpy as np

from benchmarks.readme_defaults_qualification import qualify


def read_archive(path: Path) -> dict:
    """Decode a JSON receipt or exact-text bundle retained beside this reader."""
    return json.loads(lzma.decompress(path.read_bytes()))


def paired_difference(first: dict, second: dict) -> tuple[float, float]:
    """Compare corresponding cold/warm samples, retaining the largest error."""
    pairs = list(zip(first["samples"], second["samples"], strict=True))
    return (
        max(abs(a["energy"] - b["energy"]) for a, b in pairs),
        max(
            float(np.max(np.abs(np.asarray(a["forces"]) - b["forces"])))
            for a, b in pairs
        ),
    )


def main() -> None:
    """Check artifact integrity and scientific gates, not current source hashes."""
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-directory",
        type=Path,
        default=root / ".artifacts/remote",
        help="Local copy of the retained campaign root (see README).",
    )
    raw = parser.parse_args().raw_directory
    summary = read_archive(directory / "summary.json.xz")
    names = [population["file"] for population in summary["populations"]]
    names += [
        f"moved/{population['variant']}-{population['case']}.json"
        for population in summary["moved"]
    ]
    texts = {name: (raw / name).read_text() for name in names}
    records = {name: json.loads(value) for name, value in texts.items()}
    for variant in ("baseline", "candidate"):
        text = (raw / f"{variant}-source.json").read_text()
        assert hashlib.sha256(text.encode()).hexdigest() == summary["sources"][
            variant
        ]["manifest_sha256"]
    for name, expected in summary["references"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    reference = directory.parent / "readme-final-20261022"
    oracles = read_archive(reference / "oracles.json.xz")
    reuse = json.loads((reference / "reference-reuse.json").read_text())
    count = 0
    for population in summary["populations"]:
        name = population["file"]
        assert hashlib.sha256(texts[name].encode()).hexdigest() == population["sha256"]
        # Qualify one fresh-process population at a time: repeated rounds share
        # the same scientific key and must not be silently overwritten.
        result = qualify([records[name]], oracles, reference_reuse=reuse)[0]
        assert result["status"] == "qualified", (name, result)
        count += len(records[name]["samples"])

    diagnostic_failures = 0
    for comparison in summary["comparisons"]:
        de, df = paired_difference(
            records[comparison["baseline"]], records[comparison["candidate"]]
        )
        assert de == comparison["energy_abs"] and df == comparison["force_max_abs"]
        passed = de <= 1e-9 and df <= 1e-8
        assert passed == comparison["paired_diagnostic_pass"]
        diagnostic_failures += not passed
    for comparison in summary["fast_within_binary_variation"]:
        de, df = paired_difference(
            records[comparison["first"]], records[comparison["second"]]
        )
        assert de == comparison["energy_abs"] and df == comparison["force_max_abs"]

    for population in summary["moved"]:
        name = f"moved/{population['variant']}-{population['case']}.json"
        assert hashlib.sha256(texts[name].encode()).hexdigest() == population[
            "receipt_sha256"
        ]
        record = records[name]
        references = records[f"moved/candidate-{population['case']}.json"]["references"]
        assert record["status"] == "complete"
        for sample in record["samples"]:
            oracle = references[sample["geometry"]]
            difference = np.asarray(sample["forces"]) - oracle["forces"]
            atoms = len(record["geometries_bohr"][sample["geometry"]])
            errors = {
                "energy_per_atom": abs(sample["energy"] - oracle["energy"]) / atoms,
                "force_max": float(np.max(np.abs(difference))),
                "force_rms": float(np.sqrt(np.mean(difference**2))),
            }
            assert all(errors[key] <= gate for key, gate in record["gates"].items())
            count += 1
    assert count == summary["qualified_complete_calls"] == 170
    print(f"Independent scientific gates: {count} complete E/F calls passed.")
    print(f"Stricter baseline/candidate paired diagnostic failures retained: {diagnostic_failures}.")
    print("Recorded same-binary Fast variation also reproduced from the receipts.")


if __name__ == "__main__":
    main()
