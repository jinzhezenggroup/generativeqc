"""Requalify the retained DIIS endpoints and scientific references without a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.readme_defaults_qualification import qualify


def read_archive(path: Path) -> Any:
    """Decode a lossless JSON or exact-text receipt bundle."""
    return json.loads(lzma.decompress(path.read_bytes()))


def _require(condition: bool, message: object) -> None:
    # Acceptance must remain active under python -O and PYTHONOPTIMIZE.
    if not condition:
        raise RuntimeError(str(message))


def validate(summary: dict, texts: dict[str, str], oracles: list, reuse: dict) -> dict:
    """Check every original population, including rejected scheduling experiments."""
    records = {name: json.loads(value) for name, value in texts.items()}
    counts = {}
    for group in ("populations", "exploratory_populations", "profile_populations"):
        entries = summary[group]
        _require(bool(entries), f"empty {group}")
        calls = 0
        for population in entries:
            name = population["file"]
            _require(
                hashlib.sha256(texts[name].encode()).hexdigest() == population["sha256"],
                f"receipt identity changed: {name}",
            )
            result = qualify([records[name]], oracles, reference_reuse=reuse)[0]
            _require(result["status"] == "qualified", (name, result))
            calls += len(records[name]["samples"])
        counts[group] = calls

    moved_calls = 0
    for population in summary["moved"]:
        name = population["file"]
        _require(
            hashlib.sha256(texts[name].encode()).hexdigest() == population["sha256"],
            f"moved receipt identity changed: {name}",
        )
        record = records[name]
        _require(record["status"] == "complete", f"incomplete moved endpoint: {name}")
        references = records[name.replace("baseline", "candidate")]["references"]
        for sample in record["samples"]:
            oracle = references[sample["geometry"]]
            delta = np.asarray(sample["forces"]) - oracle["forces"]
            errors = {
                "energy_per_atom": abs(sample["energy"] - oracle["energy"])
                / len(record["geometries_bohr"][sample["geometry"]]),
                "force_max": float(np.max(np.abs(delta))),
                "force_rms": float(np.sqrt(np.mean(delta**2))),
            }
            _require(
                all(errors[key] <= gate for key, gate in record["gates"].items()),
                (name, errors),
            )
            moved_calls += 1
    counts["moved"] = moved_calls
    _require(
        counts["populations"] + moved_calls == summary["qualified_complete_calls"],
        "integrated endpoint count changed",
    )

    failures = 0
    for pair in summary["comparisons"]:
        samples = zip(
            records[pair["baseline"]]["samples"],
            records[pair["candidate"]]["samples"],
            strict=True,
        )
        errors = [
            (
                abs(a["energy"] - b["energy"]),
                float(np.max(np.abs(np.asarray(a["forces"]) - b["forces"]))),
            )
            for a, b in samples
        ]
        de, df = max(x[0] for x in errors), max(x[1] for x in errors)
        _require(
            de == pair["energy_abs"] and df == pair["force_max_abs"],
            "paired diagnostic changed",
        )
        passed = de <= 1e-9 and df <= 1e-8
        _require(passed == pair["strict_diagnostic_pass"], "diagnostic status changed")
        failures += not passed
    counts["strict_diagnostic_failures"] = failures
    return counts


def main() -> None:
    """Check compact metadata against exact retained raw bundles."""
    directory = Path(__file__).resolve().parent
    root = directory.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=root / ".artifacts/remote")
    args = parser.parse_args()
    summary = read_archive(directory / "summary.json.xz")
    for name, receipt in summary["raw_bundles"].items():
        _require(
            hashlib.sha256((args.raw_directory / name).read_bytes()).hexdigest()
            == receipt["sha256"],
            f"raw bundle identity changed: {name}",
        )
    texts = read_archive(args.raw_directory / "records.json.xz")
    for name, expected in summary["references"].items():
        _require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name)
    sources = read_archive(args.raw_directory / "sources.json.xz")
    for name, expected in summary["source_manifests"].items():
        _require(hashlib.sha256(sources[name].encode()).hexdigest() == expected, name)
    reference = directory.parent / "readme-final-20261022"
    result = validate(
        summary,
        texts,
        read_archive(reference / "oracles.json.xz"),
        json.loads((reference / "reference-reuse.json").read_text()),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
