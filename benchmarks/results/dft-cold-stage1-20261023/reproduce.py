"""Recompute retained cold-endpoint evidence without running a GPU calculation.

Original receipts remain immutable. Stationary projections of the moved
trajectory are explicitly labeled views used by the existing static qualifier;
the moved comparison uses its separately measured GPU4PySCF geometry.
"""

from __future__ import annotations

import copy
import json
import lzma
from pathlib import Path
from statistics import median

from benchmarks.readme_defaults_accuracy import errors
from benchmarks.readme_defaults_qualification import qualify
from benchmarks.readme_defaults_report import checked_sample, load_records

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "readme-final-20261022"


def read(name: str) -> dict:
    """Load a losslessly retained set of complete named receipts."""
    return json.loads(lzma.decompress((ROOT / name).read_bytes()))


def timing(first: list[float], last: list[float]) -> dict:
    """Keep every observation and the unchanged median/relative-MAD gate."""
    before, after = median(first), median(last)
    noise = 2 * (
        median(abs(x - before) for x in first) / before
        + median(abs(x - after) for x in last) / after
    )
    gain = 1 - after / before
    floor = max(0.02, noise)
    return {
        "before_seconds": first,
        "after_seconds": last,
        "before_median": before,
        "after_median": after,
        "gain_fraction": gain,
        "gain_floor": floor,
        "passes_gain_gate": gain > floor,
    }


def main() -> None:
    oracles = load_records(
        [REFERENCE / "oracles.json.xz"],
        schema="generativeqc.readme-defaults-oracle.v1",
    )
    reuse = json.loads((REFERENCE / "reference-reuse.json").read_text())
    result = {"qualification": {}, "timing": {}, "moved_default_reference": {}}
    for cohort in ("cooperative-clean", "cooperative-moved"):
        records = read(cohort + ".json.xz")
        for name, record in records.items():
            if record["arm"] == "gpu4pyscf":
                continue
            view = copy.deepcopy(record)
            if cohort == "cooperative-moved":
                view["samples"] = view["samples"][:2]
                view["warm_repeats"] = 1
            qualified = qualify([view], oracles, reference_reuse=reuse)[0]
            assert qualified["status"] == "qualified", (cohort, name, qualified)
            result["qualification"][cohort + "/" + name] = {
                "view": "stationary cold/warm" if cohort.endswith("moved") else "complete",
                **qualified,
            }
        for atoms in (48, 96):
            phases = ("cold", "warm", "moved", "moved-warm")
            if cohort.endswith("clean"):
                phases = phases[:2]
            for phase in phases:
                populations = []
                for mapping in ("shell_warp", "nucleus_cooperative"):
                    chosen = sorted(
                        (name, r)
                        for name, r in records.items()
                        if r["atoms"] == atoms and mapping in name
                    )
                    assert len(chosen) == 5, (cohort, atoms, mapping)
                    samples = [
                        sample
                        for _, r in chosen
                        for sample in r["samples"]
                        if sample["phase"] == phase
                    ]
                    assert len(samples) == 5
                    populations.append([s["seconds"] for s in samples])
                result["timing"][f"{cohort}/{atoms}/{phase}"] = timing(*populations)
            if cohort.endswith("moved"):
                reference = records[f"pbe0-direct-{atoms}-gpu4pyscf.json"]
                for name, record in records.items():
                    if record["arm"] == "gpu4pyscf" or record["atoms"] != atoms:
                        continue
                    assert record["moved_geometry_bohr"] == reference["geometry_bohr"]
                    checks = []
                    for sample in record["samples"][2:]:
                        checked_sample(sample, record)
                        check = errors(
                            sample["energy"], sample["forces"], reference["samples"][0], atoms
                        )
                        assert check["passed"], (name, check)
                        checks.append(check)
                    result["moved_default_reference"][name] = checks
    # The final DF trajectory has its own five pairs and source identities.
    moved_path = ROOT / "df-moved.json.xz"
    if moved_path.exists():
        records = read(moved_path.name)
        moved_references = read("cooperative-moved.json.xz")
        for name, record in records.items():
            view = copy.deepcopy(record)
            view["samples"] = view["samples"][:2]
            view["warm_repeats"] = 1
            qualified = qualify([view], oracles, reference_reuse=reuse)[0]
            assert qualified["status"] == "qualified", (name, qualified)
            result["qualification"]["df-moved/" + name] = qualified
            reference = moved_references[f"pbe0-direct-{record['atoms']}-gpu4pyscf.json"]
            assert record["moved_geometry_bohr"] == reference["geometry_bohr"]
            checks = []
            for sample in record["samples"][2:]:
                checked_sample(sample, record)
                check = errors(
                    sample["energy"], sample["forces"], reference["samples"][0], record["atoms"]
                )
                assert check["passed"], (name, check)
                checks.append(check)
            result["moved_default_reference"]["df-moved/" + name] = checks
        for atoms in (48, 96):
            for phase in ("cold", "warm", "moved", "moved-warm"):
                populations = [
                    [
                        s["seconds"]
                        for name, r in sorted(records.items())
                        if variant in name and r["atoms"] == atoms
                        for s in r["samples"]
                        if s["phase"] == phase
                    ]
                    for variant in ("baseline", "stage1")
                ]
                assert all(len(population) == 5 for population in populations)
                result["timing"][f"df-moved/{atoms}/{phase}"] = timing(*populations)
    # Final combined-source cohorts, when present, are independent populations.
    for path in sorted(ROOT.glob("stage1-*.json.xz")):
        records = read(path.name)
        for name, record in records.items():
            qualified = qualify([record], oracles, reference_reuse=reuse)[0]
            assert qualified["status"] == "qualified", (path.name, name, qualified)
            result["qualification"][path.name + "/" + name] = qualified
        if path.name.startswith("stage1-pairs-"):
            for phase in ("cold", "warm"):
                populations = []
                for variant in ("baseline", "stage1"):
                    # Each process contributes one statistic; its three warm
                    # calls are not three independent cold-start replicates.
                    values = [
                        median(s["seconds"] for s in r["samples"] if s["phase"] == phase)
                        for name, r in sorted(records.items())
                        if variant in name
                    ]
                    assert len(values) == 5
                    populations.append(values)
                result["timing"][path.name + "/" + phase] = timing(*populations)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
