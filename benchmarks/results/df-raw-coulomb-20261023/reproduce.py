"""Recheck immutable raw-J endpoints and their separate source populations.

Stationary samples use the independently qualified dense references. Moved
samples use fresh same-geometry GPU4 DF controls; this is schedule equivalence,
not a moved dense-grid convergence claim. No original receipt is rewritten.
"""

from __future__ import annotations

import copy
import hashlib
import json
import lzma
from pathlib import Path
from statistics import median

from benchmarks.readme_defaults_qualification import _comparison, qualify
from benchmarks.readme_defaults_report import load_records

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "readme-final-20261022"


def read(name: str) -> dict:
    """Read losslessly retained named receipts."""
    return json.loads(lzma.decompress((ROOT / (name + ".json.xz")).read_bytes()))


def timing(first: list[float], last: list[float]) -> dict:
    """Keep every observation and the unchanged median/relative-MAD gate."""
    before, after = median(first), median(last)
    floor = max(
        0.02,
        2
        * (
            median(abs(x - before) for x in first) / before
            + median(abs(x - after) for x in last) / after
        ),
    )
    gain = 1 - after / before
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
    moved_references = read("stage2-moved-reference")
    result = {"qualification": {}, "moved_default_reference": {}, "timing": {}}
    cohorts = [("stage2-pairs", 20), ("stage2-moved", 40), ("stage2-matrix", 12)]
    # Integrated-default populations, if retained, remain distinct from opt-in.
    cohorts += [
        (path.name.removesuffix(".json.xz"), None)
        for path in sorted(ROOT.glob("default-*.json.xz"))
    ]
    bridge_path = ROOT / "merge-bridge.json.xz"
    if bridge_path.is_file():
        bridge = read("merge-bridge")
        inputs = read("stage2-default-source-inputs")["files"].copy()
        delta = bridge["source_input_delta"]
        inputs.update(delta["changed"])
        for name in delta["removed"]:
            del inputs[name]
        assert len(inputs) == bridge["source_input_count"]
        manifest = {"source_revision": bridge["source_revision"], "files": dict(sorted(inputs.items()))}
        original_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
        assert hashlib.sha256(original_bytes).hexdigest() == bridge["source_manifest_file_sha256"]
        assert len(bridge["records"]) == 12
        for name, record in bridge["records"].items():
            assert record["source_revision"] == bridge["source_revision"]
            qualified = qualify([record], oracles, reference_reuse=reuse)[0]
            assert qualified["status"] == "qualified", (name, qualified)
            result["qualification"]["merge-matrix/" + name] = qualified
    for cohort, count in cohorts:
        records = read(cohort)
        if count is not None:
            assert len(records) == count, cohort
        for name, record in records.items():
            assert record["status"] == "complete", (name, record.get("error"))
            view = copy.deepcopy(record)
            if "moved" in cohort:
                assert [s["phase"] for s in record["samples"]] == [
                    "cold", "warm", "moved", "moved-warm"
                ]
                view["samples"] = view["samples"][:2]
                view["warm_repeats"] = 1
                ref = moved_references[
                    f"{record['method']}-df-{record['atoms']}-gpu4pyscf.json"
                ]
                assert record["moved_geometry_bohr"] == ref["geometry_bohr"]
                for key in (
                    "method", "mode", "orbital_basis", "basis_representation",
                    "charge", "spin", "auxiliary_input_sha256",
                ):
                    assert record[key] == ref[key], (name, key)
                checks = [
                    _comparison(s, ref["samples"][0], record["atoms"])
                    for s in record["samples"][2:]
                ]
                assert all(c["passed"] for c in checks), (name, checks)
                result["moved_default_reference"][cohort + "/" + name] = checks
            qualified = qualify([view], oracles, reference_reuse=reuse)[0]
            assert qualified["status"] == "qualified", (name, qualified)
            result["qualification"][cohort + "/" + name] = qualified
        if "matrix" in cohort:
            continue
        variants = ("stage1", "stage2") if cohort.startswith("stage2-") else ("off", "unset")
        for method in ("pbe", "r2scan"):
            for atoms in sorted({r["atoms"] for r in records.values()}):
                phases = ("cold", "warm", "moved", "moved-warm") if "moved" in cohort else ("cold", "warm")
                for phase in phases:
                    populations = []
                    for variant in variants:
                        population = [
                            median(s["seconds"] for s in r["samples"] if s["phase"] == phase)
                            for name, r in sorted(records.items())
                            if variant in name and r["method"] == method and r["atoms"] == atoms
                        ]
                        assert len(population) == 5, (cohort, method, atoms, variant)
                        populations.append(population)
                    result["timing"][f"{cohort}/{method}/{atoms}/{phase}"] = timing(*populations)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
