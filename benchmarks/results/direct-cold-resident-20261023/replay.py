"""Recheck retained endpoint arrays and medians offline; no GPU or compilation.

Run from the repository root with PYTHONPATH=python:. . Historical sources are
bound by base plus patch; this replay never requires current source spellings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import statistics
from pathlib import Path

import numpy as np

from benchmarks.readme_defaults import COMMON_GATES
from benchmarks.readme_defaults_qualification import qualify
from benchmarks.readme_defaults_report import checked_sample


def cpu_check(record: dict, reference: dict) -> dict:
    """Bind the exact scientific inputs before comparing all exported forces."""
    assert record["geometry_bohr"] == reference["geometry_bohr"]
    assert record["orbital_basis"].lower() == reference["basis"] == "def2-svp"
    assert record["basis_representation"] == "spherical"
    assert (record["method"], record["mode"], record["charge"], record["spin"]) == (
        "hf", "direct", 0, 0
    )
    assert reference["cartesian"] is False
    assert reference["controls"] == {
        "energy": 1e-11, "gradient": 1e-8, "direct_scf": False
    }
    assert reference["orbital_gradient_norm"] < 1e-8
    assert record["status"] == "complete"
    assert len(record["samples"]) == record["warm_repeats"] + 1
    assert record["protocol"]["common_observable_gates"] == COMMON_GATES
    rows = []
    for index, sample in enumerate(record["samples"]):
        checked_sample(sample, record)
        assert (sample["phase"], sample["repeat"]) == (
            ("cold", 0) if index == 0 else ("warm", index - 1)
        )
        assert sample["precision"]["operator_work_counters_valid"]
        assert sample["fock_builds"] > 0
        actual = np.asarray(sample["forces"])
        expected = np.asarray(reference["forces"])
        assert actual.shape == expected.shape == (record["atoms"], 3)
        delta = actual - expected
        assert np.isfinite(delta).all()
        errors = {
            "energy_abs_per_atom_hartree": abs(sample["energy"] - reference["energy"])
            / record["atoms"],
            "force_max_abs_hartree_per_bohr": float(np.abs(delta).max()),
            "force_rms_hartree_per_bohr": float(np.sqrt(np.mean(delta**2))),
        }
        rows.append({**errors, "passed": all(errors[k] <= v for k, v in COMMON_GATES.items())})
    return {"status": "qualified" if all(x["passed"] for x in rows) else "failed", "errors": rows}


def main() -> None:
    """Recompute gates, preserving rejected pilots outside final populations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--extract", type=Path, help="extract original scripts and source patches")
    args = parser.parse_args()
    payload = json.loads(lzma.decompress(Path(__file__).with_name("evidence.json.xz").read_bytes()))
    for name, patch in payload["provenance"]["patches"].items():
        assert hashlib.sha256(patch.encode()).hexdigest() == payload["provenance"]["patch_sha256"][name]
    for path, expected in payload["legacy_reference_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    old = Path("benchmarks/results/readme-final-20261022")
    oracles = json.loads(lzma.decompress((old / "oracles.json.xz").read_bytes()))
    reuse = json.loads((old / "reference-reuse.json").read_bytes())
    results = {}
    for name, record in payload["records"].items():
        reference = next(
            (r for r in payload["cpu_oracles"].values() if r["geometry_bohr"] == record["geometry_bohr"]),
            None,
        )
        result = cpu_check(record, reference) if reference else qualify([record], oracles, reference_reuse=reuse)[0]
        # The v1 force defect is retained deliberately, never silently promoted.
        expected = "failed" if name.startswith("trial/values-1-") else "qualified"
        assert result["status"] == expected, (name, result)
        results[name] = result
    timings = {}
    for case in ("water12", "water24", "benzene"):
        timings[case] = {}
        for arm in ("0", "auto"):
            records = [payload["records"][f"default/{case}-{arm}-r{i}.json"] for i in (1, 2, 3)]
            timings[case][arm] = {
                phase: statistics.median(s["seconds"] for r in records for s in r["samples"] if s["phase"] == phase)
                for phase in ("cold", "warm")
            }
        timings[case]["cold_reduction_percent"] = 100 * (1 - timings[case]["auto"]["cold"] / timings[case]["0"]["cold"])
    for name, trace in payload["traces"].items():
        if name.startswith("fallback/"):
            assert "cuda_rhf_resident_values_prepare" not in trace["scope_names"]
    for case, actions in (("water12", 12), ("benzene", 11)):
        events = payload["traces"][f"diagnostic/{case}.trace.jsonl"]["resident_events"]
        values = {r["key"]: r["value"] for r in events if r["event"] == "VALUE"}
        assert values["selection"] == "auto"
        assert values["source_values_built"] > 0
        assert values["phase_resident_fock_actions"] == actions
        assert values["phase_direct_values_per_action"] == 0
    assert "ERROR SUMMARY: 0 errors" in payload["logs"]["memcheck.log"]
    summary = {"qualifications": results, "timings": timings}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if args.extract:
        args.extract.mkdir(parents=True, exist_ok=True)
        for name, content in {**payload["scripts"], **payload["provenance"]["patches"]}.items():
            assert Path(name).name == name
            (args.extract / name).write_text(content)
    print(json.dumps({"checked_receipts": len(results), "timings": timings}, indent=2))


if __name__ == "__main__":
    main()
