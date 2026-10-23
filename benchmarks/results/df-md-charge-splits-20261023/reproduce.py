"""Recheck separate charge-partition populations and independent endpoint gates."""

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


def timing(before: list[float], after: list[float]) -> dict:
    """Use one observation per process, including its median warm invocation."""
    baseline, candidate = median(before), median(after)
    baseline_mad = median(abs(x - baseline) for x in before)
    candidate_mad = median(abs(x - candidate) for x in after)
    floor = max(0.02, 2 * (baseline_mad / baseline + candidate_mad / candidate))
    gain = 1 - candidate / baseline
    return {
        "before_seconds": before,
        "after_seconds": after,
        "before_median": baseline,
        "after_median": candidate,
        "before_mad": baseline_mad,
        "after_mad": candidate_mad,
        "gain_fraction": gain,
        "gain_floor": floor,
        "passes_gain_gate": gain > floor,
    }


def main() -> None:
    """Authenticate retained provenance, then qualify every native call."""
    data = json.loads(lzma.decompress((ROOT / "receipts.json.xz").read_bytes()))
    manifest = lzma.decompress((ROOT / data["source_manifest_path"]).read_bytes())
    assert hashlib.sha256(manifest).hexdigest() == data["source_manifest_sha256"]
    assert len(json.loads(manifest)["files"]) == 5984
    oracles = load_records(
        [REFERENCE / "oracles.json.xz"], schema="generativeqc.readme-defaults-oracle.v1"
    )
    reuse = json.loads((REFERENCE / "reference-reuse.json").read_text())
    result = {"qualification": {}, "moved": {}, "timing": {}, "native_calls": 0}
    for cohort, records in data["cohorts"].items():
        for name, record in records.items():
            assert record["status"] == "complete"
            if record["arm"] == "gpu4pyscf":
                continue
            assert record["source_revision"] == data["source_revision"]
            result["native_calls"] += len(record["samples"])
            view = copy.deepcopy(record)
            if "moved_geometry_bohr" in record:
                assert [s["phase"] for s in record["samples"]] == [
                    "cold", "warm", "moved", "moved-warm"
                ]
                view["samples"] = view["samples"][:2]
                view["warm_repeats"] = 1
                reference = records["r2scan-3c-df-moved-gpu4pyscf.json"]
                assert record["moved_geometry_bohr"] == reference["geometry_bohr"]
                for key in (
                    "method", "mode", "orbital_basis", "basis_representation",
                    "charge", "spin", "auxiliary_input_sha256",
                ):
                    assert record[key] == reference[key], (name, key)
                errors = [
                    _comparison(sample, reference["samples"][0], record["atoms"])
                    for sample in record["samples"][2:]
                ]
                assert all(error["passed"] for error in errors), (name, errors)
                result["moved"][name] = errors
            qualified = qualify([view], oracles, reference_reuse=reuse)[0]
            assert qualified["status"] == "qualified", (name, qualified)
            result["qualification"][cohort + "/" + name] = qualified
    assert result["native_calls"] == 104
    pairs = data["cohorts"]["stage7-v2-pairs"]
    for phase in ("cold", "warm"):
        populations = []
        for variant in (1, 32):
            values = [
                median(s["seconds"] for s in record["samples"] if s["phase"] == phase)
                for name, record in sorted(pairs.items())
                if f"splits-{variant}-" in name
            ]
            assert len(values) == 5
            populations.append(values)
        result["timing"][phase] = timing(*populations)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
