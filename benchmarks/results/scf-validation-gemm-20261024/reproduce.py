"""Recheck every retained endpoint and paired timing gate without GPU execution."""
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
METHODS = ("pbe", "pbe0", "b3lyp", "r2scan", "r2scan-3c", "m06-2x", "wb97m-v")


def read(name: str) -> dict:
    """Read the complete lossless JSON packet."""
    return json.loads(lzma.decompress((ROOT / name).read_bytes()))


def timing(baseline: list[float], candidate: list[float]) -> dict:
    """Keep one cold or warm-median observation per independent process."""
    b, c = median(baseline), median(candidate)
    bm = median(abs(x - b) for x in baseline)
    cm = median(abs(x - c) for x in candidate)
    gate = max(0.02, 2 * (bm / b + cm / c))
    gain = 1 - c / b
    return dict(baseline=baseline, candidate=candidate, baseline_median=b,
                candidate_median=c, baseline_mad=bm, candidate_mad=cm,
                gain=gain, gate=gate, passed=gain > gate)


def main() -> None:
    """Authenticate provenance, independently qualify all forces and count work."""
    if not __debug__:
        raise SystemExit("Evidence verification requires assertions; do not use -O.")
    packet, provenance = read("endpoints.json.xz"), read("provenance.json.xz")
    revision = packet["source_revision"]
    assert revision == provenance["source_revision"]
    source_raw = provenance["source_manifest_raw"]
    assert hashlib.sha256(source_raw.encode()).hexdigest() == provenance["source_manifest_sha256"]
    source = json.loads(source_raw)
    assert source["source_revision"] == revision
    verified = json.loads(provenance["support"]["validation-gemm-pr-source-verification.json"])
    assert verified["source_revision"] == revision and not verified["mismatches"]
    assert verified["files_verified"] == len(source["files"])
    for name, raw in provenance["support"].items():
        assert hashlib.sha256(raw.encode()).hexdigest() == provenance["support_sha256"][name]
    references = ROOT.parent / "readme-final-20261022"
    oracles = load_records([references / "oracles.json.xz"], schema="generativeqc.readme-defaults-oracle.v1")
    reuse = json.loads((references / "reference-reuse.json").read_text())
    report = dict(qualified_calls=0, qualification={}, work={}, timing={})
    decoded = {}
    for cohort, records in packet["cohorts"].items():
        decoded[cohort] = {}
        for name, raw in records.items():
            key = cohort + "/" + name
            assert hashlib.sha256(raw.encode()).hexdigest() == packet["raw_record_sha256"][key]
            record = json.loads(raw)
            decoded[cohort][name] = record
            assert record["status"] == "complete" and record["source_revision"] == revision
            assert record["native_build"]["library_sha256"] in provenance["support"]["validation-gemm-pr-package-sha256.txt"]
            view = copy.deepcopy(record)
            if "moved_geometry_bohr" in record:
                ref = packet["moved_references"][record["method"] + "/" + record["mode"]]
                assert record["moved_geometry_bohr"] == ref["geometry_bohr"]
                for field in ("method", "mode", "orbital_basis", "basis_representation", "charge", "spin", "auxiliary_input_sha256"):
                    assert record.get(field) == ref.get(field), (key, field)
                assert [s["phase"] for s in record["samples"]] == ["cold", "warm", "moved", "moved-warm"]
                moved = [_comparison(s, ref["samples"][0], record["atoms"]) for s in record["samples"][2:]]
                assert all(v["passed"] for v in moved), (key, moved)
                view["samples"], view["warm_repeats"] = view["samples"][:2], 1
            verdict = qualify([view], oracles, reference_reuse=reuse)[0]
            assert verdict["status"] == "qualified", (key, verdict)
            report["qualification"][key] = verdict
            report["qualified_calls"] += len(record["samples"])
            report["work"][key] = [(s["phase"], s["iterations"], s["fock_builds"]) for s in record["samples"]]
    assert report["qualified_calls"] == packet["complete_calls"] == 328
    for mode in ("df", "direct"):
        matrix = decoded[f"validation-gemm-pr-gemm-matrix-{mode}"]
        assert {(r["method"], r["arm"]) for r in matrix.values()} == {(m, a) for m in METHODS for a in ("standard", "fast")}
        pairs = decoded[f"validation-gemm-pr-gemm-pairs-{mode}"]
        assert len(pairs) == 30
        for method in ("pbe", "pbe0", "r2scan-3c"):
            arms = {a: [pairs[f"{method}-{mode}-pair-{i}-gemm-{a}.json"] for i in range(5)] for a in ("0", "1")}
            for arm, records in arms.items():
                for record in records:
                    control = record["validation_gemm_control"]
                    assert control["selection"] == arm
                    runner = provenance["support"]["validation-gemm-pr-gemm-screen.py"]
                    assert control["runner_sha256"] == hashlib.sha256(runner.encode()).hexdigest()
            for b, c in zip(arms["0"], arms["1"], strict=True):
                for field in ("source_revision", "native_build", "protocol", "geometry_bohr", "orbital_basis", "basis_representation", "slurm_job_id"):
                    assert b[field] == c[field], field
                assert {k: v for k, v in b["accelerator"].items() if k != "nvidia_smi"} == {k: v for k, v in c["accelerator"].items() if k != "nvidia_smi"}
                for bs, cs in zip(b["samples"], c["samples"], strict=True):
                    assert bs["settings"] == cs["settings"] and _comparison(bs, cs, b["atoms"])["passed"]
            for phase in ("cold", "warm"):
                values = [[median(s["seconds"] for s in r["samples"] if s["phase"] == phase) for r in arms[a]] for a in ("0", "1")]
                report["timing"][f"{method}/{mode}/{phase}"] = timing(*values)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
