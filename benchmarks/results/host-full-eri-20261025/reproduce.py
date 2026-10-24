"""Recheck source-bound full-range RAM evidence and independent E+F gates."""

from __future__ import annotations

import hashlib
import json
import lzma
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

from benchmarks import readme_defaults_report as reader
from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parent


def unpack(path: Path) -> dict:
    return json.loads(lzma.decompress(path.read_bytes()))


def digest(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def main() -> None:
    if not __debug__:
        raise SystemExit("Run evidence verification without -O.")
    packet = unpack(ROOT / "measurements.json.xz")
    assert packet["schema"] == "generativeqc.host-full-eri.v1"
    for name, raw in packet["support"].items():
        assert digest(raw) == packet["support_sha256"][name]
    for binding in packet["source_bindings"].values():
        inventory = "".join(f"{n}:{h}\n" for n, h in sorted(binding["native_inventory"].items()))
        assert digest(inventory) == binding["native_identity"]
    oracles = reader.load_records(
        [ROOT.parent / "readme-final-20261022/oracles.json.xz"],
        schema="generativeqc.readme-defaults-oracle.v1",
    )
    reuse = json.loads(unpack(ROOT.parent / "fitted-range-cold-20261024/integration.json.xz")["support"]["stage37-endpoint-qualification.reference-reuse.json"])
    records, gates = {}, {}
    for name, raw in packet["records"].items():
        assert digest(raw) == packet["record_sha256"][name]
        r = json.loads(raw)
        binding = packet["source_bindings"][name.split("/")[0]]
        assert r["status"] == "complete" and r["source_revision"] == binding["commit"]
        assert r["native_build"]["library_sha256"] == binding["library_sha256"]
        assert r["native_build"]["source_identity"] == binding["native_identity"]
        assert r["host_eri_experiment"]["runner_sha256"] == binding["runner_sha256"]
        gates[name] = qualify([r], oracles, reference_reuse=reuse)[0]
        assert gates[name]["status"] == "qualified", (name, gates[name])
        records[name] = r
    tests = {}
    for name, count in (("raw-full", 4), ("endpoint-full", 3), ("phase-regression", 2)):
        cases = list(ET.fromstring(packet["support"][f"n5-source/{name}.xml"]).iter("testcase"))  # noqa: S314
        assert len(cases) == count
        assert not any(c.find(tag) is not None for c in cases for tag in ("failure", "error", "skipped"))
        tests[name] = count
    assert "ERROR SUMMARY: 0 errors" in packet["support"]["n5-source/memcheck.log"]
    assert "LEAK SUMMARY: 0 bytes leaked" in packet["support"]["n5-source/memcheck.log"]
    summary = {}
    for cohort, prefix in (("n5-natural", "full"), ("n5-source", "source")):
        arms = {}
        for tier in ("gpu", "host"):
            selected = [records[f"{cohort}/{prefix}-{tier}-{i}-96.json"] for i in (1, 2)]
            assert all(not r["host_eri_experiment"]["trace"] for r in selected)
            assert all((r["host_eri_experiment"]["controls"].get("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK") == "0") == (cohort == "n5-source") for r in selected)
            arms[tier] = {
                phase: [next(s for s in r["samples"] if s["phase"] == phase)["seconds"] for r in selected]
                for phase in ("cold", "warm")
            }
            arms[tier]["work"] = [[{k: s[k] for k in ("phase", "iterations", "fock_builds")} for s in r["samples"]] for r in selected]
        arms["cold_reduction_percent"] = 100 * (1 - statistics.median(arms["host"]["cold"]) / statistics.median(arms["gpu"]["cold"]))
        summary[cohort] = arms
    actions = packet["profiles"]["n5-source/source-profile.jsonl"]["actions"]
    cached = [a for a in actions if a["counters"].get("host_cached_values")]
    assert len(actions) == 12 and len(cached) == 11
    for action in cached:
        c = action["counters"]
        assert c["recurrence_evaluations"] == 0
        assert c["canonical_values_visited"] == c["gpu_cached_values"] + c["host_cached_values"]
        assert c["host_h2d_bytes"] == 8 * c["host_cached_values"]
    print(json.dumps({"tests": tests, "endpoints": summary, "cached_action_counters": [a["counters"] for a in cached], "qualification": gates}, indent=2))


if __name__ == "__main__":
    main()
