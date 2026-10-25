"""Audit temporary-file ERI evidence, including the losing full-range cohort."""

from __future__ import annotations

import hashlib
import json
import lzma
import xml.etree.ElementTree as ET
from pathlib import Path

from benchmarks import readme_defaults_report as reader
from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parent


def unpack(path: Path) -> dict:
    return json.loads(lzma.decompress(path.read_bytes()))


def digest(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def physical_io(record: dict) -> dict:
    """Process I/O also includes input/library traffic; ERI bytes are separate."""
    io = record["host_eri_experiment"]["process_io"]
    return {key: io["after"][key] - io["before"][key] for key in io["before"]}


def main() -> None:
    if not __debug__:
        raise SystemExit("Run evidence verification without -O.")
    p = unpack(ROOT / "measurements.json.xz")
    assert p["schema"] == "generativeqc.temporary-eri-spill.v1"
    for name, raw in p["support"].items():
        assert digest(raw) == p["support_sha256"][name], name
    for binding in p["source_bindings"].values():
        text = "".join(f"{name}:{sha}\n" for name, sha in sorted(binding["native_inventory"].items()))
        assert digest(text) == binding["native_identity"]
    measured, policy = (p["source_bindings"][key] for key in ("final", "policy"))
    assert {name for name in measured["native_inventory"] if measured["native_inventory"][name] != policy["native_inventory"][name]} == {"src/dft/cuda_ks.cpp"}
    tests = {}
    for cohort, families in (("n2", (("raw-disk", 12), ("endpoint-range", 8), ("endpoint-full", 3))), ("n2-policy", (("endpoint-range", 8), ("endpoint-full", 3)))):
        for name, expected in families:
            key = f"{cohort}/{name}.xml"
            cases = list(ET.fromstring(p["support"][key]).iter("testcase"))  # noqa: S314
            assert len(cases) == expected
            assert not any(c.find(tag) is not None for c in cases for tag in ("failure", "error", "skipped"))
            tests[key] = expected
    for mode in ("full-disk-oracle", "range-tiered-oracle"):
        log = p["support"][f"n2/memcheck-{mode}.log"]
        assert "ERROR SUMMARY: 0 errors" in log and "LEAK SUMMARY: 0 bytes leaked" in log
    oracles = reader.load_records([ROOT.parent / "readme-final-20261022/oracles.json.xz"], schema="generativeqc.readme-defaults-oracle.v1")
    old = ROOT.parent / "fitted-range-cold-20261024"
    oracles.append(json.loads(unpack(old / "provenance.json.xz")["support"]["stage29-larger-oracle/oracle.json"]))
    reuse = json.loads(unpack(old / "integration.json.xz")["support"]["stage37-endpoint-qualification.reference-reuse.json"])
    reader.SIZES = (*reader.SIZES, 108)
    records, gates = {}, {}
    for name, raw in p["records"].items():
        assert digest(raw) == p["record_sha256"][name]
        r = json.loads(raw)
        assert r["status"] == "complete" and r["source_revision"] == measured["commit"]
        assert r["native_build"]["source_identity"] == measured["native_identity"]
        assert r["native_build"]["library_sha256"] == measured["library_sha256"]
        assert r["host_eri_experiment"]["runner_sha256"] == measured["runner_sha256"]
        gates[name] = qualify([r], oracles, reference_reuse=reuse)[0]
        assert gates[name]["status"] == "qualified", (name, gates[name])
        records[name] = r
    timings = {}
    for name in ([f"n4/{tier}-108.json" for tier in ("gpu", "host", "disk", "tiered")] + [f"n5/full-{tier}-96.json" for tier in ("gpu", "disk")]):
        r = records[name]
        assert not r["host_eri_experiment"]["trace"]
        timings[name] = {
            "samples": [{k: s[k] for k in ("phase", "seconds", "iterations", "fock_builds")} for s in r["samples"]],
            "process_io": physical_io(r),
        }
    work = {}
    for profile, receipt, expected in (("n4/disk-profile.jsonl", "n4/disk-primer-108.json", 28), ("n4/tiered-profile.jsonl", "n4/tiered-profile-108.json", 28), ("n5/full-disk-profile.jsonl", "n5/full-disk-primer-96.json", 11)):
        actions = p["profiles"][profile]["actions"]
        cached = [a for a in actions if a["counters"].get("disk_cached_values")]
        assert len(cached) == expected
        for a in cached:
            c = a["counters"]
            assert c["recurrence_evaluations"] == 0
            assert c["canonical_values_visited"] == sum(c.get(k, 0) for k in ("gpu_cached_values", "host_cached_values", "disk_cached_values"))
            assert c["disk_read_bytes"] == 8 * c["disk_cached_values"]
            assert c["host_h2d_bytes"] == 8 * (c.get("host_cached_values", 0) + c["disk_cached_values"])
        logical = sum(a["counters"]["disk_read_bytes"] for a in cached)
        physical = physical_io(records[receipt])["read_bytes"]
        # This checks these retained measurements, not an OS guarantee about
        # fadvise: the experiment must actually have exercised physical storage.
        assert physical > 0.9 * logical, (profile, physical, logical)
        work[profile] = {"cached_actions": expected, "logical_disk_bytes": logical, "process_physical_read_bytes": physical, "first_cached_action": cached[0]["counters"]}
    small = p["profiles"]["n4/auto-small-progress.jsonl"]["values"]
    assert any(r.get("key") == "source_values_built" and r["value"] > 0 for r in small)
    assert not any(r["value"] for r in small if r.get("key") == "disk_source_value_bytes")
    print(json.dumps({"tests": tests, "timings": timings, "work": work, "qualification": gates}, indent=2))


if __name__ == "__main__":
    main()
