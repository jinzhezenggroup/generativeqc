"""Audit automatic RAM backing without relabeling the original measurements.

The executed script supplied an abbreviated Git revision. Raw receipts remain
byte-for-byte intact. The separately verified archive/build binding resolves
only that abbreviation for the qualification input view; neither the measured
values nor independent oracle producer identities are changed.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

from reproduce import digest, profile_summary, unpack

from benchmarks import readme_defaults_report as reader
from benchmarks.readme_defaults_qualification import identity, qualify

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Recompute provenance, independent gates and paired endpoint summaries."""
    if not __debug__:
        raise SystemExit("Run evidence verification without -O.")
    packet = unpack(ROOT / "default.json.xz")
    assert packet["schema"] == "generativeqc.host-lr-default.v1"
    for name, raw in packet["support"].items():
        assert digest(raw) == packet["support_sha256"][name], name
    binding = json.loads(packet["support"]["source-binding.json"])
    assert binding["commit"].startswith(binding["abbreviation"])
    inventory = "".join(
        f"{name}:{value}\n"
        for name, value in sorted(binding["native_inventory"].items())
    )
    assert hashlib.sha256(inventory.encode()).hexdigest() == binding["native_source_identity"]
    assert binding["verified_archive_files"] > len(binding["native_inventory"])
    tests = {}
    for name, count in (("raw-host", 4), ("endpoint-host", 9), ("range-regression", 40)):
        cases = list(ET.fromstring(packet["support"][f"n4/{name}.xml"]).iter("testcase"))  # noqa: S314
        assert len(cases) == count
        assert not any(c.find(tag) is not None for c in cases for tag in ("failure", "error", "skipped"))
        tests[name] = count
    previous = ROOT.parent / "fitted-range-cold-20261024"
    oracles = reader.load_records(
        [ROOT.parent / "readme-final-20261022/oracles.json.xz"],
        schema="generativeqc.readme-defaults-oracle.v1",
    )
    oracles.append(json.loads(unpack(previous / "provenance.json.xz")["support"]["stage29-larger-oracle/oracle.json"]))
    reuse = json.loads(unpack(previous / "integration.json.xz")["support"]["stage37-endpoint-qualification.reference-reuse.json"])
    reader.SIZES = (*reader.SIZES, 108)
    records, gates = {}, {}
    for name, raw in packet["records"].items():
        assert digest(raw) == packet["record_sha256"][name]
        record = json.loads(raw)
        assert record["status"] == "complete"
        assert record["source_revision"] == binding["abbreviation"]
        assert record["native_build"]["library_sha256"] == binding["library_sha256"]
        assert record["native_build"]["source_identity"] == binding["native_source_identity"]
        assert record["host_eri_experiment"]["runner_sha256"] == binding["runner_files"]["benchmarks/host_eri_cache.py"]
        # This explicitly identified view expands a proven abbreviation. The
        # original receipt and its digest above remain the evidence authority.
        view = {**record, "source_revision": binding["commit"]}
        gate = qualify([view], oracles, reference_reuse=reuse)[0]
        assert gate["status"] == "qualified", (name, gate)
        gates[name] = {
            "raw_receipt_sha256": identity(record),
            "qualification_view_sha256": identity(view),
            "only_view_edit": {"source_revision": [binding["abbreviation"], binding["commit"]]},
            "qualification": gate,
        }
        records[name] = record
    summary = {}
    for size in (12, 96, 108):
        arms = {}
        for tier in ("gpu", "auto"):
            selected = [r for name, r in records.items() if name.startswith(f"default-{tier}-") and r["atoms"] == size]
            assert len(selected) == (2 if size == 108 else 1)
            assert all(not r["host_eri_experiment"]["trace"] for r in selected)
            arms[tier] = {
                phase: [next(s for s in r["samples"] if s["phase"] == phase)["seconds"] for r in selected]
                for phase in ("cold", "warm")
            }
            arms[tier]["cold_median"] = statistics.median(arms[tier]["cold"])
            arms[tier]["work"] = [[{k: s[k] for k in ("phase", "iterations", "fock_builds")} for s in r["samples"]] for r in selected]
        arms["cold_reduction_percent"] = 100 * (1 - arms["auto"]["cold_median"] / arms["gpu"]["cold_median"])
        summary[size] = arms
    profile = packet["profiles"]["default-profile.jsonl"]
    assert profile["status"] == "valid"
    print(json.dumps({"tests": tests, "endpoints": summary, "profile": profile_summary(profile["actions"]), "qualification": gates}, indent=2))


if __name__ == "__main__":
    main()
