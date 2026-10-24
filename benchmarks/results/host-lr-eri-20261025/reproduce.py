"""Recheck the opt-in host ERI screen from complete retained receipts.

Run with ``PYTHONPATH=python:. python <this-file>`` from the repository root.
No GPU or native library is loaded. Timing populations remain separate from
primers and profiled diagnostics, and legacy independent oracles retain their
original producer identities through the previously audited reuse manifest.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from benchmarks import readme_defaults_report as reader
from benchmarks.readme_defaults_qualification import qualify

ROOT = Path(__file__).resolve().parent


def unpack(path: Path) -> dict:
    """Read a losslessly compressed evidence packet."""
    return json.loads(lzma.decompress(path.read_bytes()))


def digest(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def distribution(values: list[float]) -> dict:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "sum": sum(values),
    }


def profile_summary(actions: list[dict]) -> dict:
    """Keep raw per-action counters and CUDA-event bucket times reproducible.

    Host replay regions include CPU staging/wait gaps and contraction. Their
    event durations must not be reported as isolated PCIe DMA durations.
    """
    if not actions:
        return {"actions": 0}
    counters = {}
    for key in (
        "canonical_values_visited",
        "recurrence_evaluations",
        "gpu_cached_values",
        "host_cached_values",
        "host_h2d_bytes",
        "host_device_staging_bytes",
    ):
        counters[key] = distribution([a["counters"].get(key, 0) for a in actions])
    times = {}
    for prefix in ("lr_direct_recompute/", "lr_gpu_replay/", "lr_host_replay/"):
        times[prefix] = distribution(
            [
                sum(r["gpu_ms"] for r in a["regions"] if r["name"].startswith(prefix))
                for a in actions
            ]
        )
    return {"actions": len(actions), "counters": counters, "region_ms": times}


def main() -> None:
    if not __debug__:
        raise SystemExit("Evidence verification requires assertions; do not use -O.")
    packet = unpack(ROOT / "measurements.json.xz")
    assert packet["schema"] == "generativeqc.host-lr-eri-screen.v1"
    support = packet["support"]
    for name, raw in support.items():
        assert digest(raw) == packet["support_sha256"][name], name
    manifest = json.loads(support["source-manifest.json"])
    assert manifest["kind"] == "git-tree"
    assert manifest["tree"] == packet["source_tree"]
    assert packet["library_sha256"] in support["library-sha256.txt"]
    launcher = json.loads(support["launcher-verification.json"])
    assert launcher["compile_commands"] > 0 and not launcher["uncached_commands"]
    tests = {}
    for name, expected in (
        ("raw-host", 4),
        ("endpoint-host", 7),
        ("range-regression", 40),
    ):
        # These are the hash-verified local JUnit receipts retained above.
        cases = list(ET.fromstring(support[f"n4/{name}.xml"]).iter("testcase"))  # noqa: S314
        assert len(cases) == expected, name
        assert not any(
            case.find(tag) is not None
            for case in cases
            for tag in ("failure", "error", "skipped")
        ), name
        tests[name] = len(cases)
    assert "ERROR SUMMARY: 0 errors" in support["n4/memcheck.log"]
    assert "LEAK SUMMARY: 0 bytes leaked" in support["n4/memcheck.log"]

    previous = ROOT.parent / "fitted-range-cold-20261024"
    oracles = reader.load_records(
        [ROOT.parent / "readme-final-20261022/oracles.json.xz"],
        schema="generativeqc.readme-defaults-oracle.v1",
    )
    provenance = unpack(previous / "provenance.json.xz")
    oracles.append(
        json.loads(provenance["support"]["stage29-larger-oracle/oracle.json"])
    )
    integration = unpack(previous / "integration.json.xz")
    reuse = json.loads(
        integration["support"]["stage37-endpoint-qualification.reference-reuse.json"]
    )
    reader.SIZES = (*reader.SIZES, 108)
    records, qualification = {}, {}
    for name, raw in packet["records"].items():
        assert digest(raw) == packet["record_sha256"][name], name
        record = json.loads(raw)
        assert record["status"] == "complete", name
        assert record["source_revision"] == packet["source_tree"], name
        assert record["host_eri_experiment"]["source_identity_kind"] == "tree", name
        assert record["native_build"]["library_sha256"] == packet["library_sha256"], (
            name
        )
        assert (
            record["host_eri_experiment"]["runner_sha256"]
            == manifest["files"]["benchmarks/host_eri_cache.py"]
        ), name
        assert [s["phase"] for s in record["samples"]] == ["cold", "warm"], name
        # Each tier has the same public method/arm identity, so qualify them
        # individually rather than merging duplicate populations.
        q = qualify([record], oracles, reference_reuse=reuse)[0]
        assert q["status"] == "qualified", (name, q)
        records[name], qualification[name] = record, q

    timed = {
        f"n5/{tier}-{size}.json"
        for size in (96, 108)
        for tier in ("direct", "gpu", "host")
    }
    assert set(packet["timed_records"]) == timed
    parity, screen = {}, {}
    for size in (96, 108):
        baseline = records[f"n5/direct-{size}.json"]
        for tier in ("direct", "gpu", "host"):
            name = f"n5/{tier}-{size}.json"
            record = records[name]
            for key in (
                "geometry_bohr",
                "orbital_basis",
                "basis_representation",
                "charge",
                "spin",
                "auxiliary_input_sha256",
                "slurm_job_id",
                "thread_controls",
                "protocol",
                "native_build",
            ):
                assert record.get(key) == baseline.get(key), (name, key)
            for key in ("packages", "host", "runtime"):
                assert record["environment"][key] == baseline["environment"][key], (
                    name,
                    key,
                )
            assert record["host_eri_experiment"]["trace"] is None, name
            checks = []
            for sample, reference in zip(
                record["samples"], baseline["samples"], strict=True
            ):
                assert sample["settings"] == reference["settings"], name
                delta_e = abs(sample["energy"] - reference["energy"])
                delta_f = float(
                    np.max(np.abs(np.array(sample["forces"]) - reference["forces"]))
                )
                assert delta_e <= 1e-8 and delta_f <= 1e-7, (name, delta_e, delta_f)
                checks.append(
                    {
                        "phase": sample["phase"],
                        "energy_abs": delta_e,
                        "force_max_abs": delta_f,
                    }
                )
            parity[name] = checks
            screen[name] = [
                {
                    key: s[key]
                    for key in ("phase", "seconds", "iterations", "fock_builds")
                }
                for s in record["samples"]
            ]

    profiles = {}
    profile_actions = {}
    for name, entry in packet["profiles"].items():
        raw = entry["lr_actions_jsonl"]
        assert digest(raw) == entry["lr_actions_sha256"], name
        actions = [json.loads(line) for line in raw.splitlines()]
        profile_actions[name] = actions
        assert all(
            a["valid"] and not a["cuda_error"] and not a["dropped_regions"]
            for a in actions
        ), name
        record = records[entry["endpoint_record"]]
        assert len(actions) == sum(s["fock_builds"] for s in record["samples"]), name
        scf = record["samples"][0]["precision"]["strict_stage_fock_builds"]
        cold_end = record["samples"][0]["fock_builds"]
        # Existing cached execution has one extra cold post-SCF Fock. Direct
        # does not. Frozen-density warm is a single strict Direct Fock in
        # these receipts; it must not be counted as a cached SCF replay.
        assert (
            cold_end - scf == record["samples"][0]["precision"]["post_scf_fock_builds"]
        ), name
        profiles[name] = {
            "cold_scf": profile_summary(actions[:scf]),
            "cold_post_scf": profile_summary(actions[scf:cold_end]),
            "warm": profile_summary(actions[cold_end:]),
        }

    # Compare the actual admitted GPU population within each diagnostic pair.
    # Automatic capacity fitting can differ between nodes, even with one model
    # of GPU, so no cross-node equality of residency is assumed.
    for size in (96, 108):
        gpu = profile_actions[f"n4/profile-gpu-{size}.jsonl"][:28]
        host = profile_actions[f"n4/profile-host-{size}.jsonl"][:28]
        for first, second in zip(gpu, host, strict=True):
            a, b = first["counters"], second["counters"]
            assert a["gpu_cached_values"] == b["gpu_cached_values"]
            assert a["canonical_values_visited"] == b["canonical_values_visited"]
            assert b["host_h2d_bytes"] == 8 * b["host_cached_values"]
            assert a["recurrence_evaluations"] == b["host_cached_values"]
            assert b["recurrence_evaluations"] == 0
    nsight = json.loads(support["n4/nsys-summary.json"])
    assert (
        nsight["h2d_bytes"]
        == profile_actions["n4/nsys-host-108.jsonl"][0]["counters"]["host_h2d_bytes"]
    )
    assert (
        0
        < nsight["h2d_resident_contraction_overlap_ns"]
        <= min(nsight["h2d_active_ns"], nsight["resident_contraction_active_ns"])
    )
    assert set(nsight["copy_stream_ids"]).isdisjoint(nsight["kernel_stream_ids"])
    print(
        json.dumps(
            {
                "tests": tests,
                "screen": screen,
                "cache_parity": parity,
                "qualification": qualification,
                "profiles": profiles,
                "nsight": nsight,
                "timing_scope": "single observations; primers/profiles excluded; no default promotion",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
