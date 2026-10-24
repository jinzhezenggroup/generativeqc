"""Audit the historical source-matched automatic VV10 packet qualification."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path

from defusedxml import ElementTree as ET


def verify() -> None:
    """Recompute gates and complete populations without pinning current sources."""
    root = Path(__file__).resolve().parent
    summary = json.loads((root / "summary.json").read_text())
    metadata = summary["evidence"]
    compressed = (root / metadata["path"]).read_bytes()
    assert len(compressed) == metadata["bytes"]
    assert hashlib.sha256(compressed).hexdigest() == metadata["sha256"]
    raw = lzma.decompress(compressed)
    assert hashlib.sha256(raw).hexdigest() == metadata["uncompressed_sha256"]
    evidence = json.loads(raw)
    assert evidence["summary"] == {
        key: value for key, value in summary.items() if key != "evidence"
    }
    files = {}
    for name, item in evidence["files"].items():
        assert hashlib.sha256(item["text"].encode()).hexdigest() == item["sha256"]
        files[name] = item["text"]
    for name, expected in summary["passed_tests"].items():
        suites = ET.fromstring(files[name]).findall(".//testsuite")
        assert suites
        assert all(int(suite.attrib["errors"]) == 0 for suite in suites)
        assert all(int(suite.attrib["failures"]) == 0 for suite in suites)
        assert expected == sum(
            int(suite.attrib["tests"]) - int(suite.attrib["skipped"])
            for suite in suites
        )
    assert summary["passed_tests"]["admission-oracle.xml"] == 752
    assert summary["passed_tests"]["default-memcheck.xml"] == 6
    assert summary["passed_tests"]["default-synccheck.xml"] == 6
    assert summary["passed_tests"]["molecular-oracle.xml"] == 6
    deployment = json.loads(files["deployment.json"])
    assert deployment["native_source_identity"] == summary["native_source_identity"]
    assert (
        deployment["runtime_files"]["libgenerativeqc.so"] == summary["library_sha256"]
    )
    expected_order = [
        (phase, repeat)
        for phase, count in (("cold", 1), ("warm", 5), ("moved", 1), ("moved-warm", 5))
        for repeat in range(count)
    ]
    calls = 0
    for point in summary["points"]:
        prefix = point["prefix"]
        native = json.loads(files[prefix + "native.json"])
        reference = json.loads(files[prefix + "reference.json"])
        assert json.loads(files[prefix + "native.outcome"]) == {
            "exit_code": 0,
            "time_limit_seconds": 1800,
        }
        assert native["stage"] == reference["stage"] == "complete"
        assert native["protocol"] == reference["protocol"]
        protocol = native["protocol"]
        assert protocol["atoms"] == point["atoms"]
        assert protocol["repeats"] == 5
        assert protocol["density_fitting"] is False
        assert (protocol["energy_gate"], protocol["force_gate"]) == (1e-8, 1e-7)
        assert tuple(
            protocol["grid"][name]
            for name in ("radial_points", "angular_polar", "angular_azimuth")
        ) == (48, 16, 32)
        assert (
            "GENERATIVEQC_VV10_PAIR_SCHEDULE" not in native["native_schedule_settings"]
        )
        assert native["native_build"]["library_sha256"] == summary["library_sha256"]
        assert (
            native["native_build"]["probe"]["source_identity"]
            == summary["native_source_identity"]
        )
        assert (
            native["reference_sha256"]
            == evidence["files"][prefix + "reference.json"]["sha256"]
        )
        oracles = {
            row["geometry"]: row
            for row in reference["records"]
            if row["phase"] in ("cold", "moved") and row["converged"]
        }
        rows = native["records"]
        assert [(row["phase"], row["repeat"]) for row in rows] == expected_order
        for row in rows:
            assert (
                row["converged"] is True and row["gate"] is True and row["status"] == 0
            )
            assert 0 < row["complete_seconds"] < 1800
            oracle = oracles[row["geometry"]]
            assert len(row["forces"]) == len(oracle["forces"]) == point["atoms"]
            energy_error = abs(row["energy"] - oracle["energy"])
            force_error = max(
                abs(actual - expected)
                for actual_vector, expected_vector in zip(
                    row["forces"], oracle["forces"], strict=True
                )
                for actual, expected in zip(actual_vector, expected_vector, strict=True)
            )
            assert energy_error <= 1e-8 and force_error <= 1e-7
            assert math.isclose(energy_error, row["energy_error"], abs_tol=1e-15)
            assert math.isclose(force_error, row["force_error"], abs_tol=1e-15)
        assert point["phases_seconds"] == {
            phase: statistics.median(
                row["complete_seconds"] for row in rows if row["phase"] == phase
            )
            for phase in ("cold", "warm", "moved", "moved-warm")
        }
        assert point["energy_error_max"] == max(row["energy_error"] for row in rows)
        assert point["force_error_max"] == max(row["force_error"] for row in rows)
        assert point["work_by_phase"] == {
            phase: {
                "calls": sum(row["phase"] == phase for row in rows),
                "iterations": sum(
                    row["iterations"] for row in rows if row["phase"] == phase
                ),
                "fock_builds": sum(
                    row["fock_builds"] for row in rows if row["phase"] == phase
                ),
                "scf_pair_sweeps": sum(
                    row["native_ks_diagnostic"]["nonlocal_work"]["pair_sweeps"]
                    for row in rows
                    if row["phase"] == phase
                ),
            }
            for phase in ("cold", "warm", "moved", "moved-warm")
        }
        calls += len(rows)
    assert calls == 24 and summary["default_enabled"] is True
    print(
        "Verified automatic selection, independent gates and all 24 complete 3/12-atom calls"
    )


if __name__ == "__main__":
    verify()
