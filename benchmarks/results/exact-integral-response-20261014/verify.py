"""Opt-in historical evidence audit; never a current-source fingerprint test."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path

import defusedxml.ElementTree as ET


def verify() -> None:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    stored = (root / manifest["evidence"]["path"]).read_bytes()
    assert len(stored) == manifest["evidence"]["bytes"]
    assert hashlib.sha256(stored).hexdigest() == manifest["evidence"]["sha256"]
    raw = lzma.decompress(stored)
    assert (
        hashlib.sha256(raw).hexdigest() == manifest["evidence"]["uncompressed_sha256"]
    )
    evidence = json.loads(raw)
    files = {}
    for name, item in evidence["files"].items():
        data = item["text"].encode()
        assert len(data) == item["bytes"], name
        assert hashlib.sha256(data).hexdigest() == item["sha256"], name
        files[name] = item["text"]
    phases = (("cold", 1), ("warm", 5), ("moved", 1), ("moved-warm", 5))
    order = [(phase, repeat) for phase, count in phases for repeat in range(count)]
    accepted = 0
    assert evidence["points"] == manifest["points"]
    for point in evidence["points"]:
        prefix = point["prefix"]
        native = json.loads(files[prefix + "native.json"])
        reference = json.loads(files[prefix + "reference.json"])
        outcome = json.loads(files[prefix + "native.outcome"])
        assert outcome == {"exit_code": 0, "time_limit_seconds": 1800}
        assert native["status"] == "measured" and native["stage"] == "complete"
        assert native["protocol"] == reference["protocol"]
        assert (
            native["reference_sha256"]
            == evidence["files"][prefix + "reference.json"]["sha256"]
        )
        protocol = native["protocol"]
        assert protocol["max_iterations"] == 100
        assert (protocol["energy_gate"], protocol["force_gate"]) == (1e-8, 1e-7)
        assert tuple(
            protocol["grid"][key]
            for key in ("radial_points", "angular_polar", "angular_azimuth")
        ) == (48, 16, 32)
        assert native["native_build"]["library_sha256"] == manifest["library_sha256"]
        assert (
            native["native_build"]["probe"]["source_identity"]
            == manifest["native_source_identity"]
        )
        assert native["native_schedule_settings"][
            "GENERATIVEQC_KS_INTEGRAL_RESPONSE_REUSE"
        ] == ("1" if point["arm"] == "on" else "0")
        oracles = {
            row["geometry"]: row
            for row in reference["records"]
            if row["phase"] in ("cold", "moved") and row["converged"]
        }
        for filename in (
            "accuracy-reference-cold.json",
            "accuracy-reference-moved.json",
        ):
            if prefix + filename in files:
                oracle = json.loads(files[prefix + filename])
                assert oracle["converged"] and oracle["protocol"] == protocol
                oracles.setdefault(oracle["geometry"], oracle)
        rows = native["records"]
        assert [(row["phase"], row["repeat"]) for row in rows] == order
        energy_errors, force_errors = [], []
        for row in rows:
            assert row["converged"] and row["status"] == 0 and row["gate"] is True
            oracle = oracles[row["geometry"]]
            assert len(row["forces"]) == len(oracle["forces"]) == point["atoms"]
            assert all(len(vector) == 3 for vector in row["forces"])
            assert all(
                math.isfinite(value) for vector in row["forces"] for value in vector
            )
            energy_error = abs(row["energy"] - oracle["energy"])
            force_error = max(
                abs(a - b)
                for va, vb in zip(row["forces"], oracle["forces"], strict=True)
                for a, b in zip(va, vb, strict=True)
            )
            assert energy_error <= 1e-8 and force_error <= 1e-7
            assert math.isclose(energy_error, row["energy_error"], abs_tol=1e-15)
            assert math.isclose(force_error, row["force_error"], abs_tol=1e-15)
            energy_errors.append(energy_error)
            force_errors.append(force_error)
            accepted += 1
        assert max(energy_errors) == point["energy_error_max"]
        assert max(force_errors) == point["force_error_max"]
        assert {
            phase: statistics.median(
                row["complete_seconds"] for row in rows if row["phase"] == phase
            )
            for phase, _ in phases
        } == point["phases_seconds"]
    assert accepted == manifest["accepted_calls"] == 72
    junit = {
        "qualification/device-original/device-911.xml": (9, 1),
        "qualification/device-final/cache-912.xml": (2, 0),
        "qualification/sanitizers/memcheck-8445.xml": (3, 0),
        "qualification/sanitizers/synccheck-8446.xml": (3, 0),
    }
    for name, (tests, failures) in junit.items():
        cases = ET.fromstring(files[name]).findall(".//testcase")
        assert len(cases) == tests
        assert sum(case.find("failure") is not None for case in cases) == failures
        assert all(
            case.find("skipped") is None and case.find("error") is None
            for case in cases
        )
    for name in ("response-memcheck-n1.log", "response-synccheck-n1.log"):
        assert "ERROR SUMMARY: 0 errors" in files["qualification/" + name]
    assert (
        "Driver/library version mismatch"
        in files["qualification/response-device-n2.log"]
    )
    print(
        f"Verified {accepted} complete E/F calls, six populations, and retained qualification outcomes"
    )


if __name__ == "__main__":
    verify()
