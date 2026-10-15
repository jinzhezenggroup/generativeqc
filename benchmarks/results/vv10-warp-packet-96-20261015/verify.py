"""Audit historical 96-atom endpoint evidence without requiring completion."""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path


def verify() -> None:
    """Recompute independent gates, phase populations, work and partial outcomes."""
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
    shared = evidence.get("shared_files", {})
    if shared:
        campaign = root.parent / "vv10-warp-packet-20261015"
        original_summary = json.loads((campaign / "summary.json").read_text())
        original_bytes = (campaign / "evidence.json.xz").read_bytes()
        assert (
            hashlib.sha256(original_bytes).hexdigest()
            == original_summary["evidence"]["sha256"]
        )
        original = json.loads(lzma.decompress(original_bytes))
        assert set(shared) == {"deployment.json"}
        for name, receipt in shared.items():
            assert receipt["campaign"] == campaign.name and receipt["file"] == name
            item = original["files"][name]
            assert item["sha256"] == receipt["sha256"]
            evidence["files"][name] = item
    for name, item in evidence["files"].items():
        assert hashlib.sha256(item["text"].encode()).hexdigest() == item["sha256"]
        files[name] = item["text"]
    deployment = json.loads(files["deployment.json"])
    assert deployment["native_source_identity"] == summary["native_source_identity"]
    assert (
        deployment["runtime_files"]["libgenerativeqc.so"] == summary["library_sha256"]
    )
    origin = json.loads(files["oracle-origin.json"])
    phases = (("cold", 1), ("warm", 5), ("moved", 1), ("moved-warm", 5))
    expected_order = [
        (phase, repeat) for phase, count in phases for repeat in range(count)
    ]
    assert [point["arm"] for point in summary["points"]] == ["off", "on"]
    settings = []
    protocols = []
    observations = 0
    for point in summary["points"]:
        prefix = point["prefix"]
        native = json.loads(files[prefix + "native.json"])
        outcome = json.loads(files[prefix + "native.outcome"])
        assert outcome == point["outcome"]
        assert outcome["time_limit_seconds"] == 1800
        protocol = native["protocol"]
        protocols.append(protocol)
        assert protocol["atoms"] == 96 and protocol["aos"] == 768
        assert protocol["method"] == "WB97M-V/RKS"
        assert protocol["density_fitting"] is False
        assert protocol["repeats"] == 5
        assert (protocol["energy_gate"], protocol["force_gate"]) == (1e-8, 1e-7)
        assert tuple(
            protocol["grid"][key]
            for key in ("radial_points", "angular_polar", "angular_azimuth")
        ) == (48, 16, 32)
        assert native["native_build"]["library_sha256"] == summary["library_sha256"]
        assert (
            native["native_build"]["probe"]["source_identity"]
            == summary["native_source_identity"]
        )
        assert native["native_experiment"]["precision_policy"] == "fp64"
        schedule = dict(native["native_schedule_settings"])
        assert schedule.pop("GENERATIVEQC_VV10_PAIR_SCHEDULE") == (
            "parallel" if point["arm"] == "off" else "parallel-packet"
        )
        settings.append(schedule)
        assert native["reference_sha256"] is None
        oracles = {}
        for binding in native["accuracy_reference_files"]:
            name = prefix + binding["file"]
            assert binding["sha256"] == evidence["files"][name]["sha256"]
            assert binding["sha256"] == origin["receipts"][binding["file"]]
            receipt = json.loads(files[name])
            assert receipt["protocol"] == protocol
            assert receipt["basis_sha256"] == native["basis_file_sha256"]
            assert receipt["status"] == "complete" and receipt["converged"] is True
            assert (
                receipt["initial_density"]
                == "GPU4PySCF default atomic guess; no native density, orbitals, energy or force input"
            )
            assert receipt["scheduler"]["SLURM_JOB_ID"]
            convergence = receipt["convergence"]
            residuals = convergence["final_residuals"]
            assert convergence["converged"] is True
            assert 0 < convergence["iterations"] <= protocol["max_iterations"]
            assert (
                abs(residuals["energy_change_hartree"]) <= protocol["energy_tolerance"]
            )
            assert 0 <= residuals["density_rms"] <= protocol["density_tolerance"]
            assert (
                0
                <= residuals["orbital_gradient_norm"]
                <= protocol["reference_gradient_tolerance"]
            )
            assert receipt["geometry"] not in oracles
            oracles[receipt["geometry"]] = receipt
        assert set(oracles) == {0, 1}
        rows = native["records"]
        assert len(rows) == point["completed_calls"] <= point["planned_calls"] == 12
        assert [(row["phase"], row["repeat"]) for row in rows] == expected_order[
            : len(rows)
        ]
        for row in rows:
            assert (
                row["converged"] is True and row["gate"] is True and row["status"] == 0
            )
            assert 0 < row["complete_seconds"] < 1800
            oracle = oracles[row["geometry"]]
            assert len(row["forces"]) == len(oracle["forces"]) == 96
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
        for phase, expected in phases:
            samples = [row for row in rows if row["phase"] == phase]
            assert point["phases"][phase] == {
                "completed_calls": len(samples),
                "planned_calls": expected,
                "complete": len(samples) == expected,
                "median_complete_seconds": statistics.median(
                    row["complete_seconds"] for row in samples
                )
                if len(samples) == expected
                else None,
                "recorded_complete_seconds": [
                    row["complete_seconds"] for row in samples
                ],
                "iterations": sum(row["iterations"] for row in samples),
                "fock_builds": sum(row["fock_builds"] for row in samples),
                "scf_pair_sweeps": sum(
                    row["native_ks_diagnostic"]["nonlocal_work"]["pair_sweeps"]
                    for row in samples
                ),
                "scf_pair_cache_hits": sum(
                    row["native_ks_diagnostic"]["nonlocal_work"]["pair_cache_hits"]
                    for row in samples
                ),
            }
        assert point["energy_error_max"] == max(
            (row["energy_error"] for row in rows), default=None
        )
        assert point["force_error_max"] == max(
            (row["force_error"] for row in rows), default=None
        )
        assert point["all_recorded_gates_pass"] == all(
            row["gate"] is True for row in rows
        )
        assert (
            point["journal_status"] == native["status"]
            and point["stage"] == native["stage"]
        )
        assert point["status"] == (
            "timeout" if outcome["exit_code"] == 124 else native["status"]
        )
        assert point["complete_population"] == (
            len(rows) == 12 and outcome["exit_code"] == 0
        )
        if outcome["exit_code"] == 124:
            assert len(rows) < 12 and native["stage"] != "complete"
        observations += len(rows)
    assert protocols[0] == protocols[1]
    assert settings[0] == settings[1]
    assert summary["default_enabled"] is False
    print(
        f"Verified {observations}/24 completed 96-atom calls, independent gates and original-limit outcomes"
    )


if __name__ == "__main__":
    verify()
