"""Opt-in audit of retained range experiments and default publication inputs.

Recompute numerical errors from complete retained force vectors and independent
geometry oracles. This audits historical evidence, not a newer native build.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import statistics
from pathlib import Path


def verify() -> None:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    for name, identity in manifest["files"].items():
        raw = (root / name).read_bytes()
        assert len(raw) == identity["bytes"], name
        assert hashlib.sha256(raw).hexdigest() == identity["sha256"], name
    evidence = json.loads(lzma.decompress((root / "campaigns.json.xz").read_bytes()))
    files = {}
    for name, receipt in evidence["files"].items():
        raw = receipt["text"].encode()
        assert len(raw) == receipt["bytes"], name
        assert hashlib.sha256(raw).hexdigest() == receipt["sha256"], name
        files[name] = json.loads(raw)
    phases = (("cold", 1), ("warm", 5), ("moved", 1), ("moved-warm", 5))
    order = [(phase, repeat) for phase, count in phases for repeat in range(count)]
    accepted = complete = rejected = 0
    plotted = json.loads(
        lzma.decompress(
            (root.parent / "readme-dft-20261013/summary.json.xz").read_bytes()
        )
    )
    for point in evidence["points"]:
        folder = f"{point['campaign']}/none/wb97m-v/{point['atoms']}/"
        native = files[folder + "native.json"]
        outcome = files[folder + "native.outcome"]
        assert outcome == point["outcome"]
        assert outcome["time_limit_seconds"] == 1800
        assert (
            point["raw_sha256"] == evidence["files"][folder + "native.json"]["sha256"]
        )
        protocol = native["protocol"]
        grid = protocol["grid"]
        assert (
            grid["radial_points"],
            grid["angular_polar"],
            grid["angular_azimuth"],
        ) == (48, 16, 32)
        assert protocol["max_iterations"] == 100
        assert (protocol["energy_gate"], protocol["force_gate"]) == (1e-8, 1e-7)
        source = manifest["campaign_sources"][point["campaign"]]
        assert (
            native["native_build"]["probe"]["source_identity"]
            == source["native_source_identity"]
        )
        assert native["native_build"]["library_sha256"] == source["library_sha256"]
        reference = files[folder + "reference.json"]
        assert reference["protocol"] == protocol
        assert (
            native["reference_sha256"]
            == evidence["files"][folder + "reference.json"]["sha256"]
        )
        oracles = {
            row["geometry"]: row
            for row in reference["records"]
            if row["phase"] in ("cold", "moved") and row.get("gate") is True
        }
        for name in ("accuracy-reference-cold.json", "accuracy-reference-moved.json"):
            if folder + name in files:
                oracle = files[folder + name]
                assert oracle["protocol"] == protocol and oracle["converged"]
                oracles.setdefault(oracle["geometry"], oracle)
        rows = native["records"]
        assert [(row["phase"], row["repeat"]) for row in rows] == order[: len(rows)]
        assert len(rows) == point["returned_calls"]
        population_complete = native["stage"] == "complete"
        assert (
            population_complete
            == point["complete_population"]
            == (outcome["exit_code"] == 0)
        )
        if population_complete:
            assert len(rows) == 12
            complete += 1
        for row in rows:
            # The first inexact-inner launch reached SCF but failed to compile
            # forces. Preserve that endpoint failure without giving it a timing
            # median or an independent E/F pass.
            if row["gate"] is not True:
                assert point["campaign"] == "vv10-inexact-inner-launch-failure"
                assert not population_complete and outcome["exit_code"] == 1
                assert row["status"] == 5 and row["forces"] is None
                assert "no usable compiler cache launcher" in row["detail"]
                rejected += 1
                continue
            assert row["converged"] and row["status"] == 0 and row["gate"] is True
            assert row["iterations"] <= 100
            assert row["complete_seconds"] >= row["seconds"] > 0
            oracle = oracles[row["geometry"]]
            assert len(row["forces"]) == len(oracle["forces"]) == point["atoms"]
            force_error = 0.0
            for candidate, target in zip(row["forces"], oracle["forces"], strict=True):
                assert len(candidate) == len(target) == 3
                for actual, expected in zip(candidate, target, strict=True):
                    assert math.isfinite(actual) and math.isfinite(expected)
                    force_error = max(force_error, abs(actual - expected))
            energy_error = abs(row["energy"] - oracle["energy"])
            assert (
                math.isfinite(energy_error)
                and energy_error <= 1e-8
                and force_error <= 1e-7
            )
            assert (
                energy_error == row["energy_error"]
                and force_error == row["force_error"]
            )
            accepted += 1
        medians = {}
        for phase, count in phases:
            phase_rows = [
                row for row in rows if row["phase"] == phase and row["gate"] is True
            ]
            if len(phase_rows) == count:
                medians[phase] = statistics.median(
                    row["complete_seconds"] for row in phase_rows
                )
        assert medians == point["phase_medians"]
        if point["campaign"] == manifest["default_campaign"]:
            assert population_complete
            assert set(native["native_schedule_settings"]) == {
                "GENERATIVEQC_STATIONARY_CACHE"
            }
            assert native["native_experiment"]["preliminary_provider"] == "none"
            plotted_point = next(
                p for p in plotted["methods"]["wb97m-v"] if p["atoms"] == point["atoms"]
            )
            assert (
                plotted_point["engines"]["native"]["raw_sha256"] == point["raw_sha256"]
            )
    # The retained failed default 96-atom population stays visible and cannot
    # be replaced by a faster explicit-mask or convergence experiment.
    large = next(p for p in plotted["methods"]["wb97m-v"] if p["atoms"] == 96)
    assert (
        large["engines"]["native"]["raw_sha256"]
        == manifest["unchanged_default_96_sha256"]
    )
    print(
        f"Verified {complete} complete populations and {accepted} independent E/F calls; "
        f"{rejected} failed endpoints and default/experiment boundaries retained."
    )


if __name__ == "__main__":
    verify()
