"""Recheck exact provenance and every retained matched endpoint/error gate."""

from __future__ import annotations

import hashlib
import json
import lzma
import sys
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from benchmarks.readme_pbe0 import SCHEMA
from tools.render_omol25_benchmarks import validate


def main() -> None:
    """Verify all 144 native and 72 freshly timed independent reference calls."""
    directory = Path(__file__).resolve().parent
    storage = json.loads((directory / "storage.json").read_text())
    compressed = (directory / "campaign.json.xz").read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == storage["sha256"]
    raw = lzma.decompress(compressed)
    assert hashlib.sha256(raw).hexdigest() == storage["uncompressed_sha256"]
    members = json.loads(raw)
    assert set(members) == set(storage["members"])
    for name, value in members.items():
        assert hashlib.sha256(value.encode()).hexdigest() == storage["members"][name]
    summary = json.loads(members["receipts/tiles-summary.json"])
    expected = {("cold", 0, 0), ("moved", 1, 0)} | {
        (phase, geometry, repeat)
        for phase, geometry in (("warm", 0), ("moved-warm", 1))
        for repeat in range(5)
    }
    cases = {case["atoms"]: case for case in summary["cases"]}
    assert set(cases) == {3, 6, 12, 24, 48, 96}
    native_samples = reference_samples = 0
    for atoms, case in sorted(cases.items()):
        regime = "large" if atoms == 96 else "default"
        prefix = f"endpoints-{regime}/{atoms}/"
        reference_text = members[prefix + "reference.json"]
        reference = json.loads(reference_text)
        reference_campaign = json.loads(members[prefix + "reference.campaign.json"])
        modes = ("256", "1024" if atoms == 96 else "auto")
        for mode in ("reference", *modes):
            stem = "reference" if mode == "reference" else "tile-" + mode
            raw_record = members[prefix + stem + ".json"]
            record = json.loads(raw_record)
            campaign = json.loads(members[prefix + stem + ".campaign.json"])
            retained = case["variants"][mode]
            assert record["status"] == "measured" and record["stage"] == "complete"
            assert hashlib.sha256(raw_record.encode()).hexdigest() == retained["sha256"]
            validate(record, reference, schema=SCHEMA)
            rows = record["records"]
            assert len(rows) == 12
            assert {
                (row["phase"], row["geometry"], row["repeat"]) for row in rows
            } == expected
            for row in rows:
                assert row["converged"] and row["status"] == 0 and row["gate"]
            errors = {"energy": 0.0, "force": 0.0}
            for row in rows:
                for oracle in reference["records"]:
                    if oracle["geometry"] != row["geometry"]:
                        continue
                    errors["energy"] = max(
                        errors["energy"], abs(row["energy"] - oracle["energy"])
                    )
                    errors["force"] = max(
                        errors["force"],
                        float(
                            np.max(np.abs(np.asarray(row["forces"]) - oracle["forces"]))
                        ),
                    )
            assert errors["energy"] <= 1e-8 and errors["force"] <= 1e-7
            assert (
                errors == retained["maximum_errors_all_same_geometry_reference_pairs"]
            )
            for key in ("source_identity", "library_sha256"):
                assert campaign[key] == retained[key] == storage[key]
            for key in (
                "job",
                "cuda_visible_devices",
                "harness_sha256",
                "force_budget",
            ):
                assert campaign[key] == reference_campaign[key]
            assert campaign["force_budget"] == regime
            for phase in ("cold", "warm", "moved", "moved-warm"):
                selected = [row for row in rows if row["phase"] == phase]
                assert (
                    median(row["complete_seconds"] for row in selected)
                    == retained["phase_medians_seconds"][phase]
                )
                assert [row["iterations"] for row in selected] == retained[
                    "phase_iterations"
                ][phase]
            if mode == "reference":
                reference_samples += len(rows)
                assert [row.get("reference_xc_backend") for row in rows] == retained[
                    "xc_backends"
                ]
                continue
            native_samples += len(rows)
            assert (
                record["reference_sha256"]
                == hashlib.sha256(reference_text.encode()).hexdigest()
            )
            assert json.loads(members[prefix + stem + ".outcome"])["exit_code"] == 0
            assert (
                record["native_build"]["library_sha256"] == campaign["library_sha256"]
            )
            assert (
                record["native_build"]["probe"]["source_identity"]
                == campaign["source_identity"]
            )
            points = atoms * 48 * 16 * 32
            grid = retained["grid_work_plan"]
            assert grid["grid_points"] == points
            assert (
                grid["grid_pair_visits"] == (1 + 2 * points) * atoms * (atoms - 1) // 2
            )
            assert all(
                row["native_force_components"]["grid_work_plan"] == grid for row in rows
            )
            assert retained["dense_G_M2"] == points * reference["protocol"]["aos"] ** 2
            assert (
                rows[0]["native_force_components"]["resource_bounds"]
                == retained["resources"]
            )
        before, after = (case["variants"][mode] for mode in modes)
        reduction = (
            1
            - after["phase_medians_seconds"]["warm"]
            / before["phase_medians_seconds"]["warm"]
        )
        assert reduction == case["warm_endpoint_reduction"]
        print(f"water{atoms}: warm reduction {100 * reduction:.3f}% ({regime} caps)")
    assert (
        native_samples
        == storage["native_samples"]
        == summary["native_endpoints_checked"]
        == 144
    )
    assert (
        reference_samples
        == storage["reference_samples"]
        == summary["reference_endpoints_checked"]
        == 72
    )
    print("PASS: all source/member identities, complete samples and numerical gates")


if __name__ == "__main__":
    main()
