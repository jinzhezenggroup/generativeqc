"""Recheck every retained endpoint against the unchanged independent references."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from benchmarks.readme_pbe0 import SCHEMA
from tools.render_omol25_benchmarks import validate


def main() -> None:
    """Validate byte identities, all-repeat gates, and paired timing provenance."""
    directory = Path(__file__).resolve().parent
    storage = json.loads((directory / "storage.json").read_text())
    summary = json.loads(
        gzip.decompress((directory / "endpoints-summary.json.gz").read_bytes())
    )
    by_atoms = {point["atoms"]: point for point in summary["results"]}
    campaigns = {}
    samples = 0
    for entry in storage["files"]:
        compressed = (directory / entry["path"]).read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == entry["sha256"]
        raw = gzip.decompress(compressed)
        assert hashlib.sha256(raw).hexdigest() == entry["uncompressed_sha256"]
        if "atoms" not in entry:
            continue
        atoms, variant = entry["atoms"], entry["variant"]
        record = json.loads(raw)
        reference_path = (
            directory / storage["reference_root"] / f"water{atoms}-reference.json.gz"
        )
        reference_bytes = gzip.decompress(reference_path.read_bytes())
        reference = json.loads(reference_bytes)
        assert record["reference_sha256"] == hashlib.sha256(reference_bytes).hexdigest()
        assert record["stage"] == "complete" and record["status"] == "measured"
        assert entry["outcome"]["exit_code"] == 0
        validate(reference, reference, schema=SCHEMA)
        validate(record, reference, schema=SCHEMA)
        campaign = entry["campaign"]
        assert record["environment"]["git"]["commit"] == campaign["base_commit"]
        assert record["environment"]["git"]["dirty"] == (variant == "candidate")
        assert record["native_build"]["library_sha256"] == campaign["library_sha256"]
        assert (
            record["native_build"]["probe"]["source_identity"]
            == campaign["source_identity"]
        )
        assert campaign["bounded_schwarz_schedule"] == (
            "0" if variant == "baseline" else "1"
        )
        retained = by_atoms[atoms]["variants"][variant]
        assert retained["provenance"] == campaign
        assert retained["native_sha256"] == entry["uncompressed_sha256"]
        rows = record["records"]
        assert len(rows) == 12 and all(row["converged"] for row in rows)
        for phase in ("cold", "warm", "moved", "moved-warm"):
            selected = [row for row in rows if row["phase"] == phase]
            times = [row["complete_seconds"] for row in selected]
            assert times == retained["phases"][phase]["complete_seconds"]
            assert median(times) == retained["phases"][phase]["median"]
            assert [row["iterations"] for row in selected] == retained["phases"][phase][
                "iterations"
            ]
        assert (
            max(row["energy_error"] for row in rows)
            == retained["energy_gate_max_error"]
        )
        assert (
            max(row["force_error"] for row in rows) == retained["force_gate_max_error"]
        )
        assert (atoms, variant) not in campaigns
        campaigns[atoms, variant] = campaign
        samples += len(rows)
    assert set(campaigns) == {
        (atoms, variant)
        for atoms in (3, 6, 12, 24, 48, 96)
        for variant in ("baseline", "candidate")
    }
    assert samples == summary["native_samples"] == 144
    for atoms, point in sorted(by_atoms.items()):
        for key in ("host", "job", "cuda_visible_devices", "base_commit"):
            assert (
                campaigns[atoms, "baseline"][key] == campaigns[atoms, "candidate"][key]
            )
        before = point["variants"]["baseline"]["phases"]
        after = point["variants"]["candidate"]["phases"]
        for phase, key in (
            ("warm", "warm_time_reduction"),
            ("moved-warm", "moved_warm_time_reduction"),
        ):
            assert 1 - after[phase]["median"] / before[phase]["median"] == point[key]
        print(
            f"{atoms:2} atoms: {before['warm']['median']:.6f} -> {after['warm']['median']:.6f} s"
        )
    print(
        f"PASS: {samples} complete endpoints, all-repeat independent energy/force gates"
    )


if __name__ == "__main__":
    main()
