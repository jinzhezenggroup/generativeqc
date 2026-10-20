"""Replay the frozen default campaign without a GPU or a reference engine."""

from __future__ import annotations

import json
import lzma
from pathlib import Path

from benchmarks.readme_defaults import ARMS, METHODS, SIZES
from benchmarks.readme_defaults_report import reduce_record
from tools.generativeqc_validation.publication import validate_publication


def test_default_campaign_retains_complete_inventory_and_actual_work() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/readme-defaults-20261020"
    )
    manifest = json.loads((directory / "publication.json").read_bytes())
    files = {
        entry["path"]: (directory / entry["path"]).read_bytes()
        for entry in manifest["files"]
    }
    validate_publication(manifest, files)
    assert manifest["decision"]["status"] == "inconclusive"
    records = json.loads(lzma.decompress(files["receipts.json.xz"]))
    expected = {
        (arm, method, mode, size)
        for arm in ARMS
        for method in METHODS
        for mode in ("direct", "df")
        for size in SIZES
    }
    assert {(r["arm"], r["method"], r["mode"], r["atoms"]) for r in records} == expected
    assert len(records) == len(expected)
    assert sum(len(record["samples"]) for record in records) == 765
    for record in records:
        reduced = reduce_record(record)
        assert record["protocol"]["grid_overrides"] is False
        assert record["protocol"]["force_response_overrides"] is False
        assert record["protocol"]["moved_geometry"] is False
        if record["arm"] in ("standard", "fast") and record["mode"] == "df":
            assert reduced["status"] == "unsupported"
            assert reduced["cold"] is None and reduced["warm"] is None
        for sample in record["samples"]:
            if record["method"] != "hf":
                grid = sample["settings"]["xc_grid"]
                if record["arm"] in ("standard", "fast"):
                    assert grid["version"] == 4 and grid["radial_points"] == 3
                    assert grid["partition"] == "becke-treutler"
                    assert grid["pruning"] == "nwchem"
                else:
                    assert grid["level"] == 3
                    assert grid["angular_pruning"] == "nwchem_prune"
                    if record["method"] == "wb97m-v":
                        assert sample["settings"]["nonlocal_grid"]["level"] == 3
            if record["arm"] == "fast":
                report = sample["precision_validation"]
                assert report["solve_count"] == len(report["solves"])
                assert all(solve["error"] is None for solve in report["solves"])
    failed = next(
        r
        for r in records
        if r["arm"] == "gpu4pyscf-fast"
        and r["method"] == "hf"
        and r["mode"] == "direct"
        and r["atoms"] == 96
    )
    assert reduce_record(failed)["cold"] is not None
    assert reduce_record(failed)["warm"] is None
