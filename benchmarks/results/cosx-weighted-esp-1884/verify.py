"""Verify weighted ESP endpoint samples, complete work, and resource receipts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
MASKS = (0, 4, 3, 7)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def effective_tile(record: dict) -> int:
    """Validate plan metadata, deriving it for historical pre-fix receipts."""
    points, requested = record["points"], record["tile"]
    assert type(points) is int and points > 0
    assert type(requested) is int and 0 < requested <= 4096
    expected = min(points, requested)
    actual = record.get("effective_tile", expected)
    assert type(actual) is int and actual == expected
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    provenance = json.loads((HERE / "provenance.json").read_text())
    records = []
    for name, sha in provenance["data_sha256"].items():
        assert digest(HERE / name) == sha, name
        records.extend(json.loads(gzip.decompress((HERE / name).read_bytes())))
    groups: dict[tuple, dict] = {}
    for record in records:
        assert record["schema"] == "cosx-weighted-endpoint-v1"
        n, points, tile, mask = (record[k] for k in ("nao", "points", "tile", "mask"))
        plan_tile = effective_tile(record)
        assert (record["atoms"], n) in ((48, 384), (96, 768))
        assert record["grid"] == [3, 3, 6]
        assert points == record["atoms"] * math.prod(record["grid"])
        assert mask in MASKS and record["geometry"] in (0, 1)
        key = (n, tuple(record["grid"]), tile, record["geometry"])
        group = groups.setdefault(key, {})
        assert mask not in group
        group[mask] = record
        samples = record["samples"]
        assert len(samples) == 6
        for sample in samples:
            assert math.isfinite(sample["energy"])
            assert math.isfinite(sample["seconds"]) and sample["seconds"] > 0
            # Absolute receipt gates are tighter than the runtime relative gate.
            assert all(math.isfinite(x) and 0 <= x < 1e-9 for x in sample["errors"])
        assert all(math.isfinite(x) and 0 <= x < 1e-9 for x in record["oracle_errors"])
        assert record["provider_allowance"] == (96 << 20 if mask else 0)
        assert 0 <= record["retained_provider_bytes"] <= record["provider_allowance"]
        assert record["host_reservation"] == 160 << 10
        for field in ("geometry_prepare_s", "prepare_s", "provider_prepare_s"):
            assert math.isfinite(record[field]) and record[field] >= 0
        assert record["runtime_version"] == provenance["runtime_version"]
        if mask:
            assert record["provider_version"] == provenance["provider_version"]
        assert record["compute_capability"] == "12.0"
        assert len(record["sites"]) == 6
        for slot, site in enumerate(record["sites"]):
            full = slot < 2 or slot == 4
            count = points // plan_tile if full else int(points % plan_tile != 0)
            # A missing tail prepares the full descriptor but executes it zero times.
            extent = plan_tile if full or not points % plan_tile else points % plan_tile
            operation = slot % 2 if slot < 4 else 2
            library = bool(mask & (1 << operation))
            calls = 6 * count
            assert site["calls"] == calls
            assert site["summands"] == calls * extent * n * n
            assert site["scaled_elements"] == (calls * extent * n if slot >= 4 else 0)
            assert site["publication_passes"] == (calls if slot >= 4 and library else 0)
            assert (site["m"], site["n"], site["k"]) == (
                (n, 1, n)
                if slot >= 4
                else (extent, n, n)
                if slot % 2 == 0
                else (n, n, extent)
            )
            assert site["provider"] == ("cublas" if library else "generated.cuda")
            if slot >= 4:
                assert site["algorithm"] == (
                    "batch-scaled-contraction-and-publication"
                    if library
                    else "batch-scaled-fused"
                )
            offers = site["offers"]
            assert {offer["provider"] for offer in offers} >= {
                "generated.cuda",
                "cublas",
            }
            assert offers[site["selected"]]["identity"] == site["candidate"]
            for offer in offers:
                if offer["provider"] not in ("generated.cuda", "cublas"):
                    assert offer["rejection"]
            for identity in ("candidate", "scientific", "semantic", "precision"):
                assert len(site[identity]) == 64 and int(site[identity], 16) >= 0
    summary = []
    for key, group in sorted(groups.items()):
        assert set(group) == set(MASKS)
        baseline = group[0]
        warm = {
            mask: statistics.median(s["seconds"] for s in group[mask]["samples"][1:])
            for mask in MASKS
        }
        for record in group.values():
            assert (
                record["device_bytes"] - record["provider_allowance"]
                == baseline["device_bytes"]
            )
            for a, b in zip(record["sites"], baseline["sites"], strict=True):
                for field in (
                    "scientific",
                    "semantic",
                    "precision",
                    "calls",
                    "summands",
                    "scaled_elements",
                    "m",
                    "n",
                    "k",
                ):
                    assert a[field] == b[field]
        summary.append(
            {
                "nao": key[0],
                "grid": key[1],
                "tile": key[2],
                "geometry": key[3],
                "points": baseline["points"],
                "warm_median_s_masks_0_4_3_7": [warm[mask] for mask in MASKS],
                "time_change_percent_masks_0_4_3_7": [
                    100 * (warm[m] / warm[0] - 1) for m in MASKS
                ],
                "esp_time_change_percent_on_generated_projection_update": 100
                * (warm[4] / warm[0] - 1),
                "esp_time_change_percent_on_library_projection_update": 100
                * (warm[7] / warm[3] - 1),
                "summands_per_operation_per_evaluation": baseline["points"]
                * key[0] ** 2,
                "scaled_elements_per_evaluation": baseline["points"] * key[0],
                "split_esp_publication_passes_per_evaluation": math.ceil(
                    baseline["points"] / effective_tile(baseline)
                ),
                "esp_symmetric_integrals_per_evaluation": baseline["points"]
                * key[0]
                * (key[0] + 1)
                // 2,
                "esp_materialized_elements_per_evaluation": baseline["points"]
                * key[0] ** 2,
                "max_paired_errors": [
                    max(s["errors"][i] for r in group.values() for s in r["samples"])
                    for i in range(3)
                ],
                "max_oracle_errors": [
                    max(r["oracle_errors"][i] for r in group.values()) for i in range(3)
                ],
            }
        )
    assert len(groups) == 8 and len(records) == provenance["records"] == 32
    if args.write_summary:
        # Keep generated summary rows compact; the README supplies the review table.
        rows = ",\n".join("  " + json.dumps(row) for row in summary)
        (HERE / "summary.json").write_text("[\n" + rows + "\n]\n")
    else:
        expected = json.loads((HERE / "summary.json").read_text())
        assert json.loads(json.dumps(summary)) == expected
    print(f"Verified {len(records)} records / {6 * len(records)} endpoint samples")


if __name__ == "__main__":
    main()
