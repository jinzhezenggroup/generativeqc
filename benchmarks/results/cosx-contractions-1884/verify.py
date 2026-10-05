"""Verify all endpoint samples, identities, work and resource receipts."""

from __future__ import annotations

import gzip
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        assert record["schema"] == "cosx-endpoint-v1"
        n, points, tile, mask = (record[k] for k in ("nao", "points", "tile", "mask"))
        effective_tile = record.get("effective_tile", min(points, tile))
        assert type(effective_tile) is int and effective_tile == min(points, tile)
        assert points == record["atoms"] * math.prod(record["grid"])
        assert 0 <= mask < 4 and record["geometry"] in (0, 1)
        key = (n, tuple(record["grid"]), tile, record["geometry"])
        group = groups.setdefault(key, {})
        assert mask not in group
        group[mask] = record
        samples = record["samples"]
        assert len(samples) == 6
        for sample in samples:
            assert math.isfinite(sample["energy"])
            assert math.isfinite(sample["seconds"]) and sample["seconds"] > 0
            # Receipts are tighter than the runtime absolute+relative gate.
            assert all(math.isfinite(x) and 0 <= x < 1e-9 for x in sample["errors"])
        assert all(math.isfinite(x) and 0 <= x < 1e-9 for x in record["oracle_errors"])
        assert record["provider_allowance"] == (96 << 20 if mask else 0)
        assert record["retained_provider_bytes"] <= record["provider_allowance"]
        assert record["host_reservation"] == 128 << 10
        for field in ("geometry_prepare_s", "prepare_s", "provider_prepare_s"):
            assert math.isfinite(record[field]) and record[field] >= 0
        assert record["runtime_version"] == provenance["runtime_version"]
        if mask:
            assert record["provider_version"] == provenance["provider_version"]
        assert record["compute_capability"] == "12.0"
        assert len(record["sites"]) == 4
        for slot, site in enumerate(record["sites"]):
            count = points // effective_tile if slot < 2 else int(points % effective_tile != 0)
            extent = effective_tile if slot < 2 or not points % effective_tile else points % effective_tile
            assert site["calls"] == 6 * count
            assert site["summands"] == 6 * count * extent * n * n
            assert (site["m"], site["n"], site["k"]) == (
                (extent, n, n) if slot % 2 == 0 else (n, n, extent)
            )
            assert site["provider"] == ("cublas" if mask & (1 << (slot % 2)) else "generated.cuda")
            assert len(site["offers"]) == 3
            assert site["offers"][site["selected"]]["identity"] == site["candidate"]
            assert site["offers"][2]["rejection"]
            for identity in ("candidate", "scientific", "semantic", "precision"):
                assert len(site[identity]) == 64 and int(site[identity], 16) >= 0
    summary = []
    for key, group in sorted(groups.items()):
        assert set(group) == set(range(4))
        baseline = group[0]
        warm = [statistics.median(s["seconds"] for s in group[m]["samples"][1:]) for m in range(4)]
        for record in group.values():
            assert record["device_bytes"] - record["provider_allowance"] == baseline["device_bytes"]
            for a, b in zip(record["sites"], baseline["sites"], strict=True):
                for field in ("scientific", "semantic", "precision", "calls", "summands", "m", "n", "k"):
                    assert a[field] == b[field]
        summary.append({
            "nao": key[0], "grid": key[1], "tile": key[2], "geometry": key[3],
            "points": baseline["points"],
            "warm_median_s_masks_0_1_2_3": warm,
            "time_change_percent_masks_0_1_2_3": [100 * (v / warm[0] - 1) for v in warm],
            "projection_summands_per_evaluation": baseline["points"] * key[0] ** 2,
            "accumulation_summands_per_evaluation": baseline["points"] * key[0] ** 2,
            "esp_symmetric_integrals_per_evaluation": baseline["points"] * key[0] * (key[0] + 1) // 2,
            "esp_materialized_elements_per_evaluation": baseline["points"] * key[0] ** 2,
            "max_paired_errors": [max(s["errors"][i] for r in group.values() for s in r["samples"]) for i in range(3)],
            "max_oracle_errors": [max(r["oracle_errors"][i] for r in group.values()) for i in range(3)],
        })
    if args.write_summary:
        (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    else:
        expected = json.loads((HERE / "summary.json").read_text())
        assert json.loads(json.dumps(summary)) == expected
    assert len(records) == provenance["records"]
    print(f"Verified {len(records)} records / {6 * len(records)} endpoint samples")


if __name__ == "__main__":
    main()
