"""Verify all endpoint samples, identities, work and resource receipts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Historical four-site receipts retain their three-, four-, and five-offer catalogs.
# Pin identities so accepting further rejected offers cannot admit arbitrary
# providers or relabel the historical measurements as a different program.
PROVIDERS = ("cublas", "generated.cuda", "cutensor", "cublaslt", "cutlass-aot")
OFFER_IDENTITIES = (
    (
        "2a0fd6bf4976be9afc848fea6eef8d04ca0e861245645403c83a3ddbc4b1d5f2",
        "db2668397f341cbd3c818f7c00d94b0ab79665b6702a4f542661f30b71d575da",
        "319ac640bb643d612dbf8da7ead024277cb2989395f108e864a5e22a4efe389e",
        "00dff5462534cb1a8b434bd927d8143e0feb0a496273e6e357fd62cb57422bac",
        "ff53e1fd39838e003a268984b0bcb1c858b9b1eeef30b0e836bf52f03365b78d",
    ),
    (
        "f6313629029b73fadfcf61d7ad65f62136490adff1373871f0dcbcb013a50871",
        "231c45ec6392ee44f3253292e115ef320e6d63afc76f1c56c79aa36999d6610b",
        "3b563fbe632ffaa540a238d3c3e829f2dafbc7aa4c98d53d87ffc3a53b2013d5",
        "9056f036ca8a3bc2c405e11f0e1d8c7d5f5ce4160f05359a99d380bd2d927887",
        "60f1ad777b0005edf568ce034691f964d00ff2d77941cacfcba176b40af6322a",
    ),
)


# V2 is a separate six-site contract. New receipts cannot borrow a historical
# catalog prefix or silently discard the weighted ESP sites. The exact fifth
# CUTLASS offer remains valid only as a rejected, canonical pending candidate.
CURRENT_OFFER_IDENTITIES = (
    (
        "2a0fd6bf4976be9afc848fea6eef8d04ca0e861245645403c83a3ddbc4b1d5f2",
        "db2668397f341cbd3c818f7c00d94b0ab79665b6702a4f542661f30b71d575da",
        "319ac640bb643d612dbf8da7ead024277cb2989395f108e864a5e22a4efe389e",
        "00dff5462534cb1a8b434bd927d8143e0feb0a496273e6e357fd62cb57422bac",
        "ff53e1fd39838e003a268984b0bcb1c858b9b1eeef30b0e836bf52f03365b78d",
    ),
    (
        "f6313629029b73fadfcf61d7ad65f62136490adff1373871f0dcbcb013a50871",
        "231c45ec6392ee44f3253292e115ef320e6d63afc76f1c56c79aa36999d6610b",
        "3b563fbe632ffaa540a238d3c3e829f2dafbc7aa4c98d53d87ffc3a53b2013d5",
        "9056f036ca8a3bc2c405e11f0e1d8c7d5f5ce4160f05359a99d380bd2d927887",
        "60f1ad777b0005edf568ce034691f964d00ff2d77941cacfcba176b40af6322a",
    ),
    (
        "1813f841f7e2cb03f3d001f9bcb81b59d0bc4615e700f0cc8e2aba06d56f7465",
        "401baaa0e0e3f00a5287d6dc32d7dbdfbd1c87f663bc2d43d5ebb28b31f83c23",
        "a29e8d733708fca672a468a015ceddf9f831b0429e238a4534d9e068a3a1a2ed",
        "e2c09f2786983a3ef0c3487e8e4ab687f1c51a73d8b0d312ff79ae02d6ed728f",
        "0885c0253604e789b5514a57d4de0b800fb00670afff03c7fd17bd966fa0ed54",
    ),
)
SITE_IDENTITIES = (
    (
        "03dfb1c5496cfffef3e7ddb62b0acbfcd6a372b604fcc2f5fce89e6912270339",
        "abd36a81d37285b726b880fdc80580032fd9aaf5064a8bed58659b73dca44c83",
        "34bae240924bdaefe13d7358fe22cba339b4a78ac6982d61888fa3643f720bb8",
    ),
    (
        "b5882279d6a3186840ea53c7ba93ae431d1901f99bf217f8311e70b312701c31",
        "9296736bb5188393e05c1a5ebc553b4a9a94b8f2ae296c5e837abe482aa39973",
        "11b61c5f8b75a8884525c910808ddd3239f1c821ec07ac36fc4242d244fd39a8",
    ),
    (
        "2251e26bab5c445f0a4449901bfd5ab1c16ca9d66a9e98acd0750cb07c5c9bb7",
        "b987caf8108d93391ae0daabacfbbd372219949cbd31dbe886ca3e113f8b3677",
        "11b61c5f8b75a8884525c910808ddd3239f1c821ec07ac36fc4242d244fd39a8",
    ),
)
SITE_LABELS = (
    "projection_full",
    "accumulation_full",
    "projection_tail",
    "accumulation_tail",
    "esp_application_full",
    "esp_application_tail",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_records(records: list[dict], provenance: dict) -> list[dict]:
    """Validate historical V1 or current V2 receipts without changing attribution."""
    groups: dict[tuple, dict] = {}
    for record in records:
        schema = record["schema"]
        assert schema in ("cosx-endpoint-v1", "cosx-endpoint-v2")
        current = schema == "cosx-endpoint-v2"
        routes = 8 if current else 4
        n, points, tile, mask = (record[k] for k in ("nao", "points", "tile", "mask"))
        assert all(type(x) is int and x > 0 for x in (n, points, tile))
        effective_tile = (
            record["effective_tile"]
            if current
            else record.get("effective_tile", min(points, tile))
        )
        assert type(effective_tile) is int and effective_tile == min(points, tile)
        assert points == record["atoms"] * math.prod(record["grid"])
        assert type(mask) is int and 0 <= mask < routes and record["geometry"] in (0, 1)
        key = (n, tuple(record["grid"]), tile, record["geometry"], schema)
        group = groups.setdefault(key, {})
        assert mask not in group
        group[mask] = record
        samples = record["samples"]
        assert len(samples) == 6
        for sample in samples:
            assert math.isfinite(sample["energy"])
            assert math.isfinite(sample["seconds"]) and sample["seconds"] > 0
            assert len(sample["errors"]) == 3
            # Receipts are tighter than the runtime absolute+relative gate.
            assert all(math.isfinite(x) and 0 <= x < 1e-9 for x in sample["errors"])
        assert len(record["oracle_errors"]) == 3
        assert all(math.isfinite(x) and 0 <= x < 1e-9 for x in record["oracle_errors"])
        assert record["provider_allowance"] == (96 << 20 if mask else 0)
        assert 0 <= record["retained_provider_bytes"] <= record["provider_allowance"]
        assert record["host_reservation"] == (160 << 10 if current else 128 << 10)
        for field in ("geometry_prepare_s", "prepare_s", "provider_prepare_s"):
            assert math.isfinite(record[field]) and record[field] >= 0
        assert record["runtime_version"] == provenance["runtime_version"]
        if mask:
            assert record["provider_version"] == provenance["provider_version"]
        assert record["compute_capability"] == "12.0"
        assert len(record["sites"]) == (6 if current else 4)
        for slot, site in enumerate(record["sites"]):
            weighted = slot >= 4
            family = 2 if weighted else slot % 2
            full = slot < 2 or slot == 4
            count = (
                points // effective_tile if full else int(points % effective_tile != 0)
            )
            extent = (
                effective_tile
                if full or not points % effective_tile
                else points % effective_tile
            )
            library = bool(mask & (1 << family))
            for field in ("calls", "summands", "m", "n", "k"):
                assert type(site[field]) is int
            assert site["calls"] == 6 * count
            assert site["summands"] == 6 * count * extent * n * n
            shape = (
                (n, 1, n)
                if weighted
                else (extent, n, n)
                if family == 0
                else (n, n, extent)
            )
            assert (site["m"], site["n"], site["k"]) == shape
            assert site["provider"] == ("cublas" if library else "generated.cuda")
            if current:
                algorithm = (
                    "batch-scaled-contraction-and-publication"
                    if weighted and library
                    else "batch-scaled-fused"
                    if weighted
                    else "prepared-affine-region"
                )
                assert site["algorithm"] == algorithm
                assert site["label"] == SITE_LABELS[slot]
                assert (
                    tuple(
                        site[field] for field in ("scientific", "semantic", "precision")
                    )
                    == SITE_IDENTITIES[family]
                )
                for field in ("batches", "scaled_elements", "publication_passes"):
                    assert type(site[field]) is int
                assert site["batches"] == (extent if weighted else 1)
                assert site["scaled_elements"] == (
                    6 * count * extent * n if weighted else 0
                )
                assert site["publication_passes"] == (
                    6 * count if weighted and library else 0
                )
            offers = site["offers"]
            identities = (
                CURRENT_OFFER_IDENTITIES[family]
                if current
                else OFFER_IDENTITIES[family]
            )
            assert len(offers) in ((4, 5) if current else (3, 4, 5))
            assert tuple(o["provider"] for o in offers) == PROVIDERS[: len(offers)]
            assert tuple(o["identity"] for o in offers) == identities[: len(offers)]
            selected = 0 if library else 1
            assert type(site["selected"]) is int and site["selected"] == selected
            assert offers[selected]["identity"] == site["candidate"]
            assert not offers[selected]["rejection"] and not offers[1]["rejection"]
            assert bool(offers[0]["rejection"]) == bool(selected)
            assert all(o["rejection"] for o in offers[2:])
            for identity in ("candidate", "scientific", "semantic", "precision"):
                assert len(site[identity]) == 64 and int(site[identity], 16) >= 0
    summary = []
    for key, group in sorted(groups.items()):
        current = key[4] == "cosx-endpoint-v2"
        routes = 8 if current else 4
        assert set(group) == set(range(routes))
        baseline = group[0]
        warm = [
            statistics.median(s["seconds"] for s in group[m]["samples"][1:])
            for m in range(routes)
        ]
        for record in group.values():
            assert (
                record["device_bytes"] - record["provider_allowance"]
                == baseline["device_bytes"]
            )
            for a, b in zip(record["sites"], baseline["sites"], strict=True):
                fields = (
                    "scientific",
                    "semantic",
                    "precision",
                    "calls",
                    "summands",
                    "m",
                    "n",
                    "k",
                )
                if current:
                    fields += ("label", "batches", "scaled_elements")
                for field in fields:
                    assert a[field] == b[field]
        masks = "_".join(map(str, range(routes)))
        result = {
            "nao": key[0],
            "grid": key[1],
            "tile": key[2],
            "geometry": key[3],
            "points": baseline["points"],
            f"warm_median_s_masks_{masks}": warm,
            f"time_change_percent_masks_{masks}": [
                100 * (v / warm[0] - 1) for v in warm
            ],
            "projection_summands_per_evaluation": baseline["points"] * key[0] ** 2,
            "accumulation_summands_per_evaluation": baseline["points"] * key[0] ** 2,
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
        if current:
            result.update(
                {
                    "schema": key[4],
                    "esp_application_summands_per_evaluation": sum(
                        s["summands"] for s in baseline["sites"][4:]
                    )
                    // 6,
                    "esp_scaled_elements_per_evaluation": sum(
                        s["scaled_elements"] for s in baseline["sites"][4:]
                    )
                    // 6,
                    f"esp_publication_passes_per_evaluation_masks_{masks}": [
                        sum(s["publication_passes"] for s in group[m]["sites"][4:]) // 6
                        for m in range(routes)
                    ],
                }
            )
        summary.append(result)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    provenance = json.loads((HERE / "provenance.json").read_text())
    records = []
    for name, sha in provenance["data_sha256"].items():
        assert digest(HERE / name) == sha, name
        records.extend(json.loads(gzip.decompress((HERE / name).read_bytes())))
    summary = verify_records(records, provenance)
    if args.write_summary:
        (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    else:
        expected = json.loads((HERE / "summary.json").read_text())
        assert json.loads(json.dumps(summary)) == expected
    assert len(records) == provenance["records"]
    print(f"Verified {len(records)} records / {6 * len(records)} endpoint samples")


if __name__ == "__main__":
    main()
