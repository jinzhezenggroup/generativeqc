"""Verify and summarize the actual-map fixed-density XC crossover receipts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def summarize() -> dict:
    """Require paired work, all-sample parity and the retained tile-256 census."""
    records = []
    for atoms in (48, 96):
        data_path = HERE / f"endpoint-{atoms}.json.gz"
        raw = data_path.read_bytes()
        provenance = json.loads((HERE / "provenance.json").read_text())
        assert hashlib.sha256(raw).hexdigest() == provenance["data_sha256"][data_path.name]
        data = json.loads(gzip.decompress(raw))
        assert data["status"] == "completed" and data["exit_status"] == 0
        parsed = data["records"]
        assert len(parsed) == 12
        baseline = json.loads(gzip.decompress(
            (HERE.parent / "pbe0-composed-baseline-20261005" / f"components-{atoms}.json.gz").read_bytes()
        ))
        for geometry in (0, 1):
            for tile in (128, 256, 512):
                pair = [row for row in parsed if row["geometry"] == geometry and row["tile"] == tile]
                assert len(pair) == 2
                generated, library = pair
                assert generated["provider"] == "generated.cuda" and library["provider"] == "cublas"
                for row in pair:
                    assert all(math.isfinite(value) for value in row.values() if isinstance(value, float))
                    assert row["atoms"] == atoms and row["nao"] == atoms * 8
                    assert row["evaluations"] == 6
                    assert row["points"] == atoms * 48 * 16 * 32
                    assert row["max_energy_parity_error"] <= 1e-8
                    assert row["max_v_parity_error"] <= 1e-9
                    assert row["max_population_parity_error"] <= 1e-8
                    samples = [row[f"sample{i}_s"] for i in range(6)]
                    assert all(value > 0 for value in samples)
                    assert math.isclose(statistics.median(samples[1:]), row["warm_median_s"], rel_tol=1e-10)
                    assert samples[0] == row["cold_s"]
                for field in ("tiles", "empty_tiles", "min_active", "max_active", "active_sum",
                              "summands", "dense_summands"):
                    assert generated[field] == library[field]
                assert generated["gemm_calls"] == generated["gathered_elements"] == 0
                assert generated["matrix_cache_bytes"] == generated["provider_allowance"] == 0
                assert library["gemm_calls"] == 6 * (library["tiles"] - library["empty_tiles"])
                assert library["gathered_elements"] == library["summands"] / tile
                assert library["gather_logical_bytes"] == library["gathered_elements"] * 24
                assert library["matrix_cache_bytes"] == (atoms * 8) ** 2 * 8
                assert library["provider_allowance"] == 96 * 2**20
                if tile == 256:
                    work = baseline["records"][geometry]["scf_ao_work"]
                    for field in ("tiles", "empty_tiles", "max_active", "active_sum"):
                        assert library[field] == work[field]
                    assert library["summands"] == 6 * work["point_ao_square_sum"]
                    assert library["dense_summands"] == 6 * work["dense_point_ao_square_sum"]
                records.append({
                    "atoms": atoms, "aos": atoms * 8, "geometry": geometry, "tile": tile,
                    "generated_warm_s": generated["warm_median_s"],
                    "gemm_warm_s": library["warm_median_s"],
                    "gemm_time_change_percent": 100 * (library["warm_median_s"] / generated["warm_median_s"] - 1),
                    "mapped_summands_per_evaluation": int(library["summands"] / 6),
                    "gather_logical_bytes_per_evaluation": int(library["gather_logical_bytes"] / 6),
                    "max_energy_parity_error": library["max_energy_parity_error"],
                    "max_v_parity_error": library["max_v_parity_error"],
                })
    return {"scope": "ordinary-stream complete fixed-density XC; diagnostic positive density; no SCF/force timing",
            "oracle": "large-system generated parity; independent scalar/CPU qualification retained by #1961",
            "production_promotion": False, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    summary = summarize()
    target = HERE / "summary.json"
    if args.write:
        target.write_text(json.dumps(summary, indent=2) + "\n")
    else:
        assert json.loads(target.read_text()) == summary
    print("Verified all 24 endpoint records / 144 samples and both geometries' retained tile-256 AO census")


if __name__ == "__main__":
    main()
