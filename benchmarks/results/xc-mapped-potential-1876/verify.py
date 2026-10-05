"""Verify complete mapped-XC potential crossover receipts and semantic work."""

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
    """Check all paired samples, cache/scatter costs and retained AO domains."""
    records = []
    provenance = json.loads((HERE / "provenance.json").read_text())
    for atoms in (48, 96):
        data_path = HERE / f"endpoint-{atoms}.json.gz"
        raw = data_path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == provenance["data_sha256"][data_path.name]
        data = json.loads(gzip.decompress(raw))
        assert data["status"] == "completed" and data["exit_status"] == 0
        parsed = data["records"]
        assert len(parsed) == 12
        baseline_path = HERE.parent / "pbe0-composed-baseline-20261005" / f"components-{atoms}.json.gz"
        baseline_raw = baseline_path.read_bytes()
        assert hashlib.sha256(baseline_raw).hexdigest() == provenance["baseline_census_sha256"][baseline_path.name]
        baseline = json.loads(gzip.decompress(baseline_raw))
        for geometry in (0, 1):
            for tile in (128, 256, 512):
                pair = [row for row in parsed if row["geometry"] == geometry and row["tile"] == tile]
                assert len(pair) == 2
                generated, library = pair
                assert generated["provider"] == "generated.cuda"
                assert generated["algorithm"] == "symmetric-cross-generated"
                assert library["provider"] == "cublas"
                assert library["algorithm"] == "indexed-symmetric-cross-rank2k"
                for row in pair:
                    assert all(math.isfinite(value) for value in row.values() if isinstance(value, float))
                    assert row["atoms"] == atoms and row["nao"] == atoms * 8
                    assert row["evaluations"] == 6
                    assert row["points"] == atoms * 48 * 16 * 32
                    assert row["tiles"] * tile == row["points"]
                    assert row["point_ao_visits"] == tile * row["active_sum"]
                    assert row["max_energy_parity_error"] <= 1e-8
                    assert row["max_v_parity_error"] <= 1e-9
                    assert row["max_population_parity_error"] <= 1e-8
                    samples = [row[f"sample{i}_s"] for i in range(6)]
                    assert all(value > 0 for value in samples)
                    assert math.isclose(statistics.median(samples[1:]), row["warm_median_s"], rel_tol=1e-10)
                    assert samples[0] == row["cold_s"]
                    assert row["calls"] == 6 * (row["tiles"] - row["empty_tiles"])
                    assert row["summands"] == 6 * (row["point_ao_square_sum"] + row["point_ao_visits"])
                    assert row["dense_summands"] == 6 * row["points"] * row["nao"] * (row["nao"] + 1)
                    assert row["logical_weighted_panel_elements"] == 6 * row["point_ao_visits"]
                    assert row["host_bytes"] == 16 * 2**10
                for field in ("tiles", "empty_tiles", "min_active", "max_active", "active_sum",
                              "point_ao_square_sum", "point_ao_visits", "summands", "dense_summands", "calls"):
                    assert generated[field] == library[field]
                for field in ("rank2k_calls", "compact_write_bytes", "scatter_matrix_logical_bytes",
                              "scatter_index_logical_bytes", "matrix_cache_bytes", "provider_allowance"):
                    assert generated[field] == 0
                assert library["rank2k_calls"] == library["calls"]
                # The scalar count includes both bilinear terms per triangle.
                triangles = library["summands"] // (2 * tile)
                assert library["compact_write_bytes"] == triangles * 8
                assert library["scatter_matrix_logical_bytes"] == triangles * 4 * 8
                assert library["scatter_index_logical_bytes"] == triangles * 2 * 8
                assert library["matrix_cache_bytes"] == (atoms * 8) ** 2 * 8
                assert library["provider_allowance"] == 96 * 2**20
                if tile == 256:
                    work = baseline["records"][geometry]["scf_ao_work"]
                    for field in ("tiles", "empty_tiles", "max_active", "active_sum",
                                  "point_ao_square_sum", "point_ao_visits"):
                        assert library[field] == work[field]
                records.append({
                    "atoms": atoms, "aos": atoms * 8, "geometry": geometry, "tile": tile,
                    "generated_warm_s": generated["warm_median_s"],
                    "rank2k_warm_s": library["warm_median_s"],
                    "rank2k_time_change_percent": 100 * (library["warm_median_s"] / generated["warm_median_s"] - 1),
                    "mapped_summands_per_evaluation": library["summands"] // 6,
                    "compact_write_bytes_per_evaluation": library["compact_write_bytes"] // 6,
                    "scatter_matrix_logical_bytes_per_evaluation": library["scatter_matrix_logical_bytes"] // 6,
                    "scatter_index_logical_bytes_per_evaluation": library["scatter_index_logical_bytes"] // 6,
                    "max_energy_parity_error": library["max_energy_parity_error"],
                    "max_v_parity_error": library["max_v_parity_error"],
                })
    return {"scope": "ordinary-stream complete fixed-density XC; diagnostic positive density; no SCF/force timing",
            "oracle": "large-system generated parity; independent scalar/CPU qualification retained by #1963",
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
