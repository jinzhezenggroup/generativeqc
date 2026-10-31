"""Report compact observations; optionally replay gates against local raw receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path
from statistics import median, stdev

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    """Keep summary checks separate from independent full-vector verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--raw-root", type=Path, help="Original per-device campaign directories")
    args = parser.parse_args()
    if args.verify and args.raw_root is None:
        parser.error("--verify requires --raw-root; full vectors are retained locally")
    bundles = json.loads(lzma.decompress(Path(__file__).with_name("observations.json.xz").read_bytes()))
    for device, bundle in bundles.items():
        analysis = bundle["analysis"]
        rows = {row["name"]: row for row in analysis["populations"]}
        for row in rows.values():
            if "qualification" in row:
                assert row["warm_median"] == median(row["warm_seconds"])
                assert row["warm_stdev"] == stdev(row["warm_seconds"])
            if args.verify:
                from benchmarks.readme_defaults_qualification import identity, qualify

                record = json.loads((args.raw_root / device / row["name"]).read_text())
                assert identity(record) == row["receipt_sha256"]
                for key in ("status", "method", "atoms", "arm", "native_build", "scheduler", "auxiliary_input_sha256"):
                    assert row[key] == record.get(key)
                assert row["scientific_task"] == {key: record.get(key) for key in row["scientific_task"]}
                geometry = json.dumps(record.get("geometry_bohr"), sort_keys=True, separators=(",", ":")).encode()
                assert row["geometry_sha256"] == hashlib.sha256(geometry).hexdigest()
                if "qualification" in row:
                    path = ROOT / "benchmarks/results/readme-refresh-20261026/oracles.json.xz"
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == bundle["oracle_file_sha256"]
                    oracles = json.loads(lzma.decompress(path.read_bytes()))
                    assert qualify([record], oracles, reference_reuse=analysis["reference_reuse"])[0] == row["qualification"]
                    for phase in ("cold", "warm"):
                        assert row[phase + "_seconds"] == [s["seconds"] for s in record["samples"] if s["phase"] == phase]
                    assert row["iterations"] == [s["iterations"] for s in record["samples"]]
                    assert row["fock_builds"] == [s.get("fock_builds") for s in record["samples"]]
                    assert row["executed_settings"] == [s.get("settings") for s in record["samples"]]
            print(device, row["name"], row.get("qualification", {}).get("status", row["status"]),
                  "cold", row.get("cold_seconds"), "warm", row.get("warm_seconds"))

        def warm_times(names: list[str]) -> list[float]:
            return [sample for name in names for sample in rows[name]["warm_seconds"]]

        for comparison in analysis["comparisons"]:
            if "native_over_reference" in comparison:
                native = median(warm_times([comparison["native"]]))
                reference = median(warm_times([comparison["reference"]]))
                assert comparison["native_over_reference"] == native / reference
            else:
                baseline = median(warm_times(comparison["baseline_populations"]))
                candidate = median(warm_times(comparison["explicit_populations"]))
                assert comparison["explicit_over_baseline"] == candidate / baseline
            print(json.dumps({k: v for k, v in comparison.items() if k not in ("observed", "errors")}, sort_keys=True))


if __name__ == "__main__":
    main()
