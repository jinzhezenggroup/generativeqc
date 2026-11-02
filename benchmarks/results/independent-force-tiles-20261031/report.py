"""Authenticate retained original receipts and optionally recompute oracle gates."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
from pathlib import Path
from statistics import median, stdev

ROOT = Path(__file__).resolve().parents[3]


def paired_errors(candidate: dict, control: dict, *, tile_change: bool) -> list[dict]:
    """Recheck identical science/work before comparing a scheduling-only pair."""
    import numpy as np

    assert candidate["native_build"] == control["native_build"]
    for key in (
        "geometry_bohr", "orbital_basis", "basis_representation", "charge",
        "spin", "mode", "method", "protocol", "auxiliary_input_sha256",
    ):
        assert candidate[key] == control[key], key
    errors = []
    for a, b in zip(candidate["samples"], control["samples"], strict=True):
        assert (a["phase"], a["repeat"]) == (b["phase"], b["repeat"])
        left, right = dict(a["settings"]), dict(b["settings"])
        if tile_change:
            assert left.pop("force_tile_points") == 1024
            assert right.pop("force_tile_points") is None
            assert left["scf_tile_points"] == 256
        assert left == right
        assert a["iterations"] == b["iterations"]
        assert a["fock_builds"] == b["fock_builds"]
        delta = np.asarray(a["forces"]) - np.asarray(b["forces"])
        error = {
            "phase": a["phase"], "repeat": a["repeat"],
            "energy_abs": abs(a["energy"] - b["energy"]),
            "force_max": float(abs(delta).max()),
        }
        if tile_change:
            error["force_rms"] = float(np.sqrt(np.mean(delta * delta)))
        assert error["energy_abs"] < 1e-7 and error["force_max"] < 1e-7
        errors.append(error)
    return errors


def main() -> None:
    """Use embedded original bytes; optional local copies must agree exactly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--raw-root", type=Path, help="Original per-device campaign directories")
    args = parser.parse_args()
    provenance = json.loads(lzma.decompress(Path(__file__).with_name("provenance.json.xz").read_bytes()))
    for value in provenance.values():
        if isinstance(value, dict) and "sha256" in value:
            if "text" in value:
                raw = value["text"].encode()
            elif "base64" in value:
                raw = base64.b64decode(value["base64"], validate=True)
            else:
                continue
            assert hashlib.sha256(raw).hexdigest() == value["sha256"]
    bundles = json.loads(lzma.decompress(Path(__file__).with_name("receipts.json.xz").read_bytes()))
    for device, bundle in bundles.items():
        analysis = bundle["analysis"]
        rows = {row["name"]: row for row in analysis["populations"]}
        assert rows.keys() == bundle["records"].keys()
        records = {}
        for row in rows.values():
            retained = bundle["records"][row["name"]]
            raw = retained["json"].encode()
            assert hashlib.sha256(raw).hexdigest() == retained["original_sha256"]
            record = json.loads(raw)
            records[row["name"]] = record
            if args.raw_root is not None:
                assert (args.raw_root / device / row["name"]).read_bytes() == raw
            if "qualification" in row:
                assert row["warm_median"] == median(row["warm_seconds"])
                assert row["warm_stdev"] == stdev(row["warm_seconds"])
            if args.verify:
                from benchmarks.readme_defaults_qualification import identity, qualify

                assert identity(record) == row["receipt_sha256"]
                for key in ("status", "method", "atoms", "arm", "native_build", "scheduler", "auxiliary_input_sha256"):
                    assert row[key] == record.get(key)
                if "qualification" in row or "provenance_status" in row:
                    path = ROOT / "benchmarks/results/readme-refresh-20261026/oracles.json.xz"
                    assert hashlib.sha256(path.read_bytes()).hexdigest() == bundle["oracle_file_sha256"]
                    oracles = json.loads(lzma.decompress(path.read_bytes()))
                if "provenance_status" in row:
                    # The first integration driver emitted a short Git tag.
                    # Preserve its original bytes and require the qualification
                    # rejection, independently of the corrected rerun's gates.
                    assert row["provenance_status"] == "rejected-abbreviated-source-revision"
                    try:
                        qualify([record], oracles, reference_reuse=analysis["reference_reuse"])
                    except ValueError as error:
                        assert str(error) == "missing measurement source revision"
                    else:
                        raise AssertionError("abbreviated source revision unexpectedly qualified")
                if "qualification" in row:
                    assert qualify([record], oracles, reference_reuse=analysis["reference_reuse"])[0] == row["qualification"]
                    for phase in ("cold", "warm"):
                        assert row[phase + "_seconds"] == [s["seconds"] for s in record["samples"] if s["phase"] == phase]
                    assert row["iterations"] == [s["iterations"] for s in record["samples"]]
                    assert row["fock_builds"] == [s.get("fock_builds") for s in record["samples"]]
            print(device, row["name"], row.get("qualification", {}).get("status", row.get("provenance_status", row["status"])),
                  "cold", row.get("cold_seconds"), "warm", row.get("warm_seconds"))

        def warm_times(names: list[str]) -> list[float]:
            return [sample for name in names for sample in rows[name]["warm_seconds"]]

        for comparison in analysis["comparisons"]:
            if "native_over_reference" in comparison:
                native = median(warm_times([comparison["native"]]))
                reference = median(warm_times([comparison["reference"]]))
                assert comparison["native_over_reference"] == native / reference
                left, right = (rows[comparison[key]] for key in ("native", "reference"))
                assert left["pci_bus_id"] == right["pci_bus_id"]
                assert left["auxiliary_input_sha256"] == right["auxiliary_input_sha256"]
                if left["accelerator_uuid"] and right["accelerator_uuid"]:
                    assert left["accelerator_uuid"] == right["accelerator_uuid"]
                else:
                    assert device == "a100"
                    for key in ("LSB_JOBID", "CUDA_VISIBLE_DEVICES"):
                        assert left["scheduler"][key] == right["scheduler"][key]
            else:
                baseline = median(warm_times(comparison["baseline_populations"]))
                candidate = median(warm_times(comparison["explicit_populations"]))
                assert comparison["explicit_over_baseline"] == candidate / baseline
                if args.verify and "observed" in comparison:
                    # ABBA order matters for its retained paired differences.
                    a1, a2 = comparison["baseline_populations"]
                    b1, b2 = comparison["explicit_populations"]
                    ordered = [records[name] for name in (a1, b1, b2, a2)]
                    assert comparison["errors"] == [
                        error for receipt in ordered[1:]
                        for error in paired_errors(receipt, ordered[0], tile_change=False)
                    ]
                    assert comparison["observed"] == [
                        receipt["fixed_map_diagnostic"]["observed"] for receipt in ordered
                    ]
            print(json.dumps({k: v for k, v in comparison.items() if k not in ("observed", "errors")}, sort_keys=True))
        if args.verify:
            for pair in analysis["paired_api_errors"]:
                assert pair["errors"] == paired_errors(
                    records[pair["candidate"]], records[pair["baseline"]], tile_change=True
                )


if __name__ == "__main__":
    main()
