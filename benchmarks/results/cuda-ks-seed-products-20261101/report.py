"""Replay original endpoint receipts, independent gates and paired errors."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]


def checked_file(relative: str, expected: str) -> bytes:
    """Require the original retained oracle/reuse dependency before replay."""
    raw = (ROOT / relative).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, relative
    return raw


def phase_times(record: dict, phase: str) -> list[float]:
    """Keep every measured repeat; never rank an incomplete population."""
    assert record["status"] == "complete"
    times = [
        sample["seconds"] for sample in record["samples"] if sample["phase"] == phase
    ]
    assert len(times) == (1 if phase == "cold" else record["warm_repeats"])
    return times


def same_task(left: dict, right: dict) -> None:
    """Authenticate inputs and the stable node/PCI device locator."""
    for key in (
        "method",
        "mode",
        "atoms",
        "geometry_bohr",
        "orbital_basis",
        "basis_representation",
        "charge",
        "spin",
        "auxiliary_input_sha256",
        "protocol",
        "thread_controls",
    ):
        assert left[key] == right[key], key
    assert left["environment"]["host"]["node"] == right["environment"]["host"]["node"]
    for key in ("name", "pci_bus_id"):
        assert left["accelerator"][key] == right["accelerator"][key], key
    left_uuid = (left["accelerator"].get("nvidia_smi") or {}).get("uuid")
    right_uuid = (right["accelerator"].get("nvidia_smi") or {}).get("uuid")
    if left_uuid and right_uuid:
        assert left_uuid == right_uuid
    else:
        assert left["scheduler"]["LSB_JOBID"]
        # ECNU CUDA exposes the node/BDF and assigned visibility but its
        # nvidia-smi receipt has no UUID. Retain distinct allocation IDs;
        # these comparisons are explicitly cross-allocation observations.
        assert right["scheduler"]["LSB_JOBID"]
        assert (
            left["scheduler"]["CUDA_VISIBLE_DEVICES"]
            == right["scheduler"]["CUDA_VISIBLE_DEVICES"]
        )


def paired_errors(left: dict, right: dict) -> list[dict]:
    """Compare unchanged solver work and all output components across builds."""
    import numpy as np

    same_task(left, right)
    errors = []
    for a, b in zip(left["samples"], right["samples"], strict=True):
        for key in ("phase", "repeat", "iterations", "fock_builds", "settings"):
            assert a[key] == b[key], key
        for key in a["initial_guess"]:
            if key != "preparation_seconds":
                assert a["initial_guess"][key] == b["initial_guess"][key], key
        for key in (
            "grid_points",
            "tile_points",
            "ao_order",
            "scf_domain",
            "coulomb_exchange_work",
            "nonlocal_work",
            "nonlocal_history_scopes",
        ):
            assert a["ks_diagnostic"].get(key) == b["ks_diagnostic"].get(key), key
        delta = np.asarray(a["forces"]) - np.asarray(b["forces"])
        row = {
            "phase": a["phase"],
            "repeat": a["repeat"],
            "energy_abs": abs(a["energy"] - b["energy"]),
            "force_max": float(abs(delta).max()),
        }
        assert row["energy_abs"] < 1e-7 and row["force_max"] < 1e-7
        errors.append(row)
    return errors


def main() -> None:
    """Print complete populations and optionally recompute independent checks."""
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE", "") not in ("", "0"):
        raise SystemExit(
            "report.py requires assertions; omit -O/-OO and unset PYTHONOPTIMIZE"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    family = Path(__file__).parent.name
    manifest = json.loads(
        (ROOT / "benchmarks/evidence-reviews" / (family + ".json")).read_text()
    )
    relative = f"benchmarks/results/{family}/receipts.json.xz"
    entries = [row for row in manifest["files"] if row["path"] == relative]
    assert len(entries) == 1, "receipts.json.xz manifest entry"
    archive = checked_file(relative, entries[0]["sha256"])
    assert len(archive) == entries[0]["bytes"], "receipts.json.xz size"
    bundle = json.loads(lzma.decompress(archive))
    oracle_bytes = checked_file(bundle["oracle_path"], bundle["oracle_sha256"])
    reuse_bytes = checked_file(bundle["reuse_path"], bundle["reuse_sha256"])
    prior = json.loads(
        lzma.decompress(checked_file(bundle["prior_path"], bundle["prior_sha256"]))
    )
    oracles = json.loads(lzma.decompress(oracle_bytes))
    reuse = json.loads(lzma.decompress(reuse_bytes))
    records = {}
    for name, entry in bundle["records"].items():
        # Parent populations already have a hash-bound retained archive. Read
        # their exact original serialization there instead of storing it twice.
        raw = (
            prior["records"][name[6:]]["json"]
            if name.startswith("prior/")
            else entry["json"]
        ).encode()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"], name
        record = records[name] = json.loads(raw)
        if entry.get("stage") == "final" and record["arm"] == "standard":
            assert record["native_build"]["source_identity"] == bundle["source_identity"]
        if args.verify and "qualification" in entry:
            from benchmarks.readme_defaults_qualification import qualify

            assert (
                qualify(
                    [record],
                    oracles,
                    reference_reuse=reuse[entry["device"]]["analysis"][
                        "reference_reuse"
                    ],
                )[0]
                == entry["qualification"]
            ), name
        if record["status"] != "complete":
            print(name, record["status"], "not ranked")
    for pair in bundle["comparisons"]:
        native, reference = (records[pair[key]] for key in ("native", "reference"))
        gates = "/".join(
            bundle["records"][pair[key]]["qualification"]["status"]
            for key in ("native", "reference")
        )
        same_task(native, reference)
        assert gates == "qualified/qualified", (pair, gates)
        if pair["kind"] in ("native-change", "native-only-control") and args.verify:
            assert paired_errors(native, reference) == pair["paired_errors"]
        for phase in ("cold", "warm"):
            n, r = (
                median(phase_times(native, phase)),
                median(phase_times(reference, phase)),
            )
            assert pair[phase + "_ratio"] == n / r
            print(
                pair["device"],
                native["method"],
                pair["kind"],
                phase,
                f"{n:.6f}",
                f"{r:.6f}",
                f"{100 * (n / r - 1):+.2f}%",
                gates,
            )
    provenance_bytes = checked_file(
        f"benchmarks/results/{family}/provenance.json.xz",
        bundle["provenance_sha256"],
    )
    assert len(provenance_bytes) == bundle["provenance_bytes"]
    provenance = json.loads(lzma.decompress(provenance_bytes))
    for name, entry in provenance.items():
        assert hashlib.sha256(entry["text"].encode()).hexdigest() == entry["sha256"], (
            name
        )


if __name__ == "__main__":
    main()
