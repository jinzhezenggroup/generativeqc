"""Verify raw E/F arrays and retain complete endpoint populations, not best samples.

Raw force vectors for every replay remain in the hash-bound ignored run directory.
The compact record retains independent cold/moved oracle vectors and native
vectors for the four baseline ablations, plus every timing/error/work observation,
protocol and source identity. All force arrays are checked before compaction.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
from pathlib import Path
from statistics import median

import numpy as np

PHASES = [
    "cold",
    "warm",
    "warm",
    "warm",
    "moved",
    "moved-warm",
    "moved-warm",
    "moved-warm",
]
FIELDS = [
    "phase",
    "geometry",
    "repeat",
    "energy",
    "complete_seconds",
    "prepare_seconds",
    "iterations",
    "fock_builds",
    "scf_jk_builds",
    "energy_error",
    "force_error",
    "gate",
    "converged",
    "status",
    "energy_change",
    "density_rms",
    "physical_residual_rms",
]
CASES = [
    (method, atoms)
    for atoms in (48, 96)
    for method in ("pbe", "pbe0", "b3lyp", "r2scan", "r2scan-3c", "m06-2x", "wb97m-v")
    if (method, atoms) != ("wb97m-v", 96)
]


def read(path: Path) -> dict:
    """Require the entire scientific cohort before admitting any timing row."""
    raw = path.read_bytes()
    data = json.loads(raw)
    assert data["status"] == "measured", path
    assert [row["phase"] for row in data["records"]] == PHASES, path
    assert all(row["gate"] is True and row["converged"] for row in data["records"]), (
        path
    )
    data["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def compact(data: dict, *, vectors: bool) -> dict:
    """Keep input identities and all observations; omit repeated SCF histories."""
    result = {
        key: data.get(key)
        for key in (
            "raw_sha256",
            "source_file_sha256",
            "reference_sha256",
            "basis_file_sha256",
            "protocol",
            "native_experiment",
            "native_schedule_settings",
            "scheduler",
        )
    }
    build = data.get("native_build")
    if build:
        result["native_build"] = {
            key: build[key] for key in ("library_sha256", "probe")
        }
    environment = data["environment"]
    result["environment"] = {
        key: environment.get(key) for key in ("packages", "accelerator")
    }
    result["records"] = [[row.get(key) for key in FIELDS] for row in data["records"]]
    if vectors:
        result["anchor_forces"] = {
            str(row["geometry"]): row["forces"]
            for row in data["records"]
            if row["phase"] in ("cold", "moved")
        }
    return result


def qualify(data: dict, reference: dict) -> None:
    """Recompute every observable gate against the independent frozen geometry."""
    for key in (
        "geometries_bohr",
        "grid",
        "basis_identity",
        "method",
        "atoms",
        "aos",
        "density_fitting",
    ):
        assert data["protocol"][key] == reference["protocol"][key], key
    assert data["reference_sha256"] == reference["raw_sha256"]
    anchors = {
        row["geometry"]: row
        for row in reference["records"]
        if row["phase"] in ("cold", "moved")
    }
    for row in data["records"]:
        anchor = anchors[row["geometry"]]
        actual = np.asarray(row["forces"])
        expected = np.asarray(anchor["forces"])
        assert actual.shape == expected.shape == (data["protocol"]["atoms"], 3)
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
        energy_error = abs(row["energy"] - anchor["energy"])
        force_error = float(np.max(np.abs(actual - expected)))
        assert energy_error <= 1e-8 and force_error <= 1e-7
        assert abs(row["energy_error"] - energy_error) < 1e-15
        assert abs(row["force_error"] - force_error) < 1e-15


def main() -> None:
    """Publish verified local measurements or print medians from retained data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--candidate-arm", default="head")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("observations.json.xz")
    )
    args = parser.parse_args()
    if args.raw:
        assert args.references
        cohorts = {}
        for method, atoms in CASES:
            key = f"{method}-{atoms}"
            reference = read(args.references / f"{key}.json")
            native = read(args.raw / args.candidate_arm / f"{key}.json")
            qualify(native, reference)
            case = {
                "reference": compact(reference, vectors=True),
                "native": compact(
                    native, vectors=(args.raw / "baseline" / f"{key}.json").exists()
                ),
            }
            baseline = args.raw / "baseline" / f"{key}.json"
            if baseline.exists():
                previous = read(baseline)
                qualify(previous, reference)
                case["baseline"] = compact(previous, vectors=False)
            cohorts[key] = case
        extras = {}
        for name, key in [
            ("minimum-r2scan-3c-48", "r2scan-3c-48"),
            ("tile1024-r2scan-3c-96", "r2scan-3c-96"),
            ("tile1024-m06-2x-96", "m06-2x-96"),
        ]:
            data = read(args.raw / f"{name}.json")
            reference = read(args.references / f"{key}.json")
            qualify(data, reference)
            extras[name] = compact(data, vectors=False)
            extras[name]["experimental_tile_points"] = data["records"][0].get(
                "experimental_tile_points"
            )
            extras[name]["minimum_workspace_capacity"] = data["records"][0].get(
                "minimum_workspace_capacity"
            )
        payload = {
            "schema": "generativeqc.ecnu-stationary-performance.v1",
            "scope": "matched-grid implementation comparison; no grid-converged production ranking",
            "row_fields": FIELDS,
            "cohorts": cohorts,
            "additional_controls": extras,
        }
        # Store full-precision anchor vectors in a standard numeric NPY array.
        # Sharing one array avoids decimal/vector duplication across protocols;
        # no object serialization, rounding, or custom numerical codec is used.
        arrays = []
        offset = 0
        for case in cohorts.values():
            for data in case.values():
                for geometry, forces in data.get("anchor_forces", {}).items():
                    array = np.asarray(forces, dtype="<f8")
                    arrays.append(array)
                    data["anchor_forces"][geometry] = {
                        "row": offset,
                        "atoms": len(array),
                    }
                    offset += len(array)
        stream = io.BytesIO()
        np.save(stream, np.concatenate(arrays), allow_pickle=False)
        stored = lzma.compress(stream.getvalue(), preset=6)
        anchor_path = args.output.with_name("anchor-forces.npy.xz")
        anchor_path.write_bytes(stored)
        payload["anchor_array"] = {
            "path": anchor_path.name,
            "sha256": hashlib.sha256(stored).hexdigest(),
            "rows": offset,
        }
        args.output.write_bytes(
            lzma.compress(
                (
                    json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
                ).encode(),
                preset=6,
            )
        )
    payload = json.loads(lzma.decompress(args.output.read_bytes()))
    stored = (args.output.parent / payload["anchor_array"]["path"]).read_bytes()
    assert hashlib.sha256(stored).hexdigest() == payload["anchor_array"]["sha256"]
    anchors = np.load(io.BytesIO(lzma.decompress(stored)), allow_pickle=False)
    assert anchors.shape == (payload["anchor_array"]["rows"], 3)
    fields = payload["row_fields"]
    for key, case in payload["cohorts"].items():
        values = {}
        for arm, data in case.items():
            rows = [dict(zip(fields, row, strict=True)) for row in data["records"]]
            assert [row["phase"] for row in rows] == PHASES
            assert all(row["gate"] is True for row in rows)
            values[arm] = {
                phase: median(
                    row["complete_seconds"] for row in rows if row["phase"] == phase
                )
                for phase in ("cold", "warm", "moved", "moved-warm")
            }
        print(key, json.dumps(values))


if __name__ == "__main__":
    main()
