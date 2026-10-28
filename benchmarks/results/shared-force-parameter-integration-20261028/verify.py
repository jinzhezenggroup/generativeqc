"""Replay retained integration receipts without admitting a different compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def verify(phase: str) -> dict:
    """Authenticate exact stored bytes and replay the recorded force gates.

    Center-energy, density and Fock gates execute in the retained test driver;
    these receipts retain force stencils and artifact ownership for offline replay.
    Neither historical phase grants capability admission to the current checkout.
    """
    manifest = json.loads((ROOT / f"{phase}-manifest.json").read_text())
    records = {}
    for name, identity in manifest["files"].items():
        stored = (ROOT / name).read_bytes()
        assert len(stored) == identity["stored_bytes"], name
        assert hashlib.sha256(stored).hexdigest() == identity["stored_sha256"], name
        original = lzma.decompress(stored)
        assert len(original) == identity["original_bytes"], name
        assert hashlib.sha256(original).hexdigest() == identity["original_sha256"], name
        if name.endswith(".json.xz"):
            records[Path(name).name.removesuffix(".xz")] = json.loads(original)

    summary = records["summary.json"]
    expected = {(row["name"], row["spins"]) for row in summary}
    assert len(summary) == len(expected)
    assert all(row["status"] == "passed" for row in summary)
    seen = set()
    libraries = set()
    endpoints = directions = 0
    for record in records.values():
        if not isinstance(record, dict) or record.get("schema") != (
            "generativeqc.native-automatic-cuda-forces.v1"
        ):
            continue
        key = (record["name"], record["spins"])
        assert key not in seen
        seen.add(key)
        library = record["native_library_sha256"]
        libraries.add(library)
        assert [row["phase"] for row in record["rows"]] == ["cold", "warm", "moved"]
        for row in record["rows"]:
            receipt = row["artifact"]
            assert receipt["force_completed"] and receipt["fallback_reason"] == "none"
            assert receipt["spin_blocks"] == record["spins"]
            assert receipt["source_count"] == 7
            for owner in ("point_artifact", "force_artifact"):
                assert receipt[owner]["sha256"] == library
            assert receipt["capacity"]["minimum_device_bytes"] > 0
            assert receipt["stationary_work_v1"][3] > 0
            forces = np.asarray(row["forces"])
            projected = (
                forces.ravel()
                if len(row["independent_fd"]) == forces.size
                else [forces[-1] @ np.array([0.2, -0.3, 0.4])]
            )
            for field in ("native_fd", "independent_fd", "analytic_directions"):
                np.testing.assert_allclose(row[field], projected, atol=3e-5, rtol=0)
            np.testing.assert_allclose(forces.sum(axis=0), 0, atol=2e-8, rtol=0)
            endpoints += 1
            directions += len(projected)
    assert seen == expected and len(libraries) == 1
    assert not manifest["build_input_verification"]["mismatches"]
    assert manifest["build_input_verification"]["source_commit"] == manifest["source_commit"]
    for name, binding in manifest.get("drivers", {}).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == binding["sha256"]
    if phase == "final":
        assert (len(seen), endpoints, directions) == (18, 54, 186)
        assert libraries == {manifest["library"]["sha256"]}
        assert not any(manifest["qualification_status"]["exit_codes"].values())

    cli = records["composite-cli.json"]
    assert len(cli["rows"]) == 4
    if phase == "final":
        assert cli["cli_sha256"] == manifest["cli"]["sha256"]
    for row in cli["rows"]:
        assert row["energy_error"] < 2e-8
        assert len(row["finite_difference_errors"]) == 2
        for errors in row["finite_difference_errors"]:
            assert errors["gpu"] < 2e-5 and errors["cpu"] < 2e-5
    return {
        "phase": phase,
        "source_commit": manifest["source_commit"],
        "force_cases": len(seen),
        "force_endpoints": endpoints,
        "finite_difference_directions": directions,
        "cli_cases": len(cli["rows"]),
        "current_compiler_admission": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pre-r1", "final"), default="final")
    print(json.dumps(verify(parser.parse_args().phase), indent=2))
