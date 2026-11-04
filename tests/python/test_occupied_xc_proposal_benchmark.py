"""Offline benchmark gates must remain effective without native/CUDA imports."""

import copy

import pytest

from benchmarks.occupied_xc_proposals import replay


def receipt() -> dict:
    """Build a complete minimal population independent of production numerics."""
    oracles = [
        {
            "spin": spin,
            "geometry": geometry,
            "energy": -1.0,
            "forces": [[0.0, 0.0, 0.0]],
        }
        for spin in ("rks", "uks")
        for geometry in (0, 1)
    ]
    return {
        "schema": "generativeqc.occupied-proposals.v1",
        "completed": True,
        "source_commit": "a" * 40,
        "library_sha256": "b" * 64,
        "repeats": 3,
        "energy_gate_eh": 2e-9,
        "force_gate_eh_per_bohr": 1e-7,
        "oracles": oracles,
        "observations": [
            {
                "spin": spin,
                "repeat": repetition,
                "selector": selector,
                "phase": phase,
                "energy": -1.0,
                "forces": [[0.0, 0.0, 0.0]],
                "iterations": 4,
                "fock_builds": 5,
                "seconds": 2.0 if selector == "0" else 1.0,
            }
            for spin in ("rks", "uks")
            for repetition in range(3)
            for selector in ("0", "1")
            for phase in ("cold", "warm", "moved", "moved_warm")
        ],
    }


def test_replay_recomputes_complete_endpoint_and_work_populations() -> None:
    summary = replay(receipt())
    assert len(summary) == 8
    assert all(row["speedup"] == 2.0 for row in summary.values())
    assert summary["rks-cold"]["iterations_fock_builds"]["1"] == [(4, 5)] * 3


@pytest.mark.parametrize(
    "corruption",
    [
        "energy",
        "force",
        "nan",
        "work",
        "missing",
        "duplicate",
        "gate",
        "source",
        "incomplete",
    ],
)
def test_replay_rejects_corrupted_scientific_or_work_evidence(corruption: str) -> None:
    record = copy.deepcopy(receipt())
    first = record["observations"][0]
    if corruption == "energy":
        first["energy"] += 1e-8
    elif corruption == "force":
        first["forces"][0][2] = 2e-7
    elif corruption == "nan":
        first["forces"][0][1] = float("nan")
    elif corruption == "work":
        first["fock_builds"] = 0
    elif corruption == "missing":
        record["observations"].pop()
    elif corruption == "duplicate":
        record["observations"].append(first)
    elif corruption == "gate":
        record["force_gate_eh_per_bohr"] = 1e-3
    elif corruption == "source":
        record["source_commit"] = "unknown"
    else:
        record["completed"] = False
    with pytest.raises((ValueError, KeyError)):
        replay(record)
