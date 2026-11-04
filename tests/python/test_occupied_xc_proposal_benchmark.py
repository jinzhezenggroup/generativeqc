"""Offline benchmark gates must remain effective without native/CUDA imports."""

import copy
import sys
from pathlib import Path

import pytest

from benchmarks.occupied_xc_proposals import replay


@pytest.mark.parametrize("destination_kind", ["retained", "alias", "traversal"])
def test_retained_outputs_reject_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    destination_kind: str,
) -> None:
    from benchmarks import _retention, occupied_xc_proposals

    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    original = retained / "existing.json"
    original.write_text("original evidence")
    destination = retained / "new/result.json"
    if destination_kind == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(retained, target_is_directory=True)
        destination = alias / "new/result.json"
    elif destination_kind == "traversal":
        destination = Path(".artifacts/../benchmarks/results/new/result.json")
    monkeypatch.setattr(
        sys, "argv", ["occupied_xc_proposals", "--output", str(destination)]
    )
    with pytest.raises(SystemExit) as error:
        occupied_xc_proposals.main()
    assert error.value.code == 2
    assert "invalid raw_output_path value" in capsys.readouterr().err
    with pytest.raises(ValueError, match="raw benchmark output"):
        occupied_xc_proposals.save(original, {"overwrite": True})
    assert original.read_text() == "original evidence"
    assert not (retained / "new").exists()
    assert not (tmp_path / ".artifacts").exists()


def test_scratch_output_round_trips(tmp_path: Path) -> None:
    from benchmarks.occupied_xc_proposals import load, save

    destination = tmp_path / "endpoint.json.xz"
    record = {"energy": -1.0, "previous_energy": float("inf")}
    save(destination, record)
    assert load(destination) == record


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


@pytest.mark.parametrize("selector", ("unset", "1"))
def test_replay_preserves_actual_default_and_explicit_control_provenance(
    selector: str,
) -> None:
    """New qualification identifies an absent variable, not a relabeled opt-in."""
    record = receipt()
    record.update(
        schema="generativeqc.occupied-proposals.v2",
        proposal_selector=selector,
        controls={"0": "0", selector: None if selector == "unset" else "1"},
    )
    for row in record["observations"]:
        if row["selector"] == "1":
            row["selector"] = selector
        row["occupied_xc_env"] = record["controls"][row["selector"]]
    summary = replay(record)
    assert summary["rks-cold"]["iterations_fock_builds"][selector] == [(4, 5)] * 3
    record["observations"][1]["occupied_xc_env"] = "invalid"
    with pytest.raises(ValueError, match="control does not match"):
        replay(record)
    record["controls"][selector] = "invalid"
    with pytest.raises(ValueError, match="control provenance"):
        replay(record)


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
