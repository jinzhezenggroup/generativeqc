"""Guard multi-functional endpoint qualification without importing a GPU stack."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.occupied_xc_functionals import PHASES, SCHEMA, replay
from benchmarks.occupied_xc_proposals import save


def receipts(tmp_path: Path, atoms: int = 3) -> list[Path]:
    """Supply deterministic complete pairs with independent full force vectors."""
    reference = {"energy": -10.0, "forces": [[0.0] * 3 for _ in range(atoms)]}
    paths = []
    for repeat in range(3):
        for control in ("0", None):
            record = {
                "schema": SCHEMA,
                "completed": True,
                "traced": False,
                "method": "mn15",
                "spin": "rks",
                "atoms": atoms,
                "basis": "def2-svp",
                "strict_solver": True,
                "source_commit": "a" * 40,
                "source_identity": "b" * 64,
                "library_sha256": "c" * 64,
                "occupied_xc_env": control,
                "oracle": {
                    "method": "mn15",
                    "spin": "rks",
                    "atoms": atoms,
                    "basis": "def2-svp",
                    "observations": [reference, reference],
                },
                "observations": [
                    dict(
                        copy.deepcopy(reference),
                        phase=phase,
                        seconds=10.0 if control == "0" else 9.0,
                        iterations=10,
                        fock_builds=10,
                        settings={"grid": "default"},
                    )
                    for phase in PHASES
                ],
            }
            path = tmp_path / f"r{repeat}-{control}.json"
            save(path, record)
            paths.append(path)
    return paths


def corrupt(path: Path, mutation: str) -> None:
    """Mutate one receipt at a time so no unrelated rejection hides the gate."""
    record = json.loads(path.read_text())
    if mutation == "energy":
        record["observations"][0]["energy"] += 1e-6
    elif mutation == "force":
        record["observations"][0]["forces"][0][0] = 1e-5
    elif mutation == "nan":
        record["observations"][0]["forces"][0][0] = float("nan")
    elif mutation == "time":
        record["observations"][0]["seconds"] = float("inf")
    elif mutation == "phase":
        record["observations"].pop()
    elif mutation == "work":
        record["observations"][0]["fock_builds"] = 0
    elif mutation == "settings":
        record["observations"][0]["settings"]["grid"] = "changed"
    elif mutation == "oracle":
        record["oracle"]["method"] = "m06-2x"
    elif mutation == "source":
        record["source_identity"] = "d" * 64
    elif mutation == "control":
        record["occupied_xc_env"] = "1"
    else:
        record[mutation] = not record[mutation]
    save(path, record)


def test_complete_pairs_replay_work_vectors_and_variability(tmp_path: Path) -> None:
    summary = replay(receipts(tmp_path))
    assert summary["independent_max_errors"] == [0.0, 0.0]
    for phase in summary["phases"].values():
        assert phase["profitable"] and not phase["regresses"]
        assert phase["improvement_fraction"] == pytest.approx(0.1)
        assert phase["variability_floor"] == 0.02
        assert phase["iterations_fock_builds"]["default"] == [[10, 10]] * 3


@pytest.mark.parametrize(
    "mutation",
    [
        "energy",
        "force",
        "nan",
        "time",
        "phase",
        "work",
        "settings",
        "oracle",
        "source",
        "control",
        "completed",
        "traced",
    ],
)
def test_replay_rejects_corrupted_observations(tmp_path: Path, mutation: str) -> None:
    paths = receipts(tmp_path)
    corrupt(paths[-1], mutation)
    with pytest.raises(ValueError):
        replay(paths)


def test_large_independent_gate_does_not_relax_paired_gate(tmp_path: Path) -> None:
    paths = receipts(tmp_path, atoms=12)
    corrupt(paths[-1], "force")
    with pytest.raises(ValueError, match="force acceptance"):
        replay(paths)


def test_incomplete_populations_cannot_claim_a_gain(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="three complete"):
        replay(receipts(tmp_path)[:-2])


def test_replay_gates_survive_optimized_python(tmp_path: Path) -> None:
    paths = receipts(tmp_path)
    command = [
        sys.executable,
        "-O",
        "-m",
        "benchmarks.occupied_xc_functionals",
        "replay",
        *map(str, paths),
    ]
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    good = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=False
    )
    assert good.returncode == 0, good.stderr
    corrupt(paths[-1], "energy")
    bad = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=False
    )
    assert bad.returncode != 0 and "energy gate failed" in bad.stderr


def test_retained_survey_preserves_wins_and_counterexamples() -> None:
    """Protect the replicated endpoint claim without deleting failed cohorts."""
    from benchmarks.occupied_xc_proposals import load

    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/occupied-xc-functionals-20261105/replay.py"
    )
    spec = importlib.util.spec_from_file_location("occupied_functional_survey", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = module.survey(load(path.parent / "survey.json.xz"))
    assert summary["scaling"]["discovery48"]["cold"]["profitable"]
    assert summary["scaling"]["confirmation48"]["cold"]["profitable"]
    assert not summary["scaling"]["size96"]["cold"]["profitable"]
    assert summary["functional"]["mn15-48"]["phases"]["moved"]["regresses"]
    assert not summary["functional"]["m06-2x-12-rks"]["qualified"]
