"""Moved-reference acceptance is independent of self-reported receipt gates."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ordered_diis_recheck",
    ROOT / "benchmarks/results/ordered-diis-cold-20261023/recheck.py",
)
assert SPEC and SPEC.loader
RECHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECHECK)
GATES = {"energy_per_atom": 2.5e-6, "force_max": 2.5e-5, "force_rms": 7.5e-6}


def run_case(gates: dict | None, energy: float = 0.0, force: float = 0.0) -> dict:
    """Exercise the complete moved branch with fresh matching receipt hashes."""
    record = {
        "status": "complete",
        "geometries_bohr": {"moved": [[0.0, 0.0, 0.0]]},
        "samples": [{"geometry": "moved", "energy": energy, "forces": [[force, 0, 0]]}],
        "references": {"moved": {"energy": 0.0, "forces": [[0.0, 0.0, 0.0]]}},
    }
    if gates is not None:
        record["gates"] = gates
    texts = {"candidate": json.dumps(record), "ordinary": json.dumps({"samples": []})}

    def entry(name: str) -> dict:
        return {
            "file": name,
            "sha256": hashlib.sha256(texts[name].encode()).hexdigest(),
        }

    summary = {
        "populations": [entry("ordinary")],
        "exploratory_populations": [entry("ordinary")],
        "profile_populations": [entry("ordinary")],
        "moved": [entry("candidate")],
        "qualified_complete_calls": 1,
        "comparisons": [],
    }
    # The ordinary qualification owner is unchanged; isolate moved admission.
    original = RECHECK.qualify
    RECHECK.qualify = lambda *args, **kwargs: [{"status": "qualified"}]
    try:
        return RECHECK.validate(summary, texts, [], {})
    finally:
        RECHECK.qualify = original


@pytest.mark.parametrize(
    "gates",
    [None, {}, {"force_max": 2.5e-5}, {**GATES, "extra": 1.0}]
    + [
        {**GATES, key: value}
        for key in GATES
        for value in (1.0, float("inf"), float("nan"))
    ],
)
def test_rejects_changed_gate_contract_with_valid_hash(gates: dict | None) -> None:
    with pytest.raises(RuntimeError, match="invalid moved gates"):
        run_case(gates)


@pytest.mark.parametrize(
    "energy,force", [(2.6e-6, 0.0), (0.0, 2.6e-5), (float("nan"), 0.0)]
)
def test_fixed_gates_reject_numerical_failure(energy: float, force: float) -> None:
    with pytest.raises(RuntimeError):
        run_case(GATES, energy, force)


def test_valid_moved_receipt_is_counted() -> None:
    assert run_case(GATES)["moved"] == 1
