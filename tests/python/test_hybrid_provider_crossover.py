"""Offline controls for #2054 benchmark acceptance and provider isolation."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tools.benchmark_hybrid_provider_crossover import (
    ARMS,
    SCHEMA,
    case_named,
    frozen_problem,
    moved,
    summarize,
)


def _records() -> tuple[list[dict], list[dict], list[dict]]:
    problem = {"method": "pbe0-rks", "geometry_bohr": [["H", [0, 0, 0]]], "auxiliary_basis": "def2-svp", "df_memory_budget_bytes": 1024}
    source = {"revision": "a" * 40, "dirty": False}
    library = {"sha256": "b" * 64}
    attempts = []
    for phase in ("cold", "warm-0", "changed-geometry", "moved-warm"):
        attempts.append({"phase": phase, "status": "PASS", "seconds": 1.0, "energy_hartree": -1.0, "forces_hartree_per_bohr": [[0.0, 0.0, 0.0]], "physical_residual_rms": 1e-10})
    grids = {"original": {"sha256": "c" * 64}, "moved": {"sha256": "d" * 64}}
    records = []
    oracles = []
    profiles = []
    for arm in ("direct", "df-jk-occupied"):
        records.append({
            "schema": SCHEMA, "kind": "native", "case": "water-3", "arm": arm,
            "source": deepcopy(source), "library": deepcopy(library), "device": {"name": "GPU"},
            "problem": deepcopy(problem), "status": "MEASURED", "attempts": deepcopy(attempts),
            "counts": {"nao": 24, "naux": 24 if arm == "df-jk-occupied" else None, "nocc": 5},
            "provider_proof": dict(zip(("coulomb", "exchange"), ARMS[arm]["proof"], strict=True)),
            "oracle_grids": deepcopy(grids),
        })
        oracles.append({
            "schema": SCHEMA, "kind": "pyscf-oracle", "case": "water-3", "arm": arm,
            "source": deepcopy(source), "library": deepcopy(library), "problem": deepcopy(problem),
            "grid_exports": deepcopy(grids), "status": "MEASURED",
            "samples": [
                {"geometry": label, "status": "PASS", "energy_hartree": -1.0,
                 "forces_hartree_per_bohr": [[0.0, 0.0, 0.0]]}
                for label in ("original", "moved")
            ],
        })
        profiles.append({
            "schema": SCHEMA, "kind": "diagnostic-profile", "case": "water-3", "arm": arm,
            "source": deepcopy(source), "library": deepcopy(library), "problem": deepcopy(problem),
            "status": "MEASURED", "occupied_reuse_verified": arm == "df-jk-occupied",
            "executed_counters": None, "metric_diagnostics": None,
        })
    records.append({
        "schema": SCHEMA, "kind": "native", "case": "water-3", "arm": "df-j-exact-k",
        "source": deepcopy(source), "problem": deepcopy(problem), "status": "UNSUPPORTED",
    })
    return records, oracles, profiles


def test_frozen_cases_preserve_large_basis_geometry() -> None:
    case = case_named("water-48")
    assert len(case.atoms) == 48
    assert len(moved(case).atoms) == 48
    problem = frozen_problem(case, auxiliary="def2-svp", budget=1 << 30)
    assert problem["method"] == "pbe0-rks"
    assert problem["grid_spec"]["radial_points"] == 48


def test_supported_arms_require_oracle_and_never_claim_full_crossover() -> None:
    records, oracles, profiles = _records()
    summary = summarize(records, oracles, profiles)
    assert summary["status"] == "PILOT_ACCEPTED"
    assert summary["arms"]["df-j-exact-k"]["status"] == "UNSUPPORTED"
    assert summary["crossover_claim_eligible"] is False
    assert summarize(records, [oracles[0]], profiles)["status"] == "INCOMPLETE"
    assert summarize(records, oracles, [profiles[0]])["status"] == "INCOMPLETE"


@pytest.mark.parametrize("mutate", [
    lambda rows, refs: rows[1]["provider_proof"].update(exchange="exact"),
    lambda rows, refs: rows[1]["source"].update(revision="new-head"),
    lambda rows, refs: rows[0]["attempts"][1].update(status="FAIL"),
    lambda rows, refs: rows[0]["attempts"][1].update(physical_residual_rms=1e-4),
    lambda rows, refs: refs[1]["samples"][0].update(energy_hartree=-0.9),
    lambda rows, refs: refs[0]["grid_exports"]["original"].update(sha256="wrong"),
])
def test_mismatch_or_failure_cannot_pass(mutate) -> None:
    records, oracles, profiles = _records()
    mutate(records, oracles)
    assert summarize(records, oracles, profiles)["status"] == "INCOMPLETE"
