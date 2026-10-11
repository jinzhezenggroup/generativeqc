"""Synthetic receipt/schema tests, not numerical force qualification."""

from __future__ import annotations

from copy import deepcopy
from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.method import resolve_bulk_ks
from generativeqc_compiler.xc.libxc_work import LIBXC_WORK_DOMAIN

from tools.qualify_libxc_forces import (
    FD_STEP,
    GATES,
    PHASES,
    RESULT_SCHEMA,
    SPINS,
    force_stage_evidence,
)

NAME = "GGA_X_APBE"

if TYPE_CHECKING:
    from generativeqc_compiler.method.bulk_ks import BulkKsResolution


@cache
def _resolution(spin: str) -> BulkKsResolution:
    return resolve_bulk_ks(NAME, spin=spin)


def _endpoint(spin: str, coordinates: np.ndarray) -> dict[str, Any]:
    resolution = _resolution(spin)
    identity = {
        field: "a" * 64
        for field in (
            "model_identity",
            "geometry_identity",
            "basis_identity",
            "overlap_identity",
            "grid_identity",
            "topology_identity",
            "provider_identity",
            "regularization_identity",
        )
    }
    identity.update(
        {
            field: 1
            for field in (
                "owner",
                "solve_epoch",
                "density_generation",
                "fock_generation",
                "orbital_generation",
            )
        }
    )
    identity.update(
        {
            "schema": "generativeqc.stationary-ks-state/v2",
            "spin": spin,
            "ingredients": list(resolution.required_ingredients),
            "functional_identity": resolution.plan.semilocal.functional.identity,
            "geometry_identity": canonical_hash(coordinates.tolist()),
        }
    )
    return {
        "energy": -1.0,
        "reference_energy": -1.0,
        "forces": np.zeros((2, 3)).tolist(),
        "coordinates": coordinates.tolist(),
        "state_identity": identity,
    }


def _result() -> dict[str, Any]:
    rows = []
    for spin in SPINS:
        for phase in PHASES:
            coordinates = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 1.4]])
            if phase == "changed-geometry":
                coordinates[1, 2] += 0.07
            stencil = []
            for atom in range(2):
                for axis in range(3):
                    minus, plus = coordinates.copy(), coordinates.copy()
                    minus[atom, axis] -= FD_STEP
                    plus[atom, axis] += FD_STEP
                    stencil.append(
                        {
                            "atom": atom,
                            "axis": axis,
                            "minus": _endpoint(spin, minus),
                            "plus": _endpoint(spin, plus),
                        }
                    )
            endpoint = _endpoint(spin, coordinates)
            if phase == "warm-replay":
                endpoint["state_identity"]["solve_epoch"] = 2
            elif phase == "changed-geometry":
                endpoint["state_identity"]["owner"] = 2
            rows.append(
                {
                    "backend": "cpu",
                    "spin": spin,
                    "phase": phase,
                    "resolution": _resolution(spin).to_payload(),
                    "stencil": stencil,
                    "native_fd": np.zeros((2, 3)).tolist(),
                    "reference_fd": np.zeros((2, 3)).tolist(),
                    **endpoint,
                }
            )
    result = {
        "schema": RESULT_SCHEMA,
        "name": NAME,
        "subject_identity": resolve_bulk_ks(NAME).capability.identity,
        "point_domain": LIBXC_WORK_DOMAIN,
        "library_identity": "b" * 64,
        "reference": "PySCF-2.14.0/Libxc-7.0.0",
        "gates": dict(GATES),
        "fd_step": FD_STEP,
        "rows": rows,
    }
    return _rehash(result)


def _rehash(result: dict[str, Any]) -> dict[str, Any]:
    result["identity"] = canonical_hash(
        {key: value for key, value in result.items() if key != "identity"}
    )
    return result


def _stage(result: dict[str, Any]) -> dict[str, Any]:
    return force_stage_evidence(NAME, result, evidence="artifact://force-result.json")


def test_complete_dual_spin_lifecycle_force_receipt_has_exact_cpu_coverage() -> None:
    result = _result()
    envelope = _stage(result)
    assert envelope["status"] == "pass"
    assert envelope["evidence"].endswith(f"#sha256={result['identity']}")
    assert all(row["backend"] == "cpu" for row in envelope["qualification"]["coverage"])


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-phase",
        "duplicate-phase",
        "wrong-backend",
        "wrong-spin",
        "wrong-method",
        "wrong-functional",
        "wrong-domain",
        "missing-stencil",
        "duplicate-stencil",
        "wrong-displacement",
        "invented-fd",
        "wrong-force",
        "failed-energy",
        "unchanged-geometry",
        "missing-epoch",
        "forged-warm-solve",
        "forged-changed-owner",
    ],
)
def test_rehashed_bad_force_coverage_cannot_promote(mutation: str) -> None:
    result = deepcopy(_result())
    row = result["rows"][0]
    if mutation == "missing-phase":
        result["rows"].pop()
    elif mutation == "duplicate-phase":
        result["rows"][-1] = deepcopy(result["rows"][0])
    elif mutation == "wrong-backend":
        row["backend"] = "cuda"
    elif mutation == "wrong-spin":
        row["state_identity"]["spin"] = "unpolarized"
    elif mutation == "wrong-method":
        row["resolution"]["method_identity"] = "d" * 64
    elif mutation == "wrong-functional":
        row["state_identity"]["functional_identity"] = "d" * 64
    elif mutation == "wrong-domain":
        row["resolution"]["required_ingredients"] = ["rho", "lapl"]
    elif mutation == "missing-stencil":
        row["stencil"].pop()
    elif mutation == "duplicate-stencil":
        row["stencil"][-1] = deepcopy(row["stencil"][0])
    elif mutation == "wrong-displacement":
        row["stencil"][0]["plus"]["coordinates"][0][0] += FD_STEP
    elif mutation == "invented-fd":
        row["reference_fd"][0][0] = 1e-6
    elif mutation == "wrong-force":
        row["forces"][0][0] = 1
    elif mutation == "failed-energy":
        row["stencil"][0]["plus"]["reference_energy"] += 0.1
    elif mutation == "unchanged-geometry":
        result["rows"][2]["state_identity"]["geometry_identity"] = row[
            "state_identity"
        ]["geometry_identity"]
    elif mutation == "missing-epoch":
        del row["state_identity"]["solve_epoch"]
    elif mutation == "forged-warm-solve":
        result["rows"][1]["state_identity"]["solve_epoch"] = row["state_identity"][
            "solve_epoch"
        ]
    elif mutation == "forged-changed-owner":
        result["rows"][2]["state_identity"]["owner"] = row["state_identity"]["owner"]
    envelope = _stage(_rehash(result))
    assert envelope["status"] == "fail"
    assert "qualification" not in envelope


def test_stale_force_receipt_digest_is_rejected() -> None:
    result = _result()
    result["rows"][0]["forces"][0][0] = 1
    with pytest.raises(ValueError, match="identity"):
        _stage(result)


def test_snapshot_tuple_and_json_list_ingredients_bind_identically() -> None:
    result = _result()
    for row in result["rows"]:
        row["state_identity"]["ingredients"] = tuple(
            row["state_identity"]["ingredients"]
        )
        for entry in row["stencil"]:
            for key in ("minus", "plus"):
                endpoint = entry[key]
                endpoint["state_identity"]["ingredients"] = tuple(
                    endpoint["state_identity"]["ingredients"]
                )
    assert _stage(_rehash(result))["status"] == "pass"


@pytest.mark.parametrize(
    "field,value",
    [
        ("subject_identity", "d" * 64),
        ("point_domain", "interior-only"),
        ("library_identity", "unknown"),
        ("reference", "same-graph-interpreter"),
        ("fd_step", 1),
        ("gates", {"force_atol": 1}),
    ],
)
def test_rehashed_wrong_force_execution_contract_is_rejected(
    field: str, value: Any
) -> None:
    result = _result()
    result[field] = value
    with pytest.raises(ValueError, match="execution contract"):
        _stage(_rehash(result))
