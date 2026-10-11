"""Qualify automatic CPU semilocal forces against independently reconverged SCF.

Both native-energy and PySCF/Libxc finite differences use the exact native moving
quadrature. Public cold, prepared replay and changed-geometry forces consume the
existing bounded stationary owner; no reference code enters production execution.
Receipts describe the measured functional/backend/spin/product only, not a public
positive allow-list, CUDA support, or second derivatives.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.common.provenance import atomic_json, canonical_hash
from generativeqc_compiler.method import resolve_bulk_ks
from generativeqc_compiler.xc.endpoint_capability import ENDPOINT_COVERAGE_SCHEMA
from generativeqc_compiler.xc.libxc_bulk_capabilities import (
    STAGE_EVIDENCE_SCHEMA,
    functional_capability,
)
from generativeqc_compiler.xc.libxc_work import LIBXC_WORK_DOMAIN

RESULT_SCHEMA = "generativeqc.libxc-automatic-force-result/v1"
QUALIFICATION_SCHEMA = "generativeqc.libxc-automatic-force-qualification/v1"
SPINS = ("polarized", "unpolarized")
PHASES = ("cold", "warm-replay", "changed-geometry")
GATES = {"energy_atol": 2e-7, "force_atol": 3e-5, "translation_atol": 2e-8}
FD_STEP = 1e-4
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _fixture(spin: str) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Use a genuine open-shell UKS fixture rather than two equal spin blocks."""
    return (
        ("He" if spin == "polarized" else "H", (0.0, 0.0, 0.0)),
        ("H", (0.1, 0.2, 1.4)),
    )


def _atoms(fixture: Any, coordinates: np.ndarray) -> tuple[Any, ...]:
    return tuple(
        (symbol, tuple(position))
        for (symbol, _), position in zip(fixture, coordinates, strict=True)
    )


def _reference_energy(name: str, atoms: Any, spin: str, grid: Any) -> float:
    """Reconverge the independent pinned Libxc oracle on the native quadrature."""
    from pyscf import __version__, dft, gto

    if __version__ != "2.14.0" or dft.libxc.__version__ != "7.0.0":
        raise RuntimeError("force qualification requires PySCF 2.14.0 / Libxc 7.0.0")
    molecule = gto.M(
        atom=atoms,
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
        spin=1 if spin == "polarized" else 0,
    )
    oracle = dft.UKS(molecule) if spin == "polarized" else dft.RKS(molecule)
    oracle.xc = name
    oracle.grids.coords = np.array(grid.points, copy=True)
    oracle.grids.weights = np.array(grid.weights, copy=True)
    oracle.small_rho_cutoff = 0
    oracle.conv_tol = 1e-12
    oracle.conv_tol_grad = 1e-10
    oracle.max_cycle = 200
    oracle.init_guess = "1e"
    energy = float(oracle.kernel())
    if not oracle.converged or not math.isfinite(energy):
        raise RuntimeError("independent force-oracle SCF did not converge")
    return energy


def _endpoint(
    calculator: Any,
    batch: Any,
    atoms: Any,
    name: str,
    spin: str,
    *,
    forces: bool,
    coordinates: Any = None,
) -> dict[str, Any]:
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO

    properties = ("energy", "forces") if forces else ("energy",)
    item = batch.execute(
        strict=True, properties=properties, coordinates=coordinates
    ).items[0]
    if not item.converged or item.executed_backend != "cpu_reference":
        raise RuntimeError(
            "force qualification requires a converged native CPU endpoint"
        )
    with NativeAO(
        atoms, basis="sto-3g", multiplicity=2 if spin == "polarized" else 1
    ) as basis:
        state = StationaryKsState.from_native(batch, basis)
        try:
            if (
                state._source.backend != "cpu"
                or state._source.method_ir != calculator.method_ir
            ):
                raise RuntimeError("force snapshot execution identity mismatch")
            return {
                "energy": float(item.energy),
                "reference_energy": _reference_energy(name, atoms, spin, state.grid),
                "forces": None if item.forces is None else item.forces.tolist(),
                "state_identity": state.identity.to_payload(),
                "coordinates": [list(position) for _, position in atoms],
            }
        finally:
            state._source.close()


def _finite_differences(
    calculator: Any, fixture: Any, coordinates: np.ndarray, name: str, spin: str
) -> dict[str, Any]:
    """Fresh native and independent solves at every displaced nuclear coordinate."""
    native = np.empty_like(coordinates)
    reference = np.empty_like(coordinates)
    stencil = []
    for atom in range(len(fixture)):
        for axis in range(3):
            endpoints = []
            for direction in (-1, 1):
                displaced = coordinates.copy()
                displaced[atom, axis] += direction * FD_STEP
                atoms = _atoms(fixture, displaced)
                with calculator.prepare_batch(
                    [atoms],
                    warm_start=False,
                    multiplicities=[2 if spin == "polarized" else 1],
                ) as batch:
                    endpoint = _endpoint(
                        calculator, batch, atoms, name, spin, forces=False
                    )
                endpoints.append(endpoint)
            minus, plus = endpoints
            native[atom, axis] = -(plus["energy"] - minus["energy"]) / (2 * FD_STEP)
            reference[atom, axis] = -(
                plus["reference_energy"] - minus["reference_energy"]
            ) / (2 * FD_STEP)
            stencil.append({"atom": atom, "axis": axis, "minus": minus, "plus": plus})
    return {
        "native_fd": native.tolist(),
        "reference_fd": reference.tolist(),
        "stencil": stencil,
    }


def _endpoint_matches(endpoint: Mapping[str, Any], resolution: Any) -> bool:
    """Require the same spin, ingredient, and functional source in every solve."""
    identity = endpoint.get("state_identity")
    if not isinstance(identity, Mapping):
        return False
    functional = resolution.plan.semilocal.functional
    if (
        identity.get("schema") != "generativeqc.stationary-ks-state/v2"
        or identity.get("spin") != resolution.method.spin
        or tuple(identity.get("ingredients", ())) != resolution.required_ingredients
        or identity.get("functional_identity") != functional.identity
    ):
        return False
    for field in (
        "owner",
        "solve_epoch",
        "density_generation",
        "fock_generation",
        "orbital_generation",
    ):
        if type(identity.get(field)) is not int or identity[field] < 1:
            return False
    for field in (
        "model_identity",
        "geometry_identity",
        "basis_identity",
        "overlap_identity",
        "grid_identity",
        "topology_identity",
        "provider_identity",
        "regularization_identity",
    ):
        if (
            not isinstance(identity.get(field), str)
            or _SHA256.fullmatch(identity[field]) is None
        ):
            return False
    energy, reference = endpoint.get("energy"), endpoint.get("reference_energy")
    return bool(
        type(energy) in (int, float)
        and type(reference) in (int, float)
        and math.isfinite(energy)
        and math.isfinite(reference)
        and abs(energy - reference) <= GATES["energy_atol"]
    )


def _row_passes(name: str, row: Mapping[str, Any]) -> bool:
    """Recompute numerical gates; a caller-supplied pass flag is not evidence."""
    try:
        if row.get("backend") != "cpu" or row.get("spin") not in SPINS:
            return False
        resolution = resolve_bulk_ks(name, backend="cpu", spin=row["spin"])
        if row.get("resolution") != resolution.to_payload() or not _endpoint_matches(
            row, resolution
        ):
            return False
        coordinates = np.asarray(row["coordinates"], dtype=float)
        if coordinates.shape != (2, 3) or not np.isfinite(coordinates).all():
            return False
        stencil = row["stencil"]
        if not isinstance(stencil, list) or len(stencil) != 6:
            return False
        seen = set()
        reconstructed_native = np.empty((2, 3))
        reconstructed_reference = np.empty((2, 3))
        for entry in stencil:
            atom, axis = entry["atom"], entry["axis"]
            if (
                type(atom) is not int
                or type(axis) is not int
                or atom not in (0, 1)
                or axis not in (0, 1, 2)
                or (atom, axis) in seen
            ):
                return False
            seen.add((atom, axis))
            for key, direction in (("minus", -1), ("plus", 1)):
                endpoint = entry[key]
                expected_coordinates = coordinates.copy()
                expected_coordinates[atom, axis] += direction * FD_STEP
                if not _endpoint_matches(endpoint, resolution) or not np.array_equal(
                    endpoint.get("coordinates"), expected_coordinates
                ):
                    return False
            minus, plus = entry["minus"], entry["plus"]
            reconstructed_native[atom, axis] = -(plus["energy"] - minus["energy"]) / (
                2 * FD_STEP
            )
            reconstructed_reference[atom, axis] = -(
                plus["reference_energy"] - minus["reference_energy"]
            ) / (2 * FD_STEP)
        forces = np.asarray(row["forces"], dtype=float)
        native = np.asarray(row["native_fd"], dtype=float)
        reference = np.asarray(row["reference_fd"], dtype=float)
        energy = float(row["energy"])
        oracle = float(row["reference_energy"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        forces.shape == native.shape == reference.shape == (2, 3)
        and np.array_equal(native, reconstructed_native)
        and np.array_equal(reference, reconstructed_reference)
        and all(np.isfinite(values).all() for values in (forces, native, reference))
        and math.isfinite(energy)
        and math.isfinite(oracle)
        and abs(energy - oracle) <= GATES["energy_atol"]
        and np.max(np.abs(forces - native)) <= GATES["force_atol"]
        and np.max(np.abs(forces - reference)) <= GATES["force_atol"]
        and np.max(np.abs(forces.sum(axis=0))) <= GATES["translation_atol"]
    )


def force_stage_evidence(
    name: str, result: Mapping[str, Any], *, evidence: str
) -> dict[str, Any]:
    """Bind a complete measured CPU force campaign to the capability identity."""
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("force evidence requires a non-empty artifact reference")
    capability = functional_capability(name)
    payload = {key: value for key, value in result.items() if key != "identity"}
    if (
        payload.get("schema") != RESULT_SCHEMA
        or payload.get("name") != capability.name
        or payload.get("subject_identity") != capability.identity
        or payload.get("point_domain") != LIBXC_WORK_DOMAIN
        or payload.get("gates") != GATES
        or payload.get("fd_step") != FD_STEP
        or payload.get("reference") != "PySCF-2.14.0/Libxc-7.0.0"
        or not isinstance(payload.get("library_identity"), str)
        or _SHA256.fullmatch(payload["library_identity"]) is None
        or result.get("identity") != canonical_hash(payload)
    ):
        raise ValueError("force result identity or execution contract mismatch")
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("force result rows must be a sequence of mappings")
    expected = {(spin, phase) for spin in SPINS for phase in PHASES}
    observed = [(row.get("spin"), row.get("phase")) for row in rows]
    passed = (
        len(observed) == len(expected)
        and set(observed) == expected
        and all(_row_passes(name, row) for row in rows)
    )
    if passed:
        by_endpoint = {(row["spin"], row["phase"]): row for row in rows}
        for spin in SPINS:
            cold, warm, changed = (by_endpoint[spin, phase] for phase in PHASES)
            passed = passed and (
                cold["coordinates"] == warm["coordinates"]
                and cold["state_identity"]["owner"] == warm["state_identity"]["owner"]
                and warm["state_identity"]["solve_epoch"]
                > cold["state_identity"]["solve_epoch"]
                and cold["state_identity"]["geometry_identity"]
                == warm["state_identity"]["geometry_identity"]
                and changed["coordinates"] != cold["coordinates"]
                and changed["state_identity"]["geometry_identity"]
                != cold["state_identity"]["geometry_identity"]
                and (
                    changed["state_identity"]["owner"]
                    != warm["state_identity"]["owner"]
                    or changed["state_identity"]["solve_epoch"]
                    > warm["state_identity"]["solve_epoch"]
                )
            )
    envelope = {
        "schema": STAGE_EVIDENCE_SCHEMA,
        "subject_identity": capability.identity,
        "stage": "forces",
        "status": "pass" if passed else "fail",
        "reason": None
        if passed
        else "incomplete lifecycle matrix or force numerical gates failed",
        "evidence": f"{evidence}#sha256={result['identity']}",
    }
    if passed:
        envelope["qualification"] = {
            "schema": ENDPOINT_COVERAGE_SCHEMA,
            "coverage": [
                {"backend": "cpu", "spin": spin, "products": ["energy", "forces"]}
                for spin in SPINS
            ],
            "result_schema": RESULT_SCHEMA,
            "result_identity": result["identity"],
            "qualification_schema": QUALIFICATION_SCHEMA,
        }
    return envelope


def qualify_forces(name: str, *, evidence: str) -> dict[str, Any]:
    """Exercise one structurally admitted registration through both spin layouts."""
    from generativeqc import Calculator, GridSpec, KsOptions, _native

    capability = functional_capability(name)
    rows = []
    library = _native.load_library(device="cpu")
    with Path(library._name).open("rb") as stream:
        library_identity = hashlib.file_digest(stream, "sha256").hexdigest()
    for spin in SPINS:
        resolution = resolve_bulk_ks(name, spin=spin, backend="cpu")
        calculator = Calculator(
            method=f"libxc-{'uks' if spin == 'polarized' else 'rks'}:{name}",
            basis="sto-3g",
            device="cpu",
            initial_guess=None,
            max_iterations=200,
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-14,
            ks_options=KsOptions(
                grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
            ),
        )
        fixture = _fixture(spin)
        coordinates = np.asarray([position for _, position in fixture])
        with calculator.prepare_batch(
            [fixture], warm_start=True, multiplicities=[2 if spin == "polarized" else 1]
        ) as batch:
            for phase in PHASES:
                replay = None
                if phase == "changed-geometry":
                    coordinates = coordinates.copy()
                    coordinates[1] += (0.03, -0.02, 0.07)
                    replay = [coordinates]
                atoms = _atoms(fixture, coordinates)
                endpoint = _endpoint(
                    calculator,
                    batch,
                    atoms,
                    name,
                    spin,
                    forces=True,
                    coordinates=replay,
                )
                differences = _finite_differences(
                    calculator, fixture, coordinates, name, spin
                )
                rows.append(
                    {
                        "backend": "cpu",
                        "spin": spin,
                        "phase": phase,
                        "resolution": resolution.to_payload(),
                        "coordinates": coordinates.tolist(),
                        **endpoint,
                        **differences,
                    }
                )
    result = {
        "schema": RESULT_SCHEMA,
        "name": capability.name,
        "subject_identity": capability.identity,
        "point_domain": LIBXC_WORK_DOMAIN,
        "library_identity": library_identity,
        "reference": "PySCF-2.14.0/Libxc-7.0.0",
        "gates": dict(GATES),
        "fd_step": FD_STEP,
        "rows": rows,
    }
    result["identity"] = canonical_hash(result)
    return {
        "result": result,
        "stage_evidence": force_stage_evidence(name, result, evidence=evidence),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--functional", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = qualify_forces(args.functional, evidence=str(args.output))
    atomic_json(args.output, result)
    return 0 if result["stage_evidence"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
