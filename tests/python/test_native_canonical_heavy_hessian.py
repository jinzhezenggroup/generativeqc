"""Opt-in real-element native CPU curvature, not TS or scale qualification."""

from __future__ import annotations

import ctypes as ct
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import Atom, Calculator, KsOptions, analyze_harmonic_modes
from generativeqc._dft_gradient import StationaryKsState
from generativeqc._native_cpu_hessian import execute_batch
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.dft import NativeAO
from generativeqc_compiler.dft.grid import MolecularGrid
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method._generated_parameters import d3_parameters
from test_global_hybrid_ecp_basis import canonical_records
from test_global_hybrid_ecp_forces import independent_solver
from test_integral_autodiff_second import require_native_library  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests/data/global_hybrid_ecp/native_curvature_manifest.json"
INDEPENDENT_POLICY = ROOT / "tests/data/global_hybrid_ecp/independent_scf_policy.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _independent_vibrational_modes(
    matrix: np.ndarray, positions: np.ndarray, masses: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve the physical mass-weighted null space without production mode code.

    One SVD of all geometric rigid vectors defines the complement, unlike the
    production translation/rotation/QR construction. This is an explicit mode
    convention, not a modification of the independently qualified raw Hessian.
    """
    centered = positions - np.average(positions, axis=0, weights=masses)
    translations = np.tile(np.eye(3), (len(positions), 1))
    rotations = np.column_stack(
        [np.cross(axis, centered).ravel() for axis in np.eye(3)]
    )
    roots = np.sqrt(np.repeat(masses, 3))
    rigid = roots[:, None] * np.column_stack((translations, rotations))
    left, singular, _ = np.linalg.svd(rigid, full_matrices=True)
    rank = int(np.count_nonzero(singular > singular.max() * 1e-10))
    complement = left[:, rank:]
    weighted = (matrix + matrix.T) * 0.5 / roots[:, None] / roots[None, :]
    values, vectors = np.linalg.eigh(complement.T @ weighted @ complement)
    factor = np.sqrt(4.3597447222060e-18 / (1.66053906892e-27 * 5.29177210544e-11**2))
    factor /= 2 * np.pi * 299792458 * 100
    frequencies = np.sign(values) * np.sqrt(abs(values)) * factor
    return frequencies, (complement @ vectors).T, rank


def test_independent_modes_preserve_signs_and_rotation_covariance() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [2.6, 0.0, -0.8], [0.3, 0.0, 4.5]])
    masses = np.array([57.93534241, 1.00782503223, 126.9044719])
    matrix = np.diag(np.array([-1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))
    frequencies, modes, rank = _independent_vibrational_modes(matrix, positions, masses)
    assert rank == 6 and frequencies.shape == (3,)
    np.testing.assert_allclose(modes @ modes.T, np.eye(3), atol=1e-13, rtol=0)
    rotation = np.array([[0.8, 0.0, 0.6], [0.0, 1.0, 0.0], [-0.6, 0.0, 0.8]])
    transform = np.kron(np.eye(3), rotation)
    rotated, _, rotated_rank = _independent_vibrational_modes(
        transform @ matrix @ transform.T, positions @ rotation.T, masses
    )
    assert rotated_rank == rank
    np.testing.assert_allclose(rotated, frequencies, atol=1e-10, rtol=0)
    negative, _, _ = _independent_vibrational_modes(-matrix, positions, masses)
    np.testing.assert_allclose(negative, -frequencies[::-1], atol=1e-10, rtol=0)


def _gradient_offsets(order: int) -> tuple[int, ...]:
    """Freeze the complete independent-gradient stencil before any SCF work."""
    if type(order) is not int:
        raise TypeError("independent stencil order must be an integer")
    if order == 2:
        return (-1, 1)
    if order == 4:
        return (-2, -1, 1, 2)
    raise ValueError("independent gradient stencil order must be 2 or 4")


def _gradient_column(
    samples: dict[int, np.ndarray], step: float, order: int
) -> np.ndarray:
    """Differentiate independently converged gradients, never production SCFs."""
    offsets = _gradient_offsets(order)
    if set(samples) != set(offsets):
        raise ValueError("incomplete independent gradient stencil")
    if isinstance(step, bool) or not np.isfinite(step) or step <= 0:
        raise ValueError("independent gradient step must be positive and finite")
    values = {offset: np.asarray(samples[offset]) for offset in offsets}
    shape = values[offsets[0]].shape
    if any(
        value.shape != shape
        or not value.size
        or value.dtype.kind not in "fiu"
        or not np.isfinite(value).all()
        for value in values.values()
    ):
        raise ValueError("invalid independent gradient samples")
    values = {
        offset: value.astype(np.float64, copy=False) for offset, value in values.items()
    }
    if order == 2:
        return (values[1] - values[-1]) / (2 * step)
    return (8 * (values[1] - values[-1]) - (values[2] - values[-2])) / (12 * step)


def _independent_state(solver: Any, gradient: np.ndarray, correction: float) -> dict:
    """Retain the cold solver's actual state, not just its convergence flag."""
    return {
        "geometry": solver.mol.atom_coords(),
        "density": solver.make_rdm1(),
        "coefficients": solver.mo_coeff,
        "occupations": solver.mo_occ,
        "orbital_energies": solver.mo_energy,
        "overlap": solver.get_ovlp(),
        "electronic_energy": solver.e_tot,
        "d3_energy": correction,
        "gradient": gradient,
        "converged": solver.converged,
        "scf_iterations": solver.cycles,
        "orbital_gradient_norm": np.linalg.norm(
            solver.get_grad(solver.mo_coeff, solver.mo_occ)
        ),
    }


def _occupied_root_overlap(central: Any, displaced: Any) -> float:
    """Compare occupied subspaces in the cross-geometry AO metric without seeding."""
    from pyscf import gto

    nao = central.mol.nao_nr()
    central_occupations = np.asarray(central.mo_occ).reshape(-1, nao)
    displaced_occupations = np.asarray(displaced.mo_occ).reshape(-1, nao)
    if not np.array_equal(central_occupations, displaced_occupations):
        return 0.0
    central_coefficients = np.asarray(central.mo_coeff).reshape(-1, nao, nao)
    displaced_coefficients = np.asarray(displaced.mo_coeff).reshape(-1, nao, nao)
    cross = gto.intor_cross("int1e_ovlp_sph", central.mol, displaced.mol)
    minimum = 1.0
    for occupations, before, after in zip(
        central_occupations,
        central_coefficients,
        displaced_coefficients,
        strict=True,
    ):
        occupied = occupations > 0
        if np.any(occupied):
            singular = np.linalg.svd(
                before[:, occupied].T @ cross @ after[:, occupied], compute_uv=False
            )
            minimum = min(minimum, float(singular.min()))
    return minimum


def _native_snapshot_state_metrics(snapshot: dict, reference: Any) -> dict:
    """Compare actual same-geometry native orbitals, not reconstructed roots.

    Coefficients use the canonical shared real-spherical AO order and native
    row-major AO-by-MO packing. A root overlap alone is insufficient: scaled or
    otherwise nonorthonormal native coefficients must fail the independent
    density, metric-orthonormality and electron-count gates as well.
    """
    if not np.array_equal(snapshot["coordinates_bohr"], reference.mol.atom_coords()):
        raise ValueError("native orbital snapshot has a different geometry")
    basis = reference.mol.nao_nr()
    reference_coefficients = np.asarray(reference.mo_coeff).reshape(-1, basis, basis)
    reference_occupations = np.asarray(reference.mo_occ).reshape(-1, basis)
    spins = len(reference_coefficients)
    if snapshot["basis"] != basis or snapshot["spins"] != spins:
        raise ValueError("native orbital snapshot has a different spin/AO shape")
    occupied_counts = snapshot["occupied_counts"]
    if occupied_counts != np.count_nonzero(reference_occupations, axis=1).tolist():
        raise ValueError("native orbital snapshot has different occupied populations")
    occupation = 2.0 if spins == 1 else 1.0
    if not np.all((reference_occupations == 0) | (reference_occupations == occupation)):
        raise ValueError("independent orbitals do not have integral spin occupations")
    coefficients = np.asarray(snapshot["coefficients"], dtype=float)
    if coefficients.size != spins * basis * basis or not np.all(
        np.isfinite(coefficients)
    ):
        raise ValueError("invalid native orbital snapshot coefficients")
    coefficients = coefficients.reshape(spins, basis, basis)
    overlap = np.asarray(reference.get_ovlp())
    reference_density = np.asarray(reference.make_rdm1()).reshape(spins, basis, basis)
    metrics = {
        "density": 0.0,
        "native_orthonormality": 0.0,
        "native_electron_count": 0.0,
        "minimum_occupied_root_overlap": 1.0,
    }
    for count, native, independent, occupations, density in zip(
        occupied_counts,
        coefficients,
        reference_coefficients,
        reference_occupations,
        reference_density,
        strict=True,
    ):
        native_density = occupation * native[:, :count] @ native[:, :count].T
        metrics["density"] = max(
            metrics["density"], float(np.max(abs(native_density - density)))
        )
        metrics["native_orthonormality"] = max(
            metrics["native_orthonormality"],
            float(np.max(abs(native.T @ overlap @ native - np.eye(basis)))),
        )
        metrics["native_electron_count"] = max(
            metrics["native_electron_count"],
            abs(float(np.trace(native_density @ overlap)) - occupation * count),
        )
        if count:
            singular = np.linalg.svd(
                native[:, :count].T @ overlap @ independent[:, occupations > 0],
                compute_uv=False,
            )
            metrics["minimum_occupied_root_overlap"] = min(
                metrics["minimum_occupied_root_overlap"], float(singular.min())
            )
    return metrics


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("change", ["phase", "occupied-rotation", "root", "scale"])
def test_native_snapshot_uses_metric_and_checks_each_occupied_spin(
    unrestricted: bool,
    change: str,
) -> None:
    """Gauge changes pass; wrong spin roots and scaled orbitals remain visible."""
    overlap = np.diag([2.0, 3.0, 5.0])
    spins = 2 if unrestricted else 1
    coefficients = np.tile(np.diag(1 / np.sqrt(np.diag(overlap))), (spins, 1, 1))
    occupations = (
        np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        if unrestricted
        else np.array([[2.0, 2.0, 0.0]])
    )
    density = np.einsum("sik,sk,sjk->sij", coefficients, occupations, coefficients)
    positions = np.array([[0.1, -0.2, 0.3]])
    reference = SimpleNamespace(
        mol=SimpleNamespace(nao_nr=lambda: 3, atom_coords=lambda: positions.copy()),
        mo_coeff=coefficients,
        mo_occ=occupations,
        get_ovlp=lambda: overlap.copy(),
        make_rdm1=lambda: density.copy(),
    )
    native = coefficients.copy()
    if change == "phase":
        native *= -1
    elif change == "occupied-rotation":
        rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
        native[0, :, :2] = native[0, :, :2] @ rotation
    elif change == "root":
        native[-1] = native[-1][:, [2, 1, 0]]
    else:
        native[-1] *= 2
    snapshot = {
        "coordinates_bohr": positions.tolist(),
        "basis": 3,
        "spins": spins,
        "occupied_counts": [2, 1] if unrestricted else [2],
        "coefficients": native.ravel().tolist(),
    }
    metrics = _native_snapshot_state_metrics(snapshot, reference)
    if change in ("phase", "occupied-rotation"):
        assert metrics["density"] < 1e-14
        assert metrics["native_orthonormality"] < 1e-14
        assert metrics["native_electron_count"] < 1e-14
        assert metrics["minimum_occupied_root_overlap"] == pytest.approx(1.0)
    elif change == "root":
        assert metrics["minimum_occupied_root_overlap"] == 0
        assert metrics["density"] > 0.1
    else:
        assert metrics["native_orthonormality"] > 2
        assert metrics["native_electron_count"] > 2
        assert metrics["density"] > 0.1


@pytest.mark.parametrize(
    "change", ["geometry", "population", "spins", "basis", "nonfinite"]
)
def test_native_snapshot_refuses_unmatched_state_binding(change: str) -> None:
    """No overlap metric should be reported for a differently bound state."""
    positions = np.zeros((1, 3))
    reference = SimpleNamespace(
        mol=SimpleNamespace(nao_nr=lambda: 2, atom_coords=lambda: positions.copy()),
        mo_coeff=np.eye(2),
        mo_occ=np.array([2.0, 0.0]),
        get_ovlp=lambda: np.eye(2),
        make_rdm1=lambda: np.diag([2.0, 0.0]),
    )
    snapshot = {
        "coordinates_bohr": positions.tolist(),
        "basis": 2,
        "spins": 1,
        "occupied_counts": [1],
        "coefficients": np.eye(2).ravel().tolist(),
    }
    if change == "geometry":
        snapshot["coordinates_bohr"][0][0] = 0.1
    elif change == "population":
        snapshot["occupied_counts"] = [2]
    elif change in ("spins", "basis"):
        snapshot[change] = 3
    else:
        snapshot["coefficients"][0] = np.nan
    with pytest.raises(ValueError):
        _native_snapshot_state_metrics(snapshot, reference)


def test_independent_state_retains_solver_and_dispersion_separately() -> None:
    """Raw displaced receipts must retain enough data to audit the SCF root."""
    solver = SimpleNamespace(
        mol=SimpleNamespace(atom_coords=lambda: np.zeros((1, 3))),
        make_rdm1=lambda: np.eye(2),
        mo_coeff=np.eye(2),
        mo_occ=np.array([2.0, 0.0]),
        mo_energy=np.array([-0.5, 0.2]),
        get_ovlp=lambda: np.eye(2),
        e_tot=-1.0,
        converged=True,
        cycles=7,
        get_grad=lambda *args: np.array([3e-11, 4e-11]),
    )
    gradient = np.array([[0.1, 0.2, 0.3]])
    receipt = _independent_state(solver, gradient, -0.01)
    np.testing.assert_array_equal(receipt["density"], np.eye(2))
    np.testing.assert_array_equal(receipt["occupations"], solver.mo_occ)
    np.testing.assert_array_equal(receipt["gradient"], gradient)
    assert receipt["electronic_energy"] == -1.0
    assert receipt["d3_energy"] == -0.01
    assert receipt["converged"] and receipt["scf_iterations"] == 7
    assert receipt["orbital_gradient_norm"] == pytest.approx(5e-11)


@pytest.mark.parametrize("change", ["phase", "occupation", "beta-subspace"])
def test_occupied_root_overlap_checks_both_spins(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Orbital phases are harmless; changed occupations or beta roots are not."""
    gto = pytest.importorskip("pyscf.gto")
    monkeypatch.setattr(gto, "intor_cross", lambda *args: np.eye(3))
    central = SimpleNamespace(
        mol=SimpleNamespace(nao_nr=lambda: 3),
        mo_coeff=np.array([np.eye(3), np.eye(3)]),
        mo_occ=np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    displaced = SimpleNamespace(
        mol=central.mol,
        mo_coeff=central.mo_coeff.copy(),
        mo_occ=central.mo_occ.copy(),
    )
    if change == "phase":
        displaced.mo_coeff *= -1
    elif change == "occupation":
        displaced.mo_occ[1] = [1.0, 1.0, 0.0]
    else:
        displaced.mo_coeff[1] = displaced.mo_coeff[1, :, [2, 1, 0]].T
    assert _occupied_root_overlap(central, displaced) == (
        1.0 if change == "phase" else 0.0
    )


def _independent_gradient(
    raw: dict,
    atoms: list,
    charge: int,
    multiplicity: int,
    grid_spec: Any,
    atomic_weights: np.ndarray,
    damping: Any,
    gate: float,
    *,
    max_iterations: int = 240,
    orbital_gradient_tolerance: float = 1e-10,
    final_diagonalization: bool = True,
    diis_factory: Any = None,
) -> tuple:
    """Cold independent SCF/gradient on the same moving discretized quadrature."""
    from dftd3 import interface
    from pyscf.dft import gen_grid

    grid = MolecularGrid(tuple(atoms), grid_spec, charge, multiplicity).explicit()
    source = SimpleNamespace(grid_spec=grid_spec, atomic_weights=atomic_weights)
    solver = independent_solver(
        raw,
        atoms,
        charge,
        multiplicity,
        SimpleNamespace(grid=grid, _source=source),
        max_iterations=max_iterations,
        orbital_gradient_tolerance=orbital_gradient_tolerance,
        final_diagonalization=final_diagonalization,
        diis_factory=diis_factory,
    )
    points, weights = gen_grid.get_partition(
        solver.grids.mol,
        solver.grids.gen_atomic_grids(),
        radii_adjust=solver.grids.radii_adjust,
        atomic_radii=solver.grids.atomic_radii,
    )
    np.testing.assert_allclose(points, grid.points, atol=1e-12, rtol=0)
    envelope = np.max(abs(solver._numint.eval_ao(solver.mol, points, deriv=1)), axis=0)
    bound = envelope.T @ (abs(weights - grid.weights)[:, None] * envelope)
    scale = envelope.T @ (abs(weights)[:, None] * envelope)
    assert np.max(bound) <= gate * max(1.0, float(np.max(scale)))
    gradient = solver.nuc_grad_method()
    gradient.grid_response = True
    electronic = gradient.kernel()
    numbers = np.array([Atom.from_value(atom).atomic_number for atom in atoms])
    correction = interface.DispersionModel(
        numbers, np.asarray([center for _, center in atoms])
    ).get_dispersion(damping, grad=True)
    return solver, electronic + correction["gradient"], float(correction["energy"])


@pytest.mark.ci_nightly
@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RUN_NATIVE_CANONICAL_HEAVY") != "1",
    reason="complete real-element native CPU qualification is opt-in",
)
@pytest.mark.parametrize(
    "case", ["cs-rks", "cs-uks", "i-rks", "i-uks", "ni-uks", "ni-i-stationary-rks"]
)
def test_native_canonical_heavy_curvature(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Qualify retained-batch or complete public endpoints against cold states."""
    pytest.importorskip("pyscf")
    interface = pytest.importorskip("dftd3.interface")
    manifest = json.loads(MANIFEST.read_text())
    specification = manifest["cases"][case]
    independent_policy = json.loads(INDEPENDENT_POLICY.read_text())
    independent_policy = {
        **independent_policy["defaults"],
        **independent_policy["cases"].get(case, {}),
    }
    stencil_order = independent_policy.pop("gradient_stencil_order")
    independent_max_iterations = independent_policy.pop(
        "max_iterations", specification["native_scf_max_iterations"]
    )
    diis_name = independent_policy.pop("diis", None)
    diis_factory = None
    if diis_name is not None:
        if diis_name != "common-gram-normalization":
            raise ValueError("unknown independent DIIS policy")
        from independent_diis import normalized_diis_factory

        diis_factory = normalized_diis_factory
    offsets = _gradient_offsets(stencil_order)
    gates = manifest["gates"]
    public = os.environ.get("GENERATIVEQC_NATIVE_CANONICAL_ENDPOINT", "batch")
    if public not in ("batch", "public"):
        raise ValueError(
            "GENERATIVEQC_NATIVE_CANONICAL_ENDPOINT must be batch or public"
        )
    public = public == "public"
    basis_manifest, raw, table = canonical_records.__wrapped__()
    atoms = [(symbol, tuple(center)) for symbol, center in specification["atoms_bohr"]]
    native_atoms = tuple(Atom.from_value(atom) for atom in atoms)
    numbers = {atom.atomic_number for atom in native_atoms}
    table = replace(
        table,
        elements=tuple(
            item for item in table.elements if item.atomic_number in numbers
        ),
    )
    assert table.elements[0].ecp_core_electrons == 0
    assert {item.ecp_core_electrons for item in table.elements} == {
        0,
        specification["ecp_core"],
    }
    charge, multiplicity = specification["charge"], specification["multiplicity"]
    calculator = Calculator(
        method=resolve_method(
            manifest["method"], spin="polarized" if multiplicity == 2 else "unpolarized"
        ),
        basis=table,
        basis_representation="spherical",
        device="cpu",
        ks_options=KsOptions(),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=specification["native_scf_max_iterations"],
    )
    grid_spec = calculator.ks_options.grid
    assert grid_spec.version == 4 and grid_spec.radial_points == 3
    library = Path(calculator._library._name).resolve()
    calculator._library.generativeqc_get_source_identity.restype = ct.c_char_p
    started = time.perf_counter()
    receipt = {
        "accepted": False,
        "case": case,
        "public_domain_qualified": False,
        "requested_endpoint": "public" if public else "batch",
        "ts_qualified": False,
        "scale_qualified": False,
        "installed_qualified": False,
        "production_displaced_scfs": 0,
        "oracle_scf_cross_seeding": False,
        "method_identity": calculator.method_ir.identity,
        "basis_manifest": basis_manifest,
        "atoms_bohr": atoms,
        "charge": charge,
        "multiplicity": multiplicity,
        "max_eri_primitives": specification["max_eri_primitives"],
        "native_scf_policy": {
            "energy_tolerance": 1e-13,
            "density_tolerance": 1e-11,
            "max_iterations": specification["native_scf_max_iterations"],
        },
        "independent_scf_policy": {
            "energy_tolerance": 1e-13,
            **independent_policy,
            "gradient_stencil_order": stencil_order,
            "max_iterations": independent_max_iterations,
            "diis": diis_name or "upstream",
        },
        "gates": gates,
        "source_hashes": {
            str(path.relative_to(ROOT)): _sha(path)
            for path in (
                Path(__file__),
                MANIFEST,
                INDEPENDENT_POLICY,
                ROOT / "tests/python/test_global_hybrid_ecp_forces.py",
                ROOT / "python/generativeqc/_native_cpu_hessian.py",
                *((ROOT / "tests/python/independent_diis.py",) if diis_factory else ()),
            )
        },
        "library_path": str(library),
        "library_sha256": _sha(library),
        "native_source_identity": calculator._library.generativeqc_get_source_identity().decode(),
    }

    def retain(stage: str) -> None:
        receipt.update(stage=stage, seconds=time.perf_counter() - started)
        (tmp_path / "receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("native production entered a Python numerical owner")

    from generativeqc.batch import PreparedBatch
    from generativeqc.composite_rks_hessian import CompositeRKSResponse

    monkeypatch.setattr(PreparedBatch, "_public_dft_cpu_force", forbidden)
    monkeypatch.setattr(CompositeRKSResponse, "from_native", forbidden)
    monkeypatch.setattr("generativeqc.d3_hessian.compose_second_order", forbidden)
    parameters = d3_parameters(manifest["method"])
    damping = interface.RationalDampingParam(
        **{name: parameters[name] for name in ("s6", "s8", "a1", "a2", "s9")}
    )
    positions = np.asarray([atom.position for atom in native_atoms])
    directions = tuple(
        np.ascontiguousarray(
            function(np.arange(positions.size) + 1).reshape(positions.shape) * 0.1
        )
        for function in (np.sin, np.cos)
    )
    retain("frozen-manifest")
    try:
        with (
            calculator.prepare_batch(
                [native_atoms],
                charges=[charge],
                multiplicities=[multiplicity],
                warm_start=False,
            ) as batch,
            NativeAO(
                atoms,
                basis=table,
                representation="spherical",
                charge=charge,
                multiplicity=multiplicity,
            ) as basis,
        ):
            primal_result = batch.execute(strict=False, properties=("energy", "forces"))
            primal = primal_result.items[0]
            receipt["native_primal"] = {
                "status": primal.status,
                "status_message": primal.status_message,
                "energy": primal.energy,
                "iterations": primal.iterations,
                "fock_builds": primal.fock_builds,
                "ks_diagnostic": (
                    None
                    if primal.ks_diagnostic is None
                    else primal.ks_diagnostic.to_payload()
                ),
            }
            retain("native-primal-completed")
            primal_result.raise_for_failures()
            state = StationaryKsState.from_native(batch, basis)
            try:
                atomic_weights = np.asarray(state._source.atomic_weights).copy()
                np.savez_compressed(
                    tmp_path / "native-state.npz",
                    density=state.density,
                    fock=state.fock,
                    coefficients=state.coefficients,
                    occupations=state.occupations,
                    orbital_energies=state.orbital_energies,
                    overlap=state.overlap,
                    grid_points=state.grid.points,
                    grid_weights=state.grid.weights,
                    grid_owners=state.grid.owners,
                    atomic_weights=atomic_weights,
                    ao_packed=basis.packed,
                    forces=primal.forces,
                    energy=primal.energy,
                )
                receipt.update(
                    nao=basis.nao,
                    grid_points=len(state.grid.points),
                    scf_iterations=primal.iterations,
                    scf_fock_builds=primal.fock_builds,
                    occupations=np.count_nonzero(state.occupations, axis=1).tolist(),
                    physical_residual=state.physical_residual,
                )
                retain("native-state-retained")
                reference, gradient, correction = _independent_gradient(
                    raw,
                    atoms,
                    charge,
                    multiplicity,
                    grid_spec,
                    atomic_weights,
                    damping,
                    gates["partition_ao_envelope"],
                    max_iterations=independent_max_iterations,
                    diis_factory=diis_factory,
                    **independent_policy,
                )
                errors = {
                    "energy": abs(primal.energy - reference.e_tot - correction),
                    "density": float(
                        np.max(
                            abs(
                                state.density
                                - np.asarray(reference.make_rdm1()).reshape(
                                    state.density.shape
                                )
                            )
                        )
                    ),
                    "gradient": float(np.max(abs(primal.forces + gradient))),
                    "independent_orbital_gradient": float(
                        np.linalg.norm(
                            reference.get_grad(reference.mo_coeff, reference.mo_occ)
                        )
                    ),
                }
                receipt["central_errors"] = errors
                np.savez_compressed(
                    tmp_path / "independent-state.npz",
                    **_independent_state(reference, gradient, correction),
                )
                retain("independent-central-state")
                for name, error in errors.items():
                    assert error < gates[name], (name, error)
                np.testing.assert_allclose(
                    np.asarray(reference.mo_occ).reshape(state.occupations.shape),
                    state.occupations,
                    atol=0,
                    rtol=0,
                )
                central_reference = reference
                stationary = specification.get("stationary")
                if stationary is not None:
                    assert np.max(abs(primal.forces)) < stationary["gradient_maximum"]
                    assert np.max(abs(gradient)) < stationary["gradient_maximum"]
            finally:
                state._source.close()
            native_started = time.perf_counter()
            common = {
                "charge": charge,
                "multiplicity": multiplicity,
                "max_eri_primitives": specification["max_eri_primitives"],
                "solver_options": GMRESOptions(atol=1e-12, rtol=1e-11),
            }
            matrix = (
                calculator.hessian(native_atoms, **common)
                if public
                else execute_batch(calculator, batch, native_atoms, **common)
            )
            np.savez_compressed(tmp_path / "native-matrix.npz", matrix=matrix.matrix)
            receipt["matrix_diagnostics"] = matrix.diagnostics
            retain("native-matrix-retained")
            actions = []
            for index, direction in enumerate(directions):
                action = (
                    calculator.hessian_vector_product(native_atoms, direction, **common)
                    if public
                    else execute_batch(
                        calculator, batch, native_atoms, direction=direction, **common
                    )
                )
                actions.append(action)
                np.savez_compressed(
                    tmp_path / f"native-hvp-{index}.npz",
                    direction=direction,
                    value=action.value,
                )
                receipt[f"hvp_{index}_diagnostics"] = action.diagnostics
                retain(f"native-hvp-{index}-retained")
            receipt["native_matrix_and_two_hvps_seconds"] = (
                time.perf_counter() - native_started
            )
        for result in (matrix, *actions):
            assert result.diagnostics["numerical_runtime"] == "native-cpp-cpu"
            assert result.diagnostics["public_calculator_endpoint"] == public
            assert (
                result.diagnostics["response_residual_norm"]
                < gates["response_residual"]
            )
            assert not result.diagnostics["runtime_compilation"]
            assert result.diagnostics["d3_pair_visits"] > 0
        np.testing.assert_allclose(
            matrix.matrix, matrix.matrix.T, atol=gates["raw_symmetry"], rtol=0
        )
        np.testing.assert_allclose(
            matrix.matrix @ np.tile(np.eye(3), (len(atoms), 1)),
            0,
            atol=gates["translation"],
            rtol=0,
        )
        for direction, action in zip(directions, actions, strict=True):
            np.testing.assert_allclose(
                matrix.matrix @ direction.ravel(),
                action.value.ravel(),
                atol=gates["hessian_hvp"],
                rtol=0,
            )
        assert (
            abs(
                directions[0].ravel() @ actions[1].value.ravel()
                - directions[1].ravel() @ actions[0].value.ravel()
            )
            < gates["bilinear"]
        )
        receipt["independent_matrix_errors"] = []
        receipt["minimum_occupied_root_overlap"] = 1.0
        if stationary is not None:
            native_modes = analyze_harmonic_modes(
                matrix.matrix,
                coordinates=positions,
                masses=stationary["masses_dalton"],
                hessian_units="hartree/bohr^2",
                coordinate_units="bohr",
                mass_units="dalton",
                symmetrize=True,
                raw_symmetry_tolerance=gates["raw_symmetry"],
            )
            assert native_modes.rigid_mode_count == stationary["rigid_modes"]
            assert native_modes.negative_mode_count == stationary["negative_modes"]
            receipt["stationary_modes"] = {
                "frequencies_cm1": native_modes.frequencies_cm1.tolist(),
                "raw_rotation_residual": native_modes.diagnostics[
                    "raw_rotation_residual"
                ],
                "mode_comparisons": [],
            }
        for step in manifest["difference_steps_bohr"]:
            independent = np.empty_like(matrix.matrix)
            for column, direction in enumerate(
                np.eye(positions.size).reshape(positions.size, *positions.shape)
            ):
                gradients = {}
                states = []
                for sign in offsets:
                    moved = [
                        (symbol, tuple(center))
                        for (symbol, _), center in zip(
                            atoms, positions + sign * step * direction, strict=True
                        )
                    ]
                    reference, gradient, correction = _independent_gradient(
                        raw,
                        moved,
                        charge,
                        multiplicity,
                        grid_spec,
                        atomic_weights,
                        damping,
                        gates["partition_ao_envelope"],
                        max_iterations=independent_max_iterations,
                        diis_factory=diis_factory,
                        **independent_policy,
                    )
                    gradients[sign] = gradient
                    independent_state = _independent_state(
                        reference, gradient, correction
                    )
                    independent_state["occupied_root_overlap"] = _occupied_root_overlap(
                        central_reference, reference
                    )
                    states.append(independent_state)
                independent[:, column] = _gradient_column(
                    gradients, step, stencil_order
                ).ravel()
                np.savez_compressed(
                    tmp_path / f"independent-{step}-{column}.npz",
                    offsets=np.asarray(offsets),
                    gradients=np.asarray([gradients[offset] for offset in offsets]),
                    **{
                        name: np.asarray([state[name] for state in states])
                        for name in states[0]
                    },
                )
                retain(f"independent-{step}-{column}")
                for independent_state in states:
                    assert independent_state["converged"]
                    assert (
                        independent_state["orbital_gradient_norm"]
                        < gates["independent_orbital_gradient"]
                    )
                    electrons = np.einsum(
                        "sij,ji->s",
                        np.asarray(independent_state["density"]).reshape(
                            -1, receipt["nao"], receipt["nao"]
                        ),
                        independent_state["overlap"],
                    )
                    occupations = np.asarray(independent_state["occupations"]).reshape(
                        len(electrons), -1
                    )
                    np.testing.assert_allclose(
                        electrons,
                        occupations.sum(axis=1),
                        atol=gates["independent_electron_count"],
                        rtol=0,
                    )
                    minimum = independent_state["occupied_root_overlap"]
                    receipt["minimum_occupied_root_overlap"] = min(
                        receipt["minimum_occupied_root_overlap"], minimum
                    )
                    assert minimum >= gates["minimum_occupied_root_overlap"]
            np.savez_compressed(
                tmp_path / f"independent-matrix-{step}.npz", matrix=independent
            )
            error = float(np.max(abs(matrix.matrix - independent)))
            receipt["independent_matrix_errors"].append(
                {"step": step, "maximum": error}
            )
            retain(f"independent-matrix-{step}")
            assert error < gates["raw_hessian"]
            if stationary is not None:
                np.testing.assert_allclose(
                    independent, independent.T, atol=gates["raw_symmetry"], rtol=0
                )
                np.testing.assert_allclose(
                    independent @ np.tile(np.eye(3), (len(atoms), 1)),
                    0,
                    atol=gates["translation"],
                    rtol=0,
                )
                frequencies, modes, rank = _independent_vibrational_modes(
                    independent, positions, np.asarray(stationary["masses_dalton"])
                )
                assert rank == stationary["rigid_modes"]
                assert np.count_nonzero(frequencies < 0) == stationary["negative_modes"]
                np.testing.assert_allclose(
                    native_modes.frequencies_cm1,
                    frequencies,
                    atol=stationary["frequency_absolute_tolerance_cm1"],
                    rtol=stationary["frequency_relative_tolerance"],
                )
                overlaps = abs(
                    np.einsum(
                        "ij,ij->i",
                        native_modes.mass_weighted_modes.reshape(modes.shape),
                        modes,
                    )
                )
                assert np.min(overlaps) > stationary["minimum_mode_overlap"]
                receipt["stationary_modes"]["mode_comparisons"].append(
                    {
                        "step_bohr": step,
                        "minimum_mode_overlap": float(np.min(overlaps)),
                        "frequencies_cm1": frequencies.tolist(),
                    }
                )
            for direction, action in zip(directions, actions, strict=True):
                np.testing.assert_allclose(
                    independent @ direction.ravel(),
                    action.value.ravel(),
                    atol=gates["raw_hvp"],
                    rtol=0,
                )
        assert _sha(library) == receipt["library_sha256"]
        receipt["accepted"] = True
        receipt["public_domain_qualified"] = public
        retain("accepted")
    except BaseException as error:
        receipt.update(failure_type=type(error).__name__, failure=str(error))
        retain("failed")
        raise
