"""Original #2840 DF-primary/Direct-LR E/F/state, without a VV10 substitution."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._ks_snapshot import NativeKsSnapshot
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc_compiler.dft.grid import MolecularGrid
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.spec import RangeSeparatedExchangePrimitive

from benchmarks.ks_matrix_provider_qualification import final_state_gates

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_KS_STATE_CUDA_TEST") != "1",
    reason="requires explicitly scheduled original-input CUDA qualification",
)

_PRIMITIVES = (
    ("3.425250914", "0.1543289673"),
    ("0.6239137298", "0.5353281423"),
    ("0.168855404", "0.4446345422"),
)
_GRID = GridSpec(24, 12, 24)


def _basis(representation: str) -> BasisSet:
    """Preserve the native fixture's decimals, including its auxiliary basis."""
    return BasisSet(
        "original-2840-hydrogens",
        (
            ElementBasis(
                1,
                (
                    BasisShell(
                        0,
                        tuple(exponent for exponent, _ in _PRIMITIVES),
                        (tuple(coefficient for _, coefficient in _PRIMITIVES),),
                    ),
                ),
            ),
        ),
        BasisProvenance(
            "tests/native/test_ks_cuda.cpp::hydrogens",
            "original-2840",
            "GPL-3.0-or-later",
            hashlib.sha256(repr(_PRIMITIVES).encode()).hexdigest(),
        ),
        representation=representation,
    )


def _atoms(unrestricted: bool, moved: bool) -> list[tuple[str, tuple[float, ...]]]:
    """Match hydrogens(count, restricted, shift=0.05), not a new geometry."""
    return [
        ("H", (0.15 * index * index, 0.13 * index, (1.55 if moved else 1.5) * index))
        for index in range(3 if unrestricted else 2)
    ]


def _reference(atoms: Any, unrestricted: bool) -> Any:
    """Reconverge independently with DF full-range J/K and exact LR-K only.

    PySCF's ordinary density_fit RSH would also fit LR exchange. Overriding
    just get_jk keeps that different Hamiltonian out of this reference. PBE's
    semilocal kernel has no range parameter; the two exchange primitives do.
    """
    from pyscf import df, dft, gto, lib, scf

    lib.num_threads(1)
    molecule = gto.M(
        atom=atoms,
        basis={
            "H": [
                [
                    0,
                    *[
                        (float(exponent), float(coefficient))
                        for exponent, coefficient in _PRIMITIVES
                    ],
                ]
            ]
        },
        unit="Bohr",
        spin=int(unrestricted),
        verbose=0,
    )
    fitted = df.DF(molecule, auxbasis=molecule._basis).build()
    reference = dft.UKS(molecule) if unrestricted else dft.RKS(molecule)
    reference.xc = "PBE + 0.2*SR_HF(0.33) + 0.65*LR_HF(0.33)"

    def get_jk(
        mol: Any,
        density: Any,
        hermi: int = 1,
        with_j: bool = True,
        with_k: bool = True,
        omega: float | None = None,
    ) -> Any:
        if omega:
            return scf.hf.get_jk(
                mol, density, hermi, with_j=with_j, with_k=with_k, omega=omega
            )
        return fitted.get_jk(density, hermi, with_j, with_k)

    reference.get_jk = get_jk
    grid = MolecularGrid(atoms, _GRID, multiplicity=2 if unrestricted else 1).explicit()
    reference.grids.radii_adjust = None
    reference.grids.coords = np.asarray(grid.points)
    reference.grids.weights = np.asarray(grid.weights)
    reference.small_rho_cutoff = 0
    reference.conv_tol = 1e-13
    reference.conv_tol_grad = 1e-10
    reference.max_cycle = 200
    reference.kernel(dm0=reference.get_init_guess(key="1e"))
    assert reference.converged
    return reference


@lru_cache(maxsize=4)
def _independent_forces(atoms: tuple, unrestricted: bool) -> np.ndarray:
    """Differentiate independently reconverged, moving-grid mixed energies.

    A five-point stencil includes auxiliary-center and quadrature response.
    It never differentiates a frozen density or substitutes fitted LR-K.
    """
    positions = np.asarray([position for _, position in atoms])
    forces = np.zeros_like(positions)
    step = 2e-4
    for atom in range(len(atoms)):
        for axis in range(3):
            energies = []
            for shift in (-2, -1, 1, 2):
                displaced = positions.copy()
                displaced[atom, axis] += shift * step
                current = [
                    (symbol, tuple(position))
                    for (symbol, _), position in zip(atoms, displaced, strict=True)
                ]
                energies.append(_reference(current, unrestricted).e_tot)
            forces[atom, axis] = -(
                energies[0] - 8 * energies[1] + 8 * energies[2] - energies[3]
            ) / (12 * step)
    return forces


@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("unrestricted", (False, True), ids=("rks", "uks"))
def test_original_df_direct_lr_state_and_force_admission(
    representation: str,
    unrestricted: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original mixed Hamiltonian owns all native integral and XC sources.

    This range-only case uses the compiled CUDA geometry owner, not the
    distinct resident range/VV10 controller. CPU derivatives cannot substitute
    for the native mixed integral bridge.
    """
    assert os.environ.get("SLURM_JOB_ID")
    pytest.importorskip("pyscf")
    from generativeqc.batch import PreparedBatch

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("unqualified original RSH force entered a Python fallback")

    monkeypatch.setattr(PreparedBatch, "_public_dft_cpu_force", forbidden)
    cuda_work = []
    cuda_force = PreparedBatch._public_dft_cuda_force

    def tracked_cuda_force(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = cuda_force(self, *args, **kwargs)
        cuda_work.append(result[1])
        return result

    monkeypatch.setattr(PreparedBatch, "_public_dft_cuda_force", tracked_cuda_force)
    monkeypatch.setenv("GENERATIVEQC_DF_RANGE_EXCHANGE", "direct")
    spin = "polarized" if unrestricted else "unpolarized"
    semilocal = resolve_method("PBE", spin=spin)
    method = replace(
        semilocal,
        identifier="original-2840-PBE-RSH",
        primitives=(
            *semilocal.primitives,
            RangeSeparatedExchangePrimitive(
                Fraction(1, 5), Fraction(33, 100), "short-range"
            ),
            RangeSeparatedExchangePrimitive(
                Fraction(13, 20), Fraction(33, 100), "long-range"
            ),
        ),
    )
    basis = _basis(representation)
    calculator = Calculator(
        "pbe-uks" if unrestricted else "pbe-rks",
        basis,
        device="cuda",
        precision="fp64",
        basis_representation=representation,
        density_fitting="cuda",
        auxiliary_basis=basis,
        ks_options=KsOptions(composition=method, grid=_GRID),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-12,
        max_iterations=200,
    )
    observations: list[dict[str, Any]] = []
    provenance = {
        "source_identity": calculator._library.generativeqc_get_source_identity().decode(),
        "library_sha256": hashlib.sha256(
            Path(calculator._library._name).read_bytes()
        ).hexdigest(),
        "method_identity": method.identity,
        "basis_identity": basis.identity,
        "scheduler": {
            name: os.environ.get(name)
            for name in ("SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES")
        },
        "force_disposition": "native mixed integral bridge plus compiled CUDA geometry",
    }
    references = {}
    try:
        with calculator.prepare_batch(
            [_atoms(unrestricted, False)], multiplicities=[2 if unrestricted else 1]
        ) as batch:
            assert not batch._native_dft_force_eligible(0)
            for phase in range(4):
                moved = phase >= 2
                atoms = _atoms(unrestricted, moved)
                item = batch.execute(
                    [[position for _, position in atoms]] if moved else None,
                    properties=("energy", "forces"),
                    strict=True,
                ).items[0]
                assert item.converged and item.executed_backend == "cuda"
                snapshot = NativeKsSnapshot(batch, 0)
                try:
                    gates = final_state_gates(
                        snapshot.metadata,
                        snapshot.values,
                        {"atomic_numbers": [1] * len(atoms), "charge": 0},
                    )
                    assert gates["accepted"]
                    assert gates["density_closure_error"] < 1e-8
                    assert gates["physical_residual_rms"] < 1e-9
                    dimension, spins = map(int, snapshot.metadata[1:3])
                    offset = 2 + 4 * len(atoms)
                    count = spins * dimension * dimension
                    density, fock = (
                        snapshot.values[begin : begin + count].reshape(
                            spins, dimension, dimension
                        )
                        for begin in (offset, offset + count)
                    )
                    if moved not in references:
                        references[moved] = _reference(atoms, unrestricted)
                    reference = references[moved]
                    reference_density = np.asarray(reference.make_rdm1()).reshape(
                        density.shape
                    )
                    reference_fock = np.asarray(
                        reference.get_fock(
                            dm=reference_density
                            if unrestricted
                            else reference_density[0]
                        )
                    ).reshape(fock.shape)
                    row = {
                        "phase": phase,
                        "energy": item.energy,
                        "reference_energy": float(reference.e_tot),
                        "energy_error": abs(item.energy - reference.e_tot),
                        "density_error": float(
                            np.max(np.abs(density - reference_density))
                        ),
                        "fock_error": float(np.max(np.abs(fock - reference_fock))),
                        "state": gates,
                        "precision": item.precision,
                        "history": item.ks_diagnostic.to_payload()["history"],
                    }
                    observations.append(row)
                    assert row["energy_error"] < 1e-8
                    assert row["density_error"] < 1e-6
                    assert row["fock_error"] < 1e-6
                    reference_forces = _independent_forces(tuple(atoms), unrestricted)
                    row["forces"] = np.asarray(item.forces).tolist()
                    row["reference_forces"] = reference_forces.tolist()
                    row["force_error"] = float(
                        np.max(np.abs(item.forces - reference_forces))
                    )
                    row["work"] = cuda_work[-1]
                    assert row["force_error"] < 3e-5
                    np.testing.assert_allclose(
                        np.asarray(item.forces).sum(axis=0), 0, atol=2e-8, rtol=0
                    )
                finally:
                    snapshot.close()
    finally:
        directory = os.environ.get("GENERATIVEQC_KS_STATE_EVIDENCE_DIR")
        if directory:
            output = Path(directory)
            output.mkdir(parents=True, exist_ok=True)
            (output / f"original-rsh-{representation}-{unrestricted}.json").write_text(
                json.dumps(
                    {"provenance": provenance, "observations": observations}, indent=2
                )
            )
        calculator.clear_cache()
