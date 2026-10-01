"""Public DF-KS energy endpoints against independently converged PySCF states."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
from generativeqc import Atom, Calculator, KsOptions
from generativeqc_compiler.dft.grid import GridSpec, MolecularGrid


@pytest.fixture(params=("cpu", "cuda"))
def device(request: pytest.FixtureRequest) -> str:
    if request.param == "cuda" and os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an allocated native CUDA library/device")
    return request.param


def calculator(device: str, method: str = "pbe-rks", **kwargs: Any) -> Calculator:
    """Use tight physical convergence and an explicitly selected auxiliary basis."""
    if method.startswith(("pbe0", "b3lyp")):
        kwargs.setdefault("ks_options", KsOptions(grid=GridSpec()))
    return Calculator(
        method=method,
        basis="sto-3g",
        device=device,
        density_fitting="auto",
        auxiliary_basis="def2-svp",
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        **kwargs,
    )


WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, -1.43233673, 1.10715266)),
    ("H", (0.0, 1.43233673, 1.10715266)),
]


def pyscf_energy(
    calc: Calculator, raw_atoms: Any, multiplicity: int, functional: str
) -> float:
    """Copy orbital/auxiliary primitives and quadrature; only the solvers differ."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft, gto

    pyscf.lib.num_threads(1)
    atoms = tuple(Atom.from_value(a) for a in raw_atoms)
    labels = [f"{symbol}{i}" for i, (symbol, _) in enumerate(raw_atoms)]

    def basis_dict(selected: Any) -> dict:
        result = {label: [] for label in labels}
        for shell in calc._shells_for_atoms(atoms, selected):
            result[labels[shell.atom_index]].append(
                [
                    shell.angular_momentum,
                    *[(p.exponent, p.coefficient) for p in shell.primitives],
                ]
            )
        return result

    mol = gto.M(
        atom=[(label, atom.position) for label, atom in zip(labels, atoms)],
        basis=basis_dict(calc._basis),
        unit="Bohr",
        cart=True,
        spin=multiplicity - 1,
        verbose=0,
    )
    mf = (dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)).density_fit(
        auxbasis=basis_dict(calc._auxiliary_basis)
    )
    grid = MolecularGrid(
        atoms, spec=calc.ks_options.grid, multiplicity=multiplicity
    ).explicit()
    mf.xc = functional
    mf.grids.coords = np.asarray(grid.points)
    mf.grids.weights = np.asarray(grid.weights)
    mf.small_rho_cutoff = 0
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    return mf.e_tot


def pyscf_energy_gradient(
    calc: Calculator,
    state: Any,
    raw_atoms: Any,
    multiplicity: int,
    functional: str,
) -> tuple[float, np.ndarray]:
    """Independent DF-SCF plus analytic auxiliary/metric and moving-grid response."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft, gto

    pyscf.lib.num_threads(1)
    atoms = tuple(Atom.from_value(a) for a in raw_atoms)
    labels = [f"{symbol}{i}" for i, (symbol, _) in enumerate(raw_atoms)]

    def basis_dict(selected: Any) -> dict:
        result = {label: [] for label in labels}
        for shell in calc._shells_for_atoms(atoms, selected):
            result[labels[shell.atom_index]].append(
                [
                    shell.angular_momentum,
                    *[(p.exponent, p.coefficient) for p in shell.primitives],
                ]
            )
        return result

    mol = gto.M(
        atom=[(label, atom.position) for label, atom in zip(labels, atoms, strict=True)],
        basis=basis_dict(calc._basis),
        unit="Bohr",
        cart=True,
        spin=multiplicity - 1,
        verbose=0,
    )
    mf = (dft.RKS(mol) if multiplicity == 1 else dft.UKS(mol)).density_fit(
        auxbasis=basis_dict(calc._auxiliary_basis)
    )
    points = np.asarray(state.grid.points)
    weights = np.asarray(state.grid.weights)
    owners = np.asarray(state.grid.owners)
    atomic_weights = np.asarray(state._source.atomic_weights)
    mf.xc = functional
    mf.grids.coords = points
    mf.grids.weights = weights
    mf.grids.radii_adjust = None
    atomic_grid = {
        mol.atom_symbol(a): (
            points[owners == a] - mol.atom_coord(a),
            atomic_weights[owners == a],
        )
        for a in range(mol.natm)
    }
    mf.grids.gen_atomic_grids = lambda *args, **kwargs: atomic_grid
    mf.small_rho_cutoff = 0
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged
    gradient = mf.nuc_grad_method()
    gradient.grid_response = True
    return mf.e_tot, np.asarray(gradient.kernel())


@pytest.mark.parametrize(
    "method,functional,atoms,multiplicity",
    [
        ("lda-rks", "LDA_X,LDA_C_PW", WATER, 1),
        ("pbe-rks", "PBE", WATER, 1),
        ("r2scan-rks", "R2SCAN", WATER, 1),
        ("pbe-uks", "PBE", [("Li", (0.0, 0.0, 0.0))], 2),
    ],
)
def test_df_semilocal_matches_independent_reference(
    device: str, method: str, functional: str, atoms: Any, multiplicity: int
) -> None:
    calc = calculator(device, method)
    result = calc.singlepoint(atoms, multiplicity=multiplicity, properties=("energy",))
    assert result.converged and result.forces is None
    assert result.executed_backend == ("cuda" if device == "cuda" else "cpu_reference")
    assert result.physical_residual_rms < 1e-9
    assert result.energy == pytest.approx(
        pyscf_energy(calc, atoms, multiplicity, functional), abs=1e-8
    )


@pytest.mark.parametrize(
    "method,functional", [("pbe0-rks", "PBE0"), ("b3lyp-rks", "B3LYP")]
)
def test_df_cpu_global_hybrid_matches_independent_reference(
    method: str, functional: str
) -> None:
    calc = calculator("cpu", method)
    result = calc.singlepoint(WATER, properties=("energy",))
    assert result.energy == pytest.approx(
        pyscf_energy(calc, WATER, 1, functional), abs=1e-8
    )


def test_df_batch_warm_replay_rebinds_auxiliary_centers(device: str) -> None:
    calc = calculator(device)
    moved = [
        (symbol, (x, y, z + (0.04 if i == 1 else 0.0)))
        for i, (symbol, (x, y, z)) in enumerate(WATER)
    ]
    expected = calc.singlepoint(moved, properties=("energy",)).energy
    with calc.prepare_batch([WATER, WATER], warm_start=True) as batch:
        cold = batch.execute(strict=True, properties=("energy",))
        from generativeqc._ks_snapshot import NativeKsSnapshot

        if device == "cuda":
            snapshot = NativeKsSnapshot(batch, 0)
            try:
                coulomb, exchange, threshold = snapshot.fock_provider_proof()
                assert coulomb == "density-fitted" and exchange is None
                assert threshold == calc._density_fitting_relative_threshold
            finally:
                snapshot.close()
        else:
            with pytest.raises(NotImplementedError):
                NativeKsSnapshot(batch, 0)
        warm = batch.execute(strict=True, properties=("energy",))
        if device == "cuda":
            # Ordinary SCF and same-geometry replay retain device matrices.
            assert all(
                item.matrix_d2h_bytes == 0 for item in batch.ks_transport_diagnostics
            )
        updated = batch.execute(
            coordinates=[None, [xyz for _, xyz in moved]],
            strict=True,
            properties=("energy",),
        )
        assert warm.energies == pytest.approx(cold.energies, abs=1e-9)
        assert all(item.warm_start_used for item in warm.items)
        assert updated.items[1].energy == pytest.approx(expected, abs=1e-9)
        if device == "cuda":
            metrics = batch.last_density_fitting_metric_diagnostics()
            assert [item.system_index for item in metrics] == [0, 1]
            assert [item.bucket_id for item in metrics] == [0, 1]
            assert all(
                item.effective_rank > 0 and item.device_resident_bytes > 0
                for item in metrics
            )
            transport = batch.ks_transport_diagnostics
            # Rebuilding only the moved item's owner exports its last-good RKS
            # density once; WATER/STO-3G has seven orbital basis functions.
            density_bytes = 7 * 7 * np.dtype(np.float64).itemsize
            assert [item.matrix_d2h_bytes for item in transport] == [0, density_bytes]


def test_df_rejects_unqualified_force_precision_and_resource_consumers(
    device: str,
) -> None:
    calc = calculator(device)
    if device == "cpu":
        with pytest.raises(ValueError, match="does not support.*forces"):
            calc.singlepoint(WATER, properties=("energy", "forces"))
    else:
        assert "forces" in calc.capabilities.supported_properties
    with pytest.raises(NotImplementedError, match="resource plans"):
        calc.estimate_resources([WATER])
    if device == "cuda":
        with pytest.raises(NotImplementedError, match="fp64"):
            calculator(device, precision="auto")
        with pytest.raises(ValueError, match="backend must match"):
            Calculator(method="pbe-rks", device=device, density_fitting="cpu")


def test_df_cuda_provider_budget_is_not_silently_enlarged(device: str) -> None:
    if device != "cuda":
        pytest.skip("DF device allocation budget")
    with pytest.raises(
        (MemoryError, RuntimeError), match="budget|memory|fit|workspace"
    ):
        calculator(device, density_fitting_memory_budget_bytes=1).singlepoint(WATER)


def test_df_default_auxiliary_ragged_batch(device: str) -> None:
    calc = Calculator(
        method="pbe-rks",
        basis="sto-3g",
        device=device,
        density_fitting="auto",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    systems = [
        [("He", (0.0, 0.0, 0.0))],
        [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
    ]
    expected = [pyscf_energy(calc, atoms, 1, "PBE") for atoms in systems]
    with calc.prepare_batch(systems) as batch:
        result = batch.execute(strict=True, properties=("energy",))
        assert result.energies == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize(
    "method,functional,atoms,multiplicity",
    [
        ("pbe-rks", "PBE", WATER, 1),
        ("r2scan-rks", "R2SCAN", WATER, 1),
        ("pbe-uks", "PBE", [("Li", (0.0, 0.0, 0.0))], 2),
    ],
)
def test_df_cuda_semilocal_force_matches_independent_response(
    method: str, functional: str, atoms: Any, multiplicity: int
) -> None:
    if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an allocated native CUDA library/device")
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO

    calc = calculator("cuda", method)
    assert "forces" in calc.capabilities.supported_properties
    with calc.prepare_batch([atoms], multiplicities=[multiplicity], warm_start=True) as batch:
        public = batch.execute(
            strict=True, properties=("energy", "forces")
        ).items[0]
        with NativeAO(
            atoms,
            basis=calc._basis,
            representation=calc._representation_name,
            multiplicity=multiplicity,
        ) as basis:
            state = StationaryKsState.from_native(batch, basis)
            try:
                coulomb, exchange, threshold = state._source.fock_provider_proof()
                assert coulomb == "density-fitted" and exchange is None
                assert threshold == calc._density_fitting_relative_threshold
                reference_energy, reference_gradient = pyscf_energy_gradient(
                    calc, state, atoms, multiplicity, functional
                )
            finally:
                state._source.close()
        assert public.executed_backend == "cuda" and public.converged
        assert public.energy == pytest.approx(reference_energy, abs=1e-8)
        np.testing.assert_allclose(
            public.forces, -reference_gradient, atol=3e-7, rtol=0
        )
        np.testing.assert_allclose(public.forces.sum(axis=0), 0, atol=2e-9, rtol=0)
        replay = batch.execute(
            strict=True, properties=("energy", "forces")
        ).items[0]
        np.testing.assert_allclose(replay.forces, public.forces, atol=2e-8, rtol=0)


def test_df_cuda_force_rebinds_auxiliary_response_on_moved_geometry() -> None:
    if os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1":
        pytest.skip("requires an allocated native CUDA library/device")
    calc = calculator("cuda", "pbe-rks")
    moved = np.asarray([xyz for _, xyz in WATER], dtype=np.float64)
    moved[1] += (0.02, -0.01, 0.03)
    moved_atoms = [
        (atom[0], tuple(position))
        for atom, position in zip(WATER, moved, strict=True)
    ]
    with calc.prepare_batch([WATER], warm_start=True) as batch:
        cold = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        replay = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        changed = batch.execute(
            coordinates=(moved,), strict=True, properties=("energy", "forces")
        ).items[0]
    fresh = calc.singlepoint(
        moved_atoms, properties=("energy", "forces")
    )
    np.testing.assert_allclose(replay.forces, cold.forces, atol=2e-8, rtol=0)
    assert changed.energy == pytest.approx(fresh.energy, abs=1e-9)
    np.testing.assert_allclose(changed.forces, fresh.forces, atol=3e-8, rtol=0)
