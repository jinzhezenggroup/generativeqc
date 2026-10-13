"""Default DF seeds preserve the requested Hamiltonian and independent endpoints."""

from __future__ import annotations

import os

import numpy as np
import pytest
from generativeqc import Atom, Calculator, GridSpec, InitialGuessSpec, KsOptions
from generativeqc_compiler.dft.grid import MolecularGrid
from test_dft_df_public import pyscf_energy_gradient
from test_huckel_cuda_hf import final_density

WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, -1.43233673, 1.10715266)),
    ("H", (0.0, 1.43233673, 1.10715266)),
]


@pytest.fixture(params=("cpu", "cuda"))
def device(request: pytest.FixtureRequest) -> str:
    if request.param == "cuda":
        if os.environ.get("GENERATIVEQC_HUCKEL_CUDA_TEST") != "1":
            pytest.skip("requires an explicit allocated-GPU opt-in")
        assert os.environ.get("SLURM_JOB_ID")
        assert os.environ.get("CUDA_VISIBLE_DEVICES")
    return request.param


def calculator(device: str, method: str, **overrides: object) -> Calculator:
    """Keep basis, DF approximation, grid and target convergence fixed across arms."""
    options = {
        "method": method,
        "basis": "sto-3g",
        "auxiliary_basis": "def2-svp",
        "device": device,
        "density_fitting": "auto",
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-12,
        "max_iterations": 100,
    }
    if method.endswith("-rks"):
        options["ks_options"] = KsOptions(grid=GridSpec(16, 8, 16))
    options.update(overrides)
    return Calculator(**options)


def independent_oracle(calc: Calculator, method: str) -> object:
    """Copy only scientific inputs; PySCF constructs and converges its own DF operator."""
    pyscf = pytest.importorskip("pyscf")
    pyscf.lib.num_threads(1)
    atoms = tuple(Atom.from_value(atom) for atom in WATER)
    labels = [f"{symbol}{index}" for index, (symbol, _) in enumerate(WATER)]

    def basis_dict(selected: object) -> dict:
        basis = {label: [] for label in labels}
        for shell in calc._shells_for_atoms(atoms, selected):
            basis[labels[shell.atom_index]].append(
                [
                    shell.angular_momentum,
                    *[(row.exponent, row.coefficient) for row in shell.primitives],
                ]
            )
        return basis

    molecule = pyscf.gto.M(
        atom=list(zip(labels, (atom.position for atom in atoms), strict=True)),
        basis=basis_dict(calc._basis),
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    oracle = (
        pyscf.scf.RHF(molecule) if method == "rhf" else pyscf.dft.RKS(molecule)
    ).density_fit(auxbasis=basis_dict(calc._auxiliary_basis))
    if method != "rhf":
        grid = MolecularGrid(atoms, spec=calc.ks_options.grid).explicit(
            max_points=100_000
        )
        oracle.xc = "PBE0" if method == "pbe0-rks" else "PBE"
        oracle.grids.coords = np.asarray(grid.points)
        oracle.grids.weights = np.asarray(grid.weights)
        oracle.small_rho_cutoff = 0
    oracle.conv_tol, oracle.conv_tol_grad, oracle.max_cycle = 1e-13, 1e-10, 100
    oracle.kernel()
    assert oracle.converged
    return oracle


@pytest.mark.parametrize("method", ["rhf", "pbe-rks", "pbe0-rks"])
def test_df_default_energy_force_density_and_residual(device: str, method: str) -> None:
    calc = calculator(device, method)
    assert calc.initial_guess == InitialGuessSpec("huckel")
    oracle = independent_oracle(calc, method)
    core = calculator(device, method, initial_guess=None).singlepoint(
        WATER, properties=("energy", "forces")
    )
    with calc.prepare_batch([WATER]) as batch:
        cold = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        if method == "rhf":
            density = final_density(batch).reshape(oracle.mol.nao, oracle.mol.nao)
            expected_forces = -oracle.nuc_grad_method().kernel()
        else:
            from generativeqc._dft_gradient import StationaryKsState
            from generativeqc_compiler.dft import NativeAO

            with NativeAO(WATER, basis=calc._basis) as basis:
                state = StationaryKsState.from_native(batch, basis)
                try:
                    density = np.asarray(state.density[0]).copy()
                    _, gradient = pyscf_energy_gradient(
                        calc, state, WATER, 1, "PBE0" if method == "pbe0-rks" else "PBE"
                    )
                    expected_forces = -gradient
                    coulomb, exchange, threshold = state._source.fock_provider_proof()
                    assert coulomb == "density-fitted"
                    assert exchange == (
                        "density-fitted" if method == "pbe0-rks" else None
                    )
                    assert threshold == calc._density_fitting_relative_threshold
                finally:
                    state._source.close()
        warm = batch.execute(strict=True, properties=("energy", "forces")).items[0]
    diagnostic = cold.initial_guess
    assert diagnostic["kind"] == "huckel" and diagnostic["outcome"] == "used"
    assert diagnostic["target_attempts"] == 1 and diagnostic["work_counters_complete"]
    assert diagnostic["preliminary_fock_builds"] > 0
    assert warm.initial_guess["outcome"] == "existing_density"
    assert warm.initial_guess["preparation_numeric_capacity"] == 0
    assert cold.energy == pytest.approx(oracle.e_tot, abs=1e-8)
    np.testing.assert_allclose(density, oracle.make_rdm1(), atol=1e-8, rtol=0)
    overlap, fock = oracle.get_ovlp(), oracle.get_fock(dm=density)
    residual = fock @ density @ overlap - overlap @ density @ fock
    assert np.sqrt(np.mean(residual**2)) <= 1e-8
    for result in (cold, warm):
        assert result.energy == pytest.approx(core.energy, abs=1e-9)
        np.testing.assert_allclose(result.forces, expected_forces, atol=3e-7, rtol=0)
        np.testing.assert_allclose(result.forces, core.forces, atol=3e-7, rtol=0)
    single = calc.singlepoint(WATER, properties=("energy", "forces"))
    assert single.initial_guess["outcome"] == "used"
    assert single.energy == pytest.approx(cold.energy, abs=1e-9)
    np.testing.assert_allclose(single.forces, expected_forces, atol=3e-7, rtol=0)


@pytest.mark.parametrize("method", ["rhf", "pbe-rks", "pbe0-rks"])
@pytest.mark.parametrize(
    "policy,outcome",
    [
        (InitialGuessSpec("huckel", maximum_numeric_bytes=1), "budget_skipped"),
        (InitialGuessSpec("huckel", max_iterations=1), "preparation_failed"),
    ],
)
def test_df_bounded_preparation_preserves_core(
    device: str, method: str, policy: InitialGuessSpec, outcome: str
) -> None:
    baseline = calculator(device, method, initial_guess=None).singlepoint(WATER)
    result = calculator(device, method, initial_guess=policy).singlepoint(WATER)
    assert result.initial_guess["outcome"] == outcome
    assert result.initial_guess["target_attempts"] == 1
    assert result.energy == pytest.approx(baseline.energy, abs=1e-9)


@pytest.mark.parametrize("method", ["rhf", "pbe-rks", "pbe0-rks"])
def test_df_failed_seed_gets_one_core_attempt(device: str, method: str) -> None:
    calc = calculator(device, method, max_iterations=1)
    with calc.prepare_batch([WATER, WATER], warm_start=False) as batch:
        results = batch.execute(properties=("energy",)).items
    for result in results:
        assert not result.converged
        assert result.initial_guess["outcome"] == "target_retried"
        assert result.initial_guess["target_attempts"] == 2
        assert result.initial_guess["discarded_target_fock_builds"] > 0


@pytest.mark.parametrize("kind", ["hf", "lda", "minao"])
def test_other_preliminary_providers_do_not_gain_df_support(kind: str) -> None:
    with pytest.raises(NotImplementedError, match="preliminary|MINAO"):
        calculator("cpu", "rhf", initial_guess=InitialGuessSpec(kind))
