"""Opt-in complete default-grid KS endpoints against independent PySCF grids."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
from generativeqc import Atom, Calculator, GridSpec, KsOptions

WATER = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, -1.43233673, 1.10715266)),
    ("H", (0.0, 1.43233673, 1.10715266)),
]


def reference(
    calculator: Calculator,
    *,
    fitted: bool,
    functional: str,
    spin: int = 0,
    nonlocal_level: int | None = None,
    forces: bool = True,
    atom_values: list | None = None,
    grid_level: int | None = None,
) -> tuple[float, np.ndarray | None]:
    """Share basis input only; PySCF owns quadrature, SCF and analytic gradients."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft, gto

    pyscf.lib.num_threads(1)
    atom_values = WATER if atom_values is None else atom_values
    atoms = tuple(Atom.from_value(a) for a in atom_values)
    labels = [f"{symbol}{i}" for i, (symbol, _) in enumerate(atom_values)]

    def basis_dict(selected: Any) -> dict:
        basis = {label: [] for label in labels}
        for shell in calculator._shells_for_atoms(atoms, selected):
            basis[labels[shell.atom_index]].append(
                [
                    shell.angular_momentum,
                    *[(p.exponent, p.coefficient) for p in shell.primitives],
                ]
            )
        return basis

    mol = gto.M(
        atom=list(zip(labels, [a.position for a in atoms], strict=True)),
        basis=basis_dict(calculator._basis),
        unit="Bohr",
        cart=False,
        charge=spin,
        spin=spin,
        verbose=0,
    )
    mf = dft.UKS(mol) if spin else dft.RKS(mol)
    direct = mf
    if fitted:
        mf = mf.density_fit(auxbasis=basis_dict(calculator._auxiliary_basis))
    mf.xc = functional
    mf.small_rho_cutoff = 0
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.direct_scf_tol, mf.max_cycle = 1e-14, 100
    if grid_level is not None:
        mf.grids.level = grid_level
    if nonlocal_level is not None:
        mf.nlcgrids.level = nonlocal_level
    if fitted and functional == "WB97M_V":
        # The existing native DF contract fits full-range J/K and retains
        # exact long-range K. PySCF normally fits both: select its independent
        # direct LR provider for both SCF and gradients, including zero LR
        # auxiliary response. Quadrature is still owned entirely by PySCF.
        fitted_k = mf.get_k

        def mixed_k(
            mol: Any = None, dm: Any = None, hermi: int = 1, omega: Any = None
        ) -> Any:
            provider = direct.get_k if omega else fitted_k
            return provider(mol, dm, hermi=hermi, omega=omega)

        mf.get_k = mixed_k
    # Default-grid callers leave both prescriptions untouched. Larger schedule
    # gates may explicitly select levels while retaining the reference family.
    energy = mf.kernel()
    assert mf.converged
    if not forces:
        return float(energy), None
    gradient = mf.nuc_grad_method()
    gradient.grid_response = True
    if fitted:
        gradient.auxbasis_response = True
        if functional == "WB97M_V":
            fitted_gradient_k = gradient.get_k
            direct_gradient = direct.nuc_grad_method()

            def mixed_gradient_k(
                mol: Any = None, dm: Any = None, hermi: int = 0, omega: Any = None
            ) -> Any:
                if not omega:
                    return fitted_gradient_k(mol, dm, hermi=hermi, omega=omega)
                dm = mf.make_rdm1() if dm is None else dm
                selected_mol = mf.mol if mol is None else mol
                result = direct_gradient.get_k(
                    selected_mol, dm, hermi=hermi, omega=omega
                )
                spins = 1 if np.ndim(dm) == 2 else len(dm)
                return pyscf.lib.tag_array(
                    result, aux=np.zeros((spins, spins, selected_mol.natm, 3))
                )

            gradient.get_k = mixed_gradient_k
    return float(energy), -np.asarray(gradient.kernel())


def require_endpoint(device: str) -> None:
    if not os.environ.get("GENERATIVEQC_LIBRARY"):
        pytest.skip("requires a source-matched native library")
    if device == "cuda" and not (
        os.environ.get("SLURM_JOB_ID")
        and os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") == "1"
    ):
        pytest.skip("requires an explicitly allocated Slurm GPU")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize(
    "method,functional",
    [
        ("pbe-rks", "PBE"),
        ("pbe0-rks", "PBE0"),
        ("r2scan-rks", "r2SCAN"),
        ("wb97m-v-rks", "WB97M_V"),
    ],
)
def test_default_grid_energy_and_analytic_forces(
    device: str, fitted: bool, method: str, functional: str
) -> None:
    require_endpoint(device)
    calculator = Calculator(
        method=method,
        basis="sto-3g",
        basis_representation="spherical",
        device=device,
        density_fitting=device if fitted else "none",
        auxiliary_basis="def2-svp" if fitted else None,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-13,
        max_iterations=100,
    )
    assert calculator.ks_options.grid == GridSpec.treutler(3)
    # Public CPU meta-GGA force admission is independent of this grid change;
    # CUDA exercises complete forces for every family, CPU PBE/PBE0 do too.
    forces = device == "cuda" or method in ("pbe-rks", "pbe0-rks")
    actual = calculator.singlepoint(
        WATER, properties=("energy", "forces") if forces else ("energy",)
    )
    expected_energy, expected_forces = reference(
        calculator, fitted=fitted, functional=functional, forces=forces
    )
    assert actual.energy == pytest.approx(expected_energy, abs=2e-8)
    if forces:
        np.testing.assert_allclose(actual.forces, expected_forces, atol=3e-7, rtol=0)
        np.testing.assert_allclose(np.sum(actual.forces, axis=0), 0, atol=2e-10)


@pytest.mark.parametrize("fitted", [False, True])
def test_adjusted_spin_and_independent_vv10_grids(fitted: bool) -> None:
    require_endpoint("cuda")
    calculator = Calculator(
        method="wb97m-v-uks",
        basis="sto-3g",
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda" if fitted else "none",
        auxiliary_basis="def2-svp" if fitted else None,
        ks_options=KsOptions(nonlocal_grid=GridSpec.treutler(1)),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-13,
        max_iterations=100,
    )
    actual = calculator.singlepoint(
        WATER, charge=1, multiplicity=2, properties=("energy", "forces")
    )
    energy, forces = reference(
        calculator, fitted=fitted, functional="WB97M_V", spin=1, nonlocal_level=1
    )
    assert actual.energy == pytest.approx(energy, abs=2e-8)
    np.testing.assert_allclose(actual.forces, forces, atol=3e-7, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("fitted", [False, True])
def test_tight_xc_keeps_default_vv10_level_three(device: str, fitted: bool) -> None:
    """Compare both defaults with independent PySCF SCF and full grid responses."""
    require_endpoint(device)
    calculator = Calculator(
        method="wb97m-v-rks",
        basis="sto-3g",
        basis_representation="spherical",
        device=device,
        density_fitting=device if fitted else "none",
        auxiliary_basis="def2-svp" if fitted else None,
        ks_options=KsOptions(grid_accuracy="tight"),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-13,
        max_iterations=100,
    )
    assert calculator.ks_options.grid == GridSpec.treutler(4)
    assert calculator.ks_options.nonlocal_grid == GridSpec.treutler(3)
    assert calculator.ks_options.has_separate_nonlocal_grid

    # Public CPU meta-GGA forces remain unsupported; the CUDA check exercises
    # the full analytic force including XC and VV10 moving-grid responses.
    forces = device == "cuda"
    actual = calculator.singlepoint(
        WATER, properties=("energy", "forces") if forces else ("energy",)
    )
    reference_energy, reference_forces = reference(
        calculator,
        fitted=fitted,
        functional="WB97M_V",
        grid_level=4,
        nonlocal_level=3,
        forces=forces,
    )
    assert actual.energy == pytest.approx(reference_energy, abs=2e-8)
    if forces:
        np.testing.assert_allclose(actual.forces, reference_forces, atol=3e-7, rtol=0)
        np.testing.assert_allclose(np.sum(actual.forces, axis=0), 0, atol=2e-10)
