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
) -> tuple[float, np.ndarray]:
    """Share basis input only; PySCF owns quadrature, SCF and analytic gradients."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft, gto

    pyscf.lib.num_threads(1)
    atoms = tuple(Atom.from_value(a) for a in WATER)
    labels = [f"{symbol}{i}" for i, (symbol, _) in enumerate(WATER)]

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
    if fitted:
        mf = mf.density_fit(auxbasis=basis_dict(calculator._auxiliary_basis))
    mf.xc = functional
    mf.small_rho_cutoff = 0
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-10
    mf.direct_scf_tol, mf.max_cycle = 1e-14, 100
    if nonlocal_level is not None:
        mf.nlcgrids.level = nonlocal_level
    # Neither XC nor VV10 radial/angular/pruning defaults are replaced.
    energy = mf.kernel()
    assert mf.converged
    gradient = mf.nuc_grad_method()
    gradient.grid_response = True
    if fitted:
        gradient.auxbasis_response = True
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
    [("pbe", "PBE"), ("pbe0", "PBE0"), ("r2scan", "r2SCAN"), ("wb97m-v", "WB97M_V")],
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
        density_fitting="auto" if fitted else "none",
        auxiliary_basis="def2-svp" if fitted else None,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-13,
        max_iterations=100,
    )
    assert calculator.ks_options.grid == GridSpec.treutler(3)
    actual = calculator.singlepoint(WATER, properties=("energy", "forces"))
    expected_energy, expected_forces = reference(
        calculator, fitted=fitted, functional=functional
    )
    assert actual.energy == pytest.approx(expected_energy, abs=2e-8)
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
        density_fitting="auto" if fitted else "none",
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
