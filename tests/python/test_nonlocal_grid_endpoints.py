"""Full independent-grid E+F against upstream molecular integrators.

CUDA cases require an explicitly allocated Slurm GPU. These are same-prescription
implementation checks, not claims that these deliberately small grids converge.
"""

import os
from dataclasses import replace

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions

ATOMS = (("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.8)))
PRIMARY = GridSpec(8, 4, 8)


def _cpu_reference(spin: int, secondary: GridSpec) -> tuple[float, np.ndarray]:
    """Reconstruct product rules independently; PySCF owns XC/AO/grid response."""
    from pyscf import dft, gto

    molecule = gto.M(
        atom=ATOMS, unit="Bohr", basis="sto-3g", spin=spin, charge=spin, verbose=0
    )
    engine = (dft.UKS if spin else dft.RKS)(molecule)
    engine.xc = "WB97M_V"
    for target, spec in ((engine.grids, PRIMARY), (engine.nlcgrids, secondary)):
        radial_nodes, radial_weights = np.polynomial.legendre.leggauss(
            spec.radial_points
        )
        radial_parameter = (radial_nodes + 1) / 2
        radii = radial_parameter / (1 - radial_parameter)
        measures = (
            radial_weights / 2 * radial_parameter**2 / (1 - radial_parameter) ** 4
        )
        heights, polar_weights = np.polynomial.legendre.leggauss(spec.angular_polar)
        angles = np.arange(spec.angular_azimuth) * (2 * np.pi / spec.angular_azimuth)
        polar, azimuth = np.meshgrid(heights, angles, indexing="ij")
        directions = np.stack(
            (
                np.sqrt(1 - polar**2) * np.cos(azimuth),
                np.sqrt(1 - polar**2) * np.sin(azimuth),
                polar,
            ),
            axis=-1,
        )
        coordinates = (radii[:, None, None, None] * directions).reshape(-1, 3)
        angular_weights = np.repeat(
            polar_weights * 2 * np.pi / len(angles), len(angles)
        )
        weights = (measures[:, None] * angular_weights).ravel()
        table = {"H": (coordinates, weights)}
        target.gen_atomic_grids = lambda molecule, *args, table=table, **kwargs: table
        target.radii_adjust = None
        target.becke_scheme = dft.gen_grid.original_becke
        target.prune = None
    engine.small_rho_cutoff = 0
    engine.conv_tol, engine.conv_tol_grad, engine.direct_scf_tol = 1e-12, 1e-10, 1e-14
    energy = engine.kernel()
    assert engine.converged
    gradient = engine.nuc_grad_method()
    gradient.grid_response = True
    return float(energy), -np.asarray(gradient.kernel())


def _calculator(spin: int, secondary: GridSpec | None, device: str) -> Calculator:
    return Calculator(
        method="wb97m-v-uks" if spin else "wb97m-v",
        basis="sto-3g",
        device=device,
        basis_representation="spherical",
        ks_options=KsOptions(grid=PRIMARY, nonlocal_grid=secondary, tile_points=64),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-12,
    )


@pytest.mark.parametrize("spin", (0, 1))
@pytest.mark.parametrize(
    "secondary", (GridSpec(6, 4, 8), GridSpec(4, 8, 8), GridSpec(10, 4, 8))
)
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_independent_grid_complete_endpoint_matches_upstream(
    spin: int, secondary: GridSpec, device: str
) -> None:
    if device == "cuda" and os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    if not os.environ.get("GENERATIVEQC_LIBRARY"):
        pytest.skip("requires a source-matched native library")
    calculator = _calculator(spin, secondary, device)
    actual = calculator.singlepoint(
        ATOMS, charge=spin, multiplicity=spin + 1, properties=("energy", "forces")
    )
    if device == "cpu":
        expected_energy, expected_forces = _cpu_reference(spin, secondary)
    else:
        import cupy as cp

        from benchmarks.readme_wb97mv import reference_engine, reference_sample

        engine = reference_engine(
            ATOMS,
            "sto-3g",
            PRIMARY,
            spin=spin,
            charge=spin,
            nonlocal_spec=secondary,
            full_fock=True,
        )
        expected = reference_sample(engine, cp)
        expected_energy = expected["energies_hartree"][0]
        expected_forces = expected["forces_hartree_per_bohr"][0]
    assert actual.energy == pytest.approx(expected_energy, abs=1e-8)
    np.testing.assert_allclose(actual.forces, expected_forces, atol=1e-7, rtol=0)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize(
    "secondary", (PRIMARY, replace(PRIMARY, element_radii=((1, 1.0),)))
)
def test_equal_grid_retains_complete_endpoint_results(
    secondary: GridSpec, device: str
) -> None:
    if device == "cuda" and os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    if not os.environ.get("GENERATIVEQC_LIBRARY"):
        pytest.skip("requires a source-matched native library")
    implicit = _calculator(0, None, device).singlepoint(
        ATOMS, properties=("energy", "forces")
    )
    explicit = _calculator(0, secondary, device).singlepoint(
        ATOMS, properties=("energy", "forces")
    )
    assert implicit.energy == pytest.approx(explicit.energy, abs=1e-13)
    np.testing.assert_allclose(implicit.forces, explicit.forces, atol=1e-12, rtol=0)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_independent_partition_force_matches_reconverged_energy_difference(
    device: str,
) -> None:
    if device == "cuda" and os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    if not os.environ.get("GENERATIVEQC_LIBRARY"):
        pytest.skip("requires a source-matched native library")
    calculator = _calculator(
        0,
        replace(GridSpec(6, 4, 8), partition_iterations=2, element_radii=((1, 1.3),)),
        device,
    )
    actual = calculator.singlepoint(ATOMS, properties=("energy", "forces"))
    step = 1e-4
    energies = []
    for direction in (-1, 1):
        moved = list(ATOMS)
        moved[1] = ("H", (0.0, 0.0, 0.8 + direction * step))
        energies.append(calculator.singlepoint(moved, properties=("energy",)).energy)
    derivative = (energies[1] - energies[0]) / (2 * step)
    assert actual.forces[1, 2] == pytest.approx(-derivative, abs=1e-7)
