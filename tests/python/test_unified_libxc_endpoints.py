"""Opt-in complete composed Libxc endpoints against independent PySCF."""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from test_r1_native_forces import forbid_python_force
from test_treutler_endpoints import WATER, reference, require_endpoint


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("name", ("RSCAN0", "R2SCAN0"))
@pytest.mark.parametrize("spin", (0, 1))
def test_composed_energy_force_lifecycle(
    device: str, name: str, spin: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prepared native owner must retain EXX and all force sources on reuse."""
    require_endpoint(device)
    forbid_python_force(monkeypatch)
    functional = f"0.75*MGGA_X_{name[:-1]} + MGGA_C_{name[:-1]} + 0.25*HF"
    calculator = Calculator(
        method=f"libxc-{'uks' if spin else 'rks'}:{name}",
        basis="sto-3g",
        basis_representation="spherical",
        device=device,
        precision="fp64",
        initial_guess=None,
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-13,
        ks_options=KsOptions(grid=GridSpec.treutler(0)),
    )
    coordinates = np.array([point for _, point in WATER])
    moved = coordinates.copy()
    moved[1, 2] += 0.02
    references = {}
    with calculator.prepare_batch(
        [WATER], charges=[spin], multiplicities=[spin + 1], warm_start=True
    ) as batch:
        for phase, geometry in (
            ("cold", coordinates),
            ("warm", coordinates),
            ("moved", moved),
        ):
            result = batch.execute(
                [geometry], properties=("energy", "forces"), strict=True
            ).items[0]
            key = "moved" if phase == "moved" else "original"
            if key not in references:
                atoms = [
                    (element, tuple(point))
                    for (element, _), point in zip(WATER, geometry, strict=True)
                ]
                references[key] = reference(
                    calculator,
                    fitted=False,
                    functional=functional,
                    spin=spin,
                    atom_values=atoms,
                    grid_level=0,
                )
            energy, forces = references[key]
            assert result.converged
            assert abs(result.energy - energy) < 2e-8
            np.testing.assert_allclose(result.forces, forces, rtol=0, atol=3e-7)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize(
    "component", ("LDA_XC_TETER93", "GGA_X_APBE", "GGA_X_PBE", "GGA_X_B88")
)
def test_atomic_components_preserve_requested_functional(
    device: str, component: str
) -> None:
    require_endpoint(device)
    calculator = Calculator(
        method=f"libxc:{component}",
        basis="sto-3g",
        basis_representation="spherical",
        device=device,
        precision="fp64",
        initial_guess=None,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        ks_options=KsOptions(grid=GridSpec.treutler(0)),
    )
    result = calculator.singlepoint(WATER, properties=("energy",))
    energy, _ = reference(
        calculator, fitted=False, functional=component, forces=False, grid_level=0
    )
    assert result.converged
    assert abs(result.energy - energy) < 2e-8


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("name", ("RSCAN0", "R2SCAN0"))
@pytest.mark.parametrize("spin", (0, 1))
def test_composed_density_fitted_energy(device: str, name: str, spin: int) -> None:
    """A registered hybrid must use the same fitted approximation for J and K."""
    require_endpoint(device)
    calculator = Calculator(
        method=f"libxc-{'uks' if spin else 'rks'}:{name}",
        basis="sto-3g",
        basis_representation="spherical",
        device=device,
        density_fitting=device,
        precision="fp64",
        initial_guess=None,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        ks_options=KsOptions(grid=GridSpec.treutler(0)),
    )
    result = calculator.singlepoint(
        WATER, charge=spin, multiplicity=spin + 1, properties=("energy",)
    )
    functional = f"0.75*MGGA_X_{name[:-1]} + MGGA_C_{name[:-1]} + 0.25*HF"
    energy, _ = reference(
        calculator,
        fitted=True,
        functional=functional,
        spin=spin,
        forces=False,
        grid_level=0,
    )
    assert result.converged
    assert abs(result.energy - energy) < 2e-8
