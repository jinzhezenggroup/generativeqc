"""Independent, complete-endpoint qualification of the generic CPU force seam."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import Calculator
from generativeqc_compiler.common.provenance import atomic_json

from tools.qualify_libxc_forces import GATES, _endpoint, qualify_forces

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "name",
    (
        "LDA_C_VWN_4",
        "LDA_C_CHACHIYO",
        "LDA_C_PZ_MOD",
        "GGA_X_APBE",
        "GGA_X_PBE_SOL",
        "GGA_X_RPBE",
        "GGA_X_B86",
        "GGA_C_APBE",
        "MGGA_X_LTA",
        "MGGA_X_GVT4",
        "MGGA_C_TPSS",
    ),
)
def test_automatic_cpu_force_lifecycle_against_independent_libxc(
    name: str, tmp_path: Path
) -> None:
    pytest.importorskip("pyscf")
    artifact = tmp_path / f"{name}.json"
    payload = qualify_forces(name, evidence=str(artifact))
    atomic_json(artifact, payload)
    assert payload["stage_evidence"]["status"] == "pass"
    assert payload["stage_evidence"]["qualification"]["coverage"] == [
        {"backend": "cpu", "spin": spin, "products": ["energy", "forces"]}
        for spin in ("polarized", "unpolarized")
    ]
    for row in payload["result"]["rows"]:
        np.testing.assert_allclose(
            row["forces"], row["reference_fd"], atol=GATES["force_atol"], rtol=0
        )


def test_automatic_force_does_not_inflate_backend_neutral_or_second_order_claims() -> (
    None
):
    from generativeqc import method_capabilities

    calculator = Calculator(method="libxc:MGGA_X_LTA", basis="sto-3g", device="cpu")
    assert calculator.capabilities.supported_properties == frozenset(
        ("energy", "forces")
    )
    assert not calculator.capabilities.supported_second_order
    assert method_capabilities("libxc:MGGA_X_LTA").supported_properties == frozenset(
        ("energy",)
    )
    assert calculator._default_properties(batch=True) == frozenset(("energy",))


@pytest.mark.parametrize("name", ("LDA_C_VWN_4", "GGA_X_APBE", "MGGA_X_LTA"))
def test_full_polarization_zero_beta_density_force_is_variational(name: str) -> None:
    from generativeqc import GridSpec, KsOptions

    calculator = Calculator(
        method=f"libxc-uks:{name}",
        basis="sto-3g",
        device="cpu",
        initial_guess=None,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
        ),
    )
    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (0.1, 0.2, 1.4))]
    result = calculator.singlepoint(
        atoms, multiplicity=3, properties=("energy", "forces")
    )
    step = 1e-4
    energies = []
    for direction in (-1, 1):
        displaced = [("H", (0.0, 0.0, 0.0)), ("H", (0.1, 0.2, 1.4 + direction * step))]
        endpoint = calculator.singlepoint(
            displaced, multiplicity=3, properties=("energy",)
        )
        assert endpoint.converged
        energies.append(endpoint.energy)
    assert result.converged
    assert np.isfinite(result.forces).all()
    np.testing.assert_allclose(
        result.forces[1, 2],
        -(energies[1] - energies[0]) / (2 * step),
        atol=GATES["force_atol"],
        rtol=0,
    )


@pytest.mark.parametrize("name", ("GGA_X_APBE", "MGGA_X_LTA"))
def test_imported_s_p_force_reuses_full_moving_grid_and_pulay_sources(
    name: str,
) -> None:
    from generativeqc import GridSpec, KsOptions

    pytest.importorskip("pyscf")
    calculator = Calculator(
        method=f"libxc:{name}",
        basis="sto-3g",
        device="cpu",
        initial_guess=None,
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)
        ),
    )
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.7)), ("H", (1.6, 0.0, -0.5))]
    with calculator.prepare_batch([atoms], warm_start=False) as batch:
        central = _endpoint(calculator, batch, atoms, name, "unpolarized", forces=True)
    step = 1e-4
    stencil = []
    for direction in (-1, 1):
        displaced = [atoms[0], ("H", (direction * step, 0.0, 1.7)), atoms[2]]
        with calculator.prepare_batch([displaced], warm_start=False) as batch:
            stencil.append(
                _endpoint(
                    calculator, batch, displaced, name, "unpolarized", forces=False
                )
            )
    assert abs(central["energy"] - central["reference_energy"]) <= GATES["energy_atol"]
    for source in ("energy", "reference_energy"):
        finite_difference = -(stencil[1][source] - stencil[0][source]) / (2 * step)
        np.testing.assert_allclose(
            central["forces"][1][0], finite_difference, atol=GATES["force_atol"], rtol=0
        )
