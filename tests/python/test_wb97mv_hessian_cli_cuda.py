"""Real Slurm GPU qualification through the actual compiled C++ CLI boundary."""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.response_solver import GMRESOptions

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_WB97MV_HESSIAN_CUDA_TEST") != "1"
    or not os.environ.get("GENERATIVEQC_WB97MV_CLI"),
    reason="requires an explicitly allocated Slurm GPU and a compiled C++ CLI",
)


def invoke_cli(
    tmp_path: Path,
    atoms: list,
    *,
    operation: str = "hvp",
    direction: np.ndarray | None = None,
    extra: tuple = (),
) -> dict:
    """Preserve Slurm visibility and exercise input, process and output contracts."""
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    xyz = tmp_path / "Input With Spaces.xyz"
    xyz.write_text(
        str(len(atoms))
        + "\nBohr\n"
        + "\n".join(
            element + " " + " ".join(str(value) for value in position)
            for element, position in atoms
        )
        + "\n"
    )
    command = [
        os.environ["GENERATIVEQC_WB97MV_CLI"],
        operation,
        str(xyz),
        "--units",
        "bohr",
        "--backend",
        "cuda",
        "--representation",
        "spherical",
        "--json",
        "--grid-radial-points",
        "8",
        "--grid-polar-points",
        "4",
        "--grid-azimuth-points",
        "8",
        "--max-iterations",
        "240",
        "--energy-tolerance",
        "1e-13",
        "--density-tolerance",
        "1e-11",
        "--response-atol",
        "1e-12",
        "--response-rtol",
        "1e-11",
    ]
    if direction is not None:
        source = tmp_path / "Direction With Spaces.json"
        source.write_text(json.dumps(direction.tolist()))
        command += ["--direction", str(source)]
    result = subprocess.run(
        [*command, *extra],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
        env=dict(
            os.environ,
            PATH="/nonexistent",
            PYTHONHOME="/nonexistent",
            PYTHONPATH="/nonexistent",
        ),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    (tmp_path / f"{operation}-result.json").write_text(result.stdout)
    assert payload["operation"] == operation
    assert payload["units"] == "hartree/bohr^2"
    assert payload["backend"] == "cuda"
    diagnostics = payload["diagnostics"]
    assert diagnostics["execution_residency"] == "device-resident"
    assert diagnostics["numerical_host_fallbacks"] == 0
    assert diagnostics["intermediate_panel_downloads"] == 0
    assert diagnostics["displaced_scf_evaluations"] == 0
    assert diagnostics["runtime_python"] is False
    assert diagnostics["runtime_compilation"] is False
    assert diagnostics["scf_iterations"] > 0
    for name in (
        "first_integral_backend",
        "second_integral_backend",
        "grid_ad_backend",
        "response_backend",
    ):
        assert diagnostics[name] == "cuda"
    return payload


def test_cpp_cli_cuda_direct_rks_hessian_and_hvp(tmp_path: Path) -> None:
    atoms = [("H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    hvp = invoke_cli(tmp_path, atoms, direction=direction)
    matrix = invoke_cli(tmp_path, atoms, operation="hessian")
    assert matrix["diagnostics"]["posthoc_symmetrization"] is False
    matrix = np.asarray(matrix["matrix"])
    np.testing.assert_allclose(
        matrix @ direction.reshape(-1),
        np.asarray(hvp["value"]).reshape(-1),
        atol=2e-8,
        rtol=2e-6,
    )
    np.testing.assert_allclose(matrix, matrix.T, atol=2e-8, rtol=0)
    cpu = Calculator(
        method="wb97m-v",
        device="cpu",
        basis_representation="spherical",
        initial_guess=None,
        max_iterations=240,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    )
    energy = cpu.singlepoint(atoms, properties=("energy",)).energy
    np.testing.assert_allclose(hvp["energy_hartree"], energy, atol=5e-9, rtol=0)
    expected = cpu.hessian_vector_product(
        atoms,
        direction,
        cache=tmp_path,
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )
    np.testing.assert_allclose(hvp["value"], expected.value, atol=3e-8, rtol=2e-6)


def test_cpp_cli_cuda_uks_df_ecp_with_spherical_d_auxiliary(tmp_path: Path) -> None:
    from test_wb97mv_ecp_hessian import synthetic_basis

    atoms = [
        ("Na", (0.13, -0.21, 0.17)),
        ("H", (0.43, 0.19, 2.1)),
        ("H", (-0.3, 0.4, -1.6)),
    ]
    direction = np.array(
        [[0.02, -0.03, 0.1], [-0.04, 0.01, -0.02], [0.02, 0.02, -0.08]]
    )
    orbital = synthetic_basis()
    auxiliary = BasisSet(
        "synthetic spherical d auxiliary CLI fixture",
        (
            ElementBasis(
                11,
                (
                    BasisShell(0, ("0.8",), (("1",),)),
                    BasisShell(2, ("0.6",), (("1",),)),
                ),
            ),
            ElementBasis(1, (BasisShell(0, ("0.65",), (("1",),)),)),
        ),
        BasisProvenance(
            "synthetic test parameters", "1", "test-only fixture", "a" * 64
        ),
        representation="spherical",
    )
    orbital_file, auxiliary_file = (
        tmp_path / "Orbital ECP.json",
        tmp_path / "Auxiliary D.json",
    )
    orbital.write(orbital_file)
    auxiliary.write(auxiliary_file)
    result = invoke_cli(
        tmp_path,
        atoms,
        direction=direction,
        extra=(
            "--method",
            "wb97m-v-uks",
            "--multiplicity",
            "2",
            "--density-fitting",
            "cuda",
            "--basis",
            str(orbital_file),
            "--auxiliary-basis",
            str(auxiliary_file),
        ),
    )
    assert result["diagnostics"]["response_iterations"] > 0
    assert result["diagnostics"]["ecp_ad_backend"] == "cuda"
    assert result["diagnostics"]["df_integral_ad_backend"] == "cuda"
    cpu = Calculator(
        method="wb97m-v-uks",
        device="cpu",
        basis=orbital,
        auxiliary_basis=auxiliary,
        density_fitting="cpu",
        basis_representation="spherical",
        initial_guess=None,
        max_iterations=240,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    )
    energy = cpu.singlepoint(atoms, multiplicity=2, properties=("energy",)).energy
    np.testing.assert_allclose(result["energy_hartree"], energy, atol=5e-9, rtol=0)
    expected = cpu.hessian_vector_product(
        atoms,
        direction,
        multiplicity=2,
        cache=tmp_path,
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )
    np.testing.assert_allclose(result["value"], expected.value, atol=3e-8, rtol=2e-6)
