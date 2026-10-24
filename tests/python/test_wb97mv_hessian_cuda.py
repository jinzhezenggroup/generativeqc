"""Real-device resident compiler-AD Hessians, exclusively under Slurm."""

import os
from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, Primitive, Shell
from generativeqc.response_solver import GMRESOptions

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_WB97MV_HESSIAN_CUDA_TEST") != "1",
    reason="requires an explicitly allocated Slurm GPU",
)


@pytest.mark.parametrize("unrestricted", (False, True))
def test_cuda_composite_hvp_matches_cpu_and_complete_force_differences(
    unrestricted: bool, tmp_path: Path
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [("H", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.5))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    options = {
        "method": "wb97m-v-uks" if unrestricted else "wb97m-v",
        "basis_representation": "cartesian" if unrestricted else "spherical",
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "max_iterations": 240,
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    }
    context = {"charge": int(unrestricted), "multiplicity": 2 if unrestricted else 1}
    cpu, cuda = (Calculator(device=device, **options) for device in ("cpu", "cuda"))
    expected = cpu.hessian_vector_product(
        atoms,
        direction,
        **context,
        cache=tmp_path,
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )
    actual = cuda.hessian_vector_product(
        atoms,
        direction,
        **context,
        cache=tmp_path,
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )
    assert actual.diagnostics["backend"] == "cuda"
    assert actual.diagnostics["second_integral_backend"] == "cuda"
    assert actual.diagnostics["grid_ad_backend"] == "cuda"
    assert actual.diagnostics["first_integral_backend"] == "cuda"
    assert actual.diagnostics["response_backend"] == "cuda"
    assert actual.diagnostics["execution_residency"] == "device-resident"
    assert actual.diagnostics["numerical_host_fallbacks"] == 0
    assert actual.diagnostics["intermediate_panel_downloads"] == 0
    assert actual.diagnostics["displaced_scf_evaluations"] == 0
    np.testing.assert_allclose(actual.value, expected.value, atol=3e-8, rtol=2e-6)
    step = 3e-4
    forces = []
    for sign in (-1, 1):
        displaced = [
            (element, np.asarray(position) + sign * step * motion)
            for (element, position), motion in zip(atoms, direction, strict=True)
        ]
        result = cpu.singlepoint(displaced, **context, properties=("energy", "forces"))
        assert result.converged
        forces.append(result.forces)
    np.testing.assert_allclose(
        actual.value, -(forces[1] - forces[0]) / (2 * step), atol=3e-6, rtol=3e-5
    )
    if unrestricted:
        matrix = cuda.hessian(atoms, **context, cache=tmp_path)
        assert matrix.diagnostics["posthoc_symmetrization"] is False
        assert matrix.diagnostics["raw_symmetry_error"] < 2e-7
        np.testing.assert_allclose(
            matrix.matrix @ direction.reshape(-1), actual.value.reshape(-1), atol=3e-8
        )


@pytest.mark.parametrize("fitted,ecp", ((True, False), (False, True), (True, True)))
def test_cuda_fitted_ecp_spherical_spin_combinations(
    fitted: bool, ecp: bool, tmp_path: Path
) -> None:
    from test_wb97mv_ecp_hessian import synthetic_basis

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [("Na" if ecp else "H", (0.13, -0.21, 0.17)), ("H", (0.43, 0.19, 2.1))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    common = {
        "method": "wb97m-v-uks",
        "basis": synthetic_basis() if ecp else "sto-3g",
        "basis_representation": "spherical",
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "max_iterations": 240,
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    }
    if fitted:
        common["auxiliary_basis"] = (
            Shell(0, 0, (Primitive(0.8, 1),)),
            Shell(1, 0, (Primitive(0.65, 1),)),
        )
    context = {"charge": 1, "multiplicity": 2}
    results = []
    for device in ("cpu", "cuda"):
        calculator = Calculator(
            device=device, density_fitting=device if fitted else "none", **common
        )
        results.append(
            calculator.hessian_vector_product(
                atoms,
                direction,
                **context,
                cache=tmp_path,
                solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
            )
        )
    expected, actual = results
    np.testing.assert_allclose(actual.value, expected.value, atol=3e-8, rtol=2e-6)
    assert actual.diagnostics["backend"] == "cuda"
    assert actual.diagnostics["execution_residency"] == "device-resident"
    if fitted:
        assert actual.diagnostics["df_integral_ad_backend"] == "cuda"
    if ecp:
        assert {"ecp_local", "ecp_nonlocal"} <= set(actual.components)


def test_cuda_coupled_nonempty_alpha_beta_response(tmp_path: Path) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [("H", (0, 0.1, -1.6)), ("H", (0.3, -0.2, 0)), ("H", (-0.1, 0.4, 1.6))]
    direction = np.array(
        [[0.02, -0.03, 0.1], [-0.04, 0.01, -0.02], [0.02, 0.02, -0.08]]
    )
    options = {
        "method": "wb97m-v-uks",
        "basis_representation": "spherical",
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "max_iterations": 240,
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    }
    cpu, cuda = (Calculator(device=device, **options) for device in ("cpu", "cuda"))
    results = [
        calculator.hessian_vector_product(
            atoms,
            direction,
            multiplicity=2,
            cache=tmp_path,
            solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
        )
        for calculator in (cpu, cuda)
    ]
    np.testing.assert_allclose(results[1].value, results[0].value, atol=3e-8, rtol=2e-6)
    forces = []
    step = 3e-4
    for sign in (-1, 1):
        displaced = [
            (element, np.asarray(position) + sign * step * motion)
            for (element, position), motion in zip(atoms, direction, strict=True)
        ]
        result = cpu.singlepoint(
            displaced, multiplicity=2, properties=("energy", "forces")
        )
        assert result.converged
        forces.append(result.forces)
    np.testing.assert_allclose(
        results[1].value, -(forces[1] - forces[0]) / (2 * step), atol=3e-6, rtol=3e-5
    )
