"""Compiler spin curvature and complete open-shell molecular response gates."""

from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.stationary_grid_second import (
    StationaryGridSecondOrder,
)


def test_polarized_grid_hessian_keeps_cross_spin_and_total_density_nonlocal() -> None:
    method = resolve_method("WB97M-V", spin="polarized")
    model = StationaryGridSecondOrder(
        method.primitives[0].functional, method.primitives[-1]
    )
    generator = np.random.default_rng(3199)
    fields = generator.normal(size=(14, 5)) * 0.1
    fields[[0, 4, 5, 9]] = generator.uniform(0.1, 0.7, size=(4, 5))
    fields[10] = np.linspace(0.1, 0.8, 5)
    left, right = (generator.normal(size=fields.shape) * 0.03 for _ in range(2))
    result, transpose = model.evaluate(fields, right), model.evaluate(fields, left)
    np.testing.assert_allclose(
        np.sum(left * result.directional_coefficients),
        np.sum(right * transpose.directional_coefficients),
        atol=2e-12,
    )
    np.testing.assert_array_equal(
        result.nonlocal_correlation[0][:5], result.nonlocal_correlation[0][5:10]
    )
    assert np.linalg.norm(result.directional_coefficients[:5]) > 0
    for step in (3e-3, 1e-3, 3e-4):
        plus, minus = (
            model.evaluate(fields + sign * step * right, np.zeros_like(fields))
            for sign in (1, -1)
        )
        np.testing.assert_allclose(
            (plus.energy - minus.energy) / (2 * step),
            np.sum(result.coefficients * right),
            atol=3e-10,
            rtol=2e-7,
        )
        np.testing.assert_allclose(
            (plus.coefficients - minus.coefficients) / (2 * step),
            result.directional_coefficients,
            atol=3e-9,
            rtol=2e-5,
        )


@pytest.mark.parametrize("case", ("heh", "h2+", "h3"))
def test_open_shell_hvp_matches_reconverged_forces_with_empty_spin_support(
    case: str, tmp_path: Path
) -> None:
    atoms = [("He", (0.03, -0.02, 0.01)), ("H", (0.15, 0.13, 1.8))]
    charge = int(case == "h2+")
    if case == "h2+":
        atoms[0] = ("H", atoms[0][1])
    elif case == "h3":
        atoms = [("H", (0, 0.1, -1.6)), ("H", (0.3, -0.2, 0)), ("H", (-0.1, 0.4, 1.6))]
    calculator = Calculator(
        method="wb97m-v-uks",
        device="cpu",
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    )
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    if case == "h3":
        direction = np.array(
            [[0.02, -0.03, 0.1], [-0.04, 0.01, -0.02], [0.02, 0.02, -0.08]]
        )
    result = calculator.hessian_vector_product(
        atoms,
        direction,
        charge=charge,
        multiplicity=2,
        cache=tmp_path,
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )
    assert result.diagnostics["response_residual_norm"] < 2e-10
    assert result.diagnostics["displaced_scf_evaluations"] == 0
    np.testing.assert_allclose(result.value.sum(axis=0), 0, atol=2e-8)
    for step in (3e-3, 1e-3, 3e-4):
        forces = []
        for sign in (-1, 1):
            displaced = [
                (element, np.asarray(position) + sign * step * motion)
                for (element, position), motion in zip(atoms, direction, strict=True)
            ]
            force = calculator.singlepoint(
                displaced,
                charge=charge,
                multiplicity=2,
                properties=("energy", "forces"),
            )
            assert force.converged
            forces.append(force.forces)
        np.testing.assert_allclose(
            result.value, -(forces[1] - forces[0]) / (2 * step), atol=3e-6, rtol=3e-5
        )
