"""Fitted, not direct, WB97M-V Hessians against independent energy differences."""

from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, Primitive, Shell
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.profiles import canonical_hash
from generativeqc.response_solver import GMRESOptions
from generativeqc_compiler.integral.df_second_execute import DFSecondEvaluator


@pytest.mark.parametrize(
    "family,angular",
    (
        ("coulomb_metric", (1, 0)),
        ("coulomb_metric", (3, 2)),
        ("three_center_eri", (1, 0, 0)),
        ("three_center_eri", (2, 1, 1)),
        ("three_center_eri", (0, 0, 3)),
    ),
)
def test_physical_df_primitive_gradient_jvp(
    family: str, angular: tuple, tmp_path: Path
) -> None:
    count = len(angular)
    evaluator = DFSecondEvaluator(family, angular, (0, 1, 2), tmp_path)
    centers = np.array(((0.1, -0.2, 0.3), (0.3, 0.2, 1.1), (-0.4, 0.5, 0.8)))[:count]
    direction = np.array(
        ((0.1, -0.03, 0.07), (-0.02, 0.06, -0.04), (0.03, 0.02, 0.01))
    )[:count]
    primitives = tuple(((0.7 + 0.2 * index, 1.0),) for index in range(count))
    values = evaluator.contract(primitives, centers, direction)
    step = 1e-4
    plus = evaluator.contract(primitives, centers + step * direction, direction)
    minus = evaluator.contract(primitives, centers - step * direction, direction)
    np.testing.assert_allclose(
        values[:, 1 + 3 * count :],
        (plus[:, 1 : 1 + 3 * count] - minus[:, 1 : 1 + 3 * count]) / (2 * step),
        atol=3e-9,
        rtol=3e-8,
    )
    np.testing.assert_allclose(
        np.einsum("pc,c->p", values[:, 1 : 1 + 3 * count], direction.reshape(-1)),
        (plus[:, 0] - minus[:, 0]) / (2 * step),
        atol=3e-9,
        rtol=3e-8,
    )


@pytest.mark.parametrize("unrestricted", (False, True))
def test_fitted_hvp_matches_reconverged_energy(
    unrestricted: bool, tmp_path: Path
) -> None:
    atoms = [("H", (0.13, -0.21, 0.17)), ("H", (0.43, 0.19, 2.1))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    left = np.array([[0.03, 0.04, -0.07], [-0.03, -0.04, 0.07]])
    calculator = Calculator(
        method="wb97m-v-uks" if unrestricted else "wb97m-v",
        device="cpu",
        basis_representation="spherical",
        density_fitting="cpu",
        auxiliary_basis=(
            Shell(0, 0, (Primitive(0.8, 1),)),
            Shell(1, 0, (Primitive(0.65, 1),)),
        ),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    )
    context = {"charge": int(unrestricted), "multiplicity": 2 if unrestricted else 1}
    result = calculator.hessian_vector_product(
        atoms,
        direction,
        **context,
        cache=tmp_path,
        solver_options=GMRESOptions(atol=1e-12, rtol=1e-11),
    )
    assert (
        result.diagnostics["density_fitting"] == "full-rank-full-range-jk-plus-exact-lr"
    )
    assert result.diagnostics["response_residual_norm"] < 2e-10
    np.testing.assert_allclose(result.value.sum(axis=0), 0, atol=2e-8)
    errors = []
    for step in (3e-3, 1e-3):
        finite = 0.0
        for first in (-1, 1):
            for second in (-1, 1):
                displaced = [
                    (
                        element,
                        np.asarray(position)
                        + step * (first * lmotion + second * rmotion),
                    )
                    for (element, position), lmotion, rmotion in zip(
                        atoms, left, direction, strict=True
                    )
                ]
                energy = calculator.singlepoint(
                    displaced, **context, properties=("energy",)
                )
                assert energy.converged
                finite += first * second * energy.energy / (4 * step**2)
        errors.append(abs(np.sum(result.value * left) - finite))
    assert max(errors) < 3e-6
    if not unrestricted:
        matrix = calculator.hessian(atoms, **context, cache=tmp_path)
        assert matrix.diagnostics["posthoc_symmetrization"] is False
        assert matrix.diagnostics["raw_symmetry_error"] < 2e-7
        np.testing.assert_allclose(
            matrix.matrix @ direction.reshape(-1), result.value.reshape(-1), atol=3e-9
        )


def test_rank_truncated_fitting_metric_fails_without_direct_fallback(
    tmp_path: Path,
) -> None:
    duplicate = Shell(0, 0, (Primitive(0.8, 1),))
    calculator = Calculator(
        method="wb97m-v-uks",
        device="cpu",
        density_fitting="cpu",
        auxiliary_basis=(duplicate, duplicate),
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    )
    atoms = [("H", (0.1, -0.2, 0.3)), ("H", (0.3, 0.2, 1.8))]
    with pytest.raises(NotImplementedError, match="full-rank metric"):
        calculator.hessian_vector_product(
            atoms,
            np.array([[0, 0, 0.1], [0, 0, -0.1]]),
            charge=1,
            multiplicity=2,
            cache=tmp_path,
        )


@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
def test_explicit_auxiliary_record_keeps_its_own_representation(
    representation: str, tmp_path: Path
) -> None:
    auxiliary = BasisSet(
        "Cartesian fitting fixture",
        (
            ElementBasis(
                1,
                (
                    BasisShell(0, ("0.8",), (("1",),)),
                    BasisShell(2, ("0.6",), (("1",),)),
                ),
            ),
        ),
        BasisProvenance(
            "synthetic test",
            "1",
            "mathematical fixture",
            canonical_hash({"s": 0.8, "d": 0.6}),
        ),
        representation=representation,
    )
    calculator = Calculator(
        method="wb97m-v",
        device="cpu",
        density_fitting="cpu",
        basis_representation="spherical",
        auxiliary_basis=auxiliary,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
        max_iterations=240,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    )
    atoms = [("H", (0.1, -0.2, 0.3)), ("H", (0.3, 0.2, 1.8))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    left = np.array([[0.03, 0.04, -0.07], [-0.03, -0.04, 0.07]])
    result = calculator.hessian_vector_product(atoms, direction, cache=tmp_path)
    step = 1e-3
    finite = 0.0
    for first in (-1, 1):
        for second in (-1, 1):
            displaced = [
                (
                    element,
                    np.asarray(position) + step * (first * lmotion + second * rmotion),
                )
                for (element, position), lmotion, rmotion in zip(
                    atoms, left, direction, strict=True
                )
            ]
            finite += (
                first
                * second
                * calculator.singlepoint(displaced, properties=("energy",)).energy
                / (4 * step**2)
            )
    np.testing.assert_allclose(
        np.sum(result.value * left), finite, atol=3e-6, rtol=3e-5
    )
