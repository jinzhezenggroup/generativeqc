"""Synthetic ECP endpoint derivatives against reconverged energy differences."""

import json
from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc.basis import BasisProvenance, BasisSet, BasisShell, ElementBasis
from generativeqc.profiles import canonical_hash
from generativeqc.response_solver import GMRESOptions


def synthetic_basis() -> BasisSet:
    """A bounded mathematical fixture, not physical sodium ECP parameters."""
    terms = [
        {
            "ecp_type": "scalar_ecp",
            "angular_momentum": [channel],
            "r_exponents": [2],
            "gaussian_exponents": ["0.8"],
            "coefficients": [[coefficient]],
        }
        for channel, coefficient in ((0, "0.3"), (1, "-0.4"))
    ]
    return BasisSet(
        "synthetic Na-core/H Hessian fixture",
        (
            ElementBasis(
                11,
                (BasisShell(0, ("0.7",), (("1",),)),),
                ecp_core_electrons=10,
                ecp_data=json.dumps(terms),
            ),
            ElementBasis(1, (BasisShell(0, ("0.9",), (("1",),)),)),
        ),
        BasisProvenance(
            "synthetic test parameters",
            "1",
            "test-only mathematical fixture",
            canonical_hash(terms),
        ),
        representation="spherical",
    )


@pytest.mark.parametrize("unrestricted", (False, True))
@pytest.mark.parametrize("fitted", (False, True))
def test_ecp_hvp_includes_effective_charges_and_both_residual_channels(
    unrestricted: bool, fitted: bool, tmp_path: Path
) -> None:
    atoms = [("Na", (0.13, -0.21, 0.17)), ("H", (0.43, 0.19, 2.1))]
    direction = np.array([[0.02, -0.03, 0.1], [-0.02, 0.03, -0.1]])
    left = np.array([[0.03, 0.04, -0.07], [-0.03, -0.04, 0.07]])
    calculator = Calculator(
        method="wb97m-v-uks" if unrestricted else "wb97m-v",
        basis=synthetic_basis(),
        device="cpu",
        basis_representation="spherical",
        density_fitting="cpu" if fitted else "none",
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
    assert {"ecp_local", "ecp_nonlocal"} <= set(result.components)
    assert np.linalg.norm(result.components["ecp_local"]) > 1e-6
    assert np.linalg.norm(result.components["ecp_nonlocal"]) > 1e-6
    np.testing.assert_allclose(result.value.sum(axis=0), 0, atol=2e-8)
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
        np.testing.assert_allclose(
            np.sum(result.value * left), finite, atol=3e-6, rtol=3e-5
        )
    if fitted and unrestricted:
        matrix = calculator.hessian(atoms, **context, cache=tmp_path)
        assert matrix.diagnostics["posthoc_symmetrization"] is False
        assert matrix.diagnostics["raw_symmetry_error"] < 2e-7
        np.testing.assert_allclose(
            matrix.matrix @ direction.reshape(-1), result.value.reshape(-1), atol=3e-8
        )
