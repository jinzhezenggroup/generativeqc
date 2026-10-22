"""Molecular screening keeps physical tails and an inclusive default boundary."""

import numpy as np
import pytest
from generativeqc_compiler.dft import FixedDensityNonlocalCorrelation
from generativeqc_compiler.dft.nonlocal_policy import MOLECULAR_VV10_DENSITY_POLICY
from generativeqc_compiler.dft.nonlocal_reference import nonlocal_energy_reference
from generativeqc_compiler.method import original_nonlocal_correlation


@pytest.mark.parametrize("variant", ("vv10", "rvv10"))
def test_default_density_domain_matches_explicit_pair_selection(variant: str) -> None:
    """Both pair legs and beta exclude vacuum; retained tail features are exact."""
    spec = original_nonlocal_correlation(variant)
    owner = FixedDensityNonlocalCorrelation(
        spec, density_policy=MOLECULAR_VV10_DENSITY_POLICY
    )
    rho = np.array([0.0, np.nextafter(1e-10, 0.0), 1e-10, 1e-9, 0.4])
    weights = np.array([0.2, 0.3, -0.7, 0.8, 1.1])
    points = np.arange(15).reshape(5, 3) / 10
    gradient = rho[:, None] * np.array([[0.1, -0.2, 0.3]])
    actual_w, actual_rho, actual_g, active = owner._density_domain(
        weights, rho, gradient
    )
    np.testing.assert_array_equal(active, [False, False, True, True, True])
    np.testing.assert_array_equal(actual_w[:2], 0)
    np.testing.assert_array_equal(actual_rho[:2], 1)
    np.testing.assert_array_equal(actual_g[:2], 0)
    for actual, expected in (
        (actual_w, weights),
        (actual_rho, rho),
        (actual_g, gradient),
    ):
        np.testing.assert_array_equal(actual[2:], expected[2:])
    expected_energy = nonlocal_energy_reference(
        points[2:], weights[2:], rho[2:], gradient[2:], spec
    )
    actual_energy = nonlocal_energy_reference(
        points, actual_w, actual_rho, actual_g, spec
    )
    assert actual_energy == pytest.approx(expected_energy, abs=1e-18, rel=2e-14)


def test_old_policy_identity_cannot_silently_replay_new_domain() -> None:
    with pytest.raises(ValueError, match="unknown nonlocal density policy"):
        FixedDensityNonlocalCorrelation(
            original_nonlocal_correlation("vv10"),
            density_policy="vv10-molecular-rho-ge-1e-8-v1",
        )
