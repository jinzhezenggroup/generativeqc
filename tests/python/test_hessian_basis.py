"""Native-normalized spherical pullbacks and public molecular derivatives."""

from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, Primitive, Shell
from generativeqc.hessian_basis import HessianBasis
from generativeqc_compiler.dft import NativeAO


@pytest.mark.parametrize("angular", (1, 2, 3))
def test_native_ao_map_preserves_all_jets_and_bilinear_cotangents(angular: int) -> None:
    atoms = [("He", (0.2, -0.1, 0.3))]
    shells = [Shell(0, angular, (Primitive(0.7, 1.0),))]
    points = np.random.default_rng(149).normal(size=(7, 3))
    with NativeAO(atoms, shells, representation="spherical") as public:
        mapping = HessianBasis(public)
        try:
            jets = mapping.cartesian.evaluate(points, 3)
            np.testing.assert_allclose(
                jets @ mapping.transform, public.evaluate(points, 3), atol=2e-14
            )
            density = np.random.default_rng(213).normal(size=(public.nao, public.nao))
            density += density.T
            matrix = np.random.default_rng(311).normal(
                size=(mapping.cartesian.nao,) * 2
            )
            np.testing.assert_allclose(
                np.sum(density * mapping.matrix(matrix)),
                np.sum(mapping.density(density) * matrix),
                atol=2e-13,
            )
        finally:
            mapping.close()
        assert public._handle


def test_wb97mv_spherical_hvp_preserves_public_variational_space(
    tmp_path: Path,
) -> None:
    atoms = [("H", (0, 0, 0)), ("H", (0.15, 0.13, 1.5))]
    direction = np.array([[0.01, -0.03, 0.1], [-0.01, 0.03, -0.1]])
    options = {
        "method": "wb97m-v",
        "device": "cpu",
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "ks_options": KsOptions(
            grid=GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
        ),
    }
    cartesian = Calculator(**options)
    spherical = Calculator(**options, basis_representation="spherical")
    assert spherical.second_order_capabilities == frozenset(("hvp", "hessian"))
    expected = cartesian.hessian_vector_product(atoms, direction, cache=tmp_path)
    actual = spherical.hessian_vector_product(atoms, direction, cache=tmp_path)
    np.testing.assert_allclose(actual.value, expected.value, atol=2e-11, rtol=2e-10)
