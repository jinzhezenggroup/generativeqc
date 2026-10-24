"""Independent gates for compiler-generated meta-GGA and VV10 second partials."""

from fractions import Fraction

import numpy as np
import pytest
from generativeqc_compiler.dft.nonlocal_reference import (
    nonlocal_energy_reference,
    nonlocal_explicit_geometry_derivatives_reference,
    nonlocal_feature_derivatives_reference,
)
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.stationary_grid_second import (
    StationaryGridSecondOrder,
    features,
    nuclear_repulsion_hvp,
)
from generativeqc_compiler.xc.potential import (
    assemble_coefficients,
    assemble_coefficients_directional,
)


def _model(**kwargs: object) -> StationaryGridSecondOrder:
    method = resolve_method("WB97M-V")
    return StationaryGridSecondOrder(
        method.primitives[0].functional, method.primitives[-1], **kwargs
    )


def _fields() -> np.ndarray:
    generator = np.random.default_rng(81731)
    values = generator.normal(size=(9, 5)) * 0.1
    values[0] = np.linspace(0.2, 0.7, 5)
    values[4] = np.linspace(0.3, 0.8, 5)
    values[5] = (0.3, -0.1, 0.4, 0.2, 0.5)
    values[6:] *= 10
    return values


def _nonlocal_partials(values: np.ndarray) -> np.ndarray:
    spec = resolve_method("WB97M-V").primitives[-1].spec
    rho, gradient, weights, points = values[0], values[1:4].T, values[5], values[6:].T
    density_partial, sigma_partial = nonlocal_feature_derivatives_reference(
        points, weights, rho, gradient, spec
    )
    point_partial, weight_partial = nonlocal_explicit_geometry_derivatives_reference(
        points, weights, rho, gradient, spec
    )
    return np.vstack(
        (
            weights * density_partial,
            (weights[:, None] * 2 * sigma_partial[:, None] * gradient).T,
            np.zeros(len(rho)),
            weight_partial,
            point_partial.T,
        )
    )


def test_vv10_ad_partials_match_independent_real_space_oracle() -> None:
    values = _fields()
    direction = np.random.default_rng(714).normal(size=values.shape) * 0.03
    result = _model().evaluate(values, direction)
    np.testing.assert_allclose(
        result.nonlocal_correlation[0],
        _nonlocal_partials(values),
        atol=2e-16,
        rtol=3e-12,
    )
    for step in (3e-3, 1e-3, 3e-4):
        finite = (
            _nonlocal_partials(values + step * direction)
            - _nonlocal_partials(values - step * direction)
        ) / (2 * step)
        np.testing.assert_allclose(
            result.nonlocal_correlation[1], finite, atol=2e-12, rtol=2e-5
        )
    assert result.pair_evaluations == len(values.T) ** 2


def test_complete_grid_energy_and_hessian_bilinear_have_correct_factors() -> None:
    values = _fields()
    generator = np.random.default_rng(188)
    left, right = (generator.normal(size=values.shape) * 0.03 for _ in range(2))
    model = _model()
    primal = model.evaluate(values, right)
    transpose = model.evaluate(values, left)
    np.testing.assert_allclose(
        np.sum(primal.directional_coefficients * left),
        np.sum(transpose.directional_coefficients * right),
        atol=2e-13,
    )
    for step in (3e-3, 1e-3, 3e-4):
        plus, minus = (
            model.evaluate(values + sign * step * right, np.zeros_like(values))
            for sign in (1, -1)
        )
        np.testing.assert_allclose(
            (plus.energy - minus.energy) / (2 * step),
            np.sum(primal.coefficients * right),
            atol=3e-10,
            rtol=2e-7,
        )
        np.testing.assert_allclose(
            (plus.coefficients - minus.coefficients) / (2 * step),
            primal.directional_coefficients,
            atol=3e-9,
            rtol=2e-5,
        )


def test_nested_tensor_jvp_contains_density_and_both_ao_legs() -> None:
    generator = np.random.default_rng(83)
    jets, left, right, mixed = (generator.normal(size=(4, 5, 3)) for _ in range(4))
    density = generator.normal(size=(3, 3))
    delta = generator.normal(size=(3, 3))
    density, delta = density + density.T, delta + delta.T
    actual = features(
        jets, density, left=left, right=right, mixed=mixed, delta_density=delta
    )
    for step in (3e-3, 1e-3, 3e-4):
        samples = [
            features(
                jets + offset * step * right,
                density + offset * step * delta,
                left=left + offset * step * mixed,
            )
            for offset in (-2, -1, 1, 2)
        ]
        finite = (samples[0] - 8 * samples[1] + 8 * samples[2] - samples[3]) / (
            12 * step
        )
        np.testing.assert_allclose(actual, finite, atol=2e-10, rtol=2e-10)


def test_vv10_both_leg_mask_and_resource_work_caps() -> None:
    values = _fields()
    # Below the explicit molecular rho >= 1e-10 domain on both pair legs.
    values[0, 1] = 0.5e-10
    result = _model().evaluate(values, np.zeros_like(values))
    assert result.pair_evaluations == 16
    assert np.all(result.nonlocal_correlation[0][:, 1] == 0)
    with pytest.raises(ValueError, match="max_nonlocal_pairs"):
        _model(max_pair_evaluations=15).evaluate(values, np.zeros_like(values))
    with pytest.raises(MemoryError, match="grid_budget_bytes"):
        _model(budget_bytes=1).evaluate(values, np.zeros_like(values))
    with pytest.raises(MemoryError, match="grid_budget_bytes"):
        _model(budget_bytes=24 * values.nbytes).evaluate(values, np.zeros_like(values))


@pytest.mark.parametrize("density", (1e-10, 0.5e-8))
def test_vv10_retains_boundary_and_formerly_screened_tail(density: float) -> None:
    values = _fields()
    values[0, 1] = density
    result = _model().evaluate(values, np.zeros_like(values))
    assert result.pair_evaluations == 25
    assert np.any(result.nonlocal_correlation[0][:, 1] != 0)
    with pytest.raises(ValueError, match="max_nonlocal_pairs"):
        _model(max_pair_evaluations=24).evaluate(values, np.zeros_like(values))


@pytest.mark.parametrize("order", (0, 1, 2))
def test_ao_feature_execution_enforces_explicit_byte_budget(order: int) -> None:
    jets = np.ones((4, 5, 3))
    density = np.eye(3)
    options = {}
    if order >= 1:
        options["left"] = np.zeros_like(jets)
    if order == 2:
        options.update(right=jets, mixed=jets)
    with pytest.raises(ValueError, match="retained-byte budget"):
        features(jets, density, max_bytes=1, **options)


def test_grid_partials_are_detached_immutable_snapshots() -> None:
    values = _fields()
    result = _model().evaluate(values, np.zeros_like(values))
    for partial in (
        result.coefficients,
        result.directional_coefficients,
        *result.semilocal,
        *result.nonlocal_correlation,
    ):
        assert not partial.flags.writeable
        with pytest.raises(ValueError):
            partial.setflags(write=True)


def test_vv10_pair_weights_include_both_quadrature_legs_and_beta() -> None:
    values = _fields()
    model = _model()
    result = model.evaluate(values, np.zeros_like(values))
    primitive = resolve_method("WB97M-V").primitives[-1]
    energy = nonlocal_energy_reference(
        values[6:].T, values[5], values[0], values[1:4].T, primitive.spec
    )
    beta = float((3 / float(primitive.spec.b) ** 2) ** 0.75 / 32)
    pair_and_beta = result.nonlocal_correlation[0]
    reconstructed = (
        np.dot(pair_and_beta[5], values[5]) + beta * np.dot(values[0], values[5])
    ) * float(Fraction(1, 2))
    np.testing.assert_allclose(energy, reconstructed, atol=2e-16, rtol=3e-12)


def test_nuclear_hvp_is_the_ad_of_the_charge_pair_energy() -> None:
    centers = np.array([[0.1, -0.2, 0.3], [0.5, 1.2, -0.8], [-1.2, 0.8, 0.9]])
    charges = np.array([8, 1, 1])
    direction = np.random.default_rng(180).normal(size=centers.shape) * 0.1
    actual = nuclear_repulsion_hvp(centers, charges, direction)

    def gradient(positions: np.ndarray) -> np.ndarray:
        delta = positions[:, None] - positions[None, :]
        radius = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(radius, np.inf)
        return -np.einsum(
            "ij,ijx->ix", charges[:, None] * charges[None, :] / radius**3, delta
        )

    for step in (1e-3, 3e-4, 1e-4):
        finite = (
            gradient(centers + step * direction) - gradient(centers - step * direction)
        ) / (2 * step)
        np.testing.assert_allclose(actual, finite, atol=3e-8, rtol=3e-6)


def test_meta_gga_ao_matrix_jvp_keeps_both_kinetic_legs() -> None:
    generator = np.random.default_rng(773)
    jets, moving_jets = (generator.normal(size=(4, 5, 3)) for _ in range(2))
    weights, moving_weights = (generator.normal(size=5) for _ in range(2))
    coefficients, moving_coefficients = (
        {
            "rho": generator.normal(size=(2, 5)),
            "gradient": generator.normal(size=(2, 5, 3)),
            "tau": generator.normal(size=(2, 5)),
        }
        for _ in range(2)
    )
    actual = assemble_coefficients_directional(
        jets,
        moving_jets,
        coefficients,
        moving_coefficients,
        weights,
        moving_weights,
    )
    for step in (3e-3, 1e-3, 3e-4):
        samples = [
            assemble_coefficients(
                jets + offset * step * moving_jets,
                {
                    key: value + offset * step * moving_coefficients[key]
                    for key, value in coefficients.items()
                },
                weights + offset * step * moving_weights,
            )
            for offset in (-2, -1, 1, 2)
        ]
        finite = (samples[0] - 8 * samples[1] + 8 * samples[2] - samples[3]) / (
            12 * step
        )
        np.testing.assert_allclose(actual, finite, atol=3e-10, rtol=3e-10)
