"""Independent fixed-Hessian physics; no electronic endpoint qualification."""

import math

import numpy as np
import pytest
from generativeqc import analyze_harmonic_modes

UNITS = {
    "hessian_units": "hartree/bohr^2",
    "coordinate_units": "bohr",
    "mass_units": "dalton",
}


def analysis(
    matrix: np.ndarray, positions: np.ndarray, masses: object, **kwargs: object
) -> object:
    return analyze_harmonic_modes(
        matrix, coordinates=positions, masses=masses, **UNITS, **kwargs
    )


def conversion() -> float:
    constants = pytest.importorskip("scipy.constants")
    return math.sqrt(
        constants.physical_constants["Hartree energy"][0]
        / constants.physical_constants["atomic mass constant"][0]
    ) / (
        constants.physical_constants["Bohr radius"][0] * 2 * math.pi * constants.c * 100
    )


def diatomic(stiffness: float = 0.2) -> tuple:
    positions = np.array(((0.03, -0.02, 0.01), (0.17, 0.11, 3.4)))
    axis = positions[1] - positions[0]
    axis /= np.linalg.norm(axis)
    bond = np.concatenate((-axis, axis))
    return positions, stiffness * np.outer(bond, bond)


def triangle() -> tuple:
    positions = np.array(((0.1, -0.1, 0.0), (1.6, 0.2, 0.3), (0.2, 1.2, -0.2)))
    matrix = np.zeros((9, 9))
    for first, second, stiffness in ((0, 1, 0.3), (1, 2, 0.5), (2, 0, 0.8)):
        axis = positions[second] - positions[first]
        axis /= np.linalg.norm(axis)
        vector = np.zeros((3, 3))
        vector[first] = -axis
        vector[second] = axis
        matrix += stiffness * np.outer(vector.ravel(), vector.ravel())
    return positions, matrix


@pytest.mark.parametrize(
    "masses", ((1.00782503223, 2.01410177812), (132.905451961, 1.00782503223))
)
def test_diatomic_reduced_mass_matches_independent_units_and_isotopes(
    masses: tuple,
    record_property: object,
) -> None:
    positions, matrix = diatomic()
    result = analysis(matrix, positions, masses)
    expected = math.sqrt(0.2 * sum(1 / mass for mass in masses)) * conversion()
    record_property(
        "independent_frequency_error_cm1",
        float(np.max(abs(result.frequencies_cm1 - expected))),
    )
    assert result.linear and result.rigid_mode_count == 5
    np.testing.assert_allclose(result.frequencies_cm1, [expected], atol=1e-7, rtol=0)
    isotope = analysis(matrix, positions, (masses[0] * 2, masses[1]))
    assert isotope.frequencies_cm1[0] < result.frequencies_cm1[0]
    mode = result.cartesian_modes.reshape(1, -1)
    np.testing.assert_allclose(
        (mode * np.repeat(masses, 3)) @ mode.T, np.eye(1), atol=1e-13
    )
    assert result.diagnostics["frequency_scale_factor"] == 1
    assert not result.diagnostics["scientific_state_verified"]
    assert not result.diagnostics["stationary_point_verified"]


def test_linear_triatomic_retains_bends_and_four_vibrational_modes() -> None:
    positions = np.array(((0, 0, -1.0), (0, 0, 0), (0, 0, 1.0)))
    matrix = np.zeros((9, 9))
    bend = np.outer([1, -2, 1], [1, -2, 1])
    matrix[np.ix_([0, 3, 6], [0, 3, 6])] = 0.2 * bend
    matrix[np.ix_([1, 4, 7], [1, 4, 7])] = 0.35 * bend
    matrix[np.ix_([2, 5, 8], [2, 5, 8])] = np.array(
        ((1, -1, 0), (-1, 2, -1), (0, -1, 1))
    )
    result = analysis(matrix, positions, (1, 1, 1))
    assert result.linear and result.rigid_mode_count == 5
    np.testing.assert_allclose(
        result.frequencies_cm1,
        np.sqrt([1, 1.2, 2.1, 3]) * conversion(),
        atol=1e-7,
        rtol=0,
    )


def test_nonlinear_saddle_matches_independent_generalized_eigensystem(
    record_property: object,
) -> None:
    linalg = pytest.importorskip("scipy.linalg")
    positions, matrix = triangle()
    masses = np.array((1.00782503223, 12.0, 132.905451961))
    mass_matrix = np.diag(np.repeat(masses, 3))
    values, vectors = linalg.eigh(matrix, mass_matrix)
    selected = np.flatnonzero(values > 1e-8)[0]
    reaction = vectors[:, selected]
    mass_reaction = mass_matrix @ reaction
    saddle = matrix - 2 * values[selected] * np.outer(mass_reaction, mass_reaction)
    result = analysis(saddle, positions, masses)
    reference_values, _ = linalg.eigh(saddle, mass_matrix)
    reference = reference_values[abs(reference_values) > 1e-8]
    expected = np.sign(reference) * np.sqrt(abs(reference)) * conversion()
    assert not result.linear and result.rigid_mode_count == 6
    assert result.negative_mode_count == result.significant_imaginary_mode_count == 1
    np.testing.assert_allclose(result.frequencies_cm1, expected, atol=1e-7, rtol=0)
    overlap = abs(result.cartesian_modes[0].ravel() @ mass_matrix @ reaction)
    record_property(
        "independent_frequency_error_cm1",
        float(np.max(abs(result.frequencies_cm1 - expected))),
    )
    record_property("independent_reaction_mode_overlap", float(overlap))
    assert overlap > 1 - 1e-12
    modes = result.cartesian_modes.reshape(3, 9)
    np.testing.assert_allclose(modes @ mass_matrix @ modes.T, np.eye(3), atol=1e-13)
    assert result.diagnostics["raw_translation_residual"] < 1e-14
    assert result.diagnostics["raw_rotation_residual"] < 1e-13


def test_frequency_spectrum_and_mode_subspace_are_rigid_covariant() -> None:
    positions, matrix = triangle()
    masses = (1.0, 12.0, 14.0)
    result = analysis(matrix, positions, masses)
    angle = 0.73
    rotation = np.array(
        (
            (math.cos(angle), -math.sin(angle), 0),
            (math.sin(angle), math.cos(angle), 0),
            (0, 0, 1),
        )
    )
    transform = np.kron(np.eye(3), rotation)
    transformed = transform @ matrix @ transform.T
    moved = analysis(
        transformed, positions @ rotation.T + [4.0, -2.0, 0.7], masses, symmetrize=True
    )
    np.testing.assert_allclose(
        result.frequencies_cm1, moved.frequencies_cm1, atol=1e-7, rtol=0
    )
    original_modes = result.mass_weighted_modes.reshape(3, 9).T
    moved_modes = moved.mass_weighted_modes.reshape(3, 9).T
    np.testing.assert_allclose(
        moved_modes @ moved_modes.T,
        transform @ original_modes @ original_modes.T @ transform.T,
        atol=2e-14,
    )


def test_soft_negative_modes_are_not_deleted_or_rescaled() -> None:
    positions, matrix = diatomic(-1e-12)
    result = analysis(matrix, positions, (1, 2))
    assert (
        result.negative_mode_count == 1 and result.significant_imaginary_mode_count == 0
    )
    assert -1 < result.frequencies_cm1[0] < 0
    expected = -math.sqrt(1e-12 * 1.5) * conversion()
    np.testing.assert_allclose(result.frequencies_cm1, [expected], atol=1e-12, rtol=0)


def test_physical_cs_mass_is_not_replaced_with_ecp_ionic_charge() -> None:
    positions, matrix = diatomic()
    physical = analysis(matrix, positions, [132.905451961, 1.00782503223])
    wrong_core_count_mass = analysis(matrix, positions, [9, 1.00782503223])
    assert (
        abs(physical.frequencies_cm1[0] - wrong_core_count_mass.frequencies_cm1[0]) > 50
    )
    assert physical.masses_dalton[0] == 132.905451961


def test_monatomic_and_nearly_linear_conventions_are_explicit() -> None:
    single = analysis(np.zeros((3, 3)), np.zeros((1, 3)), [132.905451961])
    assert single.frequencies_cm1.size == 0 and single.rigid_mode_count == 3
    for deviation, linear, modes in ((1e-10, True, 4), (1e-4, False, 3)):
        positions = np.array(((0, 0, -1.0), (deviation, 0, 0), (0, 0, 1.0)))
        result = analysis(np.zeros((9, 9)), positions, [1, 1, 1])
        assert result.linear is linear and len(result.frequencies_cm1) == modes


def test_integrity_is_checked_before_optional_symmetry_cleanup() -> None:
    positions, matrix = diatomic()
    matrix[0, 1] += 1e-8
    with pytest.raises(ValueError, match="explicit symmetrize"):
        analysis(matrix, positions, [1, 2])
    result = analysis(matrix, positions, [1, 2], symmetrize=True)
    assert result.diagnostics["raw_asymmetry"] == pytest.approx(1e-8)
    assert result.diagnostics["symmetrized"]
    assert result.diagnostics["raw_translation_residual"] > 0
    matrix[0, 1] += 1e-3
    with pytest.raises(ValueError, match="raw_symmetry_tolerance"):
        analysis(matrix, positions, [1, 2], symmetrize=True)


@pytest.mark.parametrize(
    "invalid",
    ([0, 1], [-1, 1], [float("nan"), 1], [True, False], [1], [[1, 2]], [1 + 1j, 2]),
)
def test_invalid_mass_input_is_rejected(invalid: object) -> None:
    positions, matrix = diatomic()
    with pytest.raises(ValueError):
        analysis(matrix, positions, invalid)


def test_unit_dimension_and_admission_errors_do_not_return_partial_results() -> None:
    positions, matrix = diatomic()
    for key, wrong in (
        ("mass_units", "electron-mass"),
        ("hessian_units", "hartree/angstrom^2"),
        ("coordinate_units", "angstrom"),
    ):
        with pytest.raises(ValueError, match="required units"):
            analyze_harmonic_modes(
                matrix, coordinates=positions, masses=[1, 2], **{**UNITS, key: wrong}
            )
    with pytest.raises(ValueError, match="complete"):
        analysis(matrix[:, :-1], positions, [1, 2])
    with pytest.raises(ValueError, match="maximum_coordinates"):
        analysis(matrix, positions, [1, 2], maximum_coordinates=3)
    with pytest.raises(ValueError, match="coincident"):
        analysis(matrix, np.zeros((2, 3)), [1, 2])
    with pytest.raises(ValueError, match="finite real"):
        analysis(matrix.astype(complex), positions, [1, 2])


def test_receipt_is_immutable_and_visualization_preserves_input_order() -> None:
    positions, matrix = diatomic()
    result = analysis(matrix, positions, [132.905451961, 1.00782503223])
    receipt = result.input_identity
    positions[0] += 1
    matrix[:] = 0
    assert result.input_identity == receipt and np.any(result.frequencies_cm1)
    for array in (
        result.frequencies_cm1,
        result.cartesian_modes,
        result.coordinates_bohr,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    lines = result.xyz_animation(["Cs", "H"], 0, frames=4).splitlines()
    assert (
        len(lines) == 16
        and lines[0] == "2"
        and lines[2].startswith("Cs ")
        and lines[3].startswith("H ")
    )
    with pytest.raises(ValueError, match="symbols"):
        result.xyz_animation(["Cs\nH", "H"], 0)
    changed = analysis(matrix, positions, [132.905451961, 1.00782503223])
    assert changed.input_identity != receipt
