"""CPU-only contract tests for the cross-functional DFT force matrix runner."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import GridSpec

from benchmarks.dft_force_matrix import (
    QUALIFICATION_SYSTEMS,
    _atoms,
    _changed_atoms,
    _method_configuration,
)


def test_cam_b3lyp_matrix_case_uses_method_ir_range_exchange_not_name_dispatch() -> (
    None
):
    selector, options = _method_configuration(
        "cam-b3lyp-rks",
        GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8),
    )

    assert selector == "pbe-rks"
    assert options.composition is not None
    assert tuple(
        primitive.operator
        for primitive in options.composition.primitives
        if hasattr(primitive, "operator")
    ) == ("short-range", "long-range")


def test_matrix_systems_include_water_scaling_and_nonwater_holdout() -> None:
    assert len(_atoms("water-3")) == 3
    assert len(_atoms("water-6")) == 6
    assert len(_atoms("water-12")) == 12
    formaldehyde = _atoms("formaldehyde")
    assert [symbol for symbol, _ in formaldehyde] == ["C", "O", "H", "H"]


def test_changed_geometry_moves_only_last_atom_by_declared_vector() -> None:
    atoms = _atoms("formaldehyde")
    changed, coordinates = _changed_atoms(atoms)

    original = np.asarray([position for _, position in atoms])
    expected = original.copy()
    expected[-1] += np.asarray((0.0010, -0.0005, 0.0003))
    np.testing.assert_allclose(coordinates, expected, atol=0, rtol=0)
    np.testing.assert_allclose(
        np.asarray([position for _, position in changed]),
        expected,
        atol=0,
        rtol=0,
    )
    np.testing.assert_array_equal(coordinates[:-1], original[:-1])


def test_unknown_matrix_method_and_system_fail_closed() -> None:
    grid = GridSpec(radial_points=8, angular_polar=4, angular_azimuth=8)
    with pytest.raises(ValueError, match="unknown benchmark method"):
        _method_configuration("unknown-rks", grid)
    with pytest.raises(ValueError, match="unknown benchmark system"):
        _atoms("unknown-system")


def test_qualification_system_set_covers_scaling_and_holdout() -> None:
    assert QUALIFICATION_SYSTEMS == (
        "water-3",
        "water-6",
        "water-12",
        "formaldehyde",
    )
