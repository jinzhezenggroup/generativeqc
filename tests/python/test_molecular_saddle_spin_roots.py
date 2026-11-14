"""Occupied-space continuity controls, not molecular TS qualification."""

from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import Atom
from generativeqc._molecular_saddle import (
    _occupied_blocks,
    _occupied_overlaps,
    _registered_root_atoms,
)


def reference(spin: str, occupations: list[list[float]]) -> SimpleNamespace:
    """Make independent orthonormal alpha/beta gauges with explicit populations."""
    generator = np.random.default_rng(2848)
    return SimpleNamespace(
        identity=SimpleNamespace(spin=spin),
        coefficients=np.stack(
            [np.linalg.qr(generator.normal(size=(6, 6)))[0] for _ in occupations]
        ),
        occupations=np.asarray(occupations, dtype=float),
    )


@pytest.mark.parametrize(
    "spin,occupations,counts",
    (
        ("unpolarized", [[2, 2, 0, 0, 0, 0]], (2,)),
        ("polarized", [[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], (3, 2)),
        ("polarized", [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], (1, 0)),
        ("polarized", [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]], (0, 1)),
    ),
)
def test_occupied_spaces_own_their_spin_storage(
    spin: str, occupations: list[list[float]], counts: tuple[int, ...]
) -> None:
    state = reference(spin, occupations)
    blocks = _occupied_blocks(state, spin)
    assert tuple(block.shape[1] for block in blocks) == counts
    assert all(not np.shares_memory(block, state.coefficients) for block in blocks)
    state.coefficients[:] = 0
    for block in blocks:
        np.testing.assert_allclose(block.T @ block, np.eye(block.shape[1]), atol=1e-14)


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_principal_overlaps_are_invariant_to_independent_occupied_gauges(
    spin: str,
) -> None:
    occupations = (
        [[2, 2, 0, 0, 0, 0]]
        if spin == "unpolarized"
        else [[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]]
    )
    blocks = _occupied_blocks(reference(spin, occupations), spin)
    generator = np.random.default_rng(100)
    for repetition in range(20):
        rotated = tuple(
            block
            @ np.linalg.qr(generator.normal(size=(block.shape[1], block.shape[1])))[0]
            for block in blocks
        )
        np.testing.assert_allclose(
            _occupied_overlaps(rotated, blocks, np.eye(6), 0.9),
            np.ones(len(blocks)),
            atol=1e-14,
            rtol=0,
        )


@pytest.mark.parametrize("changed_spin", (0, 1))
def test_one_continuous_spin_cannot_mask_the_other_root_change(
    changed_spin: int,
) -> None:
    state = reference("polarized", [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]])
    blocks = _occupied_blocks(state, "polarized")
    changed = list(blocks)
    changed[changed_spin] = state.coefficients[changed_spin, :, 2:4]
    with pytest.raises(ValueError, match="root continuity"):
        _occupied_overlaps(tuple(changed), blocks, np.eye(6), 0.9)


def test_population_changes_fail_instead_of_using_rectangular_svd() -> None:
    state = reference("polarized", [[1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
    blocks = _occupied_blocks(state, "polarized")
    with pytest.raises(ValueError, match="populations changed"):
        _occupied_overlaps((blocks[0][:, :1], blocks[1]), blocks, np.eye(6), 0.9)
    empty = (blocks[0], blocks[1][:, :0])
    np.testing.assert_allclose(
        _occupied_overlaps(empty, empty, np.eye(6), 0.9), [1, 1], atol=1e-14
    )
    with pytest.raises(ValueError, match="populations changed"):
        _occupied_overlaps(blocks, empty, np.eye(6), 0.9)
    with pytest.raises(ValueError, match="populations changed"):
        _occupied_overlaps(blocks, blocks[:1], np.eye(6), 0.9)


def test_smallest_principal_overlap_cannot_be_hidden_by_an_unchanged_orbital() -> None:
    state = reference("polarized", [[1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
    blocks = _occupied_blocks(state, "polarized")
    changed = (state.coefficients[0][:, [0, 2]], blocks[1])
    with pytest.raises(ValueError, match="root continuity"):
        _occupied_overlaps(changed, blocks, np.eye(6), 0.9)


def test_cross_geometry_metric_is_used_for_every_spin() -> None:
    state = reference("polarized", [[1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
    blocks = _occupied_blocks(state, "polarized")
    metric = np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    current = tuple(np.linalg.solve(metric.T, block) for block in blocks)
    np.testing.assert_allclose(
        _occupied_overlaps(current, blocks, metric, 0.9), [1, 1], atol=1e-14
    )
    with pytest.raises(ValueError, match="root continuity"):
        _occupied_overlaps(current, blocks, np.eye(6), 0.9)


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_compact_core_translation_is_not_a_root_change(spin: str) -> None:
    """An analytic normalized Gaussian exposes laboratory-frame attenuation."""
    previous = (Atom.from_value(("Ni", (0.0, 0.0, 0.0))),)
    current = (Atom.from_value(("Ni", (0.04, -0.01, 0.0))),)
    registered, translation = _registered_root_atoms(
        current, previous, np.array([57.93534241])
    )
    populations = [[2.0]] if spin == "unpolarized" else [[1.0], [1.0]]
    state = SimpleNamespace(
        identity=SimpleNamespace(spin=spin),
        coefficients=np.ones((len(populations), 1, 1)),
        occupations=np.asarray(populations),
    )
    blocks = _occupied_blocks(state, spin)
    exponent = 600.0
    laboratory_overlap = np.exp(-0.5 * exponent * np.dot(translation, translation))
    with pytest.raises(ValueError, match="root continuity"):
        _occupied_overlaps(blocks, blocks, np.array([[laboratory_overlap]]), 0.9)
    displacement = np.asarray(registered[0].position) - previous[0].position
    registered_overlap = np.exp(-0.5 * exponent * np.dot(displacement, displacement))
    np.testing.assert_allclose(
        _occupied_overlaps(blocks, blocks, np.array([[registered_overlap]]), 0.9),
        np.ones(len(populations)),
        atol=1e-14,
    )


def test_root_registration_preserves_internal_motion_and_production_geometry() -> None:
    previous = (
        Atom.from_value(("Ni", (0.0, 0.0, 0.0))),
        Atom.from_value(("O", (3.2, 0.0, 0.0))),
        Atom.from_value(("H", (0.3, 2.5, 0.0))),
    )
    masses = np.array([57.93534241, 15.99491461957, 1.00782503223])
    initial = np.array([atom.position for atom in previous])
    moved = initial + [[0.04, 0.0, 0.0], [0.02, -0.04, 0.0], [0.0, 0.05, 0.0]]
    current = tuple(
        Atom(atom.atomic_number, tuple(center))
        for atom, center in zip(previous, moved, strict=True)
    )
    registered, translation = _registered_root_atoms(current, previous, masses)
    registered_positions = np.array([atom.position for atom in registered])
    np.testing.assert_allclose(
        np.average(registered_positions - initial, axis=0, weights=masses),
        np.zeros(3),
        atol=1e-15,
    )
    np.testing.assert_allclose(registered_positions + translation, moved, atol=1e-15)
    np.testing.assert_allclose(
        registered_positions[:, None] - registered_positions[None, :],
        moved[:, None] - moved[None, :],
        atol=1e-15,
    )
    np.testing.assert_array_equal([atom.position for atom in current], moved)
    np.testing.assert_array_equal([atom.position for atom in previous], initial)
    shifted = tuple(
        Atom(atom.atomic_number, tuple(np.asarray(atom.position) + [4.0, -3.0, 2.0]))
        for atom in current
    )
    shifted_registered, _translation = _registered_root_atoms(shifted, previous, masses)
    np.testing.assert_allclose(
        [atom.position for atom in shifted_registered], registered_positions, atol=1e-15
    )
    scaled, _translation = _registered_root_atoms(current, previous, masses * 1e300)
    np.testing.assert_array_equal(
        [atom.position for atom in scaled], registered_positions
    )


def test_root_registration_keeps_finite_grid_rotation_orientation() -> None:
    previous = (Atom(1, (-1.0, 0.0, 0.0)), Atom(1, (1.0, 0.0, 0.0)))
    rotated = (Atom(1, (0.0, -1.0, 0.0)), Atom(1, (0.0, 1.0, 0.0)))
    registered, translation = _registered_root_atoms(rotated, previous, np.ones(2))
    np.testing.assert_array_equal(translation, np.zeros(3))
    assert registered == rotated
    assert registered != previous


@pytest.mark.parametrize(
    "invalid",
    ("fractional", "nonfinite", "empty", "identity", "spin_axis", "shape", "spin"),
)
def test_invalid_occupied_states_fail_closed(invalid: str) -> None:
    state = reference("polarized", [[1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
    spin = "polarized"
    if invalid == "fractional":
        state.occupations[1, 0] = 0.5
    elif invalid == "nonfinite":
        state.coefficients[1, 0, 0] = np.nan
    elif invalid == "empty":
        state.occupations[:] = 0
    elif invalid == "identity":
        state.identity.spin = "unpolarized"
    elif invalid == "spin_axis":
        state.coefficients = state.coefficients[:1]
        state.occupations = state.occupations[:1]
    elif invalid == "shape":
        state.occupations = state.occupations[:, :3]
    elif invalid == "spin":
        spin = "invalid"
    with pytest.raises(ValueError, match="occup"):
        _occupied_blocks(state, spin)
