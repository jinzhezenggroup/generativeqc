"""Pruned quadrature, native parity and geometry-response acceptance gates."""

from __future__ import annotations

import ctypes
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.dft.grid import (
    GridPolicy,
    GridSpec,
    MolecularGrid,
    grid_policy_provenance,
)
from generativeqc_compiler.xc.grid_response import grid_response_tiles

ROOT = Path(__file__).resolve().parents[2]
CENTERS = np.array([[0.1, -0.2, 0.3], [1.7, 0.2, -0.4]])
MOTION = np.array([[0.2, -0.1, 0.3], [-0.1, 0.2, -0.15]])


def specification(radial: int = 24, polar: int = 8, azimuth: int = 16) -> GridSpec:
    """Use distinct explicit radii so point and measure scaling are observable."""
    return GridSpec(
        radial, polar, azimuth, ((1, 0.8), (8, 1.4)), version=3, pruning="radial-v1"
    )


def molecular(spec: GridSpec, centers: np.ndarray = CENTERS) -> MolecularGrid:
    return MolecularGrid(tuple(zip((1, 8), centers, strict=True)), spec)


def test_versioned_opt_in_identity_and_exact_work() -> None:
    base = GridPolicy(pruning="none").resolve("pbe-rks")
    pruned = replace(base, version=3, pruning="radial-v1")
    assert GridSpec(**asdict(pruned)) == pruned
    assert 0 < pruned.points_per_atom < base.points_per_atom
    assert max(polar * azimuth for polar, azimuth in pruned.angular_shells) == 16 * 32
    assert grid_policy_provenance(pruned) == {
        "policy_version": 3,
        "contract": "explicit-grid-v3",
        "pruning": "radial-v1",
    }
    atoms = ((1, (0, 0, 0)), (8, (0, 0, 1.4)))
    assert MolecularGrid(atoms, base).identity != MolecularGrid(atoms, pruned).identity
    assert base.version == 2 and base.pruning == "none"
    for version in (1, 2):
        unpruned = replace(base, version=version)
        assert unpruned.points_per_atom == 54 * 16 * 32
        with pytest.raises(ValueError, match="version 3"):
            replace(unpruned, pruning="radial-v1")
    with pytest.raises(ValueError, match="sourced radius"):
        MolecularGrid(atoms, replace(pruned, element_radii=((1, 0.8),)))


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"))
def test_qualified_pruned_default_retains_explicit_unpruned_policy(method: str) -> None:
    """The new default changes identity/work without rewriting legacy v2 grids."""
    default = GridPolicy().resolve(method)
    previous = GridPolicy(pruning="none").resolve(method)
    assert default.version == 3 and default.pruning == "radial-v1"
    assert default.radial_points == 64
    assert previous.version == 2 and previous.pruning == "none"
    assert previous.radial_points == 54
    assert default.points_per_atom < previous.points_per_atom
    assert default.element_radii == previous.element_radii
    assert grid_policy_provenance(default) == GridPolicy().provenance
    assert grid_policy_provenance(previous) == GridPolicy(pruning="none").provenance
    assert default != previous


def test_benchmark_automatic_pruning_respects_qualified_method_boundaries() -> None:
    """Both engines consume the selected grid; other methods remain opt-in."""
    from benchmarks.readme_wb97mv import benchmark_grid_spec

    atoms = (("O", (0, 0, 0)), ("H", (0, 0, 1.4)))
    for method in ("lda", "pbe"):
        assert benchmark_grid_spec(None, atoms, method=method) == GridPolicy().resolve(
            method
        )
        assert benchmark_grid_spec(None, atoms, "none", method=method) == GridPolicy(
            pruning="none"
        ).resolve(method)
    for method in ("pbe0", "b3lyp", "r2scan", "m06-2x", "wb97m-v"):
        assert benchmark_grid_spec(None, atoms, method=method) == GridSpec()
    explicit = benchmark_grid_spec((48, 16, 32), atoms, "radial-v1", method="pbe")
    assert explicit.radial_points == 48 and explicit.pruning == "radial-v1"
    assert explicit.element_radii == ((1, 1.0), (8, 1.0))


def test_pruned_point_count_is_a_shape_only_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resource admission must not materialize Legendre rules or molecular grids."""

    def unavailable(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            "quadrature construction reached during point-count planning"
        )

    monkeypatch.setattr(np.polynomial.legendre, "leggauss", unavailable)
    spec = specification(47, 15, 31)
    assert spec.points_per_atom == sum(
        polar * azimuth for polar, azimuth in spec.angular_shells
    )


@pytest.mark.parametrize("polar,azimuth", [(1, 3), (3, 5), (7, 23), (128, 512)])
def test_shells_are_complete_positive_angular_rules(polar: int, azimuth: int) -> None:
    spec = specification(polar=polar, azimuth=azimuth)
    assert all(
        1 <= count <= polar and 3 <= ring <= azimuth
        for count, ring in spec.angular_shells
    )
    assert spec.points_per_atom == sum(
        count * ring for count, ring in spec.angular_shells
    )
    for count, ring in set(spec.angular_shells):
        _nodes, weights = np.polynomial.legendre.leggauss(count)
        assert weights.sum() * ring * (2 * np.pi / ring) == pytest.approx(
            4 * np.pi, abs=2e-14
        )
    for changes in (
        {"angular_polar": 129},
        {"angular_azimuth": 513},
        {"pruning": "none"},
    ):
        with pytest.raises(ValueError):
            replace(spec, **changes)


@pytest.mark.parametrize("tile", [1, 37, 1024, 100_000])
def test_tiles_cross_shell_and_owner_boundaries_without_padding(tile: int) -> None:
    grid = molecular(specification())
    expected = grid.explicit()
    tiles = list(grid.tiles(tile))
    np.testing.assert_array_equal(
        np.concatenate([item.points for item in tiles]), expected.points
    )
    np.testing.assert_array_equal(
        np.concatenate([item.weights for item in tiles]), expected.weights
    )
    assert tuple(owner for item in tiles for owner in item.owners) == expected.owners
    assert len(expected.weights) == 2 * grid.spec.points_per_atom
    assert np.bincount(expected.owners).tolist() == [grid.spec.points_per_atom] * 2
    for storage in (grid.shell_offsets, grid.shell_direction_offsets):
        with pytest.raises(ValueError):
            storage.setflags(write=True)


@pytest.mark.parametrize("alpha", [0.01, 1.0, 100.0])
def test_pruned_gaussian_volume_and_second_moments_have_independent_exact_values(
    alpha: float,
) -> None:
    radius = 1 / np.sqrt(alpha)
    spec = replace(specification(96, 16, 32), element_radii=((1, radius),))
    grid = MolecularGrid(((1, (0, 0, 0)),), spec)
    value = 0.0
    moments = np.zeros(3)
    for tile in grid.tiles(257):
        density = np.exp(-alpha * np.sum(tile.points**2, axis=1))
        value += tile.weights @ density
        moments += (tile.weights * density) @ (tile.points**2)
    np.testing.assert_allclose(value, (np.pi / alpha) ** 1.5, rtol=3e-12)
    np.testing.assert_allclose(
        moments, (np.pi / alpha) ** 1.5 / (2 * alpha), rtol=3e-12
    )


@pytest.fixture(scope="module")
def native_grid(tmp_path_factory: pytest.TempPathFactory, native_cxx: object) -> object:
    directory = tmp_path_factory.mktemp("pruned-native-grid")
    library = directory / "grid.so"
    native_cxx.build_shared(
        (ROOT / "src/dft/grid.cpp", ROOT / "tests/native/grid_pruning_probe.cpp"),
        library,
        compile_args=(
            "-std=c++20",
            "-O2",
            "-I",
            str(ROOT / "src"),
            "-I",
            str(ROOT / "include"),
        ),
    )
    function = ctypes.CDLL(str(library)).grid_pruning_probe
    function.restype = ctypes.c_size_t
    function.argtypes = [
        ctypes.c_uint,
        *([ctypes.c_size_t] * 3),
        *([ctypes.c_void_p] * 6),
    ]
    return function


@pytest.mark.parametrize("shape", [(1, 1, 3), (5, 7, 23), (24, 8, 16), (48, 16, 32)])
@pytest.mark.parametrize("version", [1, 2, 3])
def test_native_cpu_points_weights_and_atomic_measures_match_python(
    native_grid: object, shape: tuple[int, int, int], version: int
) -> None:
    spec = specification(*shape)
    if version != 3:
        spec = replace(spec, version=version, pruning="none")
    grid = molecular(spec)
    explicit = grid.explicit()
    points = np.empty_like(explicit.points)
    weights, raw = np.empty(grid.npoint), np.empty(grid.npoint)
    owners, gradient = np.empty(grid.npoint, dtype=np.uint32), np.empty((2, 3))
    outputs = (CENTERS, points, weights, raw, owners, gradient)
    count = native_grid(version, *shape, *(value.ctypes.data for value in outputs))
    assert count == grid.npoint
    np.testing.assert_allclose(points, explicit.points, atol=5e-10, rtol=4e-13)
    np.testing.assert_array_equal(owners, explicit.owners)
    expected_raw = np.concatenate([tile.weights for tile in grid._raw_tiles(199)])
    np.testing.assert_allclose(raw, expected_raw, atol=1e-8, rtol=3e-12)
    # Far radial measures amplify tiny independent Legendre/Becke roundoff.
    # Compare ownership in its natural dimensionless units, not huge raw tails.
    np.testing.assert_allclose(
        weights / raw, explicit.weights / expected_raw, atol=3e-13, rtol=3e-10
    )
    derivative = sum(
        tile.weight_motion @ np.exp(-0.2 * np.sum(tile.grid.points**2, axis=1))
        for tile in grid_response_tiles(grid, MOTION, tile_points=199)
    )
    assert np.sum(gradient * MOTION) == pytest.approx(derivative, abs=3e-11)
    np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=3e-12)


def test_pruned_moving_grid_derivative_matches_multistep_finite_differences() -> None:
    spec = specification()
    grid = molecular(spec)
    derivative = 0.0
    for tile in grid_response_tiles(grid, MOTION, tile_points=193):
        density = np.exp(-0.2 * np.sum(tile.grid.points**2, axis=1))
        derivative += tile.weight_motion @ density
        derivative += (tile.grid.weights * density) @ (
            -0.4 * np.sum(tile.grid.points * tile.point_motion, axis=1)
        )

    def integral(centers: np.ndarray) -> float:
        return sum(
            tile.weights @ np.exp(-0.2 * np.sum(tile.points**2, axis=1))
            for tile in molecular(spec, centers).tiles(211)
        )

    errors = []
    # Start in the truncation-dominated regime. At 1e-5 the energy difference
    # is already roundoff-limited on some NumPy/BLAS builds, so a further
    # error-ratio requirement can reject an accurate analytic derivative.
    for step in (1e-2, 1e-3, 1e-4):
        finite_difference = (
            integral(CENTERS + step * MOTION) - integral(CENTERS - step * MOTION)
        ) / (2 * step)
        errors.append(abs(finite_difference - derivative))
    assert errors[-1] < 2e-8
    assert errors[-1] < errors[0] / 30
