"""Independent quadrature, native value/derivative and default-selection gates."""

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import Atom
from generativeqc.ks import KsOptions, resolve_ks_options
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.dft.grid import (
    GridPolicy,
    GridSpec,
    MolecularGrid,
)
from generativeqc_compiler.dft.treutler import (
    angular_count,
    angular_rule,
    atomic_point_count,
    default_counts,
    radial_rule,
)
from generativeqc_compiler.xc.grid_native import NativeGridContraction
from generativeqc_compiler.xc.grid_response import (
    partition_mixed_response,
    partition_response,
)
from test_grid_pruning import CENTERS, MOTION
from test_grid_pruning import native_grid as native_grid  # noqa: PLC0414 -- pytest fixture export


def canonical(array: Any) -> np.ndarray:
    """Independent generators may choose different symmetry-orbit orderings."""
    array = np.asarray(array)
    return array[np.lexsort(array[:, :3].T)]


@pytest.mark.parametrize(
    "count", [6, 14, 26, 38, 50, 74, 86, 110, 194, 302, 434, 1454, 5810]
)
def test_lebedev_matches_independent_reference_and_moments(count: int) -> None:
    lebedev = pytest.importorskip("pyscf.dft.LebedevGrid")
    actual = np.asarray(angular_rule(count))
    np.testing.assert_allclose(
        canonical(actual), canonical(lebedev.MakeAngularGrid(count)), atol=2e-16, rtol=0
    )
    np.testing.assert_allclose(actual[:, :3] ** 2 @ np.ones(3), 1, atol=3e-16)
    np.testing.assert_allclose(actual[:, 3] @ actual[:, :3], 0, atol=3e-17)
    np.testing.assert_allclose(actual[:, 3] @ actual[:, :3] ** 2, 1 / 3, atol=3e-15)
    assert actual[:, 3].sum() == pytest.approx(1, abs=3e-15)
    if count == 74:
        assert np.any(actual[:, 3] < 0)  # Legitimate signed Lebedev measure.


@pytest.mark.parametrize("z", [1, 8, 18, 36, 54, 86, 103])
@pytest.mark.parametrize("level", [0, 3, 9])
def test_radial_shells_and_nwchem_pruning_match_pyscf(z: int, level: int) -> None:
    radi = pytest.importorskip("pyscf.dft.radi")
    gen_grid = pytest.importorskip("pyscf.dft.gen_grid")
    nr, maximum = default_counts(z, level)
    assert nr == gen_grid._default_rad(z, level)
    assert maximum == gen_grid._default_ang(z, level)
    radii, measures = np.asarray(radial_rule(z, nr)).T
    expected_r, expected_dr = radi.treutler_ahlrichs(nr, z)
    np.testing.assert_allclose(radii, expected_r, rtol=3e-15, atol=1e-16)
    np.testing.assert_allclose(measures, expected_dr, rtol=3e-15, atol=1e-16)
    counts = [angular_count(z, r, maximum) for r in radii]
    np.testing.assert_array_equal(counts, gen_grid.nwchem_prune(z, radii, maximum))
    assert atomic_point_count(z, level) == sum(counts)


def test_molecular_default_matches_independent_pyscf_grid() -> None:
    gto = pytest.importorskip("pyscf.gto")
    gen_grid = pytest.importorskip("pyscf.dft.gen_grid")
    from scipy.spatial import cKDTree

    atoms = ((8, (0, 0, 0)), (1, (0, 1.4, 1.1)), (1, (0, -1.4, 1.1)))
    mol = gto.M(atom=atoms, basis="sto-3g", unit="Bohr", verbose=0)
    reference = gen_grid.Grids(mol)
    reference.level = 3
    reference.build()
    grid = MolecularGrid(atoms, GridSpec.treutler())
    actual = grid.explicit()
    distances, indices = cKDTree(reference.coords).query(actual.points)
    assert grid.npoint == 33698
    np.testing.assert_allclose(distances, 0, atol=3e-14)
    np.testing.assert_allclose(
        actual.weights, reference.weights[indices], atol=3e-12, rtol=2e-10
    )
    assert len(np.unique(indices)) == grid.npoint
    assert (
        np.count_nonzero(reference.weights) <= grid.npoint
    )  # Only zero padding is omitted.
    for values in (grid.atom_offsets, grid.centers, *grid.atomic_rules.values()):
        with pytest.raises(ValueError):
            values.setflags(write=True)


@pytest.mark.parametrize("level", [0, 3])
def test_native_cpu_values_raw_measure_and_partition_force(
    native_grid: Any, level: int
) -> None:
    spec = GridSpec.treutler(level)
    grid = MolecularGrid(tuple(zip((1, 8), CENTERS, strict=True)), spec)
    explicit = grid.explicit()
    points = np.empty_like(explicit.points)
    weights, raw = np.empty(grid.npoint), np.empty(grid.npoint)
    owners, gradient = np.empty(grid.npoint, dtype=np.uint32), np.empty((2, 3))
    count = native_grid(
        4,
        level,
        0,
        0,
        *(v.ctypes.data for v in (CENTERS, points, weights, raw, owners, gradient)),
    )
    assert count == grid.npoint
    np.testing.assert_allclose(points, explicit.points, atol=5e-13, rtol=5e-14)
    expected_raw = np.concatenate([tile.weights for tile in grid._raw_tiles(199)])
    np.testing.assert_array_equal(owners, explicit.owners)
    np.testing.assert_allclose(raw, expected_raw, rtol=2e-13, atol=1e-12)
    np.testing.assert_allclose(
        weights / raw, explicit.weights / expected_raw, atol=4e-14, rtol=2e-10
    )
    radii = spec.partition_radii(grid.atoms)
    seeds = np.exp(-0.2 * np.sum(points**2, axis=1)) * raw
    derivative = partition_response(
        points,
        CENTERS,
        point_motion=MOTION[owners],
        center_motion=MOTION,
        atomic_radii=radii,
    )
    expected = seeds @ derivative.directional[np.arange(count), owners]
    assert np.sum(gradient * MOTION) == pytest.approx(expected, abs=2e-11)


def test_generated_native_adjusted_first_and_mixed_response(tmp_path: Path) -> None:
    centers = np.array([[0.0, 0.0, 0.0], [1.6, 0.1, -0.2], [-0.3, 1.7, 0.4]])
    points = np.array([[0.3, 0.5, 0.1], [0.9, -0.5, 0.2], [-0.7, 0.9, 0.3]])
    owners = np.array([0, 1, 2], dtype=np.int64)
    left = np.array([[0.2, -0.1, 0.3], [-0.1, 0.2, -0.15], [0.1, 0.3, -0.2]])
    right = left[::-1].copy()
    radii = np.array([0.66, 1.14, 2.55])
    seeds = np.array([0.3, -0.2, 0.7])
    executor = NativeGridContraction(
        compiler=CppCompilerAdapter(Path("c++")), cache=tmp_path
    )
    gradient = executor.contract(points, centers, owners, seeds, atomic_radii=radii)
    first = partition_response(
        points,
        centers,
        point_motion=left[owners],
        center_motion=left,
        atomic_radii=radii,
    )
    selected = (np.arange(3), owners)
    assert np.sum(gradient * left) == pytest.approx(
        seeds @ first.directional[selected], abs=2e-14
    )
    np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=3e-15)
    mixed = partition_mixed_response(
        points,
        centers,
        left_point_motion=left[owners],
        left_center_motion=left,
        right_point_motion=right[owners],
        right_center_motion=right,
        atomic_radii=radii,
    )
    errors = []
    for step in (1e-3, 1e-4, 1e-5):
        responses = [
            partition_response(
                points + sign * step * right[owners],
                centers + sign * step * right,
                point_motion=left[owners],
                center_motion=left,
                atomic_radii=radii,
            ).directional
            for sign in (1, -1)
        ]
        errors.append(
            np.max(np.abs((responses[0] - responses[1]) / (2 * step) - mixed.mixed))
        )
    assert errors[-1] < 3e-10 and errors[-1] < errors[0] / 100
    for invalid in ((0.0, 1.0, 2.0), (1.0, float("nan"), 2.0)):
        with pytest.raises(ValueError):
            executor.contract(points, centers, owners, seeds, atomic_radii=invalid)


@pytest.mark.parametrize(
    "method",
    [
        "lda-rks",
        "pbe-rks",
        "pbe0-rks",
        "b3lyp-rks",
        "r2scan-rks",
        "m06-2x-rks",
        "wb97m-v-rks",
        "pbe-uks",
    ],
)
def test_all_supported_families_default_to_level_three(method: str) -> None:
    options = resolve_ks_options(method)
    assert options.grid == GridSpec.treutler(3)
    assert options.nonlocal_grid is None  # VV10 inherits the identical level-3 grid.
    assert resolve_ks_options(
        method, KsOptions(grid_accuracy="tight")
    ).grid == GridSpec.treutler(4)
    explicit = GridSpec(8, 4, 8)
    assert resolve_ks_options(method, KsOptions(grid=explicit)).grid == explicit


def test_level_encoding_element_bounds_and_metadata_only_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from generativeqc_compiler.dft import treutler

    spec = GridSpec.treutler()
    assert GridSpec(**asdict(spec)) == spec
    assert spec.level == 3 and GridSpec().level is None
    assert GridPolicy().resolve("pbe", derivative_order=1) == spec
    monkeypatch.setattr(
        treutler,
        "atomic_rule",
        lambda *args: pytest.fail("point planning expanded a grid"),
    )
    atoms = (Atom(1, (0.0, 0.0, 0.0)), Atom(8, (0.0, 0.0, 1.4)))
    assert spec.atom_offsets(atoms) == (0, 9808, 23890)
    assert spec.point_count((1, 8)) == 23890
    for level in (-1, 10, True, 3.5):
        with pytest.raises(ValueError):
            GridSpec.treutler(level)
    for z in (0, 104, 118, True):
        with pytest.raises(ValueError):
            spec.point_count((z,))
    with pytest.raises(ValueError, match="element"):
        _ = spec.points_per_atom
    for invalid in (8, False, 0.0):
        with pytest.raises(ValueError):
            replace(spec, angular_polar=invalid)
        with pytest.raises(ValueError):
            replace(spec, angular_azimuth=invalid)
