"""Check exact-content reuse without weakening native grid source validation."""

import numpy as np
import pytest
from generativeqc._snapshot_grid_cache import SnapshotGridCache
from generativeqc_compiler.dft.grid import ExplicitGrid


def inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use unequal points/weights so one changed component invalidates reuse."""
    return (
        np.arange(12, dtype=float).reshape(4, 3),
        np.arange(1, 5, dtype=float),
        np.array([0.0, 0.0, 1.0, 1.0]),
    )


def test_identical_grid_preserves_public_identity_and_owned_storage() -> None:
    points, weights, owners = inputs()
    cache = SnapshotGridCache()
    first, reused = cache.resolve(points, weights, owners, owner=7)
    expected = ExplicitGrid(
        points,
        weights,
        tuple(map(int, owners)),
        {"source": "native-ks-snapshot-v1", "owner": 7},
    )
    assert not reused and first.identity == expected.identity
    second, reused = cache.resolve(
        points.copy(), weights.copy(), owners.copy(), owner=7
    )
    assert reused and second is first
    assert 0 < cache.retained_bytes <= cache.max_bytes
    points[:] = -1
    weights[:] = -1
    assert not np.array_equal(first.points, points)
    assert not np.array_equal(first.weights, weights)
    with pytest.raises(ValueError):
        first.points.setflags(write=True)


@pytest.mark.parametrize(
    "component", ("points", "weights", "owners", "owner", "shape", "signed-zero")
)
def test_any_source_change_replaces_the_only_entry(component: str) -> None:
    points, weights, owners = inputs()
    cache = SnapshotGridCache()
    first, _ = cache.resolve(points, weights, owners, owner=7)
    owner = 7
    if component == "points":
        points[0, 0] = np.nextafter(points[0, 0], 1.0)
    elif component == "weights":
        weights[-1] = np.nextafter(weights[-1], 0.0)
    elif component == "owners":
        owners[-1] = 0
    elif component == "owner":
        owner = 8
    elif component == "signed-zero":
        points[0, 0] = -0.0
    else:
        points, weights, owners = points[:-1], weights[:-1], owners[:-1]
    second, reused = cache.resolve(points, weights, owners, owner=owner)
    assert not reused and second is not first
    assert second.identity != first.identity
    assert cache.grid is second


def test_tight_budget_uses_uncached_exact_grid_and_clear_releases_entry() -> None:
    points, weights, owners = inputs()
    cache = SnapshotGridCache(max_bytes=0)
    first, reused = cache.resolve(points, weights, owners, owner=7)
    second, reused_again = cache.resolve(points, weights, owners, owner=7)
    assert not reused and not reused_again and first is not second
    assert first.identity == second.identity
    assert cache.grid is None and cache.retained_bytes == 0
    cache = SnapshotGridCache()
    cache.resolve(points, weights, owners, owner=7)
    cache.clear()
    assert cache.grid is None and cache.owner is None and cache.retained_bytes == 0


def test_invalid_replacement_drops_cached_entry_and_still_fails() -> None:
    points, weights, owners = inputs()
    cache = SnapshotGridCache()
    cache.resolve(points, weights, owners, owner=7)
    points[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cache.resolve(points, weights, owners, owner=7)
    assert cache.grid is None
