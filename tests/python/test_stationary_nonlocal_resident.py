"""Device-free sequencing tests for the real resident composition adapter."""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "resident_geometry_under_test",
    ROOT / "python/vibeqc/_stationary_nonlocal_resident.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Grid:
    def __init__(self, events):
        self.events = events
        self.stream = 19
        self.live = False
        self.visits = 0

    @contextmanager
    def feature_task(self, points, ao_ids, ingredients):
        assert not self.live
        assert ingredients == ("rho", "gradient", "tau")
        self.live = True
        self.visits += len(points)
        task = SimpleNamespace(
            view=SimpleNamespace(stream=self.stream, npoint=len(points))
        )
        self.events.append(("lease", len(points)))
        try:
            yield task
        finally:
            self.live = False
            self.events.append(("release", len(points)))


class Nonlocal:
    def __init__(self, events, grid, count):
        self.events, self.grid, self.point_count = events, grid, count
        self.executed, self.collected_points, self.generation = False, 0, 0
        self.seed_stream = 19

    def diagnostic(self):
        return SimpleNamespace(
            executed=self.executed,
            collected_points=self.collected_points,
            generation=self.generation,
            device_bytes=29 * 8 * self.point_count + 12,
        )

    def reset(self):
        assert self.executed
        self.events.append(("nonlocal_reset",))
        self.executed, self.collected_points = False, 0

    def collect(self, task, begin):
        assert self.grid.live and not self.executed
        assert begin == self.collected_points
        self.events.append(("collect", begin))
        self.collected_points += task.view.npoint

    def execute(self):
        assert self.collected_points == self.point_count and not self.executed
        self.executed = True
        self.generation += 1
        self.events.append(("pair_enqueue",))
        return SimpleNamespace(
            pointer=0x1234,
            stride=self.point_count,
            stream=self.seed_stream,
            generation=self.generation,
        )


class Sources:
    def __init__(self, events, grid):
        self.events, self.grid = events, grid
        self.finishes = 0
        self.external_weights = []

    def geometry(self, task, owners, weights, raw, *, functional):
        assert self.grid.live and functional == 4
        self.events.append(("local", len(weights)))

    def finish(self):
        self.finishes += 1
        self.events.append(("drain", self.finishes))
        return {
            name: np.ones((2, 3)) * self.finishes
            for name in ("xc_ao", "xc_grid", "xc_weight", "nuclear")
        }

    def reset(self, tolerance, density, weighted):
        assert self.finishes >= 1
        self.events.append(("source_reset",))

    def geometry_external_device(
        self, task, owners, weights, raw, pointer, stride, begin
    ):
        assert self.grid.live and pointer == 0x1234 and stride == 5
        self.events.append(("external", begin))
        self.external_weights.extend(weights)


def fixture():
    events = []
    grid = Grid(events)
    owner = Nonlocal(events, grid, 5)
    sources = Sources(events, grid)
    args = dict(
        points=np.zeros((5, 3)),
        weights=np.arange(1, 6, dtype=float),
        owners=np.array([0, 0, 1, 1, 1]),
        atomic_weights=np.ones(5),
        ao_ids=np.arange(3, dtype=np.uintp),
        tile_points=2,
        density=np.eye(3)[None],
        weighted_density=np.eye(3)[None],
        coincident_tolerance=1e-12,
        functional=4,
    )
    return events, grid, sources, owner, args


def test_production_join_uses_one_owner_and_only_resident_leases():
    events, grid, sources, owner, args = fixture()
    parts, seconds, work = MODULE.resident_nonlocal_geometry(
        grid, sources, owner, **args
    )
    assert grid.visits == 10
    assert [x for x in events if x[0] == "collect"] == [
        ("collect", 0),
        ("collect", 2),
        ("collect", 4),
    ]
    assert [x for x in events if x[0] == "external"] == [
        ("external", 0),
        ("external", 2),
        ("external", 4),
    ]
    assert events.index(("drain", 1)) < events.index(("pair_enqueue",))
    assert events.index(("pair_enqueue",)) < events.index(("source_reset",))
    assert np.array_equal(parts["nuclear"], np.ones((2, 3)))
    assert np.array_equal(parts["nonlocal_ao"], np.full((2, 3), 2.0))
    assert sources.external_weights == [1, 2, 3, 4, 5]
    assert work["nonlocal_feature_d2h_bytes"] == work["nonlocal_seed_h2d_bytes"] == 0
    assert work["nonlocal_pair_evaluations"] is None
    assert work["nonlocal_pair_evaluations_upper_bound"] == 25
    assert "vv10_enqueue" in seconds and "vv10_pairs" not in seconds
    MODULE.resident_nonlocal_geometry(grid, sources, owner, **args)
    assert owner.generation == 2 and ("nonlocal_reset",) in events


def test_wrong_stream_fails_before_any_external_seed_use():
    events, grid, sources, owner, args = fixture()
    owner.seed_stream = 20
    with pytest.raises(ValueError, match="stream"):
        MODULE.resident_nonlocal_geometry(grid, sources, owner, **args)
    assert not any(x[0] == "external" for x in events)


def test_partial_previous_owner_is_not_silently_reused():
    _, grid, sources, owner, args = fixture()
    owner.collected_points = 2
    with pytest.raises(ValueError, match="incomplete"):
        MODULE.resident_nonlocal_geometry(grid, sources, owner, **args)


@pytest.mark.parametrize("tile", [0, -1, True, 1.5])
def test_invalid_tile_rejected_before_work(tile):
    events, grid, sources, owner, args = fixture()
    args["tile_points"] = tile
    with pytest.raises(ValueError, match="tile_points"):
        MODULE.resident_nonlocal_geometry(grid, sources, owner, **args)
    assert not events


def test_missing_consumer_is_explicit():
    _, grid, _, owner, args = fixture()
    with pytest.raises(RuntimeError, match="device-seed consumer"):
        MODULE.resident_nonlocal_geometry(grid, object(), owner, **args)
