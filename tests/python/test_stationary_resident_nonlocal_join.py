"""Device-free protocol tests for the production resident nonlocal join."""

from __future__ import annotations

import importlib.util
import typing
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "resident_nonlocal_join", ROOT / "python/generativeqc/_stationary_nonlocal_cuda.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(
    stream: int = 31, seed_stream: int = 31
) -> tuple[dict[str, typing.Any], list[tuple[typing.Any, ...]]]:
    events: list[tuple[typing.Any, ...]] = []
    source = SimpleNamespace(grid_spec=SimpleNamespace(coincident_tolerance=1e-12))
    state = SimpleNamespace(
        grid=SimpleNamespace(
            points=np.arange(15, dtype=float).reshape(5, 3),
            weights=np.arange(5, dtype=float) + 0.5,
            owners=np.array([0, 1, 0, 1, 0]),
        ),
        density=np.eye(2)[None, ...],
        weighted_density=np.eye(2)[None, ...],
        _source=source,
    )

    class Grid:
        @contextmanager
        def feature_task(
            self,
            points: np.ndarray,
            ids: typing.Any,
            ingredients: tuple[str, ...],
            *,
            defer_error_to_consumer: bool = False,
        ) -> typing.Iterator[SimpleNamespace]:
            assert ids is None
            assert ingredients == ("rho", "gradient", "tau")
            assert defer_error_to_consumer
            lease = SimpleNamespace(
                alive=True,
                view=SimpleNamespace(stream=stream),
                _owner=SimpleNamespace(device_id=0),
            )
            events.append(("borrow", len(points)))
            try:
                yield lease
            finally:
                lease.alive = False
                events.append(("release", len(points)))

        def feature_task_with_features(self, *args: typing.Any) -> typing.NoReturn:
            raise AssertionError("host feature export reintroduced")

        def xc_task(self, *args: typing.Any) -> typing.NoReturn:
            raise AssertionError("functional-name dispatch reintroduced")

    class Nonlocal:
        point_count = 5
        executed = False
        collected_points = 0
        seeds = SimpleNamespace(
            pointer=4096, stride=5, stream=seed_stream, generation=7
        )

        def diagnostic(self) -> SimpleNamespace:
            return SimpleNamespace(
                executed=self.executed, collected_points=self.collected_points
            )

        def reset(self) -> None:
            assert self.executed, "native reset rejects a never-executed owner"
            self.executed = False
            self.collected_points = 0
            events.append(("nlc_reset",))

        def collect(self, *_args: typing.Any) -> typing.NoReturn:
            raise AssertionError("tile feature collection reintroduced")

        def seed_from_snapshot(
            self, snapshot: SimpleNamespace, task: SimpleNamespace
        ) -> None:
            assert snapshot is source and task.alive
            self.collected_points = self.point_count
            events.append(("resident_seed",))

        def execute(self) -> SimpleNamespace:
            assert self.collected_points == self.point_count
            events.append(("pairs",))
            self.executed = True
            return self.seeds

    class Sources:
        def __init__(self, label: str, base: float) -> None:
            self.label = label
            self.base = base
            self.finishes = 0

        def geometry(
            self,
            task: SimpleNamespace,
            owners: np.ndarray,
            weights: np.ndarray,
            raw: np.ndarray,
            *,
            functional: int,
        ) -> None:
            assert self.label == "local"
            assert task.alive and functional == 4
            events.append(("local", len(owners)))

        def geometry_external_device(
            self,
            task: SimpleNamespace,
            owners: np.ndarray,
            weights: np.ndarray,
            raw: np.ndarray,
            pointer: int,
            stride: int,
            begin: int,
        ) -> None:
            assert self.label == "nonlocal"
            assert task.alive
            assert (pointer, stride) == (4096, 5)
            np.testing.assert_array_equal(
                weights, state.grid.weights[begin : begin + len(owners)]
            )
            events.append(("external", begin))

        def finish(self) -> dict[str, np.ndarray]:
            events.append((self.label + "_finish",))
            self.finishes += 1
            return {
                key: np.full((2, 3), self.base + index, dtype=float)
                for index, key in enumerate(
                    ("xc_ao", "xc_grid", "xc_weight", "nuclear")
                )
            }

    args = {
        "grid": Grid(),
        "sources": Sources("local", 1.0),
        "nonlocal_sources": Sources("nonlocal", 10.0),
        "nonlocal_owner": Nonlocal(),
        "state": state,
        "raw_weights": np.ones(5),
        "tile_points": 2,
        "ao_count": 2,
        "functional": 4,
        "ingredients": ("rho", "gradient", "tau"),
    }
    return args, events


def test_complete_join_collocates_each_grid_tile_once() -> None:
    args, events = fixture()
    components, seconds, work = MODULE.resident_nonlocal_geometry(**args)
    assert set(components) == {
        "xc_ao",
        "xc_grid",
        "xc_weight",
        "nuclear",
        "nonlocal_ao",
        "nonlocal_grid",
        "nonlocal_weight",
    }
    assert [event for event in events if event[0] == "borrow"] == [
        ("borrow", 2),
        ("borrow", 2),
        ("borrow", 1),
    ]
    assert [event for event in events if event[0] == "local"] == [
        ("local", 2),
        ("local", 2),
        ("local", 1),
    ]
    assert [event for event in events if event[0] == "external"] == [
        ("external", 0),
        ("external", 2),
        ("external", 4),
    ]
    assert events.index(("local", 2)) < events.index(("resident_seed",))
    assert events.index(("resident_seed",)) < events.index(("pairs",))
    assert events.index(("pairs",)) < events.index(("external", 0))
    assert work["nonlocal_feature_source"] == "exact-final-scf-device-binding"
    assert work["nonlocal_feature_d2d_bytes"] == 160
    assert work["nonlocal_feature_d2h_bytes"] == work["nonlocal_seed_h2d_bytes"] == 0
    assert work["ao_collocation_point_visits"] == 5
    assert work["geometry_point_visits"] == 10
    assert work["nonlocal_dense_pair_capacity"] == 25
    assert "nonlocal_pair_evaluations" not in work
    assert "resident_feature_seed_enqueue" in seconds
    assert "single_pass_geometry_and_pair_drain" in seconds
    np.testing.assert_array_equal(components["xc_ao"], np.full((2, 3), 1.0))
    np.testing.assert_array_equal(components["nonlocal_ao"], np.full((2, 3), 10.0))


@pytest.mark.parametrize(
    "field,value", [("pointer", 0), ("stride", 4), ("stream", 0), ("generation", 0)]
)
def test_invalid_seed_view_never_reaches_geometry(field: str, value: int) -> None:
    args, events = fixture()
    setattr(args["nonlocal_owner"].seeds, field, value)
    with pytest.raises(ValueError, match="invalid seed lease"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not any(event[0] == "external" for event in events)
    assert args["sources"].finishes == 0
    assert args["nonlocal_sources"].finishes == 0


def test_cross_stream_seed_is_rejected_before_consumption() -> None:
    args, events = fixture(seed_stream=32)
    with pytest.raises(ValueError, match="streams differ"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not any(event[0] == "external" for event in events)


def test_join_refuses_missing_consumer_dependency() -> None:
    args, events = fixture()
    args["nonlocal_sources"].geometry_external_device = None
    with pytest.raises(TypeError, match="lacks the resident"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not events


def test_join_refuses_missing_snapshot_seed_dependency() -> None:
    args, events = fixture()
    args["nonlocal_owner"].seed_from_snapshot = None
    with pytest.raises(TypeError, match="final-state feature handoff"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not events


def test_join_replays_complete_grid_after_reset() -> None:
    args, events = fixture()
    MODULE.resident_nonlocal_geometry(**args)
    MODULE.resident_nonlocal_geometry(**args)
    assert events.count(("nlc_reset",)) == 1
    assert events.count(("resident_seed",)) == 2
    assert events.count(("borrow", 2)) == 4
    assert events.count(("borrow", 1)) == 2


def test_production_driver_uses_shared_pass_accumulators() -> None:
    driver = (ROOT / "python/generativeqc/_stationary_wb97mv_cuda.py").read_text()
    join = (ROOT / "python/generativeqc/_stationary_nonlocal_cuda.py").read_text()
    assert "resident_nonlocal_geometry(" in driver
    assert "nonlocal_sources=self.nonlocal_sources" in driver
    assert "_ResidentNonlocalForceOwner(" in driver
    assert "seed_from_snapshot(state._source, task)" in join
    assert join.count("with grid.feature_task(") == 1
    assert "nonlocal_owner.collect(" not in join
    for retired in (
        "NonlocalFixedGridPlan",
        "feature_task_with_features(",
        "rho[active]",
        "seeds[:, begin:end]",
    ):
        assert retired not in driver


def test_fresh_owner_is_not_reset_before_first_seed() -> None:
    args, events = fixture()
    MODULE.resident_nonlocal_geometry(**args)
    assert ("nlc_reset",) not in events


def test_partial_previous_collection_is_not_reused() -> None:
    args, events = fixture()
    args["nonlocal_owner"].collected_points = 2
    with pytest.raises(ValueError, match="incomplete previous"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not events
