"""Device-free protocol tests for the production resident nonlocal join."""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import typing

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "resident_nonlocal_join", ROOT / "python/vibeqc/_stationary_nonlocal_cuda.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(
    stream: int = 31, seed_stream: int = 31
) -> tuple[dict[str, typing.Any], list[tuple[typing.Any, ...]]]:
    events: list[tuple[typing.Any, ...]] = []
    state = SimpleNamespace(
        grid=SimpleNamespace(
            points=np.arange(15, dtype=float).reshape(5, 3),
            weights=np.arange(5, dtype=float) + 0.5,
            owners=np.array([0, 1, 0, 1, 0]),
        ),
        density=np.eye(2)[None, ...],
        weighted_density=np.eye(2)[None, ...],
        _source=SimpleNamespace(grid_spec=SimpleNamespace(coincident_tolerance=1e-12)),
    )

    class Grid:
        @contextmanager
        def feature_task(
            self,
            points: np.ndarray,
            ids: np.ndarray,
            ingredients: tuple[str, ...],
        ) -> typing.Iterator[SimpleNamespace]:
            assert tuple(ids) == (0, 1)
            assert ingredients == ("rho", "gradient", "tau")
            lease = SimpleNamespace(alive=True, view=SimpleNamespace(stream=stream))
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

        def collect(self, task: SimpleNamespace, offset: int) -> None:
            assert task.alive
            events.append(("collect", offset))

        def execute(self) -> SimpleNamespace:
            events.append(("pairs",))
            self.executed = True
            return self.seeds

    class Sources:
        finishes = 0

        def geometry(
            self,
            task: SimpleNamespace,
            owners: np.ndarray,
            weights: np.ndarray,
            raw: np.ndarray,
            *,
            functional: int,
        ) -> None:
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
            assert task.alive
            assert (pointer, stride) == (4096, 5)
            np.testing.assert_array_equal(
                weights, state.grid.weights[begin : begin + len(owners)]
            )
            events.append(("external", begin))

        def reset(self, *args: typing.Any) -> None:
            events.append(("source_reset",))

        def finish(self) -> dict[str, np.ndarray]:
            events.append(("finish",))
            self.finishes += 1
            return {
                key: np.full((2, 3), self.finishes + index, dtype=float)
                for index, key in enumerate(
                    ("xc_ao", "xc_grid", "xc_weight", "nuclear")
                )
            }

    args = {
        "grid": Grid(),
        "sources": Sources(),
        "nonlocal_owner": Nonlocal(),
        "state": state,
        "raw_weights": np.ones(5),
        "tile_points": 2,
        "ao_count": 2,
        "functional": 4,
        "ingredients": ("rho", "gradient", "tau"),
    }
    return args, events


def test_complete_join_uses_leases_and_resets_before_pair_enqueue() -> None:
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
    assert [event for event in events if event[0] == "collect"] == [
        ("collect", 0),
        ("collect", 2),
        ("collect", 4),
    ]
    assert [event for event in events if event[0] == "external"] == [
        ("external", 0),
        ("external", 2),
        ("external", 4),
    ]
    assert events.index(("source_reset",)) < events.index(("pairs",))
    assert events[-1] == ("finish",)
    assert work["nonlocal_feature_d2h_bytes"] == work["nonlocal_seed_h2d_bytes"] == 0
    assert work["ao_collocation_point_visits"] == 10
    assert work["nonlocal_dense_pair_capacity"] == 25
    assert "nonlocal_pair_evaluations" not in work  # A bound is not a measured count.
    assert "vv10_pair_enqueue" in seconds and "vv10_pairs" not in seconds
    np.testing.assert_array_equal(components["nonlocal_ao"], np.full((2, 3), 2.0))


@pytest.mark.parametrize(
    "field,value", [("pointer", 0), ("stride", 4), ("stream", 0), ("generation", 0)]
)
def test_invalid_seed_view_never_reaches_geometry(field: str, value: int) -> None:
    args, events = fixture()
    setattr(args["nonlocal_owner"].seeds, field, value)
    with pytest.raises(ValueError, match="invalid seed lease"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not any(event[0] == "external" for event in events)
    assert args["sources"].finishes == 1


def test_cross_stream_seed_is_rejected_before_consumption() -> None:
    args, events = fixture(seed_stream=32)
    with pytest.raises(ValueError, match="streams differ"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not any(event[0] == "external" for event in events)


def test_join_refuses_missing_consumer_dependency() -> None:
    args, events = fixture()
    args["sources"].geometry_external_device = None
    with pytest.raises(TypeError, match="lacks the resident"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not events


def test_join_replays_complete_grid_after_reset() -> None:
    args, events = fixture()
    MODULE.resident_nonlocal_geometry(**args)
    MODULE.resident_nonlocal_geometry(**args)
    assert events.count(("nlc_reset",)) == 1
    assert events.count(("collect", 0)) == 2
    assert events.count(("finish",)) == 4


def test_production_driver_uses_resident_join_not_host_seed_staging() -> None:
    driver = (ROOT / "python/vibeqc/_stationary_wb97mv_cuda.py").read_text()
    assert "resident_nonlocal_geometry(" in driver
    assert "_ResidentNonlocalForceOwner(" in driver
    for retired in (
        "NonlocalFixedGridPlan",
        "feature_task_with_features(",
        "rho[active]",
        "seeds[:, begin:end]",
    ):
        assert retired not in driver


def test_fresh_owner_is_not_reset_before_first_collection() -> None:
    args, events = fixture()
    MODULE.resident_nonlocal_geometry(**args)
    assert ("nlc_reset",) not in events


def test_partial_previous_collection_is_not_reused() -> None:
    args, events = fixture()
    args["nonlocal_owner"].collected_points = 2
    with pytest.raises(ValueError, match="incomplete previous"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not events
