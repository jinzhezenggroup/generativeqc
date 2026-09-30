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
    resident_grid = SimpleNamespace(device=0, points=8192, weights=12288, point_count=6)
    source = SimpleNamespace(
        grid_spec=SimpleNamespace(coincident_tolerance=1e-12),
        cuda_resident_grid=lambda: resident_grid,
    )
    state = SimpleNamespace(
        grid=SimpleNamespace(
            points=np.arange(18, dtype=float).reshape(6, 3),
            weights=np.arange(6, dtype=float) + 0.5,
            owners=np.array([0, 0, 0, 1, 1, 1]),
        ),
        density=np.eye(2)[None, ...],
        weighted_density=np.eye(2)[None, ...],
        _source=source,
    )

    class Grid:
        device_id = 0

        @contextmanager
        def feature_task_device_points(
            self,
            pointer: int,
            point_count: int,
            ids: typing.Any,
            ingredients: tuple[str, ...],
        ) -> typing.Iterator[SimpleNamespace]:
            assert ids is None
            assert ingredients == ("rho", "gradient", "tau")
            begin = (pointer - resident_grid.points) // (3 * 8)
            assert pointer == resident_grid.points + 3 * begin * 8
            assert 0 <= begin < resident_grid.point_count
            lease = SimpleNamespace(
                alive=True,
                view=SimpleNamespace(stream=stream),
                _owner=SimpleNamespace(device_id=0),
            )
            events.append(("borrow", point_count))
            events.append(("point_pointer", begin))
            try:
                yield lease
            finally:
                lease.alive = False
                events.append(("release", point_count))

        def feature_task(
            self, *args: typing.Any, **kwargs: typing.Any
        ) -> typing.NoReturn:
            raise AssertionError("host point upload reintroduced")

        def feature_task_with_features(self, *args: typing.Any) -> typing.NoReturn:
            raise AssertionError("host feature export reintroduced")

        def xc_task(self, *args: typing.Any) -> typing.NoReturn:
            raise AssertionError("functional-name dispatch reintroduced")

    class Nonlocal:
        point_count = 6
        executed = False
        collected_points = 0
        seeds = SimpleNamespace(
            pointer=4096, stride=6, stream=seed_stream, generation=7
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
        natom = 2

        def __init__(self, label: str, base: float) -> None:
            self.label = label
            self.base = base
            self.finishes = 0

        def geometry_molecular(
            self,
            task: SimpleNamespace,
            begin: int,
            points_per_atom: int,
            weights: np.ndarray,
            raw: np.ndarray,
            *,
            functional: int,
        ) -> None:
            assert self.label == "local"
            assert task.alive and functional == 4
            assert points_per_atom == 3
            np.testing.assert_array_equal(
                weights, state.grid.weights[begin : begin + len(weights)]
            )
            events.append(("local", len(weights)))

        def geometry_external_device_molecular(
            self,
            task: SimpleNamespace,
            begin: int,
            points_per_atom: int,
            weights: np.ndarray,
            raw: np.ndarray,
            pointer: int,
            stride: int,
            seed_begin: int,
        ) -> None:
            assert self.label == "nonlocal"
            assert task.alive and points_per_atom == 3 and seed_begin == begin
            assert (pointer, stride) == (4096, 6)
            np.testing.assert_array_equal(
                weights, state.grid.weights[begin : begin + len(weights)]
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
        "raw_weights": np.ones(6),
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
        ("borrow", 2),
    ]
    assert [event for event in events if event[0] == "local"] == [
        ("local", 2),
        ("local", 2),
        ("local", 2),
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
    assert work["nonlocal_feature_d2d_bytes"] == 192
    assert work["nonlocal_feature_d2h_bytes"] == work["nonlocal_seed_h2d_bytes"] == 0
    assert work["grid_owner_source"] == "implicit-atom-major-index"
    assert work["grid_owner_h2d_bytes"] == 0
    assert work["grid_point_source"] == "exact-native-resident-grid"
    assert work["grid_point_h2d_bytes"] == 0
    assert [event[1] for event in events if event[0] == "point_pointer"] == [0, 2, 4]
    assert work["ao_collocation_point_visits"] == 6
    assert work["geometry_point_visits"] == 12
    assert work["nonlocal_dense_pair_capacity"] == 36
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
    args["nonlocal_sources"].geometry_external_device_molecular = None
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
    assert events.count(("borrow", 2)) == 6


def test_production_driver_uses_shared_pass_accumulators() -> None:
    driver = (ROOT / "python/generativeqc/_stationary_wb97mv_cuda.py").read_text()
    join = (ROOT / "python/generativeqc/_stationary_nonlocal_cuda.py").read_text()
    assert "resident_nonlocal_geometry(" in driver
    assert "nonlocal_sources=self.nonlocal_sources" in driver
    assert "_ResidentNonlocalForceOwner(" in driver
    assert "seed_from_snapshot(state._source, task)" in join
    assert join.count("with grid.feature_task_device_points(") == 1
    assert "with grid.feature_task(" not in join
    assert "except NotImplementedError:" in join
    assert join.count("nonlocal_owner.collect(") == 1
    assert "state.grid.owners" not in join
    assert "geometry_molecular(" in join
    assert "geometry_external_device_molecular(" in join
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


def fallback_fixture() -> tuple[dict[str, typing.Any], list[tuple[typing.Any, ...]]]:
    args, events = fixture()
    owner = args["nonlocal_owner"]

    def unavailable(snapshot: typing.Any, task: typing.Any) -> typing.NoReturn:
        assert snapshot is args["state"]._source and task.alive
        events.append(("resident_unavailable",))
        raise NotImplementedError("final state has no device-resident features")

    def collect(task: typing.Any, begin: int) -> None:
        assert task.alive and begin == owner.collected_points
        length = min(args["tile_points"], owner.point_count - begin)
        owner.collected_points += length
        events.append(("collect", begin))

    owner.seed_from_snapshot = unavailable
    owner.collect = collect
    return args, events


@pytest.mark.parametrize("tile_points", [1, 2, 8])
def test_missing_resident_features_preserves_bounded_device_fallback(
    tile_points: int,
) -> None:
    args, events = fallback_fixture()
    args["tile_points"] = tile_points
    components, seconds, work = MODULE.resident_nonlocal_geometry(**args)
    offsets = list(range(0, 6, tile_points))
    assert [event[1] for event in events if event[0] == "collect"] == offsets
    assert [event[1] for event in events if event[0] == "external"] == offsets
    assert sum(event[1] for event in events if event[0] == "borrow") == 12
    assert sum(event[1] for event in events if event[0] == "local") == 6
    assert events.count(("resident_unavailable",)) == 1
    assert events.index(("collect", offsets[-1])) < events.index(("pairs",))
    assert events.index(("local_finish",)) < events.index(("pairs",))
    assert events.index(("pairs",)) < events.index(("external", 0))
    assert args["sources"].finishes == args["nonlocal_sources"].finishes == 1
    assert work["ao_collocation_point_visits"] == 12
    assert work["geometry_point_visits"] == 12
    assert work["nonlocal_feature_source"] == "bounded-grid-feature-collection"
    assert work["nonlocal_feature_d2d_bytes"] == 0
    assert work["nonlocal_feature_collection_point_visits"] == 6
    assert work["nonlocal_feature_d2h_bytes"] == work["nonlocal_seed_h2d_bytes"] == 0
    assert work["grid_owner_source"] == "implicit-atom-major-index"
    assert work["grid_owner_h2d_bytes"] == 0
    assert "two_pass_geometry_and_pair_drain" in seconds
    assert "single_pass_geometry_and_pair_drain" not in seconds
    np.testing.assert_array_equal(components["xc_ao"], np.full((2, 3), 1.0))
    np.testing.assert_array_equal(components["nonlocal_ao"], np.full((2, 3), 10.0))


def test_fallback_replays_from_an_empty_collection() -> None:
    args, events = fallback_fixture()
    MODULE.resident_nonlocal_geometry(**args)
    MODULE.resident_nonlocal_geometry(**args)
    assert events.count(("nlc_reset",)) == 1
    assert events.count(("collect", 0)) == 2
    assert events.count(("pairs",)) == 2
    assert args["nonlocal_owner"].collected_points == 6


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, MemoryError])
def test_snapshot_errors_are_not_hidden_by_capability_fallback(
    error_type: type[Exception],
) -> None:
    args, events = fallback_fixture()

    def invalid(*args: typing.Any) -> typing.NoReturn:
        raise error_type("stale token, device mismatch or failed CUDA allocation")

    args["nonlocal_owner"].seed_from_snapshot = invalid
    with pytest.raises(error_type, match="stale token"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not any(event[0] in ("collect", "pairs", "external") for event in events)
    assert args["sources"].finishes == args["nonlocal_sources"].finishes == 0
    assert events[-1][0] == "release"


@pytest.mark.parametrize("fallback", [False, True])
def test_pair_failure_is_not_retried_as_a_feature_capability_miss(
    fallback: bool,
) -> None:
    args, events = fallback_fixture() if fallback else fixture()

    def invalid() -> typing.NoReturn:
        raise NotImplementedError("pair execution rejected")

    args["nonlocal_owner"].execute = invalid
    with pytest.raises(NotImplementedError, match="pair execution rejected"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not any(event[0] == "external" for event in events)
    assert args["nonlocal_sources"].finishes == 0


@pytest.mark.parametrize("field,value", [("pointer", 0), ("stream", 32)])
def test_fallback_rejects_bad_seeds_before_nonlocal_consumption(
    field: str, value: int
) -> None:
    args, events = fallback_fixture()
    setattr(args["nonlocal_owner"].seeds, field, value)
    with pytest.raises(ValueError, match="invalid seed lease|streams differ"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not any(event[0] == "external" for event in events)
    assert args["nonlocal_sources"].finishes == 0


@pytest.mark.parametrize("fallback", [False, True])
def test_both_schedules_keep_host_intervals_exclusive(
    monkeypatch: pytest.MonkeyPatch, fallback: bool
) -> None:
    args, _ = fallback_fixture() if fallback else fixture()
    ticks = iter((0.0, 1.0, 1.0, 2.0, 4.0, 4.0, 7.0, 15.0))
    monkeypatch.setattr(MODULE, "perf_counter", lambda: next(ticks))
    _, seconds, _ = MODULE.resident_nonlocal_geometry(**args)
    assert sum(seconds.values()) == pytest.approx(15.0)
    assert seconds["resident_feature_seed_enqueue"] == pytest.approx(2.0)
    assert seconds["vv10_pair_enqueue"] == pytest.approx(3.0)
    assert all(value >= 0 for value in seconds.values())


def test_capability_miss_cannot_reuse_a_partially_seeded_owner() -> None:
    args, events = fallback_fixture()
    owner = args["nonlocal_owner"]

    def partial(*args: typing.Any) -> typing.NoReturn:
        owner.collected_points = 1
        raise NotImplementedError("partially changed owner")

    owner.seed_from_snapshot = partial
    with pytest.raises(RuntimeError, match="modified the nonlocal owner"):
        MODULE.resident_nonlocal_geometry(**args)
    assert not any(event[0] == "collect" for event in events)
