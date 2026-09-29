"""Exclusive host-wall accounting for the single-pass resident nonlocal join."""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "resident_nonlocal_timing", ROOT / "python/generativeqc/_stationary_nonlocal_cuda.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "seed,pair,tail", [(2.0, 3.0, 8.0), (0.0, 0.0, 0.0), (0.25, 0.5, 1.0)]
)
def test_single_pass_host_intervals_do_not_double_count_enqueues(
    monkeypatch: pytest.MonkeyPatch, seed: float, pair: float, tail: float
) -> None:
    end = 2.0 + seed + pair + tail
    ticks = iter(
        (0.0, 1.0, 1.0, 2.0, 2.0 + seed, 2.0 + seed, 2.0 + seed + pair, end)
    )
    monkeypatch.setattr(MODULE, "perf_counter", lambda: next(ticks))
    calls: list[str] = []

    @contextmanager
    def feature_task(*args: Any, **kwargs: Any) -> Iterator[SimpleNamespace]:
        assert args[1] is None and kwargs["defer_error_to_consumer"]
        calls.append("borrow")
        yield SimpleNamespace(view=SimpleNamespace(stream=31))
        calls.append("release")

    def seed_from_snapshot(*args: Any) -> None:
        calls.append("seed")

    def execute() -> SimpleNamespace:
        calls.append("pairs")
        return SimpleNamespace(pointer=4096, stride=1, stream=31, generation=1)

    parts = {
        name: np.zeros((1, 3))
        for name in ("xc_ao", "xc_grid", "xc_weight", "nuclear")
    }
    sink = SimpleNamespace(
        geometry=lambda *args, **kwargs: None,
        geometry_external_device=lambda *args: None,
        finish=lambda: parts,
    )
    state = SimpleNamespace(
        grid=SimpleNamespace(
            points=np.zeros((1, 3)), owners=np.zeros(1), weights=np.ones(1)
        ),
        _source=object(),
    )
    components, seconds, work = MODULE.resident_nonlocal_geometry(
        grid=SimpleNamespace(feature_task=feature_task),
        sources=sink,
        nonlocal_sources=sink,
        nonlocal_owner=SimpleNamespace(
            point_count=1,
            diagnostic=lambda: SimpleNamespace(executed=False, collected_points=0),
            seed_from_snapshot=seed_from_snapshot,
            execute=execute,
        ),
        state=state,
        raw_weights=np.ones(1),
        tile_points=1,
        ao_count=1,
        functional=4,
        ingredients=("rho", "gradient", "tau"),
    )
    assert seconds["resident_feature_seed_enqueue"] == pytest.approx(seed)
    assert seconds["vv10_pair_enqueue"] == pytest.approx(pair)
    assert seconds["single_pass_geometry_and_pair_drain"] == pytest.approx(1.0 + tail)
    assert sum(seconds.values()) == pytest.approx(end)
    assert all(value >= 0.0 for value in seconds.values())
    assert calls == ["borrow", "seed", "pairs", "release"]
    assert work["ao_collocation_point_visits"] == 1
    assert len(components) == 7
