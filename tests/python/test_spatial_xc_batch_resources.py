"""Bounded queue admission and explicit matrix precision contracts."""

from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc_compiler.common.resources import ResourceBudget, plan_resources
from generativeqc_compiler.dft.spatial_xc_batch import SpatialXcOptions, batch_request


def owner(active: int = 19, points: int = 32) -> SimpleNamespace:
    return SimpleNamespace(
        plan=SimpleNamespace(
            active_ao_capacity=active, nao=active, tile_points=points, order=1
        ),
        device_id=0,
    )


def test_queue_capacity_and_budget_fallback() -> None:
    request = batch_request(owner(), SpatialXcOptions())
    plan = plan_resources((request,), ResourceBudget()).require_feasible()
    # Include AO/work, features/coefficient/weight/total rows, local matrices,
    # maps, seven-size_t descriptors, six sum words and per-slot error witnesses.
    expected = 8 * (8 * (12 * 32 * 19 + 28 * 32 + 2 * 19**2 + 19) + 6) + 56 * 8 + 4 * 9
    assert plan.peak_bytes["device"] == expected
    smaller = plan_resources(
        (request,), ResourceBudget(device_bytes=expected - 1)
    ).require_feasible()
    assert 0 < smaller.peak_bytes["device"] < expected
    fallback = plan_resources(
        (request,), ResourceBudget(device_bytes=0)
    ).require_feasible()
    assert fallback.selections == (("spatial_xc_batch", "single"),)
    assert fallback.peak_bytes["device"] == 0


@pytest.mark.parametrize(
    "options,shape",
    [
        (SpatialXcOptions(batch_tiles=1), owner()),
        (SpatialXcOptions(max_batch_bytes=0), owner()),
        (SpatialXcOptions(), owner(active=1)),
        (SpatialXcOptions(), owner(points=1)),
        (SpatialXcOptions(), owner(active=1_000_000)),
    ],
)
def test_unprofitable_or_out_of_domain_shapes_keep_bounded_fallback(
    options: SpatialXcOptions, shape: SimpleNamespace
) -> None:
    request = batch_request(shape, options)
    assert len(request.candidates) == 1
    assert plan_resources((request,), ResourceBudget()).peak_bytes["device"] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_tiles": 0},
        {"batch_tiles": 33},
        {"batch_tiles": True},
        {"max_batch_bytes": -1},
        {"matrix_precision": "auto"},
    ],
)
def test_invalid_options_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SpatialXcOptions(**kwargs)
