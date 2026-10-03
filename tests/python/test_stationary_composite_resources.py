"""Admit the whole composite force before selecting faster grid concurrency."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.dft.plan import plan_tiles
from generativeqc_compiler.method.stationary_composite_resources import (
    CompositeStationaryCudaResources,
    plan_composite_stationary_cuda_resources,
)

TARGET = cuda_target_info("sm_120")
BASIS = SimpleNamespace(
    nao=192,
    natom=24,
    nprimitive=176,
    shells=(None,) * 96,
    numeric_bytes=16384,
    packed=SimpleNamespace(size=4096),
)


def _plan(**options: object) -> CompositeStationaryCudaResources:
    defaults = {
        "grid_points": 589_824,
        "spins": 1,
        "source_count": 7,
        "nonlocal_bytes": 141_594_644,
        "target": TARGET,
        "max_device_bytes": 1 << 30,
        "max_host_bytes": 2 << 30,
    }
    defaults.update(options)
    return plan_composite_stationary_cuda_resources(
        BASIS,
        grid_plan=lambda points: plan_tiles(
            BASIS,
            backend="cuda",
            order=2,
            tile_points=points,
            active_ao_capacity=BASIS.nao,
            budget_bytes=defaults["max_device_bytes"],
        ),
        **defaults,
    )


def test_automatic_composite_schedule_prefers_more_concurrent_points() -> None:
    selected = _plan()
    assert selected.grid.tile_points == selected.sources.geometry_lanes == 1024
    assert selected.sources.becke_threads_per_point == 32
    assert selected.device_bound <= 1 << 30
    assert selected.host_bound <= 2 << 30


@pytest.mark.parametrize("scope", ["device", "host"])
def test_tight_totals_retain_old_tile_and_explicit_requests_fail(scope: str) -> None:
    baseline = _plan(tile_points=256)
    limit = getattr(baseline, f"{scope}_bound")
    automatic = _plan(**{f"max_{scope}_bytes": limit})
    assert automatic.grid.tile_points == 256
    assert getattr(automatic, f"{scope}_bound") <= limit
    with pytest.raises(ValueError, match="no admitted"):
        _plan(tile_points=1024, **{f"max_{scope}_bytes": limit})


def test_small_grid_and_target_shared_memory_keep_bounded_fallbacks() -> None:
    small = _plan(grid_points=17)
    assert small.grid.tile_points == small.sources.geometry_lanes == 17
    constrained = replace(
        TARGET, shared_memory_per_block=4096, tuning_maximum_shared_bytes=4096
    )
    direct = _plan(target=constrained)
    assert direct.grid.tile_points == 1024
    assert direct.sources.becke_threads_per_point == 1
    assert direct.sources.becke_shared_bytes == 0


def test_full_grid_nonlocal_storage_is_not_shrunk_to_fit_a_tile() -> None:
    with pytest.raises(ValueError, match="no admitted"):
        _plan(nonlocal_bytes=2 << 30)


def test_sub_256_tiles_remain_available_under_a_smaller_total_budget() -> None:
    baseline = _plan(tile_points=128)
    selected = _plan(max_host_bytes=baseline.host_bound)
    assert selected.grid.tile_points == 128
    assert selected.host_bound == baseline.host_bound


@pytest.mark.parametrize("tile", [0, True, 4097])
def test_explicit_tile_domain_is_not_silently_reinterpreted(tile: int) -> None:
    with pytest.raises(ValueError, match="tile_points"):
        _plan(tile_points=tile)
