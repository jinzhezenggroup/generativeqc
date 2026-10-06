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


def _plan(
    basis: SimpleNamespace = BASIS, **options: object
) -> CompositeStationaryCudaResources:
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
        basis,
        grid_plan=lambda points: plan_tiles(
            basis,
            backend="cuda",
            order=2,
            tile_points=points,
            active_ao_capacity=basis.nao,
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


@pytest.mark.parametrize("spins", [1, 2])
def test_primitive_reservation_charges_both_composite_owners(spins: int) -> None:
    """Opt-in cannot displace full-grid nonlocal storage or point concurrency."""
    basis = SimpleNamespace(
        **{**vars(BASIS), "natom": 48, "nao": 384, "nprimitive": 352}
    )
    baseline = _plan(basis, tile_points=256, spins=spins)
    primitive = _plan(basis, tile_points=256, spins=spins, becke_primitive=True)
    assert baseline.sources.becke_primitive is False
    assert primitive.sources.becke_primitive is True
    assert primitive.sources.phased_becke_bytes > 0
    assert primitive.grid == baseline.grid
    assert primitive.sources.geometry_lanes == baseline.sources.geometry_lanes
    assert (
        primitive.device_bound - baseline.device_bound
        == 2 * primitive.sources.phased_becke_bytes
    )
    bounded = _plan(
        basis,
        tile_points=256,
        spins=spins,
        becke_primitive=True,
        max_device_bytes=primitive.device_bound - 1,
    )
    assert bounded.sources.becke_primitive is False
    assert bounded.sources.phased_becke_bytes == 0
    assert bounded.sources.geometry_lanes == baseline.sources.geometry_lanes


def test_full_tzvpd_96_requires_explicit_complete_capacity() -> None:
    """Metadata from the unmodified offline H/O snapshot, not padded SVP.

    The nonlocal size is the native dry query for 2359296 points and a 256
    pair tile. Reducing the AO tile cannot evade the complete live inventory.
    These are capacity bounds, not measured peaks or performance estimates.
    """
    basis = SimpleNamespace(
        nao=1856,
        natom=96,
        nprimitive=1184,
        shells=(None,) * 768,
        numeric_bytes=564224,
        packed=SimpleNamespace(size=32352),
    )
    options = {"grid_points": 2_359_296, "nonlocal_bytes": 434_257_940}
    with pytest.raises(ValueError, match="no admitted"):
        _plan(basis, **options)
    selected = _plan(basis, **options, max_device_bytes=4 << 30, max_host_bytes=4 << 30)
    assert selected.grid.tile_points == 1024
    assert selected.native_bytes == 885_850_112
    assert 3 << 30 < selected.device_bound <= 4 << 30
    assert 3 << 30 < selected.host_bound <= 4 << 30
    # Full-grid storage and the native reserve remain charged at one point.
    smallest = _plan(
        basis,
        **options,
        tile_points=1,
        max_device_bytes=4 << 30,
        max_host_bytes=4 << 30,
    )
    for scope in ("device", "host"):
        # The optional center-pair cache can be dropped before rejecting a
        # device allowance. Test the mandatory floor, not a cache preference.
        optional = (
            2 * smallest.sources.center_geometry_bytes if scope == "device" else 0
        )
        with pytest.raises(ValueError, match="no admitted"):
            _plan(
                basis,
                **options,
                **{
                    "max_device_bytes": 4 << 30,
                    "max_host_bytes": 4 << 30,
                    f"max_{scope}_bytes": getattr(smallest, f"{scope}_bound")
                    - optional
                    - 1,
                },
            )


def test_sub_256_tiles_remain_available_under_a_smaller_total_budget() -> None:
    baseline = _plan(tile_points=128)
    selected = _plan(max_host_bytes=baseline.host_bound)
    assert selected.grid.tile_points == 128
    assert selected.host_bound == baseline.host_bound


@pytest.mark.parametrize("tile", [0, True, 4097])
def test_explicit_tile_domain_is_not_silently_reinterpreted(tile: int) -> None:
    with pytest.raises(ValueError, match="tile_points"):
        _plan(tile_points=tile)
