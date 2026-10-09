"""Device-free finite inventory admission; no occupancy or molecule whitelist."""

import typing

import pytest
from generativeqc_compiler.dft.ao_map_plan import ExactAoMapResources


@pytest.mark.parametrize(
    "aos,points,tile", [(1, 1, 1), (33, 15, 7), (768, 2359296, 512)]
)
def test_exact_map_reserves_only_its_actual_numeric_owner(
    aos: int, points: int, tile: int
) -> None:
    plan = ExactAoMapResources(aos, points, tile)
    tiles = 1 + (points - 1) // tile
    expected = 4 * tiles * ((aos + 31) // 32) + 16 * (tiles + 1) + 8 * aos
    assert plan.numeric_peak_bound_bytes == expected
    assert plan.admitted_bytes(expected - 1) == 0
    assert plan.admitted_bytes(expected) == expected
    assert plan.admitted_bytes(64 << 20) == expected


@pytest.mark.parametrize("dimension", ["aos", "points", "tile_points"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, (1 << 63)])
def test_invalid_exact_map_dimensions_fail_closed(
    dimension: str, value: typing.Any
) -> None:
    arguments = {"aos": 7, "points": 15, "tile_points": 7}
    arguments[dimension] = value
    with pytest.raises(ValueError):
        ExactAoMapResources(**arguments)


def test_exact_map_byte_overflow_fails_before_allocation() -> None:
    with pytest.raises(ValueError):
        ExactAoMapResources((1 << 63) - 1, 1, 1)
