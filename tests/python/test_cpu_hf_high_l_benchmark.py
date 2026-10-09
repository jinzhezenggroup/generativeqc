"""Source-work receipts must count the same canonical domain as CPU ERIs."""

from itertools import combinations_with_replacement
from types import SimpleNamespace

import pytest

from benchmarks.cpu_hf_high_l import quartet_census


@pytest.mark.parametrize(
    "angular", ((0,), (2,), (3,), (0, 1, 2, 3), (3, 2, 1, 0), (0, 4), (0, 3, 4))
)
def test_census_matches_independent_ao_pair_enumeration(
    angular: tuple[int, ...],
) -> None:
    """Include mixed/high-l fallbacks, contracted shells and reversed shell order."""
    shells = tuple(
        SimpleNamespace(angular_momentum=order, primitives=tuple(range(index % 2 + 1)))
        for index, order in enumerate(angular)
    )
    ao_shells = [
        index
        for index, order in enumerate(angular)
        for _ in range((order + 1) * (order + 2) // 2)
    ]
    pairs = tuple(combinations_with_replacement(range(len(ao_shells)), 2))
    expected = {
        "primitive_components": 0,
        "spd_primitive_components": 0,
        "f_primitive_components": 0,
        "fallback_primitive_components": 0,
    }
    for bra, ket in combinations_with_replacement(pairs, 2):
        selected = tuple(shells[ao_shells[ao]] for ao in (*bra, *ket))
        primitives = 1
        for shell in selected:
            primitives *= len(shell.primitives)
        expected["primitive_components"] += primitives
        role = (
            "spd_primitive_components"
            if all(shell.angular_momentum <= 2 for shell in selected)
            else "fallback_primitive_components"
        )
        expected[role] += primitives
        if max(shell.angular_momentum for shell in selected) == 3:
            expected["f_primitive_components"] += primitives
    actual = quartet_census(shells)
    assert {key: actual[key] for key in expected} == expected
    shell_pairs = len(shells) * (len(shells) + 1) // 2
    assert actual["shell_quartets"] == shell_pairs * (shell_pairs + 1) // 2
    expected_shared = {"spd_shared_geometries": 0, "f_shared_geometries": 0}
    pairs = tuple(combinations_with_replacement(range(len(shells)), 2))
    for bra, ket in combinations_with_replacement(pairs, 2):
        selected = tuple(shells[index] for index in (*bra, *ket))
        maximum = max(shell.angular_momentum for shell in selected)
        if maximum > 3:
            continue
        primitives = 1
        for shell in selected:
            primitives *= len(shell.primitives)
        role = "spd_shared_geometries" if maximum <= 2 else "f_shared_geometries"
        expected_shared[role] += primitives
    assert {key: actual[key] for key in expected_shared} == expected_shared
    if angular == (2,):
        assert actual["spd_shared_geometries"] == 1
        assert actual["spd_primitive_components"] == 231
    if angular == (3,):
        assert actual["f_shared_geometries"] == 1
        assert actual["f_primitive_components"] == 1540


def test_empty_census_has_no_work() -> None:
    assert all(value == 0 for value in quartet_census(()).values())
