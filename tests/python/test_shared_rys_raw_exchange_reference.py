"""Independent positive raw-K semantics, not combined RHF Fock cancellation."""

from pathlib import Path

import numpy as np
import pytest

from tools.generativeqc_validation.shared_rys_xy_fixture import (
    prepare_fixture,
    raw_exchange_reference,
)


def test_positive_raw_exchange_contracts_all_eight_unique_symmetries() -> None:
    """Four distinct AOs expose every transpose and the exchange sign."""
    density = np.arange(16, dtype=float).reshape(4, 4)
    density = density + density.T + 1.0
    expected = np.zeros((4, 4))
    for first, third, second, fourth in (
        (0, 2, 1, 3),
        (1, 2, 0, 3),
        (0, 3, 1, 2),
        (1, 3, 0, 2),
        (2, 0, 3, 1),
        (3, 0, 2, 1),
        (2, 1, 3, 0),
        (3, 1, 2, 0),
    ):
        expected[first, third] = 2.5 * density[second, fourth]
    result = raw_exchange_reference(np.full((1, 1, 1, 1), 2.5), density, (0, 1, 2, 3))
    np.testing.assert_array_equal(result, expected)


def test_coincident_orbit_does_not_count_duplicate_permutations() -> None:
    result = raw_exchange_reference(
        np.full((1, 1, 1, 1), 3.0), np.array([[2.0]]), (0, 0, 0, 0)
    )
    np.testing.assert_array_equal(result, [[6.0]])


def test_invalid_consumer_fails_before_writing_sources(tmp_path: Path) -> None:
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match="consumer"):
        prepare_fixture(destination, consumer="combined-k")
    assert not destination.exists()
