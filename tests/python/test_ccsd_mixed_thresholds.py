"""The threshold runner preserves endpoint controls and absent force results."""

from __future__ import annotations

import pytest

from benchmarks.ccsd_mixed_thresholds import controls, maximum_force_error


def test_threshold_scan_changes_only_the_warmup_policy() -> None:
    strict = controls(True, None)
    mixed = controls(True, 1e-4)
    assert strict[:27] == mixed[:27]
    assert strict[27:] == ["0", "0", "1e-05"]
    assert mixed[27:] == ["1", "0", "0.0001"]
    assert controls(False, 1e-4)[2] == "0"
    assert controls(True, None, auto=True)[27:] == ["auto", "0"]


def test_force_error_uses_flat_public_components() -> None:
    actual = {"forces_requested": 1, "forces": [0.1, -0.2, 0.3]}
    expected = {"forces_requested": 1, "forces": [0.1, -0.21, 0.3]}
    assert maximum_force_error(actual, expected) == pytest.approx(0.01)
    assert maximum_force_error({"forces_requested": 0}, {}) is None
    with pytest.raises(ValueError):
        maximum_force_error(actual, {"forces": [0.1]})
