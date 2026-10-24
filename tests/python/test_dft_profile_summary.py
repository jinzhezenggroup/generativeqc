"""Accounting gates for exclusive host and overlapping device intervals."""

import pytest

from benchmarks.summarize_dft_profile import (
    exclusive_device_intervals,
    exclusive_host_intervals,
)


def test_host_nesting_and_crossing_refusal() -> None:
    rows = [
        {"name": "endpoint", "start": 0, "end": 20},
        {"name": "scf", "start": 2, "end": 10},
        {"name": "xc", "start": 3, "end": 6},
        {"name": "force", "start": 11, "end": 18},
    ]
    assert exclusive_host_intervals(rows) == {
        "endpoint": 5e-9,
        "scf": 5e-9,
        "xc": 3e-9,
        "force": 7e-9,
    }
    rows[-1]["start"] = 9
    with pytest.raises(ValueError, match="crossing"):
        exclusive_host_intervals(rows)


def test_device_overlap_and_window_clipping_are_not_double_counted() -> None:
    rows = [
        {"phase": "scf", "start": -2, "end": 4},
        {"phase": "force", "start": 2, "end": 7},
        {"phase": "force", "start": 6, "end": 8},
        {"phase": "xc", "start": 10, "end": 30},
    ]
    result = exclusive_device_intervals(rows, 0, 12)
    assert result == {
        "scf": 2e-9,
        "force": 3e-9,
        "xc": 2e-9,
        "overlapping_kernels": 3e-9,
        "no_kernel_activity": 2e-9,
    }
    assert sum(result.values()) == pytest.approx(12e-9)
