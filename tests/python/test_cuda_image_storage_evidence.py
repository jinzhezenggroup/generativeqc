"""Replay CUDA storage evidence and reject incomplete package-load witnesses."""

import json
import lzma
import runpy
from pathlib import Path

import pytest

BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/results/cuda-image-storage-20261028"
)
REDUCE = runpy.run_path(str(BUNDLE / "reduce.py"))["reduce"]


def receipt() -> dict:
    """Read a fresh receipt because replay decodes its samples in place."""
    with lzma.open(BUNDLE / "receipts.json.xz", "rt") as handle:
        return json.load(handle)


def test_complete_package_witnesses_replay_independent_of_row_order() -> None:
    data = receipt()
    for arm in ("baseline", "final", "default"):
        data["records"][f"remote/package-load-{arm}.json"]["rows"].reverse()
    result = REDUCE(data)
    assert result["campaigns"]["mainline-default"]["failures"] == []


@pytest.mark.parametrize("arm", ["baseline", "final", "default"])
def test_duplicate_package_witness_cannot_replace_missing_libraries(arm: str) -> None:
    data = receipt()
    rows = data["records"][f"remote/package-load-{arm}.json"]["rows"]
    rows[:] = [rows[0]] * len(rows)
    with pytest.raises(AssertionError):
        REDUCE(data)
