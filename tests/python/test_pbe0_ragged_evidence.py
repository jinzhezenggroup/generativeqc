"""Ragged evidence must isolate scheduling from SCF algorithm experiments."""

import json
import os
from pathlib import Path

import pytest

from benchmarks.pbe0_xc_trajectories import check_record, prepared_snapshots, selection


def test_ragged_selection_exercises_default_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both arms use normal SCF chunks; external settings cannot leak into A/B."""
    monkeypatch.setenv("GENERATIVEQC_CUDA_KS_CHUNK", "7")
    monkeypatch.setenv("GENERATIVEQC_CUDA_XC_RAGGED_BATCH", "9")
    for enabled in (False, True):
        with selection(enabled, None, "ragged"):
            assert "GENERATIVEQC_CUDA_KS_CHUNK" not in os.environ
            assert os.environ.get("GENERATIVEQC_CUDA_XC_RAGGED_BATCH") == (
                None if enabled else "0"
            )
            assert os.environ["GENERATIVEQC_CUDA_XC_COMPACT_BATCH"] == "1"
        assert os.environ["GENERATIVEQC_CUDA_KS_CHUNK"] == "7"
        assert os.environ["GENERATIVEQC_CUDA_XC_RAGGED_BATCH"] == "9"


def test_batch_receipts_decode_completed_prepared_plans(tmp_path: Path) -> None:
    """Preparation receipts are metadata, not additive iteration counters."""
    common = {"name": "cuda_xc_batch_plan", "id": 4}
    events = [
        common | {"event": "BEGIN"},
        common | {"event": "VALUE", "key": "ragged", "value": 1},
        common | {"event": "VALUE", "key": "compact_nonempty_tiles", "value": 41},
        common | {"event": "END", "status": "returned"},
    ]
    path = tmp_path / "batch.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    assert prepared_snapshots(path, "cuda_xc_batch_plan") == [
        {"ragged": 1, "compact_nonempty_tiles": 41}
    ]


def test_failed_endpoint_without_forces_cannot_pass() -> None:
    """Failure evidence remains serializable and never masquerades as E+F."""
    reference = {"energy": -1.0, "forces": [[0.0, 0.0, 0.0]], "converged": True}
    actual = {"energy": None, "forces": None, "converged": False, "status": 1}
    assert not check_record(actual, reference)["gate"]
