"""Allocated source-sharing lifecycle checks and complete independent method gates."""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def test_shared_source_authentication_capacity_and_lifetime() -> None:
    """A compiled consumer retains its source through the last physical pullback."""
    executable = os.environ.get("GENERATIVEQC_DF_SHARED_SOURCE_PROBE")
    if not executable:
        pytest.skip("requires the compiled source-sharing probe and Slurm GPU")
    assert os.environ.get("SLURM_JOB_ID")
    result = subprocess.run(
        [executable], capture_output=True, text=True, check=True, timeout=180
    )
    receipt = json.loads(result.stdout)
    assert receipt["lifecycle"] and receipt["mismatch_fallbacks"] == 3
    assert receipt["maximum_factor_error"] < 1e-11


@pytest.mark.parametrize("shared", [False, True])
def test_shared_source_complete_mp2_independent_oracle(
    shared: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Measure actual setup elimination while retaining independent E+F acceptance."""
    if os.environ.get("GENERATIVEQC_FULL_DF_CUDA_TEST") != "1":
        pytest.skip("requires allocated full-DF CUDA qualification")
    from test_full_df_cbs_native import (
        WATER,
        test_full_df_complete_energy_force_and_companions,
    )

    if shared:
        monkeypatch.delenv("GENERATIVEQC_DF_SHARED_SOURCE", raising=False)
    else:
        monkeypatch.setenv("GENERATIVEQC_DF_SHARED_SOURCE", "0")
    trace = tmp_path / "progress.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(trace))
    test_full_df_complete_energy_force_and_companions(
        WATER, "cc-pvdz", "cc-pvdz-jkfit", "mp2", False
    )
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    counts = [row for row in records if row.get("key") == "source_constructions"]
    assert counts
    assert all(row["value"] == int(not shared) for row in counts)
