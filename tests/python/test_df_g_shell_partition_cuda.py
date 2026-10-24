"""Partition default auxiliary-g force work without omitting a derivative class."""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING

import pytest
from test_scf_df_auxiliary_g_cuda import check_endpoint

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DF_G_ENDPOINT_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires an explicit finite Slurm CUDA allocation",
)


@pytest.mark.parametrize("method,xc", [("rhf", None), ("pbe-rks", "PBE")])
@pytest.mark.parametrize(
    "route,pairs,buckets",
    [
        ("shell", "symmetric", "off"),
        ("shell", "full", "packet"),
        ("shell", "packed", "on"),
        ("shell-sp", "symmetric", "off"),
        ("generic", "full", "off"),
    ],
)
def test_complete_g_partition_matches_independent_forces(
    method: str,
    xc: str | None,
    route: str,
    pairs: str,
    buckets: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CPU E/F, translation and live work cover strict and retained fallbacks.

    The complete default fit contains both f and g auxiliary functions. The
    small case explicitly selects the shell route to exercise its six-term
    metadata independently of the larger automatic profitability threshold.
    Packed requests retain dense weights for the complete auxiliary-g domain.
    AUTO must select the same CUDA auxiliary owner as explicit CUDA DF.
    Intrusive shell diagnostics prove g work runs in the generated consumer;
    shell-sp and generic retain the independently checked generic remainder.
    """
    trace = tmp_path / "progress.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_PROGRESS_TRACE", str(trace))
    monkeypatch.setenv("GENERATIVEQC_DF_WEIGHTED_EXECUTION", route)
    monkeypatch.setenv("GENERATIVEQC_DF_DERIVATIVE_PAIRS", pairs)
    monkeypatch.setenv("GENERATIVEQC_DF_PRIMITIVE_BUCKETS", buckets)
    monkeypatch.setenv("GENERATIVEQC_DF_SHELL_WORK", "1")
    check_endpoint(
        method, xc, False, "water-def2-svp-spherical", density_fitting="auto"
    )
    values = [
        row
        for line in trace.read_text().splitlines()
        if (row := json.loads(line)).get("event") == "VALUE"
    ]
    shell_work = any(
        row.get("key") == "three_center_shell_panels" and row["value"] > 0
        for row in values
    )
    g_work = any(
        re.fullmatch(r"shell_[0-3][0-3]4_work_active_shell_tasks", row.get("key", ""))
        and row["value"] > 0
        for row in values
    )
    assert shell_work == (route != "generic")
    assert g_work == (route == "shell")
