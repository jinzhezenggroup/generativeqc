"""Actual-device independent force and positive-admission gates for exact pairs."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest
from test_cuda_hybrid_snapshot import (
    test_separate_full_range_derivatives_match_libcint as check_derivatives,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires a finite Slurm GPU allocation and independent Libcint",
)


@pytest.mark.parametrize("method", ("pbe-rks", "pbe-uks", "pbe0-uks"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
def test_exact_pair_forces_match_libcint_and_admit_complete_classes(
    method: str,
    representation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check Separate/Combined/replay forces and actual, not eligible, work.

    The pure-J dimer exercises every new class. The charged UKS monomer avoids
    an unrelated unstable dimer SCF trajectory while still exercising spin
    projection. Hybrid exchange must retain its original arbitrary cotangent.
    These intrusive censuses are science gates, never endpoint timing samples.
    """
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    census = tmp_path / "exact-pair-work.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DIRECT_FORCE_CENSUS", str(census))
    monkeypatch.setenv("GENERATIVEQC_DIRECT_FORCE_PAIR_SEPARABLE", "1")
    check_derivatives(
        method, representation, "1", method != "pbe-uks", "bounded", monkeypatch
    )
    rows = [json.loads(line) for line in census.read_text().splitlines()]
    assert rows and all(row["valid"] for row in rows)
    work = [shell for row in rows for shell in row["classes"]]
    accepted = {
        shell["class"]
        for shell in work
        if shell["scalar_pair_contraction"]["accepted_shells"]
    }
    if method == "pbe-rks":
        assert accepted == {5, 8, 9, 11, 12, 13, 16}
    elif method == "pbe-uks":
        assert accepted
    else:
        assert not accepted
    assert all(
        not shell["scalar_pair_contraction"]["numerical_rejections"] for shell in work
    )
