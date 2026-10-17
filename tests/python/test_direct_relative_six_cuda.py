"""Independent source-force qualification of shell-shared order-six reuse."""

import json
import os
from pathlib import Path

import pytest
from test_cuda_hybrid_snapshot import (
    test_separate_full_range_derivatives_match_libcint as check_derivatives,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires a finite Slurm allocation and independent Libcint",
)


@pytest.mark.parametrize("method", ("pbe-rks", "pbe0-rks", "pbe0-uks"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("fallback", (False, True))
def test_relative_six_preserves_sources_screening_and_fallback(
    method: str,
    representation: str,
    fallback: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare signed J'/K' and their combined consumer against Libcint."""
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    census = tmp_path / "six-work.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DIRECT_FORCE_CENSUS", str(census))
    monkeypatch.setenv("GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS", "relative-reuse")
    monkeypatch.setenv(
        "GENERATIVEQC_DIRECT_PAIR_COOPERATIVE_DERIVATIVES", "0" if fallback else "1"
    )
    check_derivatives(method, representation, "1", True, "bounded", monkeypatch)
    rows = [json.loads(line) for line in census.read_text().splitlines()]
    assert rows
    assert all(row["valid"] for row in rows)
    assert all(row["complete_bounded_spd_domain"] != fallback for row in rows)
    six = [
        shell for row in rows for shell in row["classes"] if shell["angular_order"] == 6
    ]
    entered = [shell for shell in six if shell["coulomb_preparations"]]
    if fallback:
        assert not entered
    else:
        assert entered, "the requested shared recurrence must actually execute"
        for shell in entered:
            assert shell["primitive_pair_products"] == shell["coulomb_preparations"]
            assert 0 < shell["executed_shells"] <= shell["admitted"]
            assert shell["published_components"] == shell["executed_components"]
