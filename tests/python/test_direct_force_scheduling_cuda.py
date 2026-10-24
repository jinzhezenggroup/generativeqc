"""Independent source-force gates for bounded task-acquisition experiments."""

import json
import os
from pathlib import Path

import pytest
from test_cuda_hybrid_snapshot import (
    test_separate_full_range_derivatives_match_libcint as check_derivatives,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires a finite Slurm CUDA allocation and independent Libcint",
)


@pytest.mark.parametrize("page", (8, 16, 32))
@pytest.mark.parametrize("class_major", (False, True))
@pytest.mark.parametrize("method", ("pbe-rks", "pbe0-rks", "pbe0-uks"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
def test_task_acquisition_preserves_independent_derivatives_and_work(
    page: int,
    class_major: bool,
    method: str,
    representation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two d-bearing centers cover both outputs, both spins and actual dddd work.

    The established fixed-density oracle independently checks J'/K', combined
    derivatives and translation. The census additionally proves that the
    requested high-order schedules execute instead of silently falling back.
    """
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    census = tmp_path / "force-work.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DIRECT_FORCE_CENSUS", str(census))
    monkeypatch.setenv("GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS", "spatial-reuse")
    monkeypatch.setenv("GENERATIVEQC_DIRECT_PAIR_COOPERATIVE_DERIVATIVES", "1")
    monkeypatch.setenv("GENERATIVEQC_EXPERIMENT_ORDER7_FORCE_PAGE", str(page))
    monkeypatch.setenv(
        "GENERATIVEQC_EXPERIMENT_DDDD_FORCE_CLASS_MAJOR", str(int(class_major))
    )
    check_derivatives(method, representation, "1", True, "bounded", monkeypatch)
    rows = [json.loads(line) for line in census.read_text().splitlines()]
    assert rows and all(row["valid"] for row in rows)
    complete = [row for row in rows if row["complete_bounded_spd_domain"]]
    assert complete, "the qualified bounded s/p/d consumer must actually execute"
    for row in complete:
        assert row["order7_page_size"] == page
        assert row["dddd_class_major"] is class_major
        for shell in row["classes"]:
            assert shell["enumerated"] == (
                shell["shell_screened"] + shell["generated_owned"] + shell["admitted"]
            )
        # These physical classes have multi-center responses in the two-water
        # fixture. Their actual recurrence products must agree with the census.
        for shell in (row["classes"][19], row["classes"][20]):
            assert shell["executed_shells"] > 0
            assert shell["executed_shells"] <= shell["admitted"]
            assert shell["primitive_pair_products"] == shell["coulomb_preparations"]
