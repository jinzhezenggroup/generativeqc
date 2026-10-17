"""Independent exact J'/K' oracles for the density/Hermite factorization."""

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


@pytest.mark.parametrize("method", ("pbe-rks", "pbe-uks", "pbe0-rks", "pbe0-uks"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("enabled", (False, True))
def test_density_projection_matches_independent_sources(
    method: str,
    representation: str,
    enabled: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain both signed source exports and combined-force realization."""
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    census = tmp_path / "density-work.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DIRECT_FORCE_CENSUS", str(census))
    monkeypatch.setenv(
        "GENERATIVEQC_DIRECT_SPD_FORCE_DOMAINS",
        "relative-density" if enabled else "relative-reuse",
    )
    monkeypatch.setenv("GENERATIVEQC_DIRECT_PAIR_COOPERATIVE_DERIVATIVES", "1")
    # The charged pure-PBE UKS dimer has an unrelated unstable SCF trajectory.
    # Its doublet monomer still exercises nonzero order-six/seven responses;
    # independent raw-AD tests retain arbitrary four-center UKS contractions.
    two_oxygens = method != "pbe-uks"
    check_derivatives(
        method, representation, "1", two_oxygens, "bounded", monkeypatch
    )
    rows = [json.loads(line) for line in census.read_text().splitlines()]
    assert rows and all(row["valid"] for row in rows)
    density = [
        shell["density_contraction"]
        for row in rows
        for shell in row["classes"]
        if shell["angular_order"] in (6, 7)
    ]
    if enabled and method.startswith("pbe-"):
        assert any(row["accepted_shells"] for row in density)
        assert all(
            row["hermite_products"] > 0 and row["coulomb_preparations"] > 0
            for row in density
            if row["accepted_shells"]
        )
    else:
        assert not any(row["attempted_shells"] for row in density)
