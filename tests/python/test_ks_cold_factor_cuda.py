"""Public DF cold-factor routing, batch lifetime and independent energy gates."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, PrecisionCalculator
from test_huckel_df import WATER, calculator, independent_oracle

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_HUCKEL_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires an explicit finite Slurm GPU allocation",
)


@pytest.mark.parametrize("fast", (False, True))
def test_public_df_cold_factor_batch_and_reset(
    fast: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both public owners preserve full endpoints and independent batch factors.

    Diagnostic matrix audits and traces deliberately stay outside performance
    qualification. The CPU/PySCF oracle supplies no density to either endpoint.
    """
    pytest.importorskip("pyscf")
    oracle_calc = calculator("cpu", "pbe0-rks")
    oracle = independent_oracle(oracle_calc, "pbe0-rks")
    results = {}
    for enabled in (False, True):
        monkeypatch.setenv("GENERATIVEQC_CUDA_KS_COLD_FACTOR", "1" if enabled else "0")
        monkeypatch.setenv("GENERATIVEQC_CUDA_KS_WARM_FACTOR_AUDIT", "1")
        trace = tmp_path / f"factor-{enabled}.jsonl"
        monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
        options = {
            "method": "pbe0-rks",
            "basis": "sto-3g",
            "auxiliary_basis": "def2-svp",
            "density_fitting": "cuda",
            "device": "cuda",
            "ks_options": KsOptions(grid=GridSpec(16, 8, 16)),
            "energy_tolerance": 1e-12,
            "density_tolerance": 1e-10,
            "screening_tolerance": 1e-12,
            "max_iterations": 100,
        }
        calc = (
            PrecisionCalculator(policy="fast", **options)
            if fast
            else Calculator(**options)
        )
        try:
            with calc.prepare_batch([WATER, WATER], warm_start=True) as batch:
                cold = batch.execute(strict=True, properties=("energy", "forces"))
                batch.set_warm_start_updates(False)
                warm = batch.execute(strict=True, properties=("energy", "forces"))
                batch.clear_warm_starts()
                reset = batch.execute(strict=True, properties=("energy", "forces"))
            results[enabled] = [
                item for phase in (cold, warm, reset) for item in phase.items
            ]
        finally:
            calc.clear_cache()
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        beginnings = [row["counters"] for row in rows if row["operation"] == "ks_begin"]
        cold_beginnings = [row for row in beginnings if row["explicit_seed"]]
        assert cold_beginnings and all(
            bool(row["cold_factor_restored"]) == enabled for row in cold_beginnings
        )
        for result in results[enabled]:
            assert result.converged and result.executed_backend == "cuda"
            assert result.energy == pytest.approx(oracle.e_tot, abs=1e-8)
    for baseline, candidate in zip(results[False], results[True], strict=True):
        assert candidate.energy == pytest.approx(baseline.energy, abs=1e-10)
        np.testing.assert_allclose(candidate.forces, baseline.forces, atol=3e-7, rtol=0)
        assert candidate.fock_builds == baseline.fock_builds
