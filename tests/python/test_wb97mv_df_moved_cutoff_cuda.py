"""H100 regression for the two real moved geometries from issue #3185."""

from __future__ import annotations

import json
import os
from contextlib import closing
from pathlib import Path

import pytest
from generativeqc import PrecisionCalculator
from generativeqc._ks_snapshot import NativeKsSnapshot

from benchmarks.readme_defaults import allocation_metadata
from benchmarks.readme_defaults_accuracy import errors

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
    reason="requires an explicitly scheduled H100",
)

FIXTURES = Path(__file__).resolve().parents[1] / "reference_data/wb97mv-df-3185"


@pytest.mark.parametrize("case", ("paper-water96", "n-c32h66"))
def test_wb97mv_df_warm_move_completes_across_lr_cutoff(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seed from the previous geometry and compare analytic forces to Direct."""
    import cupy as cp

    allocation_metadata(os.environ.get("GQC_QZ_JOB_NAME"))
    if "H100" not in cp.cuda.runtime.getDeviceProperties(0)["name"].decode():
        pytest.skip("these fixed-cutoff boundary cases are qualified on H100")
    monkeypatch.delenv("GENERATIVEQC_DF_RANGE_EXCHANGE", raising=False)
    root = FIXTURES / case
    previous, target = json.loads((root / "frames.json").read_text())["steps"]
    reference = json.loads((root / "reference.json").read_text())
    assert reference["geometry_sha256"] == target["geometry_sha256"]
    calculator = PrecisionCalculator(
        method="wb97m-v-rks",
        device="cuda",
        basis="def2-svp",
        basis_representation="spherical",
        density_fitting="cuda",
        auxiliary_basis=root / "wb97m-v-auxiliary.json",
    )
    try:
        with calculator.prepare_batch(
            [previous["geometry_bohr"]], warm_start=True
        ) as owner:
            cold = owner.execute(properties=("energy", "forces"), strict=True).items[0]
            assert cold.converged and cold.executed_backend == "cuda"
            with closing(NativeKsSnapshot(owner._owner, 0)) as snapshot:
                assert snapshot.range_density_fitted
            result = owner.execute(
                [[xyz for _, xyz in target["geometry_bohr"]]],
                properties=("energy", "forces"),
                strict=True,
            ).items[0]
            assert result.converged and result.executed_backend == "cuda"
            assert result.warm_start_used and not result.warm_start_fallback
            with closing(NativeKsSnapshot(owner._owner, 0)) as snapshot:
                assert snapshot.fock_provider_proof() == (
                    "density-fitted",
                    "density-fitted",
                    1e-10,
                )
                assert not snapshot.range_density_fitted
        gates = errors(
            result.energy, result.forces, reference, len(target["geometry_bohr"])
        )
        assert gates["passed"], gates
    finally:
        calculator.clear_cache()
