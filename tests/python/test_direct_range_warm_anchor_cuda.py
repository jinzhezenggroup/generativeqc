"""Independent full-Direct science and dense-seed deferred-anchor lifetimes."""

from __future__ import annotations

import gc
import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from test_lazy_range_warm_anchor_cuda import ATOMS

if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1",
    reason="explicit scheduled CUDA qualification required",
)


@pytest.fixture(scope="module")
def direct_reference() -> dict[str, Any]:
    """Reconverge independent full-Direct E/F on the same declared atomic rules."""
    if not (os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID")):
        pytest.fail("finite scheduler allocation required")
    import cupy as cp
    from generativeqc import GridSpec
    from generativeqc_compiler.dft.nonlocal_policy import (
        MOLECULAR_VV10_DENSITY_THRESHOLD,
    )

    from benchmarks.pbe0_precision import reference

    result = reference(
        "water-def2-svp-spherical",
        GridSpec(12, 4, 8),
        1,
        functional="WB97M_V",
        atoms=ATOMS,
        backend="gpu4pyscf",
        vv10_density_threshold=float(MOLECULAR_VV10_DENSITY_THRESHOLD),
    )
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return result


@pytest.mark.parametrize("updates", [False, True])
@pytest.mark.parametrize("invalidate", ["replay", "capacity", "eviction"])
def test_direct_dense_seed_anchor_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    updates: bool,
    invalidate: str,
    direct_reference: dict[str, Any],
) -> None:
    """Source reuse avoids cold work; source loss reconstructs its own frozen D.

    The opt-in must preserve independent E/F and first-warm work, including
    frozen warm updates. Explicit eviction may defer a counted preparation
    action, but may not revoke the already-published stationary token.
    """
    from contextlib import closing

    from generativeqc import Calculator, GridSpec, KsOptions
    from generativeqc._ks_snapshot import NativeKsSnapshot

    monkeypatch.setenv("GENERATIVEQC_KS_RANGE_RESIDENT_VALUES", "1")
    monkeypatch.setenv("GENERATIVEQC_KS_RESIDENT_BUCKET_MASK", "0xfffffff")
    monkeypatch.setenv("GENERATIVEQC_KS_RANGE_HOST_MAX_BYTES", "0")
    monkeypatch.setenv("GENERATIVEQC_KS_RANGE_DISK_MAX_BYTES", "0")
    monkeypatch.setenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", "0")
    monkeypatch.setenv("GENERATIVEQC_CUDA_KS_CHUNK", "1")
    monkeypatch.setenv("GENERATIVEQC_CUDA_KS_WARM_FACTOR", "0")
    populations = []
    receipts = []
    for selector in ("0", "1", None):
        if selector is None:
            monkeypatch.delenv("GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY", raising=False)
        else:
            monkeypatch.setenv("GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY", selector)
        monkeypatch.setenv("GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY", "1")
        monkeypatch.setenv("GENERATIVEQC_KS_RESIDENT_MAX_BYTES", str(8 << 30))
        calculator = Calculator(
            method="wb97m-v",
            basis="def2-svp",
            basis_representation="spherical",
            device="cuda",
            precision="fp64",
            density_fitting="none",
            max_iterations=100,
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-14,
            ks_options=KsOptions(grid=GridSpec(12, 4, 8)),
        )
        with calculator.prepare_batch([ATOMS], warm_start=True) as owner:
            cold = owner.execute(strict=True).items[0]
            owner.set_warm_start_updates(updates)
            retained = owner.execute(strict=True).items[0]
            if invalidate == "eviction":
                with closing(NativeKsSnapshot(owner, 0)) as snapshot:
                    energy_before = snapshot.energy()
                    receipt = snapshot.cuda_prepare_force_memory(1 << 40)
                    assert receipt["reclaimed_device_bytes"] > 0
                    snapshot.check_current()
                    assert snapshot.energy() == energy_before
            elif invalidate == "capacity":
                monkeypatch.setenv("GENERATIVEQC_KS_RESIDENT_MAX_BYTES", "1")
            else:
                monkeypatch.setenv("GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY", "0")
            fallback = owner.execute(strict=True).items[0]
            second = owner.execute(strict=True).items[0]
        calculator.clear_cache()
        population = [cold, retained, fallback, second]
        populations.append(population)
        receipts.append(
            {
                "selector": selector,
                "samples": [
                    {
                        "energy": result.energy,
                        "forces": result.forces.tolist(),
                        "iterations": result.iterations,
                        "fock_builds": result.fock_builds,
                        "precision": result.precision,
                    }
                    for result in population
                ],
            }
        )
        (tmp_path / "direct-anchor.json").write_text(
            json.dumps(
                {"reference": direct_reference, "populations": receipts}, indent=2
            )
            + "\n"
        )
        for result in population:
            assert result.converged and result.executed_backend == "cuda"
            assert abs(result.energy - direct_reference["energy"]) < 2e-8
            np.testing.assert_allclose(
                result.forces, direct_reference["forces"], atol=2e-7, rtol=0
            )
            precision = result.precision
            assert precision["complete"] and precision["operator_inventory_complete"]
            assert precision["operator_work_counters_valid"]
            assert (
                sum(
                    event["count"]
                    for event in precision["scf_fock_timeline"]
                    if event["kind"] in ("mixed_fock", "strict_fock", "post_scf_fock")
                )
                == result.fock_builds
            )
        assert all(result.iterations == 1 for result in population[1:])
        assert retained.fock_builds == retained.iterations
        assert fallback.fock_builds == fallback.iterations + int(selector == "1")
        assert second.fock_builds == second.iterations
        assert all(
            event["phase"] != "preparation"
            for event in retained.precision["scf_fock_timeline"]
        )
    eager, deferred, default = populations
    assert deferred[0].fock_builds == eager[0].fock_builds - 1
    assert default[0].fock_builds == eager[0].fock_builds
    for baseline, candidate, incumbent in zip(eager, deferred, default, strict=True):
        for result in (candidate, incumbent):
            assert abs(result.energy - baseline.energy) < 1e-10
            np.testing.assert_allclose(
                result.forces, baseline.forces, atol=1e-7, rtol=0
            )
