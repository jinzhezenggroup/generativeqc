"""Frozen-seed and work-ledger gates for deferred strict range energy anchors."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


ATOMS = [
    ("O", (0.02, -0.03, 0.04)),
    ("H", (0.1, 1.43, 1.1)),
    ("H", (-0.15, -1.45, 1.12)),
]


@pytest.fixture(scope="module")
def independent_reference() -> dict[str, Any]:
    """Independently reconverge fitted full-range J/K plus exact range K."""
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("explicit scheduled CUDA qualification required")
    if not (os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID")):
        pytest.fail("finite scheduler allocation required")
    from generativeqc import GridSpec
    from generativeqc_compiler.dft.nonlocal_policy import (
        MOLECULAR_VV10_DENSITY_THRESHOLD,
    )
    from test_df_fast_public_cuda import auxiliary_bases

    from benchmarks._cases import benchmark_cases
    from benchmarks.pbe0_precision import reference

    case = "water-def2-svp-spherical"
    _, auxiliary = auxiliary_bases(benchmark_cases()[case])
    result = reference(
        case,
        GridSpec(12, 4, 8),
        1,
        functional="WB97M_V",
        atoms=ATOMS,
        auxiliary_basis=auxiliary,
        backend="gpu4pyscf",
        vv10_density_threshold=float(MOLECULAR_VV10_DENSITY_THRESHOLD),
    )
    # The matched GPU4PySCF adapter binds methods back to its engine, creating
    # collectible cycles. Its result contains only host scalars/lists. Release
    # the oracle and idle CuPy blocks before testing native ownership; otherwise
    # separate module fixtures can retain gigabytes until interpreter teardown.
    import gc

    import cupy as cp

    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return result


@pytest.mark.parametrize(
    "invalidate",
    ["replay", "capacity", "buckets", "factor", "eviction", "failed_begin"],
)
@pytest.mark.parametrize("updates", [False, True])
def test_deferred_range_anchor_preserves_frozen_warm_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalidate: str,
    updates: bool,
    independent_reference: dict[str, Any],
) -> None:
    """Both energy schedules must certify one physical warm iteration.

    Compare independently owned eager and deferred plans through retained replay,
    invalidation, the first fallback and a second fallback. The preparation event
    is real executed J/K work, distinct from SCF proposals and final auditing.
    """
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("explicit scheduled CUDA qualification required")
    if not (os.environ.get("SLURM_JOB_ID") or os.environ.get("LSB_JOBID")):
        pytest.fail("finite scheduler allocation required")
    from generativeqc import Calculator, GridSpec, KsOptions
    from test_range_resident_ks_cuda import _auxiliary_basis

    populations = []
    evidence = []
    for lazy in (False, True, None):
        # The public default must exercise the same eviction and frozen-seed
        # behavior as explicit admission, including the counted fallback work.
        deferred = lazy is not False
        if lazy is None:
            monkeypatch.delenv("GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY", raising=False)
        else:
            monkeypatch.setenv("GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY", str(int(lazy)))
        monkeypatch.setenv("GENERATIVEQC_KS_RANGE_RESIDENT_VALUES", "1")
        monkeypatch.setenv("GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY", "1")
        monkeypatch.setenv("GENERATIVEQC_KS_RESIDENT_MAX_BYTES", str(8 << 30))
        monkeypatch.setenv("GENERATIVEQC_KS_RESIDENT_BUCKET_MASK", "0xfffffff")
        monkeypatch.setenv("GENERATIVEQC_KS_RANGE_HOST_MAX_BYTES", "0")
        monkeypatch.setenv("GENERATIVEQC_KS_RANGE_DISK_MAX_BYTES", "0")
        monkeypatch.setenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", "0")
        monkeypatch.setenv("GENERATIVEQC_CUDA_KS_CHUNK", "1")
        monkeypatch.setenv("GENERATIVEQC_CUDA_KS_WARM_FACTOR", "1")
        calculator = Calculator(
            method="wb97m-v",
            basis="def2-svp",
            basis_representation="spherical",
            device="cuda",
            precision="fp64",
            density_fitting="cuda",
            auxiliary_basis=_auxiliary_basis(True),
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
                from contextlib import closing

                from generativeqc._ks_snapshot import NativeKsSnapshot

                with closing(NativeKsSnapshot(owner, 0)) as snapshot:
                    # This is an advisory envelope, not a 1 TiB allocation.
                    # Force-boundary eviction must preserve the current lease
                    # and defer any anchor reconstruction until the next begin.
                    before = snapshot.energy()
                    receipt = snapshot.cuda_prepare_force_memory(1 << 40)
                    assert receipt["reclaimed_device_bytes"] > 0
                    snapshot.check_current()
                    assert snapshot.energy() == before
            elif invalidate == "failed_begin":
                from generativeqc import _native

                monkeypatch.setenv("GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY", "0")
                inject = owner._library.ks_cuda_fail_next_runtime_for_test_v1
                inject.argtypes = []
                inject.restype = None
                inject()
                failed = owner.execute(strict=False).items[0]
                assert failed.status == _native.STATUS_CUDA_ERROR
                assert failed.precision is None and failed.ks_diagnostic is None
                assert failed.fock_builds is None
            elif invalidate == "factor":
                monkeypatch.setenv("GENERATIVEQC_CUDA_KS_WARM_FACTOR", "0")
                monkeypatch.setenv("GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY", "0")
            else:
                key, value = {
                    "replay": ("GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY", "0"),
                    "capacity": ("GENERATIVEQC_KS_RESIDENT_MAX_BYTES", "1"),
                    "buckets": ("GENERATIVEQC_KS_RESIDENT_BUCKET_MASK", "0x1"),
                }[invalidate]
                monkeypatch.setenv(key, value)
            fallback = owner.execute(strict=True).items[0]
            second = owner.execute(strict=True).items[0]
        population = [cold, retained, fallback, second]
        evidence.append(
            {
                "lazy": lazy,
                "updates": updates,
                "invalidation": invalidate,
                "independent_reference": independent_reference,
                "samples": [
                    {
                        "energy": r.energy,
                        "forces": r.forces.tolist(),
                        "iterations": r.iterations,
                        "fock_builds": r.fock_builds,
                        "precision": r.precision,
                    }
                    for r in population
                ],
            }
        )
        (tmp_path / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")
        assert all(r.converged for r in population)
        assert all(r.iterations == 1 for r in population[1:])
        for result in population:
            precision = result.precision
            assert precision["complete"] and precision["operator_inventory_complete"]
            assert precision["operator_work_counters_valid"]
            fock_work = 0
            for kind, field in (
                ("mixed_fock", "mixed_stage_fock_builds"),
                ("strict_fock", "strict_stage_fock_builds"),
                ("post_scf_fock", "post_scf_fock_builds"),
            ):
                counted = sum(
                    event["count"]
                    for event in precision["scf_fock_timeline"]
                    if event["kind"] == kind
                )
                assert counted == precision[field]
                fock_work += counted
            assert fock_work == result.fock_builds
            assert abs(result.energy - independent_reference["energy"]) < 2e-8
            np.testing.assert_allclose(
                result.forces, independent_reference["forces"], atol=2e-7, rtol=0
            )
        for result in population[1:]:
            assert abs(result.energy - cold.energy) < 1e-10
            np.testing.assert_allclose(result.forces, cold.forces, atol=1e-7, rtol=0)
        phases = [
            [e for e in r.precision["scf_fock_timeline"] if e["phase"] == "preparation"]
            for r in population
        ]
        assert [len(p) for p in phases] == [0, 0, int(deferred), 0]
        if deferred:
            assert phases[2][0]["kind"] == "strict_fock"
            assert fallback.fock_builds == fallback.iterations + 1
            assert second.fock_builds == second.iterations
        populations.append(population)
    for eager, explicit, default in zip(*populations, strict=True):
        for candidate in (explicit, default):
            assert abs(eager.energy - candidate.energy) < 1e-10
            np.testing.assert_allclose(
                eager.forces, candidate.forces, atol=1e-7, rtol=0
            )
