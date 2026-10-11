"""Exercise the shared gate through real built-in CPU and Slurm CUDA owners."""

from __future__ import annotations

import ctypes as ct
import os
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, ResourceBudget, _native
from generativeqc._ks_snapshot import NativeKsSnapshot

H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
HE = [("He", (0.0, 0.0, 0.0))]


@pytest.fixture(params=("cpu", "cuda"))
def backend(request: Any) -> str:
    if request.param == "cuda":
        if os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1":
            pytest.skip("requires explicit real-GPU qualification")
        assert os.environ.get("SLURM_JOB_ID"), "real CUDA work must use Slurm"
    return request.param


def lifecycle(batch: Any) -> tuple[int, ...]:
    """Read the actual native owner, not a Python mirror of its lifecycle."""
    getter = batch._library.generativeqc_batch_execution_lifecycle_v1
    getter.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint64), ct.c_size_t]
    getter.restype = ct.c_int
    record = (ct.c_uint64 * 9)()
    _native.check(batch._library, getter(batch._batch, record, len(record)))
    return tuple(record)


@pytest.mark.parametrize("method", ("rhf", "pbe-rks"))
def test_real_cold_replay_geometry_reset_failure_recovery_and_close(
    method: str, backend: str
) -> None:
    calculator = Calculator(
        method=method, device=backend, energy_tolerance=1e-12, density_tolerance=1e-10
    )
    batch = calculator.prepare_batch([H2, HE])
    try:
        bound = lifecycle(batch)
        assert bound[0] == 1
        assert bound[1] == (
            _native.BACKEND_CUDA if backend == "cuda" else _native.BACKEND_CPU_REFERENCE
        )
        cold = batch.execute(properties=("energy",), strict=True)
        assert lifecycle(batch)[0] == 3
        assert all(
            item.executed_backend == ("cuda" if backend == "cuda" else "cpu_reference")
            for item in cold.items
        )
        replay = batch.execute(properties=("energy",), strict=True)
        np.testing.assert_allclose(replay.energies, cold.energies, atol=1e-9, rtol=0)
        assert all(item.warm_start_used for item in replay.items)

        snapshot = NativeKsSnapshot(batch, 0) if method == "pbe-rks" else None
        try:
            failed = batch.execute(
                [np.full((2, 3), np.nan), None], properties=("energy",)
            )
            assert failed.failure_indices == (0,)
            assert failed.items[1].succeeded
            assert failed.items[1].energy == pytest.approx(
                cold.items[1].energy, abs=1e-9
            )
            if snapshot is not None:
                with pytest.raises((RuntimeError, ValueError)):
                    snapshot.check_current()
        finally:
            if snapshot is not None:
                snapshot.close()

        moved = np.array([[0.0, 0.0, -0.8], [0.0, 0.0, 0.8]])
        actual = batch.execute([moved, None], properties=("energy",), strict=True)
        reference = calculator.singlepoint(
            [(atom[0], position) for atom, position in zip(H2, moved, strict=True)],
            properties=("energy",),
        )
        assert actual.items[0].energy == pytest.approx(reference.energy, abs=1e-9)
        assert abs(actual.items[0].energy - cold.items[0].energy) > 1e-5

        # Reject at the actual ABI validation boundary, after revoking native
        # exports but before entering any scientific method or GPU operation.
        status = batch._library.generativeqc_batch_execute(
            batch._batch, None, 0, None, 0
        )
        assert status == _native.STATUS_INVALID_ARGUMENT
        assert lifecycle(batch)[0] == 4
        recovered = batch.execute(properties=("energy",), strict=True)
        assert lifecycle(batch)[0] == 3 and lifecycle(batch)[8] > 0
        np.testing.assert_allclose(recovered.energies, cold.energies, atol=1e-9, rtol=0)
        batch.clear_warm_starts()
        assert lifecycle(batch)[0] == 1
        reset = batch.execute(properties=("energy",), strict=True)
        assert all(not item.warm_start_used for item in reset.items)
    finally:
        batch.close()
    with pytest.raises(RuntimeError, match="closed"):
        batch.execute(properties=("energy",))


@pytest.mark.parametrize("method", ("rhf", "pbe-rks"))
def test_two_admitted_budgets_and_preflight_refusal(method: str, backend: str) -> None:
    probe = Calculator(method=method, device=backend)
    plan = probe.estimate_resources([H2, HE]).require_feasible()
    energies = []
    for multiplier in (1, 2):
        budget = ResourceBudget(
            host_bytes=multiplier * plan.peak_bytes["host"],
            device_bytes=multiplier * plan.peak_bytes.get("device", 0),
        )
        with Calculator(
            method=method, device=backend, resource_budget=budget
        ).prepare_batch([H2, HE]) as batch:
            energies.append(batch.execute(properties=("energy",), strict=True).energies)
            observed = batch.resource_diagnostics["observation"]
            host_peak = observed["sampled_item_peak_host_bytes"]
            if host_peak is not None:
                assert host_peak <= budget.host_bytes
            else:
                # CUDA HF does not sample host buffers. Unknown usage must stay
                # explicit rather than being fabricated as a zero-byte peak.
                assert observed["complete_plan_peak"] is False
                assert observed["excludes"]
            if backend == "cuda":
                ledger = observed["device_ledger"]
                assert ledger["peak_bytes"] <= budget.device_bytes
                assert ledger["rejected_allocations"] == 0
    np.testing.assert_allclose(energies[0], energies[1], atol=1e-9, rtol=0)
    too_small = ResourceBudget(
        host_bytes=plan.peak_bytes["host"] - 1,
        device_bytes=plan.peak_bytes.get("device", 0),
    )
    with pytest.raises(MemoryError, match="no supported plan fits"):
        Calculator(
            method=method, device=backend, resource_budget=too_small
        ).prepare_batch([H2, HE])


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires explicit real-GPU qualification and a native test-hook build",
)
def test_real_cuda_runtime_failure_revokes_snapshot_and_preserves_neighbor() -> None:
    assert os.environ.get("SLURM_JOB_ID")
    with Calculator(method="pbe-rks", device="cuda").prepare_batch([H2, HE]) as batch:
        cold = batch.execute(properties=("energy",), strict=True)
        snapshot = NativeKsSnapshot(batch, 0)
        try:
            inject = batch._library.ks_cuda_fail_next_runtime_for_test_v1
            inject.argtypes = []
            inject.restype = None
            inject()
            failed = batch.execute(properties=("energy",))
            assert failed.failure_indices == (0,)
            assert failed.items[1].energy == pytest.approx(
                cold.items[1].energy, abs=1e-9
            )
            with pytest.raises((RuntimeError, ValueError)):
                snapshot.check_current()
            recovered = batch.execute(properties=("energy",), strict=True)
            np.testing.assert_allclose(
                recovered.energies, cold.energies, atol=1e-9, rtol=0
            )
            with pytest.raises((RuntimeError, ValueError)):
                snapshot.check_current()
        finally:
            snapshot.close()
