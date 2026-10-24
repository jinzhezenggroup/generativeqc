"""A complete prepared CUDA target-Fock census includes discarded attempts."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from generativeqc import Calculator, _native, cross_overlap
from generativeqc.ks import GridSpec, KsOptions

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_CUDA_KS_WORK_TEST") != "1",
    reason="requires exact retry-count native WIP and a finite Slurm allocation",
)
ATOMS = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]


@pytest.mark.parametrize("spins", (1, 2))
def test_actual_nonconverged_warm_retry_has_complete_target_count(spins: int) -> None:
    """The curated LDA owner exercises the same retry census without promotion."""
    run_nonconverged_warm_retry(spins, f"lda-{'rks' if spins == 1 else 'uks'}")


def run_nonconverged_warm_retry(spins: int, method: str) -> None:
    """A virtual determinant exhausts two physical steps, then cold converges.

    The warm seed is normalized/idempotent, not an invalid-input zero-work
    failure. DF J avoids phase-source warm-energy rebasing and speculative
    device chunks. Under this controlled strict-FP64 schedule,
    the discarded attempt executes exactly two Focks, not merely two exported
    history rows. No preliminary diagnostic is requested or needed for totals.
    """
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    calculator = Calculator(
        method=method,
        basis="sto-3g",
        device="cuda",
        precision="fp64",
        density_fitting="cuda",
        initial_guess=None,
        max_iterations=2,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12),
            tile_points=37,
        ),
    )
    overlap = cross_overlap(calculator, calculator, ATOMS)
    rows = []
    with calculator.prepare_batch([ATOMS], multiplicities=[1]) as batch:
        started = perf_counter()
        cold = batch.execute(properties=("energy",), strict=True).items[0]
        cold_seconds = perf_counter() - started
        assert cold.succeeded and cold.executed_backend == "cuda"
        assert cold.fock_builds == cold.ks_diagnostic.fock_builds
        assert cold.fock_builds == cold.iterations
        rows.append(
            {
                "phase": "cold",
                "endpoint_seconds": cold_seconds,
                "endpoint_fock_builds": cold.fock_builds,
                "returned_attempt_fock_builds": cold.ks_diagnostic.fock_builds,
            }
        )

        state = _native.HfWarmState(
            ctypes.sizeof(_native.HfWarmState), _native.ABI_VERSION
        )
        getter = batch._library.generativeqc_batch_get_hf_warm_state
        _native.check(batch._library, getter(batch._batch, 0, ctypes.byref(state)))
        density = np.empty(state.density_count)
        coordinates = np.empty(state.coordinate_count)
        state.density = density.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        state.coordinates = coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        _native.check(batch._library, getter(batch._batch, 0, ctypes.byref(state)))
        assert density.size == spins * 4
        occupation = 2.0 if spins == 1 else 1.0
        density[:] = (
            occupation * np.linalg.inv(overlap)[None, :, :]
            - density.reshape(spins, 2, 2)
        ).ravel()
        for channel in density.reshape(spins, 2, 2):
            np.testing.assert_allclose(
                np.trace(channel @ overlap), occupation, atol=1e-12
            )
            np.testing.assert_allclose(
                channel @ overlap @ channel, occupation * channel, atol=1e-12
            )
        states = (_native.HfWarmState * 1)(state)
        _native.check(
            batch._library,
            batch._library.generativeqc_batch_restore_hf_warm_states(
                batch._batch, states, 1
            ),
        )
        before = batch.ks_transport_diagnostics[0]
        started = perf_counter()
        retried = batch.execute(properties=("energy",), strict=True).items[0]
        retry_seconds = perf_counter() - started
        after = batch.ks_transport_diagnostics[0]
        assert retried.succeeded and retried.converged
        assert retried.warm_start_used and retried.warm_start_fallback
        assert not retried.ks_diagnostic.initial_density_used
        assert retried.ks_diagnostic.history[0].energy_change is None
        assert len(retried.ks_diagnostic.history) == retried.iterations
        assert retried.ks_diagnostic.fock_builds == retried.iterations
        assert retried.energy == pytest.approx(cold.energy, abs=1e-10)
        assert retried.fock_builds == retried.ks_diagnostic.fock_builds + 2
        assert after.iterations - before.iterations == retried.iterations + 2
        rows.append(
            {
                "phase": "nonconverged-warm-then-cold",
                "endpoint_seconds": retry_seconds,
                "endpoint_fock_builds": retried.fock_builds,
                "returned_attempt_fock_builds": retried.ks_diagnostic.fock_builds,
                "discarded_attempt_fock_builds": 2,
                "warm_start_used": retried.warm_start_used,
                "warm_start_fallback": retried.warm_start_fallback,
                "transport_iterations_before": before.iterations,
                "transport_iterations_after": after.iterations,
            }
        )

        # A later no-retry execution must own a fresh total rather than retain
        # the preceding execution's discarded-attempt count.
        started = perf_counter()
        replay = batch.execute(properties=("energy",), strict=True).items[0]
        replay_seconds = perf_counter() - started
        assert replay.succeeded and replay.converged
        assert replay.warm_start_used and not replay.warm_start_fallback
        assert replay.fock_builds == replay.ks_diagnostic.fock_builds

        rows.append(
            {
                "phase": "warm-replay-after-retry",
                "endpoint_seconds": replay_seconds,
                "endpoint_fock_builds": replay.fock_builds,
                "returned_attempt_fock_builds": replay.ks_diagnostic.fock_builds,
            }
        )

    evidence = os.environ.get("GENERATIVEQC_CUDA_KS_WORK_EVIDENCE_DIR")
    if evidence:
        destination = Path(evidence)
        destination.mkdir(parents=True, exist_ok=True)
        label = method.replace(":", "-")
        (destination / f"retry-work-{label}.json").write_text(
            json.dumps(
                {
                    "spin_channels": spins,
                    "method": method,
                    "initial_guess": None,
                    "timing_scope": "prepared execute; excludes initial batch preparation",
                    "work_scope": "all physical target attempts; no preliminary-method work",
                    "slurm_job_id": os.environ["SLURM_JOB_ID"],
                    "rows": rows,
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("schedule", ("device_fused", "host_unfused"))
def test_nonconvergence_and_uninventoried_schedule_keep_distinct_contracts(
    spins: int, schedule: str
) -> None:
    """A physical count is not a convergence or precision-audit certificate."""
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    ordinary = schedule == "device_fused"
    calculator = Calculator(
        method=f"lda-{'rks' if spins == 1 else 'uks'}",
        basis="sto-3g",
        device="cuda",
        precision="fp64",
        density_fitting="cuda",
        initial_guess=None,
        max_iterations=1 if ordinary else 50,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12),
            tile_points=37,
            xc_schedule=schedule,
        ),
    )
    with calculator.prepare_batch([ATOMS], multiplicities=[1]) as batch:
        result = batch.execute(properties=("energy",), strict=False).items[0]
        assert result.executed_backend == "cuda"
        assert result.ks_diagnostic is not None
        if ordinary:
            assert result.status == _native.STATUS_NOT_CONVERGED
            assert not result.converged
            assert result.fock_builds == result.ks_diagnostic.fock_builds == 1
            assert result.precision is not None and not result.precision["complete"]
            assert not result.precision["operator_inventory_complete"]
        else:
            assert result.succeeded and result.converged
            assert result.fock_builds is None


@pytest.mark.parametrize("spins", (1, 2))
def test_exceptional_warm_attempt_keeps_target_count_unavailable(spins: int) -> None:
    """A driver error revokes work and preserves the last-good explicit replay.

    The native one-shot fault is injected in begin(), after attempt ownership
    starts. Driver/resource errors are deliberately not auto-retried. The failed
    execution has no complete count or returned-state diagnostic. A subsequent
    explicit execute owns a fresh, independently valid total.
    """
    assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
    calculator = Calculator(
        method=f"lda-{'rks' if spins == 1 else 'uks'}",
        basis="sto-3g",
        device="cuda",
        precision="fp64",
        density_fitting="cuda",
        initial_guess=None,
        ks_options=KsOptions(
            grid=GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12),
            tile_points=37,
        ),
    )
    with calculator.prepare_batch([ATOMS], multiplicities=[1]) as batch:
        cold = batch.execute(properties=("energy",), strict=True).items[0]
        assert cold.succeeded and cold.fock_builds is not None
        inject = batch._library.ks_cuda_fail_next_runtime_for_test_v1
        inject.argtypes = []
        inject.restype = None
        inject()
        failed = batch.execute(properties=("energy",), strict=False).items[0]
        assert failed.status == _native.STATUS_CUDA_ERROR
        assert not failed.succeeded
        assert failed.fock_builds is None
        assert failed.ks_diagnostic is None and failed.precision is None
        replay = batch.execute(properties=("energy",), strict=True).items[0]
        assert replay.succeeded and replay.converged
        assert replay.warm_start_used and not replay.warm_start_fallback
        assert replay.fock_builds == replay.ks_diagnostic.fock_builds
        assert replay.energy == pytest.approx(cold.energy, abs=1e-10)
