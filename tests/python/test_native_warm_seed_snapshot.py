"""Native snapshot lifetime, scientific replay and authoritative transport errors."""

from __future__ import annotations

import ctypes as ct
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, ResourceBudget, _native
from generativeqc import _precision_batch as precision
from generativeqc import _warm_seed_native as native

SYSTEMS = [
    [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
    [("He", (0.0, 0.0, 0.0))],
]


def forbidden(*args: Any, **kwargs: Any) -> None:
    """Current-library audit staging must never reconstruct descriptor arrays."""
    raise AssertionError("native snapshot delegated to descriptor compatibility")


@pytest.fixture(params=("cpu", "cuda"))
def backend(request: pytest.FixtureRequest) -> str:
    """Require an explicitly scheduled device for the actual CUDA variants."""
    if request.param == "cuda":
        if os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1":
            pytest.skip("requires explicit CUDA qualification")
        assert os.environ.get("SLURM_JOB_ID"), "real CUDA tests require Slurm"
    return request.param


@pytest.mark.parametrize("method", ("rhf", "uhf", "pbe-rks", "pbe-uks"))
def test_snapshot_outlives_source_and_replays_moved_frozen_seeds(
    method: str, backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = (
        {"ks_options": KsOptions(grid=GridSpec(12, 6, 12))} if "pbe" in method else {}
    )
    calculator = Calculator(
        method=method, device=backend, initial_guess=None, **options
    )
    assert hasattr(calculator._library, "generativeqc_batch_capture_warm_seeds_v1")
    monkeypatch.setattr(precision._SeedLease, "_capture_compatibility", forbidden)
    monkeypatch.setattr(precision, "_restore_states", forbidden)
    moved = np.array([[0, 0, -0.7], [0.1, 0, 0.7]])
    with calculator.prepare_batch(SYSTEMS) as source:
        bound, largest = precision._seed_capacity(source)
        with pytest.raises(MemoryError, match="maximum_seed_bytes"):
            precision._SeedLease(source, bound - 1)
        cold = precision._SeedLease(source, bound)
        assert cold.numeric_bytes == 0 and cold.storage == []
        wide_limit = native.capture_snapshot(source, 1 << 40)
        assert wide_limit.numeric_bytes == 0
        wide_limit.close()
        expected = source.execute(
            [moved, None], properties=("energy", "forces"), strict=True
        )
        lease = precision._SeedLease(source, bound)
        assert lease.numeric_bytes == bound and lease.largest_density_bytes == largest
        arrays = lease.storage
        assert all(not value.flags.writeable for value in arrays)
        source.clear_warm_starts()
    calculator.clear_cache()
    # A fresh facade/context has the same model. The lease contains no source
    # handles and retains frozen source coordinates through repeated restores.
    target_calculator = Calculator(
        method=method, device=backend, initial_guess=None, **options
    )
    with target_calculator.prepare_batch(SYSTEMS) as target:
        for _ in range(2):
            lease.restore(target, updates=False)
            actual = target.execute(
                [moved, None], properties=("energy", "forces"), strict=True
            )
            for result, reference in zip(actual.items, expected.items, strict=True):
                if backend == "cuda":
                    assert result.executed_backend == "cuda"
                assert result.energy == pytest.approx(reference.energy, abs=2e-10)
                np.testing.assert_allclose(
                    result.forces, reference.forces, atol=2e-9, rtol=0
                )
            for retained, original in zip(lease.storage, arrays, strict=True):
                np.testing.assert_array_equal(retained, original)
            target.clear_warm_starts()
    target_calculator.clear_cache()
    snapshot = lease._snapshot
    with pytest.raises(ValueError, match="different native library"):
        snapshot.restore(SimpleNamespace(_library=SimpleNamespace(_handle=-1)))
    snapshot.close()
    snapshot.close()
    with pytest.raises(RuntimeError, match="closed"):
        snapshot.diagnostic_arrays()
    # Diagnostic copies remain valid after their native storage is destroyed.
    assert arrays[0].shape == ((8,) if method in ("uhf", "pbe-uks") else (4,))


def test_missing_capture_is_the_only_compatibility_boundary() -> None:
    assert (
        native.capture_snapshot(SimpleNamespace(_library=SimpleNamespace()), 0) is None
    )
    library = SimpleNamespace(generativeqc_batch_capture_warm_seeds_v1=lambda *args: 0)
    with pytest.raises(RuntimeError, match="incomplete"):
        native.capture_snapshot(SimpleNamespace(_library=library), 0)


def test_budgeted_snapshot_restore_retains_resource_diagnostics(backend: str) -> None:
    calculator = Calculator(
        method="rhf",
        device=backend,
        initial_guess=None,
        resource_budget=ResourceBudget(host_bytes=256 << 20),
    )
    assert hasattr(calculator._library, "generativeqc_batch_capture_warm_seeds_v1")
    with calculator.prepare_batch(SYSTEMS) as source:
        source.execute(properties=("energy",), strict=True)
        lease = precision._SeedLease(source, precision._seed_capacity(source)[0])
    with calculator.prepare_batch(SYSTEMS) as target:
        lease.restore(target, updates=False)
        observation = target.resource_diagnostics["warm_restore"]
        assert target.resource_diagnostics["phase"] == "warm_restore"
        # CPU restore does not run an SCF iteration, so its iteration-boundary
        # sampler may honestly report unavailable. CUDA retains its device ledger.
        assert observation["complete_plan_peak"] is False
        assert observation["status"] in ("observed", "unavailable")
        if backend == "cuda":
            assert observation["scope_owner"] == "native-prepared-owner"
            assert observation["status"] == "observed"
            assert observation["device_ledger"] is not None
        result = target.execute(properties=("energy",), strict=True)
        assert all(item.converged for item in result.items)
    calculator.clear_cache()


def test_selected_capture_refusal_is_authoritative() -> None:
    library = SimpleNamespace(
        generativeqc_batch_capture_warm_seeds_v1=lambda *args: (
            _native.STATUS_NOT_IMPLEMENTED
        ),
        generativeqc_batch_restore_warm_seeds_v1=forbidden,
        generativeqc_warm_seed_snapshot_read_v1=forbidden,
        generativeqc_warm_seed_snapshot_destroy=forbidden,
        generativeqc_status_message=lambda status: b"snapshot owner unavailable",
    )
    with pytest.raises(NotImplementedError):
        native.capture_snapshot(
            SimpleNamespace(_library=library, _batch=None, _context=None), 0
        )


def test_malformed_capture_receipt_releases_native_handle() -> None:
    released = []

    def capture(batch: Any, maximum: int, info: Any, handle: Any) -> int:
        ct.cast(handle, ct.POINTER(ct.c_void_p))[0] = ct.c_void_p(1)
        ct.cast(info, ct.POINTER(native._SnapshotInfo)).contents.numeric_bytes = (
            maximum + 1
        )
        return 0

    library = SimpleNamespace(
        generativeqc_batch_capture_warm_seeds_v1=capture,
        generativeqc_batch_restore_warm_seeds_v1=forbidden,
        generativeqc_warm_seed_snapshot_read_v1=forbidden,
        generativeqc_warm_seed_snapshot_destroy=lambda handle: released.append(
            handle.value
        ),
    )
    batch = SimpleNamespace(
        _library=library, _batch=None, _context=None, system_count=0
    )
    with pytest.raises(RuntimeError, match="invalid receipt"):
        native.capture_snapshot(batch, 0)
    assert released == [1]
