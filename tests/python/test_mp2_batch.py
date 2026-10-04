"""Public conventional MP2 prepared-batch ownership and isolation."""

import ctypes
import os
import typing
from pathlib import Path

import numpy as np
import pytest
from generativeqc import Calculator, _native, method_capabilities

H2 = [
    [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
    [("H", (0.0, 0.0, -0.8)), ("H", (0.0, 0.0, 0.8))],
]


def test_mp2_advertises_batches_reuses_hf_warm_state_and_rejects_profiling() -> None:
    assert method_capabilities("mp2").supports_batch
    calculator = Calculator(method="mp2", device="cpu")
    with calculator.prepare_batch(H2) as batch:
        cold = batch.execute(properties=("energy",), strict=True)
        assert all(not item.warm_start_used for item in cold.items)

        warm = batch.execute(properties=("energy",), strict=True)
        np.testing.assert_allclose(warm.energies, cold.energies, atol=1.0e-10, rtol=0)
        assert all(item.warm_start_used for item in warm.items)
        assert all(item.converged and item.iterations > 0 for item in warm.items)
        assert all(not item.warm_start_fallback for item in warm.items)

        changed_coordinates = np.asarray(
            [[0.0, 0.0, -0.75], [0.0, 0.0, 0.75]], dtype=np.float64
        )
        changed = batch.execute(
            [changed_coordinates, None], properties=("energy",), strict=True
        )
        assert all(item.warm_start_used for item in changed.items)

        batch.clear_warm_starts()
        cleared = batch.execute(properties=("energy",), strict=True)
        assert all(not item.warm_start_used for item in cleared.items)

    with pytest.raises(RuntimeError, match="MP2 batch does not support profiling"):
        calculator.prepare_batch(H2, warm_start=False, shell_class_profiling=True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("forces", [False, True])
def test_mp2_batch_checkpoint_restores_hf_warm_state(
    tmp_path: Path, device: str, forces: bool
) -> None:
    if device == "cuda" and os.environ.get("GENERATIVEQC_MP2_CUDA_TEST") != "1":
        pytest.skip("requires explicitly allocated CUDA device and native library")
    properties = ("energy", "forces") if forces else ("energy",)
    calculator = Calculator(method="mp2", device=device)
    checkpoint = tmp_path / "mp2-warm.vqcp"

    with calculator.prepare_batch(H2) as source:
        baseline = source.execute(properties=properties, strict=True)
        snapshots = [_snapshot(source, index) for index in range(len(H2))]
        for snapshot, atoms in zip(snapshots, H2, strict=True):
            assert snapshot is not None
            density, coordinates, energy = snapshot
            # H2/STO-3G must export the complete seed on both backends, even
            # when CUDA HF leaves its optional iterative density vector empty.
            assert density.shape == (4,)
            assert coordinates.shape == (6,)
            assert np.isfinite(density).all() and np.isfinite(coordinates).all()
            assert np.isfinite(energy)
            np.testing.assert_array_equal(
                coordinates, np.asarray([atom[1] for atom in atoms]).ravel()
            )
        source.save_checkpoint(checkpoint)

    with calculator.prepare_batch(H2) as target:
        report = target.load_checkpoint(checkpoint)
        assert all(item["restored_fields"] == ["density"] for item in report["items"])
        for index, snapshot in enumerate(snapshots):
            _assert_snapshot_equal(_snapshot(target, index), snapshot)
        replay = target.execute(properties=properties, strict=True)

    np.testing.assert_allclose(replay.energies, baseline.energies, atol=1.0e-10, rtol=0)
    assert all(item.warm_start_used for item in replay.items)
    assert all(not item.warm_start_fallback for item in replay.items)
    for item, reference in zip(replay.items, baseline.items, strict=True):
        assert item.executed_backend == (
            "cuda" if device == "cuda" else "cpu_reference"
        )
        assert item.converged and item.iterations > 0
        if forces:
            np.testing.assert_allclose(item.forces, reference.forces, atol=2e-9, rtol=0)


def test_mp2_batch_energy_force_replay_geometry_and_order_independence() -> None:
    calculator = Calculator(method="mp2", device="cpu")
    expected = [calculator.singlepoint(system) for system in H2]
    with calculator.prepare_batch(H2, warm_start=False) as batch:
        first = batch.execute(strict=True)
        second = batch.execute(strict=True)
        energy_only = batch.execute(properties=("energy",), strict=True)
        changed_coordinates = np.asarray(
            [[0.0, 0.0, -0.75], [0.0, 0.0, 0.75]], dtype=np.float64
        )
        changed = batch.execute([changed_coordinates, None], strict=True)

    for replay in (first, second):
        for item, reference in zip(replay.items, expected, strict=True):
            assert item.energy == pytest.approx(reference.energy, abs=1.0e-10)
            np.testing.assert_allclose(
                item.forces, reference.forces, atol=2.0e-9, rtol=0
            )
            assert not item.warm_start_used and not item.warm_start_fallback
    np.testing.assert_allclose(
        energy_only.energies, first.energies, atol=1.0e-10, rtol=0
    )
    assert all(item.forces is None for item in energy_only.items)
    changed_reference = calculator.singlepoint(
        [("H", (0.0, 0.0, -0.75)), ("H", (0.0, 0.0, 0.75))]
    )
    assert changed.items[0].energy == pytest.approx(
        changed_reference.energy, abs=1.0e-10
    )
    np.testing.assert_allclose(
        changed.items[0].forces, changed_reference.forces, atol=2.0e-9, rtol=0
    )
    assert changed.items[1].energy == pytest.approx(expected[1].energy, abs=1.0e-10)

    reversed_result = calculator.batch_singlepoint(list(reversed(H2)), strict=True)
    np.testing.assert_allclose(
        reversed_result.energies, first.energies[::-1], atol=1.0e-10, rtol=0
    )


def test_mp2_batch_failure_is_item_local_and_later_replay_is_clean() -> None:
    calculator = Calculator(method="mp2", device="cpu")
    with calculator.prepare_batch(H2, warm_start=False) as batch:
        baseline = batch.execute(strict=True)
        failed = batch.execute([np.zeros((1, 3)), None])
        assert failed.failure_indices == (0,)
        assert failed.items[0].forces is None
        assert failed.items[1].succeeded
        assert failed.items[1].energy == pytest.approx(
            baseline.items[1].energy, abs=1.0e-10
        )
        np.testing.assert_allclose(
            failed.items[1].forces, baseline.items[1].forces, atol=2.0e-9, rtol=0
        )
        recovered = batch.execute(strict=True)
    np.testing.assert_allclose(
        recovered.energies, baseline.energies, atol=1.0e-10, rtol=0
    )
    for item, reference in zip(recovered.items, baseline.items, strict=True):
        np.testing.assert_allclose(item.forces, reference.forces, atol=2.0e-9, rtol=0)


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_MP2_CUDA_TEST") != "1",
    reason="requires explicitly allocated CUDA device and native library",
)
@pytest.mark.parametrize("warm_start", [False, True])
def test_mp2_cuda_batch_matches_cpu_and_isolates_failed_items(warm_start: bool) -> None:
    cpu = Calculator(method="mp2", device="cpu").batch_singlepoint(H2, strict=True)
    calculator = Calculator(method="mp2", device="cuda")
    with calculator.prepare_batch(H2, warm_start=warm_start) as batch:
        cuda = batch.execute(strict=True)
        assert all(item.correlation is not None for item in cuda.items)
        assert all(
            not item.correlation.reference_execution_plan_reused
            and item.correlation.reference_execution_plan_owned_device_bytes > 0
            for item in cuda.items
        )
        if warm_start:
            # CUDA HF retains D in its physical reference, not ScfResult.density.
            # The MP2 checkpoint must still own all four AO density entries.
            density, coordinates, _ = _snapshot(batch, 0)
            assert density.shape == (4,)
            assert coordinates.shape == (6,)
            assert np.isfinite(density).all()
        replay = batch.execute(strict=True)
        assert all(item.warm_start_used == warm_start for item in replay.items)
        assert all(not item.warm_start_fallback for item in replay.items)
        assert all(
            item.correlation.reference_execution_plan_reused
            and item.correlation.reference_execution_plan_owned_device_bytes > 0
            for item in replay.items
        )
        np.testing.assert_allclose(replay.energies, cuda.energies, atol=1.0e-9, rtol=0)
        for item, reference in zip(replay.items, cpu.items, strict=True):
            assert item.executed_backend == "cuda"
            np.testing.assert_allclose(
                item.forces, reference.forces, atol=2.0e-9, rtol=0
            )
        moved = np.asarray([[0.0, 0.0, -0.75], [0.0, 0.0, 0.75]])
        changed = batch.execute([moved, None], properties=("energy",), strict=True)
        assert changed.items[0].warm_start_used == warm_start
        assert changed.items[0].correlation.reference_execution_plan_reused
        assert (
            changed.items[0].correlation.reference_execution_plan_owned_device_bytes > 0
        )
        failed = batch.execute([np.zeros((1, 3)), None])
        recovered = batch.execute(strict=True)
        assert all(
            item.correlation.reference_execution_plan_reused for item in recovered.items
        )
    np.testing.assert_allclose(cuda.energies, cpu.energies, atol=1.0e-9, rtol=0)
    for item, reference in zip(cuda.items, cpu.items, strict=True):
        assert item.executed_backend == "cuda"
        np.testing.assert_allclose(item.forces, reference.forces, atol=2.0e-9, rtol=0)
    assert failed.failure_indices == (0,)
    assert failed.items[0].forces is None
    assert failed.items[1].succeeded and failed.items[1].executed_backend == "cuda"
    np.testing.assert_allclose(
        failed.items[1].forces, cuda.items[1].forces, atol=2.0e-9, rtol=0
    )
    np.testing.assert_allclose(recovered.energies, cuda.energies, atol=1.0e-9, rtol=0)


def _snapshot(batch: typing.Any, index: int) -> typing.Any:
    state = _native.HfWarmState(ctypes.sizeof(_native.HfWarmState), _native.ABI_VERSION)
    getter = batch._library.generativeqc_batch_get_hf_warm_state
    _native.check(batch._library, getter(batch._batch, index, ctypes.byref(state)))
    if not state.present:
        return None
    density = np.empty(state.density_count)
    coordinates = np.empty(state.coordinate_count)
    state.density = density.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    state.coordinates = coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    _native.check(batch._library, getter(batch._batch, index, ctypes.byref(state)))
    return density, coordinates, state.energy


def _restore(batch: typing.Any, snapshots: typing.Any) -> int:
    states = (_native.HfWarmState * len(snapshots))()
    for state, snapshot in zip(states, snapshots, strict=True):
        state.struct_size, state.abi_version = ctypes.sizeof(state), _native.ABI_VERSION
        if snapshot is None:
            continue
        density, coordinates, energy = snapshot
        state.present = 1
        state.density = density.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        state.density_count = density.size
        state.coordinates = coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        state.coordinate_count = coordinates.size
        state.energy = energy
    return batch._library.generativeqc_batch_restore_hf_warm_states(
        batch._batch, states, len(states)
    )


def _assert_snapshot_equal(actual: typing.Any, expected: typing.Any) -> None:
    assert actual is not None and expected is not None
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


@pytest.mark.parametrize("forces", [False, True])
def test_mp2_warm_payload_is_charged_to_endpoint_budget(forces: bool) -> None:
    properties = ("energy", "forces") if forces else ("energy",)
    calculator = Calculator(method="mp2", device="cpu")
    cold = calculator.singlepoint(H2[0], properties=properties)
    base_capacity = cold.correlation.numeric_capacity_bytes
    # H2/STO-3G has a 2x2 density and six source-coordinate doubles.
    seed_bytes = (2 * 2 + 3 * 2) * 8
    with calculator.prepare_batch([H2[0]]) as batch:
        first = batch.execute(properties=properties, strict=True).items[0]
        replay = batch.execute(properties=properties, strict=True).items[0]
        batch.set_warm_start_updates(False)
        frozen = batch.execute(properties=properties, strict=True).items[0]
        assert first.correlation.numeric_capacity_bytes == base_capacity + seed_bytes
        assert (
            replay.correlation.numeric_capacity_bytes == base_capacity + 2 * seed_bytes
        )
        assert frozen.correlation.numeric_capacity_bytes == base_capacity + seed_bytes
        for item in (first, replay, frozen):
            assert item.energy == pytest.approx(cold.energy, abs=1e-10)
            if forces:
                np.testing.assert_allclose(item.forces, cold.forces, atol=2e-9, rtol=0)
                assert item.correlation.measured_endpoint_peak_bytes == 0

    tight = Calculator(
        method="mp2",
        device="cpu",
        correlation_memory_budget_bytes=base_capacity + 2 * seed_bytes - 1,
    )
    with tight.prepare_batch([H2[0]]) as batch:
        batch.execute(properties=properties, strict=True)
        retained = _snapshot(batch, 0)
        rejected = batch.execute(properties=properties).items[0]
        assert rejected.status == _native.STATUS_OUT_OF_MEMORY
        assert rejected.correlation is None
        _assert_snapshot_equal(_snapshot(batch, 0), retained)
        # Removing candidate storage (freeze) or the prior seed (clear) admits
        # the same endpoint again without relaxing any scientific threshold.
        batch.set_warm_start_updates(False)
        batch.execute(properties=properties, strict=True)
        batch.clear_warm_starts()
        batch.set_warm_start_updates(True)
        assert (
            not batch.execute(properties=properties, strict=True)
            .items[0]
            .warm_start_used
        )

    exact = Calculator(
        method="mp2",
        device="cpu",
        correlation_memory_budget_bytes=base_capacity + 2 * seed_bytes,
    )
    with exact.prepare_batch([H2[0]]) as batch:
        batch.execute(properties=properties, strict=True)
        accepted = batch.execute(properties=properties, strict=True).items[0]
        assert (
            accepted.correlation.numeric_capacity_bytes
            <= base_capacity + 2 * seed_bytes
        )


def test_mp2_frozen_changed_geometry_and_restore_are_atomic() -> None:
    calculator = Calculator(method="mp2", device="cpu")
    with calculator.prepare_batch(H2) as batch:
        batch.execute(strict=True)
        original = [_snapshot(batch, index) for index in range(2)]
        batch.set_warm_start_updates(False)
        moved_coordinates = np.array([[0.0, 0.0, -0.9], [0.0, 0.0, 0.9]])
        moved = batch.execute([moved_coordinates, None], strict=True)
        for index in range(2):
            _assert_snapshot_equal(_snapshot(batch, index), original[index])
        assert all(item.warm_start_used for item in moved.items)
        cold = calculator.singlepoint([("H", tuple(x)) for x in moved_coordinates])
        assert moved.items[0].energy == pytest.approx(cold.energy, abs=1e-10)
        np.testing.assert_allclose(
            moved.items[0].forces, cold.forces, atol=2e-9, rtol=0
        )

        # Slot zero would change, but invalid slot one rejects the entire import.
        replacement = (original[0][0], original[0][1], original[0][2] + 1.0)
        invalid = (np.full_like(original[1][0], np.nan), original[1][1], original[1][2])
        assert (
            _restore(batch, [replacement, invalid]) == _native.STATUS_INVALID_ARGUMENT
        )
        for index in range(2):
            _assert_snapshot_equal(_snapshot(batch, index), original[index])
        assert _restore(batch, [replacement, None]) == _native.STATUS_SUCCESS
        _assert_snapshot_equal(_snapshot(batch, 0), replacement)
        _assert_snapshot_equal(_snapshot(batch, 1), original[1])

    # Force correlation failure after a successful HF solve has produced a
    # candidate seed; last-good state must remain intact, with no stale result.
    failing = Calculator(method="mp2", device="cpu", mp2_denominator_threshold=100.0)
    with failing.prepare_batch(H2) as batch:
        assert _restore(batch, original) == _native.STATUS_SUCCESS
        failed = batch.execute()
        assert failed.failure_indices == (0, 1)
        for index, item in enumerate(failed.items):
            assert item.correlation is None and item.forces is None
            _assert_snapshot_equal(_snapshot(batch, index), original[index])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mp2_changed_geometry_warm_energy_and_forces_match_pyscf(device: str) -> None:
    if device == "cuda" and os.environ.get("GENERATIVEQC_MP2_CUDA_TEST") != "1":
        pytest.skip("requires explicitly allocated CUDA device and native library")
    pyscf = pytest.importorskip("pyscf")
    mp = pytest.importorskip("pyscf.mp")
    moved = [("H", (0.0, 0.0, -0.9)), ("H", (0.0, 0.0, 0.9))]
    mol = pyscf.gto.M(atom=moved, unit="Bohr", basis="sto-3g", verbose=0)
    hf = pyscf.scf.RHF(mol).run(conv_tol=1e-12)
    reference = mp.MP2(hf).run()
    with Calculator(method="mp2", device=device).prepare_batch([H2[0]]) as batch:
        batch.execute(strict=True)
        result = batch.execute([np.array([a[1] for a in moved])], strict=True).items[0]
    assert result.executed_backend == ("cuda" if device == "cuda" else "cpu_reference")
    assert result.warm_start_used and not result.warm_start_fallback
    assert result.converged and result.iterations > 0
    assert result.energy == pytest.approx(reference.e_tot, abs=1e-9)
    np.testing.assert_allclose(
        result.forces, -reference.nuc_grad_method().kernel(), atol=2e-8, rtol=0
    )


def test_mp2_nonconverged_warm_proposal_retries_cold_reference() -> None:
    calculator = Calculator(method="mp2", device="cpu", max_iterations=2)
    cold = calculator.singlepoint(H2[0], properties=("energy",))
    with calculator.prepare_batch([H2[0]]) as batch:
        batch.execute(properties=("energy",), strict=True)
        retained = _snapshot(batch, 0)
        # A localized rank-one density has valid metric occupations (2, 0)
        # and electron trace two, but cannot converge within two iterations.
        proposal = (np.array([2.0, 0.0, 0.0, 0.0]), retained[1], retained[2])
        assert _restore(batch, [proposal]) == _native.STATUS_SUCCESS
        result = batch.execute(properties=("energy",), strict=True).items[0]
        assert result.warm_start_used and result.warm_start_fallback
        assert result.converged and result.iterations == cold.iterations
        assert result.energy == pytest.approx(cold.energy, abs=1e-10)
        _assert_snapshot_equal(_snapshot(batch, 0), retained)
