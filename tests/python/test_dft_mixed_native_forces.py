"""Per-item native force admission must survive an unqualified batch neighbor."""

from __future__ import annotations

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions, _native
from generativeqc._ks_snapshot import NativeKsSnapshot

H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
PROPERTIES = ("energy", "forces")


def calculator(method: str = "pbe-rks", fitting: str = "none") -> Calculator:
    """Use a small fixed grid to exercise routing independently of grid policy."""
    return Calculator(
        method=method,
        device="cpu",
        density_fitting=fitting,
        ks_options=KsOptions(grid=GridSpec(12, 6, 12)),
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )


@pytest.mark.parametrize("method", ["pbe-rks", "pbe0-rks"])
@pytest.mark.parametrize("fitting", ["none", "cpu"])
def test_mixed_force_consumers_share_scf_and_survive_replay(
    method: str, fitting: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calc = calculator(method, fitting)
    with calc.prepare_batch([H2, H2]) as batch:
        assert batch._native_dft_force_eligible()
        # Narrow one item's admission to exercise both genuine consumers on
        # the same inexpensive fixture. This never grants a native capability.
        monkeypatch.setattr(
            batch, "_native_dft_force_eligible", lambda index=None: index == 0
        )
        compatibility = batch._public_dft_cpu_force
        visited: set[int] = set()

        def force(index: int, atoms: object) -> tuple:
            assert index == 1, (
                "qualified neighbor returned to Python force orchestration"
            )
            visited.add(index)
            return compatibility(index, atoms)

        monkeypatch.setattr(batch, "_public_dft_cpu_force", force)
        moved = np.array([[0.0, 0.0, -0.72], [0.0, 0.0, 0.75]])
        for coordinates in (None, None, [moved, moved]):
            visited.clear()
            result = batch.execute(coordinates, properties=PROPERTIES, strict=True)
            assert result.succeeded
            assert visited == {1}
            np.testing.assert_allclose(
                result.items[0].energy, result.items[1].energy, atol=2e-10
            )
            np.testing.assert_allclose(
                result.items[0].forces, result.items[1].forces, atol=2e-8, rtol=0
            )
        visited.clear()
        energy = batch.execute(properties=("energy",), strict=True)
        assert all(item.forces is None for item in energy.items)
        assert not visited
        failed = batch.execute([np.full((2, 3), np.nan), None], properties=PROPERTIES)
        assert failed.failure_indices == (0,)
        assert failed.items[0].forces is None and failed.items[1].forces is not None
        assert visited == {1}
        visited.clear()
        recovered = batch.execute(properties=PROPERTIES, strict=True)
        assert recovered.succeeded and visited == {1}


def test_real_unqualified_neighbor_preserves_native_force_and_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Thirteen He/STO-3G atoms exceed the native primitive-source work bound.
    # Their CPU compatibility consumer also retains its explicit atom bound.
    # This exercises actual prepared capability differences, not mocked flags.
    large = [("He", (float(i % 4) * 4, float(i // 4) * 4, 0.0)) for i in range(13)]
    calc = calculator()
    reference = calc.singlepoint(H2, properties=PROPERTIES)
    with calc.prepare_batch([H2, large]) as batch:
        assert batch._native_dft_force_eligible(0)
        assert not batch._native_dft_force_eligible(1)
        compatibility = batch._public_dft_cpu_force
        visited: set[int] = set()

        def force(index: int, atoms: object) -> tuple:
            assert index == 1, "unqualified neighbor displaced an admitted native force"
            visited.add(index)
            return compatibility(index, atoms)

        monkeypatch.setattr(batch, "_public_dft_cpu_force", force)
        result = batch.execute(properties=PROPERTIES)
        assert result.failure_indices == (1,)
        assert visited == {1}
        assert result.items[1].status == _native.STATUS_INVALID_ARGUMENT
        assert "dense-export domain" in result.items[1].status_message
        assert result.items[1].forces is None
        np.testing.assert_allclose(
            result.items[0].forces, reference.forces, atol=2e-9, rtol=0
        )
        snapshot = NativeKsSnapshot(batch, 0)
        try:
            selected = batch._library.generativeqc_ks_batch_execute_selected_forces_v1
            assert (
                selected(batch._batch, None, 0, None, 0)
                == _native.STATUS_INVALID_ARGUMENT
            )
            with pytest.raises((RuntimeError, ValueError)):
                snapshot.check_current()
        finally:
            snapshot.close()
        visited.clear()
        recovered = batch.execute(properties=PROPERTIES)
        assert recovered.failure_indices == (1,)
        np.testing.assert_allclose(
            recovered.items[0].forces, reference.forces, atol=2e-9, rtol=0
        )
        assert visited == {1}


def test_older_library_retains_whole_fleet_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with calculator().prepare_batch([H2, H2]) as batch:
        monkeypatch.setattr(
            batch, "_native_dft_force_eligible", lambda index=None: index == 0
        )
        monkeypatch.setattr(
            batch._library, "generativeqc_ks_batch_execute_selected_forces_v1", None
        )
        compatibility = batch._public_dft_cpu_force
        visited: set[int] = set()

        def force(index: int, atoms: object) -> tuple:
            visited.add(index)
            return compatibility(index, atoms)

        monkeypatch.setattr(batch, "_public_dft_cpu_force", force)
        result = batch.execute(properties=PROPERTIES, strict=True)
        assert result.succeeded and visited == {0, 1}


def test_selective_execution_error_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with calculator().prepare_batch([H2, H2]) as batch:
        monkeypatch.setattr(
            batch, "_native_dft_force_eligible", lambda index=None: index == 0
        )
        monkeypatch.setattr(
            batch._library,
            "generativeqc_ks_batch_execute_selected_forces_v1",
            lambda *args: _native.STATUS_OUT_OF_MEMORY,
        )

        def forbidden(*args: object) -> None:
            pytest.fail("a native execution error was treated as a capability miss")

        monkeypatch.setattr(batch, "_public_dft_cpu_force", forbidden)
        with pytest.raises(RuntimeError, match="out of memory"):
            batch.execute(properties=PROPERTIES)


@pytest.mark.parametrize("method,xc", [("pbe-uks", "PBE"), ("pbe0-uks", "PBE0")])
@pytest.mark.parametrize("fitting", ["none", "cpu"])
def test_selected_force_and_final_state_match_independent_oracle(
    method: str, xc: str, fitting: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selected native consumer preserves independent E/F/D/W acceptance."""
    pytest.importorskip("pyscf")
    from generativeqc._dft_gradient import StationaryKsState
    from generativeqc_compiler.dft import NativeAO
    from test_dft_complete_cpu import independent_global_hybrid_gradient
    from test_native_uks_forces import assert_reference_state

    calc = calculator(method, fitting)
    with (
        calc.prepare_batch([H2, H2], charges=[1, 1], multiplicities=[2, 2]) as batch,
        NativeAO(H2, charge=1, multiplicity=2) as basis,
    ):
        assert batch._native_dft_force_eligible()
        monkeypatch.setattr(
            batch, "_native_dft_force_eligible", lambda index=None: index == 0
        )
        result = batch.execute(properties=PROPERTIES, strict=True).items[0]
        state = StationaryKsState.from_native(batch, basis)
        try:
            energy, gradient, reference = independent_global_hybrid_gradient(
                basis,
                state,
                method,
                xc=xc,
                density_fitting=fitting == "cpu",
                return_mean_field=True,
            )
            assert result.energy == pytest.approx(energy, abs=2e-9)
            np.testing.assert_allclose(result.forces, -gradient, atol=2e-7, rtol=0)
            assert_reference_state(basis, state, reference)
        finally:
            state._source.close()
