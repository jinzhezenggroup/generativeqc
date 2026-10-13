"""Optional VV10 work receipts preserve unavailable and measured-zero states."""

import os
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import _native
from generativeqc.ks_diagnostics import _nonlocal_history_scopes, _nonlocal_work


def test_nonlocal_history_preserves_interleaved_operator_scopes() -> None:
    """An inner solve is not a contiguous prefix once VV10 anchors are refreshed."""

    def query(handle: int, index: int, values: Any, count: int) -> int:
        assert (handle, index, count) == (12, 3, 5)
        values[:] = (1, 0, 2, 0, 0)
        return _native.STATUS_SUCCESS

    library = SimpleNamespace(
        generativeqc_internal_batch_get_nonlocal_history_scopes_v1=query
    )
    assert _nonlocal_history_scopes(library, 12, 3, 5) == (
        "provisional-semilocal-rsh",
        "full-target",
        "provisional-frozen-vv10",
        "full-target",
        "full-target",
    )
    assert _nonlocal_history_scopes(SimpleNamespace(), 12, 3, 5) is None


def test_nonlocal_history_rejects_unknown_scope() -> None:
    def query(handle: int, index: int, values: Any, count: int) -> int:
        values[0] = 4
        return _native.STATUS_SUCCESS

    library = SimpleNamespace(
        generativeqc_internal_batch_get_nonlocal_history_scopes_v1=query
    )
    with pytest.raises(ValueError, match="operator scope"):
        _nonlocal_history_scopes(library, 12, 3, 1)


def test_nonlocal_work_keeps_exact_reuse_zeros() -> None:
    def query(handle: int, index: int, values: Any, count: int) -> int:
        assert (handle, index, count) == (12, 3, 6)
        values[:] = (1, 0, 1, 1, 1, 0)
        return _native.STATUS_SUCCESS

    library = SimpleNamespace(generativeqc_internal_batch_get_nonlocal_work_v1=query)
    assert _nonlocal_work(library, 12, 3) == {
        "provisional_focks": 0,
        "target_focks": 1,
        "pair_cache_probes": 1,
        "pair_cache_hits": 1,
        "pair_sweeps": 0,
    }


def test_nonlocal_work_does_not_invent_old_library_evidence() -> None:
    assert _nonlocal_work(SimpleNamespace(), 12, 3) is None
    library = SimpleNamespace(
        generativeqc_internal_batch_get_nonlocal_work_v1=lambda *_: (
            _native.STATUS_NOT_IMPLEMENTED
        )
    )
    assert _nonlocal_work(library, 12, 3) is None


def test_nonlocal_work_rejects_malformed_availability() -> None:
    library = SimpleNamespace(
        generativeqc_internal_batch_get_nonlocal_work_v1=lambda *_: (
            _native.STATUS_SUCCESS
        )
    )
    with pytest.raises(ValueError, match="availability"):
        _nonlocal_work(library, 12, 3)


@pytest.mark.parametrize("native_seed", ["off", "semilocal", "frozen", "frozen-range"])
def test_native_scf_exact_pair_reuse_and_geometry_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    native_seed: str,
) -> None:
    """A frozen replay removes pair work; a new geometry cannot inherit the key."""
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    assert os.environ.get("SLURM_JOB_ID")
    import numpy as np
    from generativeqc import Calculator, GridSpec, KsOptions

    monkeypatch.setenv("GENERATIVEQC_VV10_SCF_COMPONENT_REUSE", "1")
    # Each route owns its complete optional policy; a molecular-cohort opt-in
    # in the parent environment must not turn the off/range cases incremental.
    monkeypatch.setenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", "0")
    monkeypatch.setenv("GENERATIVEQC_VV10_INCREMENTAL_SEED", "0")
    monkeypatch.setenv("GENERATIVEQC_VV10_CONTINUE_INNER_DIIS", "0")
    monkeypatch.setenv(
        "GENERATIVEQC_VV10_NATIVE_SEED", str(int(native_seed == "semilocal"))
    )
    monkeypatch.setenv(
        "GENERATIVEQC_VV10_FROZEN_POTENTIAL_SEED",
        str(int(native_seed in {"frozen", "frozen-range"})),
    )
    monkeypatch.setenv(
        "GENERATIVEQC_VV10_FROZEN_RANGE_SEED", str(int(native_seed == "frozen-range"))
    )
    calculator = Calculator(
        method="wb97m-v",
        basis="sto-3g",
        device="cuda",
        basis_representation="spherical",
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        ks_options=KsOptions(grid=GridSpec(8, 6, 12)),
    )
    atoms = [("H", (0.0, 0.0, 0.0)), ("H", (0.15, 0.13, 1.5))]
    with calculator.prepare_batch([atoms], warm_start=True) as owner:
        cold = owner.execute(properties=("energy",)).items[0]
        assert cold.converged
        assert cold.ks_diagnostic.nonlocal_work["pair_sweeps"] > 0
        if native_seed != "off":
            assert cold.ks_diagnostic.nonlocal_work["provisional_focks"] > 0
            assert cold.ks_diagnostic.nonlocal_work["target_focks"] >= 2
            rows = cold.ks_diagnostic.to_payload()["history"]
            assert rows[0]["operator_scope"] == "provisional-semilocal-rsh"
            assert rows[-1]["operator_scope"] == "full-target"
        owner.set_warm_start_updates(False)
        warm = owner.execute(properties=("energy",)).items[0]
        work = warm.ks_diagnostic.nonlocal_work
        assert warm.converged and abs(warm.energy - cold.energy) < 1e-10
        assert work["pair_cache_hits"] == work["target_focks"]
        assert work["pair_sweeps"] == 0
        moved = np.array([[a[1] for a in atoms]])
        moved[0, 1, 2] += 0.001
        after = owner.execute(moved, properties=("energy",)).items[0]
        assert after.converged
        assert after.ks_diagnostic.nonlocal_work["pair_sweeps"] > 0
        if native_seed in {"frozen", "frozen-range"}:
            # Transported target densities start with a fresh physical anchor;
            # discarding VV10 first needlessly moves them away from that target.
            scopes = after.ks_diagnostic.nonlocal_history_scopes
            assert scopes[0] == "full-target"
            assert "provisional-semilocal-rsh" not in scopes


@pytest.mark.parametrize("max_iterations", [4, 100])
@pytest.mark.parametrize(
    "frozen_range,incremental,continued_diis",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, True, True),
    ],
)
def test_frozen_nonlocal_inner_solver_keeps_physical_acceptance_and_total_budget(
    monkeypatch: pytest.MonkeyPatch,
    max_iterations: int,
    frozen_range: bool,
    incremental: bool,
    continued_diis: bool,
) -> None:
    """Exercise actual inner anchors and reject a finite-budget unfinished seed."""
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    assert os.environ.get("SLURM_JOB_ID")
    from generativeqc import Calculator, GridSpec, KsOptions

    monkeypatch.setenv("GENERATIVEQC_VV10_SCF_COMPONENT_REUSE", "1")
    monkeypatch.setenv("GENERATIVEQC_VV10_NATIVE_SEED", "0")
    monkeypatch.setenv("GENERATIVEQC_VV10_FROZEN_POTENTIAL_SEED", "1")
    monkeypatch.setenv("GENERATIVEQC_VV10_FROZEN_RANGE_SEED", str(int(frozen_range)))
    monkeypatch.setenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", str(int(incremental)))
    monkeypatch.setenv("GENERATIVEQC_VV10_INCREMENTAL_SEED", str(int(incremental)))
    monkeypatch.setenv(
        "GENERATIVEQC_VV10_CONTINUE_INNER_DIIS", str(int(continued_diis))
    )
    atoms = [
        ("O", (0.02, -0.03, 0.04)),
        ("H", (0.1, 1.43, 1.1)),
        ("H", (-0.15, -1.45, 1.12)),
    ]
    calculator = Calculator(
        method="wb97m-v",
        basis="def2-svp",
        device="cuda",
        basis_representation="spherical",
        max_iterations=max_iterations,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        ks_options=KsOptions(grid=GridSpec(8, 6, 12)),
    )
    with calculator.prepare_batch([atoms], warm_start=True) as owner:
        result = owner.execute(properties=("energy",)).items[0]
        diagnostic = result.ks_diagnostic
        scopes = diagnostic.nonlocal_history_scopes
        work = diagnostic.nonlocal_work
        assert result.iterations <= max_iterations
        assert work["target_focks"] + work["provisional_focks"] == result.iterations
        assert scopes.count("full-target") == work["target_focks"]
        if max_iterations == 4:
            assert result.status == _native.STATUS_NOT_CONVERGED
            assert not result.converged
            # An unfinished provisional solve cannot become a last-good warm seed.
            retry = owner.execute(properties=("energy",)).items[0]
            assert not retry.warm_start_used and not retry.converged
        else:
            assert result.converged
            expected_inner = (
                "provisional-frozen-vv10-and-range-exchange"
                if frozen_range
                else "provisional-frozen-vv10"
            )
            assert expected_inner in scopes
            assert scopes[-2:] == ("full-target", "full-target")
            assert diagnostic.physical_residual_max < 1e-10
            assert diagnostic.density_change_max < 1e-10
            if incremental:
                # The screened lower retains one delta between full refreshes,
                # and its final force-capable state receives a full-density audit.
                receipt = result.incremental_direct_jk
                assert receipt["active"] and receipt["delta_builds"] > 0
                assert receipt["post_scf_full_builds"] > 0
                warm = owner.execute(properties=("energy",)).items[0]
                assert warm.converged and warm.iterations == 1
                assert not warm.incremental_direct_jk["active"]
                assert abs(warm.energy - result.energy) < 1e-10
            monkeypatch.setenv("GENERATIVEQC_VV10_FROZEN_POTENTIAL_SEED", "0")
            monkeypatch.setenv("GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK", "0")
            with calculator.prepare_batch([atoms]) as reference:
                full = reference.execute(properties=("energy",)).items[0]
            assert full.converged and abs(full.energy - result.energy) < 1e-10
