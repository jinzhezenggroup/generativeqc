"""Optional VV10 work receipts preserve unavailable and measured-zero states."""

import os
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import _native
from generativeqc.ks_diagnostics import _nonlocal_work


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


@pytest.mark.parametrize("native_seed", [False, True])
def test_native_scf_exact_pair_reuse_and_geometry_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    native_seed: bool,
) -> None:
    """A frozen replay removes pair work; a new geometry cannot inherit the key."""
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    assert os.environ.get("SLURM_JOB_ID")
    import numpy as np
    from generativeqc import Calculator, GridSpec, KsOptions

    monkeypatch.setenv("GENERATIVEQC_VV10_SCF_COMPONENT_REUSE", "1")
    monkeypatch.setenv("GENERATIVEQC_VV10_NATIVE_SEED", str(int(native_seed)))
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
        if native_seed:
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
