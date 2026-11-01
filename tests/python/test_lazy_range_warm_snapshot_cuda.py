"""Deferred energy anchors must stay local to their immutable warm owner."""

from __future__ import annotations

from contextlib import closing
from typing import Any

import numpy as np
import pytest
from test_lazy_range_warm_anchor_cuda import ATOMS
from test_lazy_range_warm_anchor_cuda import (
    independent_reference as independent_reference,  # noqa: PLC0414 -- pytest fixture re-export
)


@pytest.mark.parametrize("lazy", (False, True))
def test_range_seed_export_preserves_final_state_and_outlives_owner(
    monkeypatch: pytest.MonkeyPatch,
    lazy: bool,
    independent_reference: dict[str, Any],
) -> None:
    """Export must preserve a live final lease; imports rebuild their own proof.

    The independent-reference fixture admits only explicitly scheduled GPUs.
    A transported density has no right to borrow the source owner's deferred
    energy, AO factor, range cache, or physical convergence certificate.
    """
    from generativeqc import Calculator, GridSpec, KsOptions
    from generativeqc._ks_snapshot import NativeKsSnapshot
    from generativeqc._warm_seed_native import capture_snapshot
    from test_range_resident_ks_cuda import _auxiliary_basis

    for key, value in {
        "GENERATIVEQC_KS_RANGE_LAZY_WARM_ENERGY": str(int(lazy)),
        "GENERATIVEQC_KS_RANGE_RESIDENT_VALUES": "1",
        "GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY": "1",
        "GENERATIVEQC_KS_RESIDENT_MAX_BYTES": str(8 << 30),
        "GENERATIVEQC_KS_RESIDENT_BUCKET_MASK": "0xfffffff",
        "GENERATIVEQC_KS_RANGE_HOST_MAX_BYTES": "0",
        "GENERATIVEQC_KS_RANGE_DISK_MAX_BYTES": "0",
        "GENERATIVEQC_KS_INCREMENTAL_DIRECT_JK": "0",
        "GENERATIVEQC_CUDA_KS_CHUNK": "1",
        "GENERATIVEQC_CUDA_KS_WARM_FACTOR": "1",
    }.items():
        monkeypatch.setenv(key, value)
    options = {
        "method": "wb97m-v",
        "basis": "def2-svp",
        "basis_representation": "spherical",
        "device": "cuda",
        "precision": "fp64",
        "density_fitting": "cuda",
        "auxiliary_basis": _auxiliary_basis(True),
        "max_iterations": 100,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "screening_tolerance": 1e-14,
        "ks_options": KsOptions(grid=GridSpec(12, 4, 8)),
    }

    def check_result(result: Any) -> None:
        assert result.converged
        assert abs(result.energy - independent_reference["energy"]) < 2e-8
        np.testing.assert_allclose(
            result.forces, independent_reference["forces"], atol=2e-7, rtol=0
        )

    calculator = Calculator(**options)
    seed = None
    try:
        with calculator.prepare_batch([ATOMS], warm_start=True) as source:
            cold = source.execute(strict=True).items[0]
            check_result(cold)
            source.set_warm_start_updates(False)
            with closing(NativeKsSnapshot(source, 0)) as final:
                seed = capture_snapshot(source, 1 << 30)
                assert seed is not None and seed.numeric_bytes > 0
                arrays = seed.diagnostic_arrays()
                assert arrays and all(not array.flags.writeable for array in arrays)
                final.check_current()
                assert final.energy() == cold.energy
            monkeypatch.setenv("GENERATIVEQC_KS_RANGE_RESIDENT_REPLAY", "0")
            fallback = source.execute(strict=True).items[0]
            check_result(fallback)
            assert fallback.iterations == 1
            assert fallback.fock_builds == fallback.iterations + int(lazy)
        calculator.clear_cache()

        # The source context has gone away. Repeated imports retain only the
        # immutable scientific seed and must not consume its former lazy flag.
        target_calculator = Calculator(**options)
        with target_calculator.prepare_batch([ATOMS], warm_start=True) as target:
            for _ in range(2):
                seed.restore(target)
                target.set_warm_start_updates(False)
                restored = target.execute(strict=True).items[0]
                check_result(restored)
                assert not any(
                    event["phase"] == "preparation"
                    for event in restored.precision["scf_fock_timeline"]
                )
                for current, original in zip(
                    seed.diagnostic_arrays(), arrays, strict=True
                ):
                    np.testing.assert_array_equal(current, original)
                target.clear_warm_starts()
        target_calculator.clear_cache()
    finally:
        if seed is not None:
            seed.close()
        calculator.clear_cache()
