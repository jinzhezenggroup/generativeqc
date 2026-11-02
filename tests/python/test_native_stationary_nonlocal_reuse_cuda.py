"""Independent force and executed-work gates for native exact VV10 reuse."""

from __future__ import annotations

import ctypes
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from test_lazy_range_warm_anchor_cuda import ATOMS
from test_lazy_range_warm_anchor_cuda import (
    independent_reference as independent_reference,  # noqa: PLC0414 -- pytest fixture
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "control",
    ("enabled", "disabled", "default-small", "minimum", "schedule", "failure", "moved"),
)
def test_native_force_exact_reuse_and_bounded_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    control: str,
    independent_reference: dict[str, Any],
) -> None:
    """Frozen seeds may reuse equal ingredients; changed owners cannot borrow them.

    The imported oracle fixture requires a finite GPU allocation and independently
    converges GPU4PySCF. Work receipts distinguish an actual hit from an accurate
    but unnecessarily repeated pair sweep, including minimum-capacity owners with no extra allocation.
    """
    from generativeqc import Calculator, GridSpec, KsOptions, _native
    from test_range_resident_ks_cuda import _auxiliary_basis

    if control == "default-small":
        monkeypatch.delenv("GENERATIVEQC_VV10_COMPONENT_REUSE", raising=False)
    else:
        monkeypatch.setenv(
            "GENERATIVEQC_VV10_COMPONENT_REUSE", "0" if control == "disabled" else "1"
        )
    monkeypatch.delenv("GENERATIVEQC_VV10_PAIR_SCHEDULE", raising=False)
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
    calculator = Calculator(
        "wb97m-v",
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
    evidence = []

    def check(
        result: Any, expected: dict[str, Any] = independent_reference
    ) -> dict[str, int]:
        assert result.converged
        assert abs(result.energy - expected["energy"]) < 2e-8
        np.testing.assert_allclose(result.forces, expected["forces"], atol=2e-7, rtol=0)
        calls = [json.loads(line) for line in trace.read_text().splitlines()]
        work = [
            row["counters"] for row in calls if row["operation"] == "stationary_force"
        ][-1]
        record = {
            key: work[key]
            for key in (
                "nonlocal_pair_key_comparisons",
                "nonlocal_pair_cache_hits",
                "nonlocal_pair_sweeps",
            )
        }
        assert record["nonlocal_pair_cache_hits"] + record["nonlocal_pair_sweeps"] == 1
        evidence.append(
            {"energy": result.energy, "forces": result.forces.tolist(), "work": record}
        )
        trace.write_text("")
        return record

    try:
        with calculator.prepare_batch([ATOMS], warm_start=True) as owner:
            assert owner._native_dft_force_eligible(0)
            if control == "minimum":
                query = owner._library.generativeqc_ks_batch_stationary_cuda_capacity_v1
                query.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint64),
                    ctypes.c_size_t,
                ]
                query.restype = ctypes.c_int
                capacity = (ctypes.c_uint64 * 9)()
                _native.check(
                    owner._library, query(owner._batch, 0, capacity, len(capacity))
                )
                limit = owner._library.generativeqc_ks_batch_stationary_cuda_limit_v1
                limit.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64]
                limit.restype = ctypes.c_int
                _native.check(owner._library, limit(owner._batch, 0, capacity[4]))
            cold = check(owner.execute(strict=True).items[0])
            assert cold["nonlocal_pair_sweeps"] == 1
            owner.set_warm_start_updates(False)
            warm = check(owner.execute(strict=True).items[0])
            admitted = control not in ("disabled", "default-small")
            assert warm["nonlocal_pair_cache_hits"] == int(admitted)
            if control == "schedule":
                monkeypatch.setenv("GENERATIVEQC_VV10_PAIR_SCHEDULE", "ordered")
                assert (
                    check(owner.execute(strict=True).items[0])["nonlocal_pair_sweeps"]
                    == 1
                )
                assert (
                    check(owner.execute(strict=True).items[0])[
                        "nonlocal_pair_cache_hits"
                    ]
                    == 1
                )
                monkeypatch.setenv("GENERATIVEQC_VV10_COMPONENT_REUSE", "0")
                assert (
                    check(owner.execute(strict=True).items[0])["nonlocal_pair_sweeps"]
                    == 1
                )
                monkeypatch.setenv("GENERATIVEQC_VV10_COMPONENT_REUSE", "1")
                assert (
                    check(owner.execute(strict=True).items[0])["nonlocal_pair_sweeps"]
                    == 1
                )
            elif control == "failure":
                inject = owner._library.ks_cuda_fail_next_runtime_for_test_v1
                inject.argtypes, inject.restype = [], None
                inject()
                failed = owner.execute(strict=False).items[0]
                assert failed.status == _native.STATUS_CUDA_ERROR
                assert failed.ks_diagnostic is None
                check(owner.execute(strict=True).items[0])
            elif control == "moved":
                from generativeqc_compiler.dft.nonlocal_policy import (
                    MOLECULAR_VV10_DENSITY_THRESHOLD,
                )
                from test_df_fast_public_cuda import auxiliary_bases

                from benchmarks._cases import benchmark_cases
                from benchmarks.pbe0_precision import reference

                moved = np.asarray([r for _, r in ATOMS])
                moved[1, 2] += 0.001
                atoms = [
                    (symbol, tuple(r))
                    for (symbol, _), r in zip(ATOMS, moved, strict=True)
                ]
                _, auxiliary = auxiliary_bases(
                    benchmark_cases()["water-def2-svp-spherical"]
                )
                expected = reference(
                    "water-def2-svp-spherical",
                    GridSpec(12, 4, 8),
                    1,
                    functional="WB97M_V",
                    atoms=atoms,
                    auxiliary_basis=auxiliary,
                    backend="gpu4pyscf",
                    vv10_density_threshold=float(MOLECULAR_VV10_DENSITY_THRESHOLD),
                )
                # Accept the new geometry/SCF seed before freezing its replay.
                # Freezing the original geometry would rebuild this owner on
                # every displaced call and deliberately prevent reuse.
                owner.set_warm_start_updates(True)
                changed = check(owner.execute([moved], strict=True).items[0], expected)
                assert changed["nonlocal_pair_sweeps"] == 1
                owner.set_warm_start_updates(False)
                assert (
                    check(owner.execute([moved], strict=True).items[0], expected)[
                        "nonlocal_pair_cache_hits"
                    ]
                    == 1
                )
    finally:
        calculator.clear_cache()
        (tmp_path / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")
