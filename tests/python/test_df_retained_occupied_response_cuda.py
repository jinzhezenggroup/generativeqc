"""Explicit CUDA candidates require independent molecular and replay evidence.

Run only in an allocated Slurm GPU job after building THIS checkout. These
are correctness/work-census tests, not timing benchmarks or automatic promotion.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("auxiliary", ["def2-svp", "cc-pvdz-jkfit"])
@pytest.mark.parametrize("route", ["fitted", "raw-batched"])
@pytest.mark.parametrize("consumer", ["auto", "qualify"])
def test_occupied_response_independent_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    auxiliary: str,
    route: str,
    consumer: str,
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "GPU work must use a finite Slurm allocation"
    from generativeqc import Calculator
    from pyscf import gto, scf

    from benchmarks.df_component_ledger import read_trace

    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8)), ("H", (1.5, 0.0, -0.5))]
    moved = [atoms[0], ("H", (0.0, 0.0, 1.801)), atoms[2]]
    refs = []
    for geometry in (atoms, moved):
        mol = gto.M(atom=geometry, unit="Bohr", basis="def2-svp", verbose=0)
        reference = scf.RHF(mol).density_fit(auxbasis=auxiliary)
        reference.conv_tol = 1e-12
        reference.conv_tol_grad = 1e-10
        reference.max_cycle = 100
        reference.kernel()
        assert reference.converged
        refs.append((reference.e_tot, -reference.nuc_grad_method().kernel()))
    n = mol.nao_nr()
    a = reference.with_df.auxmol.nao_nr()
    fixed = 4 * a * a + 4 * n * n + 2 * a
    budget = 135000 if auxiliary == "def2-svp" else 8 * (fixed + 3 * a * n * n // 4)
    for name in (
        "GENERATIVEQC_DF_WEIGHTED_EXECUTION",
        "GENERATIVEQC_DF_DERIVATIVE_PAIRS",
        "GENERATIVEQC_DF_SHELL_SCHEDULE",
        "GENERATIVEQC_DF_PRIMITIVE_BUCKETS",
        "GENERATIVEQC_DF_RESPONSE_UPLOAD_PROBE",
        "GENERATIVEQC_DF_RESPONSE_SCATTER_PROBE",
        "GENERATIVEQC_DF_SERIAL_RESPONSE_DOT",
        "GENERATIVEQC_DF_RAW_REUSE",
        "GENERATIVEQC_DF_RESPONSE_BATCHING",
        "GENERATIVEQC_DF_SOURCE_PROJECTION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", "packed-single")
    monkeypatch.setenv("GENERATIVEQC_DF_EXCHANGE", "occupied")
    monkeypatch.setenv("GENERATIVEQC_DF_RESPONSE_SPACE", "occupied")
    monkeypatch.setenv("GENERATIVEQC_DF_RESPONSE_ALGEBRA", "blas")
    monkeypatch.setenv("GENERATIVEQC_DF_SOURCE_DERIVATIVE_SCHEDULE", consumer)
    monkeypatch.setenv("GENERATIVEQC_DF_RESPONSE_BUDGET_BYTES", str(budget))
    monkeypatch.setenv(
        "GENERATIVEQC_DF_OCCUPIED_RESPONSE_SOURCE",
        "fitted" if route == "fitted" else "raw",
    )
    if route == "raw-batched":
        monkeypatch.setenv("GENERATIVEQC_DF_SOURCE_PROJECTION", "batched")
    # The practical auxiliary is an external record, not a bundled basis name.
    auxiliary_record = (
        auxiliary
        if auxiliary == "def2-svp"
        else Path(__file__).resolve().parents[2]
        / "benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json"
    )
    calculator = Calculator(
        device="cuda",
        method="rhf",
        basis="def2-svp",
        basis_representation="spherical",
        auxiliary_basis=auxiliary_record,
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calculator.prepare_batch([atoms], warm_start=True) as batch:
        # None restores the prepared geometry, so replay moved coordinates explicitly.
        for phase, geometry in enumerate((None, None, moved, moved)):
            trace = tmp_path / f"{route}-{auxiliary}-{consumer}-{phase}.jsonl"
            monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
            result = batch.execute(
                None if geometry is None else [np.asarray([p for _, p in geometry])],
                strict=True,
            )
            item = result.items[0]
            energy, force = refs[int(phase >= 2)]
            assert item.energy == pytest.approx(energy, abs=1e-9, rel=0)
            np.testing.assert_allclose(item.forces, force, atol=1e-8, rtol=0)
            responses = [
                x for x in read_trace(trace) if x["operation"] == "force_response"
            ]
            assert len(responses) == 1
            counts = responses[0]["counters"]
            assert counts.get("response_fitted_panel_calls", 0) == 0
            assert counts.get("response_full_rank_bounded_factor_first", 0) == 0
            if route == "fitted":
                assert counts.get("response_retained_fitted_projection_passes", 0) == 1
                assert counts.get("response_retained_fitted_projection_columns", 0) == a
                assert counts.get("response_streamed_occupied_raw_passes", 0) == 0
                assert (
                    counts.get("response_borrowed_whitened_bytes", 0)
                    == n * (n + 1) // 2 * a * 8
                )
                panels = counts["response_retained_fitted_projection_panels"]
            else:
                assert counts.get("response_streamed_occupied_raw_passes", 0) == 1
                assert counts.get("response_batched_raw_source_values", 0) == n * n * a
                assert counts.get("response_batched_raw_projection_columns", 0) == a
                panels = counts["response_batched_raw_projection_panels"]
                assert counts.get("response_batched_raw_source_calls", 0) == panels
            assert panels < a
            assert (
                counts.get("response_occupied_projection_blas_calls", 0) == 2 * panels
            )
