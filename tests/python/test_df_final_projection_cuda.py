"""Exact-state final occupied-K projection reuse, with independent force gates."""

import os
import typing

import numpy as np
import pytest
from generativeqc import Calculator

from benchmarks._cases import benchmark_cases
from benchmarks.df_component_ledger import read_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires a finite Slurm GPU allocation",
)


def test_automatic_single_fitted_projection_replays(
    monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    """Automatic force selection consumes a fresh U lease or exact panel fallback.

    Force the small fixture's storage layout, leaving response selection at its
    public default. Large endpoint qualification separately covers the automatic
    value/response budget crossover without any layout override.
    """
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    for name in (
        "GENERATIVEQC_DF_EXCHANGE",
        "GENERATIVEQC_DF_FINAL_EXCHANGE",
        "GENERATIVEQC_DF_RESIDENT_EXCHANGE",
        "GENERATIVEQC_DF_RESPONSE_STORAGE",
        "GENERATIVEQC_DF_RESPONSE_SPACE",
        "GENERATIVEQC_DF_OCCUPIED_RESPONSE_SOURCE",
        "GENERATIVEQC_DF_RESPONSE_BUDGET_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GENERATIVEQC_DF_VALUE_STORAGE", "packed-single")
    case = benchmark_cases()["water-tetramer-def2-svp-spherical"]
    calc = Calculator(
        method="rhf",
        basis="def2-svp",
        auxiliary_basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=100,
    )
    with calc.prepare_batch([case.atoms], warm_start=True) as batch:
        for moved in (False, True):
            coordinates = np.asarray([r for _, r in case.atoms]).copy()
            coordinates[1, 0] += 0.001 if moved else 0
            atoms = [(z, r) for (z, _), r in zip(case.atoms, coordinates, strict=True)]
            oracle = scf.RHF(
                gto.M(atom=atoms, unit="Bohr", basis="def2-svp", verbose=0)
            ).density_fit(auxbasis="def2-svp")
            oracle.conv_tol, oracle.conv_tol_grad = 1e-12, 1e-10
            oracle.kernel()
            assert oracle.converged
            expected = -oracle.nuc_grad_method().kernel()
            for repeat, policy in enumerate(("off", "auto", "auto")):
                monkeypatch.setenv("GENERATIVEQC_DF_FINAL_PROJECTION", policy)
                trace = tmp_path / f"single-{moved}-{repeat}.jsonl"
                monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
                item = batch.execute([coordinates], strict=True).items[0]
                assert item.energy == pytest.approx(oracle.e_tot, abs=1e-9, rel=0)
                np.testing.assert_allclose(item.forces, expected, atol=1e-8, rtol=0)
                response = [
                    r for r in read_trace(trace) if r["operation"] == "force_response"
                ]
                assert len(response) == 1
                reused = response[0]["counters"].get(
                    "response_final_fitted_projection_reused", 0
                )
                assert reused == int(policy == "auto")


@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_final_projection_replay_and_geometry(
    method: typing.Any, monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    """Only singleton RHF may lend U; warm and geometry replays bind fresh states."""
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    case = benchmark_cases()["water-tetramer-def2-svp-spherical"]
    for name, value in {
        "GENERATIVEQC_DF_EXCHANGE": "occupied",
        "GENERATIVEQC_DF_FINAL_EXCHANGE": "occupied",
        "GENERATIVEQC_DF_RESIDENT_EXCHANGE": "full",
        "GENERATIVEQC_DF_RESPONSE_STORAGE": "jk-scratch",
        "GENERATIVEQC_DF_RESPONSE_SPACE": "occupied",
        "GENERATIVEQC_DF_FINAL_PROJECTION": "reuse",
    }.items():
        monkeypatch.setenv(name, value)
    calc = Calculator(
        method=method,
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=100,
    )
    with calc.prepare_batch([case.atoms]) as batch:
        batch.execute(strict=True)
        batch.set_warm_start_updates(False)
        for moved in (False, True):
            coordinates = np.array([r for _, r in case.atoms])
            coordinates[1, 0] += 0.001 if moved else 0
            atoms = [(z, r) for (z, _), r in zip(case.atoms, coordinates, strict=True)]
            mol = gto.M(atom=atoms, unit="Bohr", basis="def2-svp", verbose=0)
            oracle = (scf.RHF if method == "rhf" else scf.UHF)(mol).density_fit(
                auxbasis="def2-svp"
            )
            oracle.conv_tol, oracle.conv_tol_grad, oracle.max_cycle = 1e-12, 1e-10, 100
            oracle.kernel()
            assert oracle.converged
            expected = -oracle.nuc_grad_method().kernel()
            answers = []
            for policy in ("off", "reuse", "reuse"):
                monkeypatch.setenv("GENERATIVEQC_DF_FINAL_PROJECTION", policy)
                trace = tmp_path / f"{moved}-{len(answers)}.jsonl"
                monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
                actual = batch.execute(
                    [coordinates] if moved else None, strict=True
                ).items[0]
                assert actual.energy == pytest.approx(oracle.e_tot, abs=1e-9, rel=0)
                np.testing.assert_allclose(actual.forces, expected, atol=1e-8, rtol=0)
                answers.append(np.asarray(actual.forces))
                rows = [
                    r for r in read_trace(trace) if r["operation"] == "force_response"
                ]
                assert len(rows) == 1
                reused = rows[0]["counters"].get("response_final_projection_reused", 0)
                if method == "uhf" or policy == "off":
                    assert reused == 0
                elif not moved:
                    assert reused == 1
            np.testing.assert_allclose(answers[0], answers[1], atol=1e-10, rtol=0)
            np.testing.assert_allclose(answers[1], answers[2], atol=1e-10, rtol=0)
