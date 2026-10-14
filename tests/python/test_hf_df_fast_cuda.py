"""Complete fitted HF candidates, including strict metric/auxiliary response.

Only finite Slurm allocations may execute these tests. An independent PySCF
DF model, not direct HF or the candidate's strict sibling, supplies the oracle.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator, PrecisionCalculator, TargetErrorBudget
from generativeqc._model_resolution import snapshot_basis
from generativeqc.elements import SYMBOLS
from test_df_fast_public_cuda import auxiliary_bases
from test_hf_fast_cuda import settings

from benchmarks._cases import benchmark_cases

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1"
    or not os.environ.get("SLURM_JOB_ID"),
    reason="requires a finite Slurm GPU allocation",
)
BUDGET = TargetErrorBudget(1e-5, 1e-4, 3e-5)


def options(case_name: str, *, orbital_auxiliary: bool = False) -> dict[str, Any]:
    """Share all scientific controls and the pinned auxiliary input."""
    auxiliary, _ = auxiliary_bases(benchmark_cases()[case_name])
    return {
        **settings(case_name),
        "density_fitting": "cuda",
        "auxiliary_basis": (
            benchmark_cases()[case_name].generativeqc_basis
            if orbital_auxiliary
            else auxiliary
        ),
    }


def oracle(
    case_name: str, atoms: Any, *, orbital_auxiliary: bool = False
) -> tuple[float, np.ndarray]:
    """Independent metric, SCF, three-center, Pulay and auxiliary derivatives."""
    pyscf = pytest.importorskip("pyscf")
    pyscf.lib.num_threads(4)
    case = benchmark_cases()[case_name]
    _, auxiliary = auxiliary_bases(case)
    basis = snapshot_basis(case.generativeqc_basis, case.basis_representation)
    shells = {
        SYMBOLS[element.atomic_number - 1]: [
            [
                shell.angular_momentum,
                *[
                    [
                        float(exponent),
                        *[float(row[index]) for row in shell.coefficients],
                    ]
                    for index, exponent in enumerate(shell.exponents)
                ],
            ]
            for shell in element.shells
        ]
        for element in basis.elements
    }
    molecule = pyscf.gto.M(
        atom=atoms,
        basis=shells,
        unit="Bohr",
        cart=case.basis_representation == "cartesian",
        verbose=0,
    )
    solver = molecule.RHF().density_fit(
        auxbasis=shells if orbital_auxiliary else auxiliary
    )
    solver.conv_tol = 1e-13
    solver.conv_tol_grad = 1e-10
    solver.max_cycle = 180
    energy = solver.kernel()
    assert solver.converged
    derivative = solver.nuc_grad_method()
    derivative.auxbasis_response = True
    return energy, -derivative.kernel()


def check_result(
    result: Any, expected: tuple[float, np.ndarray], *, mixed: bool
) -> None:
    """Qualification floors are tighter than the default Fast allowance."""
    energy, force = expected
    assert result.converged
    assert abs(result.energy - energy) < 1e-7
    error = result.forces - force
    assert np.max(np.abs(error)) < (1e-5 if mixed else 1e-6)
    assert np.sqrt(np.mean(error**2)) < (3e-6 if mixed else 3e-7)
    np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=2e-8, rtol=0)
    work = (result.precision or {}).get("gradient")
    if not mixed:
        assert work is None
        return
    assert work["qualification"] == "rhf-df-shell-budget-fp32-v1"
    assert work["complete"] and work["returned_mode"] == "mixed"
    assert work["weighted_task_counters_collected"]
    assert work["attempted_fp32_weighted_tasks"] > 0
    assert work["attempted_fp32_primitive_sources"] > 0
    assert 0 <= work["weighted_task_retries"] < work["attempted_fp32_weighted_tasks"]
    assert work["weighted_fp32_orders"] == []
    assert work["attempted_fp32_grid_points"] == 0
    assert "df_metric_and_response_weights" in work["fp64_components"]
    assert "df_shell_sp_bra_contractions" in work["fp64_components"]


@pytest.mark.parametrize("buckets", ("off", "on", "packet"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("orbital_auxiliary", (False, True))
def test_complete_shell_candidate_and_return_to_generic(
    representation: str,
    buckets: str,
    orbital_auxiliary: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Diagnostic scheduling tests arithmetic, not a default endpoint speedup."""
    from test_dft_complete_cuda import no_cpu_derivatives

    name = "water-def2-svp" + ("-spherical" if representation == "spherical" else "")
    atoms = benchmark_cases()[name].atoms
    expected = oracle(name, atoms, orbital_auxiliary=orbital_auxiliary)
    mixed = representation == "spherical" or orbital_auxiliary
    monkeypatch.setenv("GENERATIVEQC_DF_WEIGHTED_EXECUTION", "shell")
    monkeypatch.setenv("GENERATIVEQC_DF_PRIMITIVE_BUCKETS", buckets)
    monkeypatch.setenv("GENERATIVEQC_DF_SHELL_SCHEDULE", "compact")
    calculator = Calculator(
        **options(name, orbital_auxiliary=orbital_auxiliary),
        precision="fp64",
        _application_precision_budget=BUDGET,
    )
    try:
        with calculator.prepare_batch([atoms]) as batch:
            with no_cpu_derivatives():
                candidate = batch.execute(strict=True).items[0]
            check_result(candidate, expected, mixed=mixed)
            if not mixed:
                (metric,) = batch.last_density_fitting_metric_diagnostics()
                assert metric.effective_rank < metric.auxiliary_tile
            batch.set_warm_start_updates(False)
            with no_cpu_derivatives():
                replay = batch.execute(strict=True).items[0]
            check_result(replay, expected, mixed=mixed)
            monkeypatch.setenv("GENERATIVEQC_DF_WEIGHTED_EXECUTION", "generic")
            with no_cpu_derivatives():
                strict = batch.execute(strict=True).items[0]
            check_result(strict, expected, mixed=False)
            assert (
                batch.execute(properties=("energy",), strict=True)
                .items[0]
                .precision.get("gradient")
                is None
            )
    finally:
        calculator.clear_cache()


def test_public_hf_df_fast_cold_frozen_moved() -> None:
    """Default schedule selects real FP32, with no precision environment override."""
    from test_dft_complete_cuda import no_cpu_derivatives

    name = "water-tetramer-def2-svp-spherical"
    atoms = benchmark_cases()[name].atoms
    moved = [(symbol, np.asarray(position).copy()) for symbol, position in atoms]
    moved[1][1][0] += 0.025
    expected = [oracle(name, geometry) for geometry in (atoms, moved)]
    positions = np.array([position for _, position in moved])
    strict = Calculator(**options(name), precision="fp64")
    fast = PrecisionCalculator(**options(name), policy="fast")
    try:
        with no_cpu_derivatives():
            baseline = strict.singlepoint(atoms)
        check_result(baseline, expected[0], mixed=False)
        with fast.prepare_batch([atoms]) as batch:
            for geometry, reference in (
                (None, expected[0]),
                ([positions], expected[1]),
            ):
                batch.set_warm_start_updates(True)
                for frozen in (False, True, True):
                    batch.set_warm_start_updates(not frozen)
                    with no_cpu_derivatives():
                        result = batch.execute(geometry, strict=True).items[0]
                    check_result(result, reference, mixed=True)
                    assert result.precision_validation.solve_count == 1
                    assert (
                        result.precision_validation.status == "qualified_mixed_gradient"
                    )
                    assert (
                        "fp32_grid_storage_unavailable"
                        not in result.precision_validation.fallback_reasons
                    )
            energy = batch.execute(properties=("energy",), strict=True).items[0]
            assert (energy.precision or {}).get("gradient") is None
    finally:
        strict.clear_cache()
        fast.clear_cache()


def test_tight_budget_keeps_strict_df_shells(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shell schedule request cannot overrule the immutable application budget."""
    name = "water-def2-svp-spherical"
    atoms = benchmark_cases()[name].atoms
    expected = oracle(name, atoms)
    monkeypatch.setenv("GENERATIVEQC_DF_WEIGHTED_EXECUTION", "shell")
    calculator = Calculator(
        **options(name),
        precision="fp64",
        _application_precision_budget=TargetErrorBudget(1e-7, 1e-6, 3e-7),
    )
    try:
        check_result(calculator.singlepoint(atoms), expected, mixed=False)
    finally:
        calculator.clear_cache()


def test_failed_peer_and_energy_only_do_not_inherit_df_precision() -> None:
    """Final-density authorization belongs to one successful fleet item/solve."""
    name = "water-tetramer-def2-svp-spherical"
    atoms = benchmark_cases()[name].atoms
    expected = oracle(name, atoms)
    calculator = PrecisionCalculator(**options(name), policy="fast")
    try:
        with calculator.prepare_batch([atoms, atoms]) as batch:
            for result in batch.execute(strict=True).items:
                check_result(result, expected, mixed=True)
            failed = batch.execute([None, np.zeros((1, 3))], strict=False)
            assert failed.failure_indices == (1,)
            assert (failed.items[1].precision or {}).get("gradient") is None
            check_result(failed.items[0], expected, mixed=True)
            for result in batch.execute(properties=("energy",), strict=True).items:
                assert (result.precision or {}).get("gradient") is None
    finally:
        calculator.clear_cache()
