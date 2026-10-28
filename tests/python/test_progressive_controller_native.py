"""Native HF endpoint coverage for the typed #192 deterministic controller."""

from typing import Any

import pytest
from generativeqc import Calculator
from generativeqc.accuracy import ObservableTarget, TargetAccuracy
from generativeqc.progressive_controller import (
    ProgressiveBudget,
    TargetProblem,
    make_deterministic_hf_plan,
    run_progressive_hf,
)

ATOMS = (("H", (0.0, 0.0, -0.7)), ("H", (0.1, 0.0, 0.7)))
ACCURACY = TargetAccuracy(
    (
        ObservableTarget("energy", "absolute", "Eh", absolute=1.0e-8),
        ObservableTarget("forces", "max_abs", "Eh/bohr", absolute=1.0e-7),
    )
)


def calculators(
    *,
    source_iterations: int = 20,
    method: str = "rhf",
    fitted: bool = False,
) -> tuple[Calculator, Calculator]:
    source = Calculator(
        basis="sto-3g",
        method=method,
        max_iterations=source_iterations,
        energy_tolerance=1.0e-7,
        density_tolerance=1.0e-6,
    )
    target = Calculator(
        basis="def2-svp",
        method=method,
        density_fitting="cpu" if fitted else "none",
        max_iterations=100,
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
        screening_tolerance=1.0e-14,
        target_accuracy=ACCURACY,
    )
    return source, target


@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("method,charge,multiplicity", [("rhf", 0, 1), ("uhf", 1, 2)])
def test_typed_plan_reaches_exact_target_without_fabricating_accuracy(
    fitted: bool, method: str, charge: int, multiplicity: int
) -> None:
    source, target = calculators(method=method, fitted=fitted)
    problem = TargetProblem.from_calculator(
        target, ATOMS, charge=charge, multiplicity=multiplicity
    )
    plan = make_deterministic_hf_plan(
        problem,
        source,
        target,
        ATOMS,
        charge=charge,
        multiplicity=multiplicity,
        budget=ProgressiveBudget(
            maximum_source_iterations=20,
            maximum_total_iterations=120,
            maximum_estimated_cost_units=5.0,
        ),
        source_estimated_cost_units=1.0,
        target_estimated_cost_units=4.0,
    )
    run = run_progressive_hf(
        plan,
        source,
        target,
        ATOMS,
        charge=charge,
        multiplicity=multiplicity,
    )
    assert run.target.converged and run.target.restart_origin == "basis_projection"
    assert run.verification.target_established
    assert run.verification.status == "unverified"
    assert run.verification.accuracy_status == "unverified"
    assert run.verification.physical_residual_source == "fixed_density_target_audit"
    assert run.verification.physical_residual_max is not None
    assert run.verification.actual_model_identity == problem.model.identity
    assert run.verification.actual_provider_identity == problem.provider_identity
    assert run.verification.physical_residual_rms is not None
    assert (
        run.verification.physical_residual_rms
        <= run.verification.physical_residual_tolerance
    )
    assert run.diagnostics["projection"]["status"] == "accepted"
    assert run.diagnostics["physical_residual_audit"]["maximum_commutator"] < 1e-10
    timed_parts = (
        "source_setup_seconds",
        "source_execution_seconds",
        "projection_seconds",
        "target_setup_seconds",
        "target_execution_seconds",
        "final_verification_seconds",
        "cleanup_seconds",
    )
    assert all(run.diagnostics[name] >= 0 for name in timed_parts)
    assert run.diagnostics["total_seconds"] >= sum(
        run.diagnostics[name] for name in timed_parts
    )
    assert run.target_density is not None and not run.target_density.flags.writeable


@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("method,charge,multiplicity", [("rhf", 0, 1), ("uhf", 1, 2)])
def test_native_schedule_target_energy_and_forces_against_pyscf(
    fitted: bool,
    method: str,
    charge: int,
    multiplicity: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qualify physical target observables, independently of callback routing."""
    import numpy as np

    pyscf = pytest.importorskip("pyscf")
    from generativeqc import progressive
    from generativeqc import progressive_controller as controller

    def forbidden_audit(*args: object, **kwargs: object) -> None:
        raise AssertionError("native progressive endpoint used Python physical audit")

    monkeypatch.setattr(
        controller, "_audit_target_physical_residual_python", forbidden_audit
    )

    def forbidden_policy(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("native progressive endpoint used Python final acceptance")

    monkeypatch.setattr(
        controller, "_finalize_hf_verification_python", forbidden_policy
    )
    monkeypatch.setattr(progressive, "_initialize_from_compatibility", forbidden_policy)
    source, target = calculators(method=method, fitted=fitted)
    problem = TargetProblem.from_calculator(
        target, ATOMS, charge=charge, multiplicity=multiplicity
    )
    plan = make_deterministic_hf_plan(
        problem,
        source,
        target,
        ATOMS,
        charge=charge,
        multiplicity=multiplicity,
    )
    result = run_progressive_hf(
        plan, source, target, ATOMS, charge=charge, multiplicity=multiplicity
    )
    assert result.diagnostics["execution_owner"] == "native-v1"
    assert result.verification.target_established
    molecule = pyscf.gto.M(
        atom=ATOMS,
        basis="def2-svp",
        unit="Bohr",
        cart=True,
        charge=charge,
        spin=multiplicity - 1,
        verbose=0,
    )
    reference = pyscf.scf.RHF(molecule) if method == "rhf" else pyscf.scf.UHF(molecule)
    if fitted:
        reference = reference.density_fit(auxbasis="def2-svp")
    reference.conv_tol = 1.0e-12
    reference.conv_tol_grad = 1.0e-10
    reference.max_cycle = 100
    reference.kernel()
    assert reference.converged
    assert result.target.energy == pytest.approx(reference.e_tot, abs=1.0e-8)
    np.testing.assert_allclose(
        result.target.forces, -reference.nuc_grad_method().kernel(), atol=1.0e-6, rtol=0
    )


def test_failed_source_falls_back_to_cold_exact_target() -> None:
    source, target = calculators(source_iterations=1)
    problem = TargetProblem.from_calculator(target, ATOMS)
    plan = make_deterministic_hf_plan(
        problem,
        source,
        target,
        ATOMS,
        budget=ProgressiveBudget(
            maximum_source_iterations=1,
            maximum_total_iterations=101,
        ),
    )
    run = run_progressive_hf(plan, source, target, ATOMS)
    assert not run.source.succeeded
    assert run.executions[0].transfer_status == "skipped_failed_source"
    assert run.target.converged and run.target.restart_origin == "cold"
    assert run.verification.target_established


def test_missing_symbol_compatibility_preserves_complete_target() -> None:
    """An old-library transport must preserve physical execution and work."""
    import numpy as np

    source, target = calculators()
    problem = TargetProblem.from_calculator(target, ATOMS)
    plan = make_deterministic_hf_plan(problem, source, target, ATOMS)
    native = run_progressive_hf(plan, source, target, ATOMS)
    selected = target._library

    class OlderLibrary:
        def __getattr__(self, name: str) -> object:
            if name == "generativeqc_progressive_hf_run_v1":
                raise AttributeError(name)
            return getattr(selected, name)

    target._library = OlderLibrary()
    compatible = run_progressive_hf(plan, source, target, ATOMS)
    assert compatible.diagnostics["execution_owner"] == "portable-compatibility"
    assert compatible.target.energy == native.target.energy
    np.testing.assert_array_equal(compatible.target.forces, native.target.forces)
    np.testing.assert_array_equal(compatible.target_density, native.target_density)
    assert compatible.verification == native.verification
    assert [(item.iterations, item.fock_builds) for item in compatible.executions] == [
        (item.iterations, item.fock_builds) for item in native.executions
    ]


def test_nonconverged_target_never_exports_density_or_audit() -> None:
    source, target = calculators()
    target._max_iterations = 1
    problem = TargetProblem.from_calculator(target, ATOMS)
    plan = make_deterministic_hf_plan(problem, source, target, ATOMS)
    result = run_progressive_hf(plan, source, target, ATOMS)
    assert not result.target.succeeded
    assert not result.target.converged
    assert result.target_density is None
    assert not result.verification.target_established
    assert result.diagnostics["physical_residual_audit"]["status"] == "unavailable"


def test_projection_memory_budget_rejection_falls_back_to_target() -> None:
    source, target = calculators()
    problem = TargetProblem.from_calculator(target, ATOMS)
    plan = make_deterministic_hf_plan(
        problem,
        source,
        target,
        ATOMS,
        budget=ProgressiveBudget(
            maximum_source_iterations=20,
            maximum_total_iterations=120,
            maximum_host_bytes=1,
        ),
    )
    run = run_progressive_hf(plan, source, target, ATOMS)
    assert run.source.succeeded
    assert run.executions[0].transfer_status == "rejected"
    assert "maximum_host_bytes" in run.diagnostics["projection"]["reason"]
    assert run.target.converged and run.target.restart_origin == "cold"
    assert run.verification.target_established


@pytest.mark.parametrize("fitted", [False, True])
@pytest.mark.parametrize("provider_limit", [False, True])
def test_verification_budget_rejects_before_fock_rebuild(
    fitted: bool, provider_limit: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import generativeqc.progressive_controller as controller

    source, target = calculators(fitted=fitted)
    problem = TargetProblem.from_calculator(target, ATOMS)
    # The provider alone fits its own peak, but the additional audit buffers
    # must also fit. This exercises real planning, not only the one-byte guard.
    limit = (
        target.estimate_resources([ATOMS]).peak_bytes["host"] if provider_limit else 1
    )
    plan = make_deterministic_hf_plan(
        problem,
        source,
        target,
        ATOMS,
        budget=ProgressiveBudget(maximum_verification_host_bytes=limit),
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("verification Fock rebuilt despite exhausted workspace budget")

    monkeypatch.setattr(controller, "FockPlan", forbidden)
    run = run_progressive_hf(plan, source, target, ATOMS)
    assert run.target.converged
    assert run.verification.status == "budget_exhausted"
    assert not run.verification.target_established
    assert not run.succeeded
    assert run.diagnostics["physical_residual_audit"]["status"] == "unavailable"
    assert "budget" in run.diagnostics["physical_residual_audit"]["reason"]
