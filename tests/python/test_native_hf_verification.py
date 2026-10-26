"""Final HF acceptance, explicit compatibility and native failure boundaries."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from generativeqc import _hf_verification_native as native
from generativeqc import _native
from generativeqc import progressive_controller as controller
from test_progressive_controller import (
    PROBLEM,
    TARGET_HASHES,
    TARGET_MODEL,
    assessment,
    execution,
    plan,
    result,
)


@pytest.fixture
def library() -> Any:
    value = native.optional_library()
    assert value is not None, "build the current native HF verification protocol"
    return value


def finalize(
    output: Any = None, *, library: Any = None, selected: Any = None, **options: Any
) -> controller.FinalVerification:
    """Use the public typed fixture while retaining selected-library provenance."""
    selected = plan() if selected is None else selected
    return controller._finalize_hf_verification(
        PROBLEM,
        selected.stages[1],
        result(accuracy=assessment(observed=True)) if output is None else output,
        TARGET_MODEL,
        TARGET_HASHES,
        selected.budget,
        (
            execution(controller.StageRole.INITIALIZATION),
            execution(controller.StageRole.TARGET),
        ),
        library=library,
        **options,
    )


def forbid_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native decisions and errors must not be repaired by the retired policy."""

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("native verification retried the Python policy")

    monkeypatch.setattr(controller, "_finalize_hf_verification_python", forbidden)


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("energy", float("nan"), "target energy is nonfinite"),
        ("energy_change", 2e-10, "target energy-change tolerance is unmet"),
        ("density_rms", float("inf"), "target density tolerance is unmet"),
        ("physical_residual_rms", None, "target physical residual is unavailable"),
        ("physical_residual_rms", 2e-8, "target physical residual tolerance is unmet"),
        ("converged", False, "target HF did not converge successfully"),
        (
            "precision",
            {"requested_mode": "fp64", "effective_bits": 32},
            "strict final arithmetic was not established",
        ),
    ],
)
def test_actual_measurements_cannot_be_promoted(
    library: Any, monkeypatch: pytest.MonkeyPatch, field: str, value: Any, reason: str
) -> None:
    forbid_compatibility(monkeypatch)
    output = result(accuracy=assessment(observed=True))
    setattr(output, field, value)
    final = finalize(output, library=library)
    assert final.status == "unmet" and not final.target_established
    assert final.accuracy_status == "observed_met"
    assert reason in final.reasons


def test_physical_audit_max_and_budget_precedence(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_compatibility(monkeypatch)
    audit = controller.HFPhysicalResidualAudit(
        TARGET_MODEL.identity,
        PROBLEM.provider_identity,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        2e-8,
        1e-12,
        -1.0,
        0.0,
        0.01,
    )
    final = finalize(
        library=library, physical_audit=audit, physical_audit_budget_exhausted=True
    )
    assert final.status == "budget_exhausted"
    assert final.reasons == (
        "target physical residual tolerance is unmet",
        "physical verification host budget was exhausted",
    )
    assert final.physical_residual_max == 2e-8
    assert final.physical_residual_rms == 1e-12
    assert finalize(
        library=library, physical_audit=replace(audit, maximum_commutator=1e-8)
    ).succeeded


def test_independent_accuracy_is_required_and_parallel_decisions_are_detached(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_compatibility(monkeypatch)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda observed: finalize(
                    result(accuracy=assessment(observed=observed)), library=library
                ),
                [True, False] * 12,
            )
        )
    assert [item.status for item in results] == ["verified", "unverified"] * 12
    assert all(item.target_established for item in results)


def test_selected_old_library_never_discovers_another_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden() -> None:
        raise AssertionError("selected old library searched for a second owner")

    monkeypatch.setattr(native, "optional_library", forbidden)
    assert finalize(library=SimpleNamespace()).succeeded


def test_native_refusal_is_not_a_compatibility_signal(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbid_compatibility(monkeypatch)

    def refusal(*args: Any) -> int:
        return _native.STATUS_INVALID_ARGUMENT

    selected = SimpleNamespace(
        generativeqc_hf_verify_final_v1=refusal,
        generativeqc_status_message=library.generativeqc_status_message,
    )
    with pytest.raises(RuntimeError, match="invalid argument"):
        finalize(library=selected)


def test_arbitrary_python_budget_does_not_wrap_into_native_integer(
    library: Any,
) -> None:
    selected = plan(budget=controller.ProgressiveBudget(maximum_total_iterations=2**80))
    final = finalize(library=library, selected=selected)
    assert final.succeeded and final.total_iterations == 6


def test_missing_force_evidence_stays_unmet(
    library: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shape/finite checks stay typed transport; native owns their disposition."""
    from test_progressive_controller import FORCE_ACCURACY, force_assessment

    forbid_compatibility(monkeypatch)
    problem = replace(PROBLEM, accuracy=FORCE_ACCURACY)
    selected = plan()
    output = result(accuracy=force_assessment())
    output.forces = np.full((2, 3), np.nan)
    final = controller._finalize_hf_verification(
        problem,
        selected.stages[1],
        output,
        TARGET_MODEL,
        TARGET_HASHES,
        selected.budget,
        (),
        library=library,
    )
    assert final.status == "unmet" and "forces" not in final.capabilities
    assert "target result omitted required observables: forces" in final.reasons
