"""Execute the public policy ABI and the Python production-consumer boundary."""

from __future__ import annotations

import ctypes
import math
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from generativeqc import TargetErrorBudget
from generativeqc import _precision_native as transport
from generativeqc import precision_policy as policy

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def library(required_native_cxx: Any, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Compile the actual public adapter; no surrogate policy implementation."""
    directory = tmp_path_factory.mktemp("precision-admission")
    path = required_native_cxx.build_shared(
        [ROOT / "src/api/c_api_precision_admission.cpp"],
        directory / "policy.so",
        compile_args=(
            "-std=c++20",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
    )
    value = ctypes.CDLL(str(path))
    value.generativeqc_status_message = lambda status: b"policy rejected"
    return value


def test_public_sdk_contract(required_native_cxx: Any, tmp_path: Path) -> None:
    executable = required_native_cxx.build_executable(
        [
            ROOT / "tests/native/test_precision_admission.cpp",
            ROOT / "src/api/c_api_precision_admission.cpp",
        ],
        tmp_path / "policy",
        compile_args=(
            "-std=c++20",
            "-I",
            str(ROOT / "include"),
            "-I",
            str(ROOT / "src"),
        ),
    )
    subprocess.run(
        [str(executable)], check=True, timeout=30, env={"PATH": "/nonexistent"}
    )


def context(library: Any) -> Any:
    """A pure semilocal completed model with strict explicit caller controls."""
    return SimpleNamespace(
        _library=library,
        _ks_options=SimpleNamespace(
            _method_ir=object(),
            coefficients=(0, 0, 0),
            has_range_exchange=False,
            execution_plan=SimpleNamespace(nonlocal_correlation=False),
        ),
        capabilities=SimpleNamespace(family="kohn_sham"),
        _density_fitting_mode=None,
        _energy_tolerance=1e-10,
        _density_tolerance=1e-8,
        _screening_tolerance=1e-12,
    )


def work() -> dict:
    """Complete strict evidence at one independently named returned state."""
    state = "cuda-ks:17:5:9"
    return {
        "requested_mode": "fp64",
        "effective_bits": 64,
        "operator_work_counters_valid": True,
        "complete": True,
        "operator_inventory_complete": True,
        "strict_stage_fock_builds": 2,
        "mixed_stage_fock_builds": 0,
        "final_residual_audits": 1,
        "strict_refinement_applied": True,
        "returned_state_identity": state,
        "scf_fock_timeline": [
            {"kind": name, "state": state, "count": 1}
            for name in ("strict_fock", "final_audit")
        ],
        "operators": [
            {
                "name": name,
                "storage": "fp64",
                "compute": "fp64",
                "accumulation": "fp64",
                "reduction": "fp64",
                "arithmetic_mode": "strict",
                "count": 1,
            }
            for name in ("coulomb_j", "xc", "fock_assembly", "physical_residual")
        ],
    }


def result(evidence: dict) -> Any:
    return SimpleNamespace(
        precision=evidence,
        warm_start_used=True,
        warm_start_fallback=False,
        converged=True,
        executed_backend="cuda",
    )


def forbid_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("selected native policy retried Python decisions")

    for name in ("budget_fock_work", "budget_fock_strict_warm", "standard_fallback"):
        monkeypatch.setattr(policy, f"_compatibility_{name}_admitted", forbidden)


@pytest.mark.parametrize(
    "missing",
    (
        None,
        "operator_work_counters_valid",
        "complete",
        "operator_inventory_complete",
        "mixed_stage_fock_builds",
        "final_residual_audits",
        "returned_state_identity",
    ),
)
def test_warm_requires_complete_returned_state(
    library: Any, monkeypatch: Any, missing: str | None
) -> None:
    forbid_compatibility(monkeypatch)
    evidence = work()
    if missing:
        evidence.pop(missing)
    assert policy._budget_fock_strict_warm_admitted(
        context(library), result(evidence)
    ) is (missing is None)


@pytest.mark.parametrize(
    "replacement", ("cuda-ks:18:5:9", "cuda-ks:17:6:9", "cuda-ks:17:5:10", None)
)
def test_old_audit_does_not_certify_returned_density(
    library: Any, replacement: str | None
) -> None:
    evidence = work()
    evidence["scf_fock_timeline"][1]["state"] = replacement
    assert not policy._budget_fock_strict_warm_admitted(
        context(library), result(evidence)
    )


@pytest.mark.parametrize(
    "trait,operator",
    (
        ("full_range_exchange", "exchange_k"),
        ("range_exchange", "exchange_k"),
        ("nonlocal", "nonlocal_correlation"),
    ),
)
def test_complete_model_requires_its_executed_operators(
    library: Any, trait: str, operator: str
) -> None:
    calculator, evidence = context(library), work()
    if trait == "full_range_exchange":
        calculator._ks_options.coefficients = (0, 0, 0.25)
    elif trait == "range_exchange":
        calculator._ks_options.has_range_exchange = True
    else:
        calculator._ks_options.execution_plan.nonlocal_correlation = True
    assert not policy._budget_fock_strict_warm_admitted(calculator, result(evidence))
    evidence["operators"].append({**evidence["operators"][0], "name": operator})
    assert policy._budget_fock_strict_warm_admitted(calculator, result(evidence))


def test_budget_fock_requires_observed_mixed_work(
    library: Any, monkeypatch: Any
) -> None:
    forbid_compatibility(monkeypatch)
    calculator, evidence = context(library), work()
    assert not policy._budget_fock_work_admitted(calculator, evidence)
    evidence["operators"][1].update(compute="fp32", arithmetic_mode="mixed")
    assert policy._budget_fock_work_admitted(calculator, evidence)
    calculator._ks_options.coefficients = (0, 0, 0.25)
    assert not policy._budget_fock_work_admitted(calculator, evidence)
    for name in ("coulomb_j", "coulomb_recurrence", "exchange_k"):
        evidence["operators"].append({**evidence["operators"][1], "name": name})
    assert policy._budget_fock_work_admitted(calculator, evidence)
    evidence["strict_refinement_applied"] = False
    assert not policy._budget_fock_work_admitted(calculator, evidence)


@pytest.mark.parametrize("valid", (True, False, None))
def test_fitted_hf_preserves_unknown_counter_boundary(
    library: Any, monkeypatch: Any, valid: Any
) -> None:
    forbid_compatibility(monkeypatch)
    calculator, evidence = context(library), work()
    calculator._ks_options = None
    calculator.capabilities.family = "hartree_fock"
    calculator._density_fitting_mode = "cuda"
    evidence["operator_work_counters_valid"] = valid
    assert policy._standard_fallback_admitted(
        calculator, result(evidence), "fp64", TargetErrorBudget(1e-5, 1e-4)
    ) is (valid is not None)


@pytest.mark.parametrize(
    "field,limit", (("energy_abs", 1e-5), ("force_max_abs", 1e-4), ("force_rms", 3e-5))
)
def test_standard_budget_neighbours(library: Any, field: str, limit: float) -> None:
    for value, expected in (
        (math.nextafter(limit, 0), False),
        (limit, True),
        (math.nextafter(limit, math.inf), True),
    ):
        budget = {
            "energy_abs": 1e-5,
            "force_max_abs": 1e-4,
            "force_rms": 3e-5,
            field: value,
        }
        assert (
            policy._standard_fallback_admitted(
                context(library), result(work()), "fp64", TargetErrorBudget(**budget)
            )
            is expected
        )


def test_selected_failure_and_malformed_receipt_cannot_retry(
    library: Any, monkeypatch: Any
) -> None:
    forbid_compatibility(monkeypatch)
    evidence = work()
    evidence["detail_version"] = 999
    with pytest.raises(NotImplementedError):
        policy._budget_fock_work_admitted(context(library), evidence)
    query = SimpleNamespace(generativeqc_precision_admission_query_v1=lambda *args: 0)
    with pytest.raises(RuntimeError, match="invalid receipt"):
        policy._budget_fock_work_admitted(context(query), work())


def test_missing_protocol_retains_compatibility() -> None:
    calculator = context(SimpleNamespace())
    assert transport.admission(calculator, work()) is None
    assert policy._budget_fock_strict_warm_admitted(calculator, result(work()))


def test_policy_queries_do_not_mutate_evidence(library: Any) -> None:
    evidence = work()
    before = deepcopy(evidence)
    transport.admission(context(library), evidence, result=result(evidence))
    assert evidence == before


@pytest.mark.parametrize(
    "field,value",
    (
        ("effective_bits", 2**32 + 64),
        ("detail_version", 2**32 + 1),
        ("strict_stage_fock_builds", -1),
        ("mixed_stage_fock_builds", 2**64),
    ),
)
def test_detached_evidence_cannot_wrap_into_admissible_values(
    library: Any, field: str, value: int
) -> None:
    """C integer conversion must not turn malformed evidence into valid proof."""
    evidence = work()
    evidence[field] = value
    with pytest.raises(ValueError, match="precision evidence must be"):
        transport.admission(context(library), evidence, result=result(evidence))
