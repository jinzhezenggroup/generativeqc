"""Observable composition, failure, snapshot and lifetime contracts for CBS."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc import CBSBasis, CCSDTCBSCalculator, Result
from generativeqc import cbs as cbs_module

if TYPE_CHECKING:
    from generativeqc import BasisSet


ATOMS = (("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4)))


def test_full_df_groups_preserve_the_explicit_reference_model(
    bases: tuple[CBSBasis, CBSBasis], fake_native: list[dict]
) -> None:
    workflow = CCSDTCBSCalculator(
        bases,
        auxiliary_bases=("sto-3g", "def2-svp"),
        hf_reference="df",
        device="cuda",
    )
    result = workflow.singlepoint(ATOMS)
    assert result.basis_metadata["hf_reference"] == "df"
    assert len(fake_native) == len(result.execution_groups) == 2
    assert all(
        event["options"]["density_fitting"] == "full_cuda" for event in fake_native
    )
    assert all(event["closed"] for event in fake_native)


@pytest.mark.parametrize("reference", ("df", "guess", None))
def test_df_reference_requires_a_valid_model_and_auxiliaries(
    bases: tuple[CBSBasis, CBSBasis], reference: str | None
) -> None:
    with pytest.raises(ValueError):
        CCSDTCBSCalculator(bases, hf_reference=reference)


def test_df_cbs_same_reference_groups_and_force_weights(
    bases: tuple[CBSBasis, CBSBasis], fake_native: list[dict]
) -> None:
    workflow = CCSDTCBSCalculator(
        bases, device="cuda", auxiliary_bases=("sto-3g", "def2-svp")
    )
    result = workflow.singlepoint(ATOMS)
    assert len(fake_native) == len(result.execution_groups) == 2
    assert fake_native[0]["method"] == "df-ccsd(t)"
    assert fake_native[0]["options"]["density_fitting"] == "cuda"
    assert fake_native[1]["method"] == "mp2"
    assert fake_native[1]["options"]["density_fitting"] == "correlation_cuda"
    assert all(event["closed"] for event in fake_native)
    assert result.energy == pytest.approx(
        sum(
            component.coefficient * component.result.energy
            for component in result.components
        )
    )
    np.testing.assert_allclose(
        result.forces,
        sum(
            component.coefficient * component.result.forces
            for component in result.components
        ),
    )
    correction_mp2 = next(
        component
        for component in result.components
        if "correction_mp2" in component.roles
    )
    assert correction_mp2.roles == ("mp2_low", "correction_mp2")


@pytest.mark.parametrize(
    "options,error,match",
    (
        ({"auxiliary_bases": ("sto-3g", "def2-svp")}, NotImplementedError, "CUDA"),
        (
            {"device": "cuda", "auxiliary_bases": ("sto-3g",)},
            ValueError,
            "two explicit",
        ),
        (
            {
                "device": "cuda",
                "auxiliary_bases": ("sto-3g", "def2-svp"),
                "correction_auxiliary_basis": "def2-svp",
            },
            ValueError,
            "share their auxiliary",
        ),
        (
            {"device": "cuda", "correction_auxiliary_basis": "sto-3g"},
            ValueError,
            "requires auxiliary_bases",
        ),
    ),
)
def test_df_cbs_auxiliary_contract(
    bases: tuple[CBSBasis, CBSBasis], options: dict, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        CCSDTCBSCalculator(bases, **options)


@pytest.fixture
def bases() -> tuple[CBSBasis, CBSBasis]:
    """Named bases here test orchestration, not a physical CBS basis sequence."""
    return CBSBasis("sto-3g", 2), CBSBasis("def2-svp", 3)


@pytest.fixture
def fake_native(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    from generativeqc import _native
    from generativeqc._api_types import CcPerformanceResult

    events = []

    class FakeCalculator:
        def __init__(self, method: str, basis: BasisSet, **options: object) -> None:
            self.method = method
            self.basis = basis
            self.options = options
            self._correlation_preconvergence = options.get(
                "_correlation_preconvergence"
            )
            self.event = {"method": method, "basis": basis.identity, "closed": False}
            events.append(self.event)
            assert all(event["closed"] for event in events[:-1])

        def singlepoint(
            self, atoms: tuple, *, properties: tuple[str, ...], **kwargs: object
        ) -> Result:
            if self.method == "ccsd(t)" and kwargs.get("charge") == 2:
                raise RuntimeError("injected CC failure")
            basis_scale = 1.0 if self.basis.name == "sto-3g" else 2.0
            method_scale = {"rhf": 1.0, "mp2": 1.2, "ccsd(t)": 1.3, "df-ccsd(t)": 1.3}[
                self.method
            ]
            scale = basis_scale * method_scale
            coordinates = np.asarray([atom.position for atom in atoms])
            self.event["properties"] = properties
            self.event["options"] = self.options
            requested = self._correlation_preconvergence == "mixed" or (
                self._correlation_preconvergence == "auto"
                and "_correlation_convergence_budget" in self.options
            )
            # Model a valid optional-provider refusal: the request is copied,
            # no FP32 work executes, and the numerical owner still completes.
            performance = (
                CcPerformanceResult(
                    **{
                        name: 0
                        for name, _ in _native.CcPerformanceDiagnostic._fields_
                        if name not in ("struct_size", "abi_version")
                    },
                    ccsd_fp32_requested=requested,
                    ccsd_fp32_executed=False,
                    ccsd_fp32_refused=requested,
                    ccsd_fp32_gemm_calls=0,
                    lambda_fp32_requested=requested if "forces" in properties else None,
                    lambda_fp32_executed=False,
                    lambda_fp32_gemm_calls=0,
                )
                if self.method == "df-ccsd(t)"
                else None
            )
            return Result(
                energy=float(-scale + scale * np.sum(coordinates**2)),
                forces=-2.0 * scale * coordinates if "forces" in properties else None,
                converged=True,
                iterations=2,
                energy_change=1e-14,
                density_rms=1e-12,
                executed_backend="cuda"
                if self.options["device"] == "cuda"
                else "cpu_reference",
                cc_performance=performance,
            )

        def clear_cache(self) -> None:
            self.event["closed"] = True

        def singlepoint_components(
            self, atoms: tuple, *, companions: tuple[str, ...], **kwargs: object
        ) -> dict[str, Result]:
            primary = self.method
            results = {}
            try:
                for method in (primary, *companions):
                    self.method = method
                    results[method] = self.singlepoint(atoms, **kwargs)
            finally:
                self.method = primary
            return results

    monkeypatch.setattr(cbs_module, "Calculator", FakeCalculator)
    return events


@pytest.mark.parametrize("hf_scheme", ("exponential", "highest"))
def test_complete_energy_and_force_use_same_weights(
    bases: tuple[CBSBasis, CBSBasis], fake_native: list[dict], hf_scheme: str
) -> None:
    workflow = CCSDTCBSCalculator(bases, device="cuda", hf_extrapolation=hf_scheme)
    result = workflow.singlepoint(iter(ATOMS))
    np.testing.assert_allclose(
        workflow.mp2_weights, (-8 / 19, 27 / 19), atol=1e-15, rtol=0
    )
    if hf_scheme == "exponential":
        ratio = math.exp(-1.63)
        np.testing.assert_allclose(
            workflow.hf_weights, (-ratio / (1 - ratio), 1 / (1 - ratio)), atol=1e-15
        )
    assert result.energy == pytest.approx(
        sum(
            component.coefficient * component.result.energy
            for component in result.components
        ),
        abs=1e-14,
    )
    np.testing.assert_allclose(
        result.forces,
        sum(
            component.coefficient * component.result.forces
            for component in result.components
        ),
        atol=1e-14,
        rtol=0,
    )
    assert result.energy == pytest.approx(
        result.hf_energy + result.mp2_correlation_energy + result.delta_ccsdt_energy
    )
    np.testing.assert_allclose(
        result.forces,
        result.hf_forces + result.mp2_correlation_forces + result.delta_ccsdt_forces,
    )
    assert result.executed_backend == "cuda" and result.converged
    assert len(result.components) == 5
    assert len(fake_native) == len(result.execution_groups) == 2
    assert all(event["closed"] for event in fake_native)
    reused = next(
        component
        for component in result.components
        if "correction_mp2" in component.roles
    )
    assert reused.roles == ("mp2_low", "correction_mp2")
    assert result.endpoint_seconds >= sum(
        group["elapsed_seconds"] for group in result.execution_groups
    )
    assert result.basis_metadata["frozen_core"] == 0


def test_explicit_third_correction_basis_and_energy_only(
    bases: tuple[CBSBasis, CBSBasis], fake_native: list[dict]
) -> None:
    result = CCSDTCBSCalculator(bases, correction_basis="def2-tzvp").singlepoint(
        ATOMS, properties=("energy",)
    )
    assert len(result.components) == 6
    assert (
        result.forces
        is result.hf_forces
        is result.mp2_correlation_forces
        is result.delta_ccsdt_forces
        is None
    )
    assert all(
        event["properties"] == ("energy",) and event["closed"] for event in fake_native
    )


def test_failure_releases_owner_and_next_geometry_can_complete(
    bases: tuple[CBSBasis, CBSBasis], fake_native: list[dict]
) -> None:
    workflow = CCSDTCBSCalculator(bases)
    with pytest.raises(RuntimeError, match="injected CC failure"):
        workflow.singlepoint(
            (("He", (0.0, 0.0, 0.0)), ("He", (0.0, 0.0, 1.4))), charge=2
        )
    assert all(event["closed"] for event in fake_native)
    assert workflow.singlepoint(ATOMS).converged


@pytest.mark.parametrize(
    "options",
    (
        {"density_fitting": "ri"},
        {"ccsd_frozen_core": 1},
        {"precision": "fp32"},
        {"method": "rhf"},
        {"unknown": True},
    ),
)
def test_inconsistent_hamiltonian_or_unknown_options_reject_before_native(
    bases: tuple[CBSBasis, CBSBasis], fake_native: list[dict], options: dict
) -> None:
    with pytest.raises(ValueError, match="unsupported CBS calculator options"):
        CCSDTCBSCalculator(bases, calculator_options=options)
    assert not fake_native


@pytest.mark.parametrize(
    "parameter,value",
    (
        ("correlation_exponent", 0.0),
        ("correlation_exponent", float("nan")),
        ("hf_exponent", float("inf")),
        ("hf_exponent", True),
        ("hf_extrapolation", "fit-at-each-geometry"),
        ("device", "auto"),
        ("device_id", -1),
    ),
)
def test_invalid_configuration(
    bases: tuple[CBSBasis, CBSBasis], parameter: str, value: object
) -> None:
    with pytest.raises(ValueError):
        CCSDTCBSCalculator(bases, **{parameter: value})


def test_basis_contracts(bases: tuple[CBSBasis, CBSBasis]) -> None:
    with pytest.raises(ValueError, match="increasing"):
        CCSDTCBSCalculator(tuple(reversed(bases)))
    with pytest.raises(ValueError, match="distinct"):
        CCSDTCBSCalculator((bases[0], CBSBasis(bases[0].basis, 3)))
    with pytest.raises(ValueError, match="representation"):
        CCSDTCBSCalculator(
            bases, correction_basis=replace(bases[0].basis, representation="cartesian")
        )
    with pytest.raises(ValueError, match="cardinal"):
        CBSBasis("sto-3g", True)
    with pytest.raises(NotImplementedError, match="all-electron"):
        element = replace(bases[0].basis.elements[0], nuclear_charge=2)
        CBSBasis(replace(bases[0].basis, elements=(element,)), 2)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"multiplicity": 3},
        {"multiplicity": True},
        {"charge": True},
        {"properties": ()},
        {"properties": ("forces",)},
        {"properties": ("energy", "hessian")},
        {"properties": "energy"},
    ),
)
def test_invalid_execution_rejects_before_native(
    bases: tuple[CBSBasis, CBSBasis], fake_native: list[dict], kwargs: dict
) -> None:
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        CCSDTCBSCalculator(bases).singlepoint(ATOMS, **kwargs)
    assert not fake_native


def test_configuration_snapshots_caller_options(
    bases: tuple[CBSBasis, CBSBasis], fake_native: list[dict]
) -> None:
    options = {"energy_tolerance": 1e-12}
    workflow = CCSDTCBSCalculator(bases, calculator_options=options)
    options["method"] = "rhf"
    result = workflow.singlepoint(ATOMS)
    saved = result.forces.copy()
    result.components[0].result.forces.fill(99)
    np.testing.assert_array_equal(result.forces, saved)
    assert all(event["closed"] for event in fake_native)


@pytest.mark.parametrize(
    "change",
    (
        {"energy": float("nan")},
        {"forces": None},
        {"forces": np.zeros((1, 3))},
        {"forces": np.full((2, 3), float("inf"))},
        {"converged": False},
        {"executed_backend": "cpu_reference"},
    ),
)
def test_invalid_completed_component_never_publishes_and_closes_owner(
    bases: tuple[CBSBasis, CBSBasis],
    fake_native: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    change: dict,
) -> None:
    original = cbs_module.Calculator.singlepoint

    def invalid(self: object, *args: object, **kwargs: object) -> Result:
        return replace(original(self, *args, **kwargs), **change)

    monkeypatch.setattr(cbs_module.Calculator, "singlepoint", invalid)
    with pytest.raises(RuntimeError, match="CBS"):
        CCSDTCBSCalculator(bases, device="cuda").singlepoint(ATOMS)
    assert all(event["closed"] for event in fake_native)


@pytest.mark.parametrize("correction", (None, "def2-svp", "def2-tzvp"))
@pytest.mark.parametrize("hf_scheme", ("exponential", "highest"))
@pytest.mark.parametrize("policy", ("standard", "fast"))
def test_routine_allocates_combined_component_budget(
    bases: tuple[CBSBasis, CBSBasis],
    fake_native: list[dict],
    correction: str | None,
    hf_scheme: str,
    policy: str,
) -> None:
    """Duplicate MP2 roles share one error; arbitrary weights must be bounded."""
    result = CCSDTCBSCalculator(
        bases,
        policy=policy,
        correction_basis=correction,
        hf_extrapolation=hf_scheme,
        correlation_exponent=2.5,
    ).singlepoint(ATOMS)
    assert len(result.execution_groups) == (3 if correction == "def2-tzvp" else 2)
    amplification = sum(abs(c.coefficient) for c in result.components)
    assert result.convergence["coefficient_l1"] == pytest.approx(amplification)
    assert result.convergence["selected"] == "relaxed"
    assert result.convergence["arithmetic"] == "fp64"
    for event in fake_native:
        options = event["options"]
        budget = options["_correlation_convergence_budget"]
        assert budget.energy_abs * amplification == pytest.approx(1e-5)
        assert budget.force_max_abs * amplification == pytest.approx(1e-4)
        assert budget.force_rms * amplification == pytest.approx(3e-5)
        assert options["ccsd_residual_tolerance"] > 1e-9
        assert options["precision"] == "fp64"
        assert event["closed"]


@pytest.mark.parametrize("policy", ("standard", "fast", "tight", "fp64"))
def test_preserves_tighter_explicit_controls(
    bases: tuple[CBSBasis, CBSBasis],
    fake_native: list[dict],
    policy: str,
) -> None:
    controls = {
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-12,
        "ccsd_energy_tolerance": 1e-13,
        "ccsd_residual_tolerance": 1e-11,
    }
    result = CCSDTCBSCalculator(
        bases, policy=policy, calculator_options=controls
    ).singlepoint(ATOMS)
    for name, value in controls.items():
        assert result.convergence["limits"][name] == value
        assert all(event["options"][name] == value for event in fake_native)


def test_fast_tight_budget_retains_strict_controls(
    bases: tuple[CBSBasis, CBSBasis],
    fake_native: list[dict],
) -> None:
    from generativeqc import PrecisionPolicy, TargetErrorBudget

    policy = PrecisionPolicy("fast", budget=TargetErrorBudget(1e-12, 1e-11))
    result = CCSDTCBSCalculator(bases, policy=policy).singlepoint(ATOMS)
    assert result.convergence["selected"] == "strict_fp64"
    assert result.convergence["reason"] == "component_budget_requires_strict"
    assert all(
        "_correlation_convergence_budget" not in event["options"]
        for event in fake_native
    )
    assert all(
        event["options"]["ccsd_residual_tolerance"] == 1e-10 for event in fake_native
    )


@pytest.mark.parametrize("policy", ("balanced", "strict", "audit_auto"))
def test_cbs_rejects_unimplemented_policy_audits(
    bases: tuple[CBSBasis, CBSBasis], policy: str
) -> None:
    with pytest.raises(NotImplementedError, match="without target audits"):
        CCSDTCBSCalculator(bases, policy=policy)


def test_convergence_metadata_is_detached(
    bases: tuple[CBSBasis, CBSBasis], fake_native: list[dict]
) -> None:
    workflow = CCSDTCBSCalculator(bases, policy="fast")
    first = workflow.singlepoint(ATOMS)
    first.convergence["limits"].clear()
    first.convergence["budget"].clear()
    second = workflow.singlepoint(ATOMS)
    assert second.convergence["limits"]["ccsd_residual_tolerance"] > 1e-9
    assert second.convergence["budget"]["energy_abs"] == 1e-5


@pytest.mark.parametrize("policy", ("standard", "fast", "tight"))
def test_explicit_fp64_preserves_independent_convergence(
    bases: tuple[CBSBasis, CBSBasis],
    fake_native: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    """Execution can be all-FP64 even when a scientific allowance is present."""
    original = cbs_module.Calculator.__init__

    def with_transport(self: object, *args: object, **kwargs: object) -> None:
        original(self, *args, **kwargs)
        self._correlation_preconvergence = kwargs.get("_correlation_preconvergence")

    monkeypatch.setattr(cbs_module.Calculator, "__init__", with_transport)
    result = CCSDTCBSCalculator(
        bases,
        auxiliary_bases=("sto-3g", "def2-svp"),
        device="cuda",
        policy=policy,
        preconvergence="fp64",
    ).singlepoint(ATOMS)
    assert result.convergence["selected"] == (
        "strict_fp64" if policy == "tight" else "relaxed"
    )
    assert result.convergence["arithmetic_request"] == "fp64"
    assert result.convergence["lambda_arithmetic_request"] == "fp64"
    assert result.convergence["arithmetic"] == "fp64"
    assert (result.convergence["scale"] > 0) is (policy != "tight")
    assert fake_native[0]["options"]["_correlation_preconvergence"] == "fp64"
    assert "_correlation_preconvergence" not in fake_native[1]["options"]


@pytest.mark.parametrize(
    "options,error,match",
    (
        ({"preconvergence": "unknown"}, ValueError, "preconvergence"),
        ({"preconvergence": "mixed"}, NotImplementedError, "DF bases"),
        ({"policy": "fp64", "preconvergence": "mixed"}, ValueError, "fp64 policy"),
    ),
)
def test_cbs_preconvergence_validation(
    bases: tuple[CBSBasis, CBSBasis],
    options: dict,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        CCSDTCBSCalculator(bases, **options)


@pytest.mark.parametrize("executed", (True, False))
@pytest.mark.parametrize("policy", ("standard", "fast", "tight"))
@pytest.mark.parametrize("lambda_executed", (True, False, None))
@pytest.mark.parametrize("forces", (True, False))
def test_mixed_reports_executed_arithmetic_and_resource_fallback(
    bases: tuple[CBSBasis, CBSBasis],
    fake_native: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    executed: bool,
    lambda_executed: bool | None,
    forces: bool,
    policy: str,
) -> None:
    """A refused FP32 request must remain an observable FP64 result."""
    from generativeqc import _native
    from generativeqc._api_types import CcPerformanceResult

    performance = CcPerformanceResult(
        **{
            name: 0
            for name, _ in _native.CcPerformanceDiagnostic._fields_
            if name not in ("struct_size", "abi_version")
        },
        ccsd_fp32_requested=True,
        ccsd_fp32_executed=executed,
        ccsd_fp32_refused=not executed,
        ccsd_fp32_evaluations=2 if executed else 0,
        ccsd_fp32_gemm_calls=12 if executed else 0,
        ccsd_fp32_cast_elements=100 if executed else 0,
        lambda_fp32_requested=None if lambda_executed is None else True,
        lambda_fp32_executed=lambda_executed,
        lambda_fp32_refused=not lambda_executed,
        lambda_fp32_gemm_calls=20 if lambda_executed else 0,
    )
    original = cbs_module.Calculator.singlepoint
    original_init = cbs_module.Calculator.__init__

    def with_transport(self: object, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)
        self._correlation_preconvergence = kwargs.get("_correlation_preconvergence")

    def measured(self: object, *args: object, **kwargs: object) -> Result:
        return replace(original(self, *args, **kwargs), cc_performance=performance)

    monkeypatch.setattr(cbs_module.Calculator, "singlepoint", measured)
    monkeypatch.setattr(cbs_module.Calculator, "__init__", with_transport)
    owner = CCSDTCBSCalculator(
        bases,
        auxiliary_bases=("sto-3g", "def2-svp"),
        device="cuda",
        policy=policy,
        preconvergence="mixed",
    )
    properties = ("energy", "forces") if forces else ("energy",)
    if forces and lambda_executed is None:
        with pytest.raises(RuntimeError, match="Lambda FP32 admission telemetry"):
            owner.singlepoint(ATOMS, properties=properties)
    else:
        result = owner.singlepoint(ATOMS, properties=properties)
        assert result.convergence["selected"] == (
            "strict" if policy == "tight" else "relaxed"
        )
        if policy == "tight":
            assert result.convergence["budget"] is None
            assert result.convergence["limits"]["ccsd_residual_tolerance"] == 1e-10
            assert all(
                "_correlation_convergence_budget" not in event["options"]
                for event in fake_native
            )
        assert result.convergence["arithmetic_request"] == "df_ccsd_fp32_preconvergence"
        assert result.convergence["arithmetic"] == (
            "mixed" if executed or (forces and lambda_executed) else "fp64"
        )
        assert result.convergence["lambda_arithmetic_request"] == (
            "df_lambda_fp32_preconvergence" if forces else "not_requested"
        )
        assert result.convergence["lambda_arithmetic"] == (
            "not_requested" if not forces else "mixed" if lambda_executed else "fp64"
        )
    assert all(event["closed"] for event in fake_native)


@pytest.mark.parametrize("policy", ("standard", "fast"))
def test_routine_missing_precision_telemetry_cannot_publish(
    bases: tuple[CBSBasis, CBSBasis],
    fake_native: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    original = cbs_module.Calculator.singlepoint

    def unmeasured(self: object, *args: object, **kwargs: object) -> Result:
        return replace(original(self, *args, **kwargs), cc_performance=None)

    monkeypatch.setattr(cbs_module.Calculator, "singlepoint", unmeasured)
    with pytest.raises(RuntimeError, match="FP32 admission telemetry"):
        CCSDTCBSCalculator(
            bases, auxiliary_bases=("sto-3g", "def2-svp"), device="cuda", policy=policy
        ).singlepoint(ATOMS)
    assert all(event["closed"] for event in fake_native)


def test_default_is_routine_and_tight_preserves_legacy_limits(
    bases: tuple[CBSBasis, CBSBasis],
    fake_native: list[dict],
) -> None:
    """Default promotion and the explicit strict escape are observable contracts."""
    default = CCSDTCBSCalculator(bases).singlepoint(ATOMS)
    fast = CCSDTCBSCalculator(bases, policy="fast").singlepoint(ATOMS)
    tight = CCSDTCBSCalculator(bases, policy="Tight").singlepoint(ATOMS)
    fp64 = CCSDTCBSCalculator(bases, policy="fp64").singlepoint(ATOMS)
    assert default.convergence["requested_policy"] == "standard"
    assert default.convergence["limits"] == fast.convergence["limits"]
    assert default.convergence["budget"] == fast.convergence["budget"]
    assert default.convergence["limits"]["ccsd_energy_tolerance"] == 1e-8
    assert default.convergence["limits"]["density_tolerance"] == 1e-11
    assert default.convergence["limits"]["ccsd_residual_tolerance"] == 1e-7
    assert tight.convergence["limits"] == fp64.convergence["limits"]
    assert tight.convergence["limits"]["ccsd_energy_tolerance"] == 1e-12
    assert tight.convergence["limits"]["ccsd_residual_tolerance"] == 1e-10
    assert tight.convergence["budget"] is None
