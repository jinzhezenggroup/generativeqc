"""Application budgets must gate actual observables and bound fallback work."""

import ctypes
import json
import typing
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc import (
    Calculator,
    GridSpec,
    KsOptions,
    MethodCapabilities,
    ObservableTarget,
    PrecisionBudgetError,
    PrecisionCalculator,
    PrecisionPolicy,
    ResolvedModel,
    Result,
    TargetAccuracy,
    TargetErrorBudget,
)
from generativeqc import precision_policy as implementation

ATOMS = (("H", (0, 0, -0.7)), ("H", (0, 0, 0.7)))
ENERGY = TargetAccuracy((ObservableTarget("energy", "absolute", "Eh", absolute=1e-6),))


def result(energy: float = -1.0, force: float = 0.0, **changes: typing.Any) -> Result:
    """A converged endpoint whose arrays are independent of native ownership."""
    forces = np.zeros((2, 3))
    forces[0, 0] = force
    return replace(
        Result(energy, forces, True, 4, 1e-12, 1e-10, "cpu_reference"), **changes
    )


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Inject endpoint observations while exercising the real policy controller.

    Native numerical/model qualification lives in the companion integration
    tests. This backend checks owner lifetimes and supplies failure conditions
    that are hard to produce reproducibly with a converged molecular solve.
    """
    state = SimpleNamespace(
        outputs=[],
        instances=[],
        calls=[],
        live=None,
        block_auto=False,
        auto_properties=None,
    )

    class Backend:
        def __init__(self, **options: typing.Any) -> None:
            if state.block_auto and options["precision"] == "auto":
                raise NotImplementedError("unavailable AUTO domain")
            self.options = options
            self._basis = options.get("basis", ())
            self._auxiliary_basis = options.get("auxiliary_basis")
            self._ks_options = options.get("ks_options")
            if options.get("method") == "pbe0-rks" and self._ks_options is None:
                from generativeqc import KsOptions

                self._ks_options = KsOptions()
            self._density_fitting_mode = 0
            self._density_fitting_relative_threshold = 1e-10
            self._energy_tolerance = options.get("energy_tolerance", 1e-10)
            self._density_tolerance = options.get("density_tolerance", 1e-8)
            self._screening_tolerance = options.get("screening_tolerance", 1e-12)
            self.capabilities = MethodCapabilities(
                "rhf",
                "hartree_fock",
                True,
                True,
                frozenset(("energy", "forces")),
                frozenset(("hessian",)),
            )
            if options["precision"] == "auto" and state.auto_properties is not None:
                self.capabilities = replace(
                    self.capabilities, supported_properties=state.auto_properties
                )
            state.instances.append(self)

        def basis_metadata(self, *args: typing.Any, **kwargs: typing.Any) -> dict:
            return {"orbital": {"mathematical_identity": "orbital"}}

        def _effective_ks_options(
            self, *args: typing.Any, **kwargs: typing.Any
        ) -> None:
            return None

        def resolved_model(
            self, *args: typing.Any, **kwargs: typing.Any
        ) -> ResolvedModel:
            return ResolvedModel("rhf", "geometry", "orbital", 2)

        def singlepoint(self, atoms: typing.Any, **options: typing.Any) -> Result:
            assert state.live is None, "previous endpoint retained its workspace"
            state.live = self
            state.calls.append((self, atoms, options))
            output = state.outputs.pop(0)
            if isinstance(output, Exception):
                raise output
            return output

        def clear_cache(self) -> None:
            if state.live is self:
                state.live = None

    monkeypatch.setattr(implementation, "Calculator", Backend)
    monkeypatch.setattr("generativeqc.ecp.resolve_ecp", lambda *args: ((0, 0), ()))
    return state


@pytest.mark.parametrize("profile", ["audit_auto", "balanced", "strict", "Balanced"])
def test_presets_are_immutable_allowances(profile: str) -> None:
    policy = PrecisionPolicy(profile)
    assert policy.profile == profile.lower()
    assert {item.observable for item in policy.target.observables} == {
        "energy",
        "forces",
    }
    assert TargetAccuracy.from_dict(policy.to_dict()["target"]) == policy.target
    with pytest.raises(FrozenInstanceError):
        policy.profile = "audit_auto"


@pytest.mark.parametrize("profile", ["fp64", "FP64", "Fp64"])
def test_fp64_executes_once_without_inventing_budget_evidence(
    backend: SimpleNamespace, profile: str
) -> None:
    """Single strict execution is distinct from a measured observable agreement."""
    native = result()
    backend.outputs = [native]
    calculator = PrecisionCalculator(policy=profile)
    output = calculator.singlepoint(ATOMS)
    report = output.precision_validation
    assert len(backend.calls) == len(backend.instances) == report.solve_count == 1
    assert report.status == "strict_fp64"
    assert report.target is None and report.errors == report.rejected_errors == ()
    assert "no-observable-audit" in report.reference
    assert report.solves[0].requested_precision == "fp64"
    assert backend.live is None
    assert calculator.policy.to_dict() == {"profile": "fp64", "target": None}
    assert json.loads(json.dumps(report.to_dict(), allow_nan=False))["target"] is None
    np.testing.assert_array_equal(output.forces, native.forces)


def test_fp64_explicit_target_retains_the_complete_budget_audit(
    backend: SimpleNamespace,
) -> None:
    """A caller's requested error proof cannot disappear inside a speed preset."""
    backend.outputs = [result(), result(energy=-1 + 1e-8)]
    output = PrecisionCalculator(
        policy=PrecisionPolicy("fp64", target=ENERGY)
    ).singlepoint(ATOMS, properties=("energy",))
    report = output.precision_validation
    assert report.status == "observed_met" and report.solve_count == 2
    assert report.target == ENERGY and report.errors[0].difference > 0


def test_fp64_tightens_loose_controls_without_loosening_explicit_strict_ones(
    backend: SimpleNamespace,
) -> None:
    calculator = PrecisionCalculator(
        policy="fp64",
        energy_tolerance=1e-4,
        density_tolerance=1e-12,
        screening_tolerance=None,
    )
    assert calculator._candidate._energy_tolerance == 1e-10
    assert calculator._candidate._density_tolerance == 1e-12
    assert calculator._candidate._screening_tolerance == 1e-12


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize(
    "name", ["energy_tolerance", "density_tolerance", "screening_tolerance"]
)
def test_fp64_rejects_invalid_controls_before_clamping(
    backend: SimpleNamespace, name: str, value: float
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        PrecisionCalculator(policy="fp64", **{name: value})
    assert not backend.instances


@pytest.mark.parametrize("profile", ["", "fp32", 7, None, []])
def test_unknown_profiles_fail_before_native_setup(profile: typing.Any) -> None:
    with pytest.raises(ValueError):
        PrecisionPolicy(profile)


def test_fast_unsupported_backend_records_explicit_fp64_audit(
    backend: SimpleNamespace,
) -> None:
    """A fallback cannot be presented as a qualified mixed-precision execution."""
    backend.outputs = [result(), result()]
    output = PrecisionCalculator(policy="fast").singlepoint(ATOMS)
    report = output.precision_validation
    assert report.solve_count == 2 and report.status == "observed_met"
    assert "application_warmup_unavailable_on_backend" in report.fallback_reasons
    assert "budget_or_domain_requires_fp64_audit" in report.fallback_reasons
    assert report.budget == TargetErrorBudget(1e-5, 1e-4, 3e-5)


def test_fast_budget_selects_native_arithmetic_without_an_extra_solve(
    backend: SimpleNamespace,
) -> None:
    work = {
        "operator_work_counters_valid": True,
        "complete": True,
        "mixed_stage_fock_builds": 9,
        "strict_stage_fock_builds": 3,
        "post_scf_fock_builds": 0,
        "strict_refinement_applied": True,
        "final_residual_audits": 1,
        "operator_inventory_complete": True,
        "operators": [
            {
                "name": "xc",
                "compute": "fp32",
                "accumulation": "fp32",
                "arithmetic_mode": "mixed",
                "count": 9,
            }
        ],
    }
    backend.outputs = [result(executed_backend="cuda", precision=work)]
    budget = TargetErrorBudget(2e-5, 4e-4)
    calculator = PrecisionCalculator(
        method="pbe0-rks", device="cuda", policy=PrecisionPolicy("fast", budget=budget)
    )
    output = calculator.singlepoint(ATOMS)
    report = output.precision_validation
    assert calculator._candidate._ks_options._precision_budget == (2e-5, 4e-4, 0)
    assert report.status == "qualified_strict_final" and report.solve_count == 1
    assert report.target is None and report.errors == () and report.budget == budget
    assert report.scope == "qualification_not_per_call_observable_audit"
    assert len(backend.calls) == 1 and backend.live is calculator._candidate
    calculator.clear_cache()
    assert backend.live is None


@pytest.mark.parametrize("retries", (0, 1))
@pytest.mark.parametrize("warmup", (0, 9))
def test_budget_gradient_is_not_misreported_as_strict(
    backend: SimpleNamespace, retries: int, warmup: int
) -> None:
    """A gradient-only route needs no approximate SCF or duplicate audit solve."""
    work = {
        "operator_work_counters_valid": True,
        "complete": True,
        "operator_inventory_complete": True,
        "mixed_stage_fock_builds": warmup,
        "strict_stage_fock_builds": 3,
        "post_scf_fock_builds": 1,
        "final_residual_audits": 1,
        "gradient": {
            "schema_version": 1,
            "complete": True,
            "attempted_fp32_geometry_points": 300,
            "attempted_fp32_grid_points": 300,
            "fp64_gradient_retries": retries,
            "returned_mode": "fp64" if retries else "mixed",
        },
    }
    backend.outputs = [result(executed_backend="cuda", precision=work)]
    calculator = PrecisionCalculator(method="pbe0-rks", device="cuda", policy="fast")
    observed = calculator.singlepoint(ATOMS)
    report = observed.precision_validation
    assert report.solve_count == 1 and not report.errors
    assert report.gradient_admission == "pbe0-rks-budget-fp32-v1"
    assert report.status == (
        "qualified_strict_final" if retries else "qualified_mixed_gradient"
    )
    assert ("native_gradient_retried_fp64" in report.fallback_reasons) == bool(retries)
    assert "fp32_gradient_not_qualified" not in report.fallback_reasons
    calculator.clear_cache()


@pytest.mark.parametrize(
    "missing", ["complete", "operator_work_counters_valid", "strict_refinement_applied"]
)
def test_fast_incomplete_native_evidence_never_claims_qualification(
    backend: SimpleNamespace, missing: str
) -> None:
    work = {
        "operator_work_counters_valid": True,
        "complete": True,
        "mixed_stage_fock_builds": 9,
        "strict_stage_fock_builds": 3,
        "strict_refinement_applied": True,
        "final_residual_audits": 1,
    }
    work[missing] = False
    backend.outputs = [
        result(executed_backend="cuda", precision=work),
        result(executed_backend="cuda"),
    ]
    report = (
        PrecisionCalculator(method="pbe0-rks", device="cuda", policy="fast")
        .singlepoint(ATOMS)
        .precision_validation
    )
    assert report.status == "observed_met" and report.solve_count == 2


def test_fast_explicit_target_is_still_an_observable_evidence_request(
    backend: SimpleNamespace,
) -> None:
    backend.outputs = [result(executed_backend="cuda"), result(executed_backend="cuda")]
    report = (
        PrecisionCalculator(
            method="pbe0-rks",
            device="cuda",
            policy=PrecisionPolicy("fast", target=ENERGY),
        )
        .singlepoint(ATOMS, properties=("energy",))
        .precision_validation
    )
    assert report.solve_count == 2 and report.target == ENERGY
    assert report.status == "observed_met"


def test_arithmetic_budget_is_not_a_target_accuracy_audit() -> None:
    with pytest.raises(TypeError, match="TargetErrorBudget"):
        PrecisionPolicy("fast", budget=ENERGY)
    with pytest.raises(ValueError, match="require Fast"):
        PrecisionPolicy("balanced", budget=TargetErrorBudget(1e-5, 1e-4))


def test_target_scope_and_precision_ownership() -> None:
    with pytest.raises(ValueError, match="relaxed-target"):
        PrecisionPolicy(target=replace(ENERGY, scope="fixed_density"))
    with pytest.raises(TypeError, match="TargetAccuracy"):
        PrecisionPolicy(target={})
    for conflicting in ("precision", "target_accuracy"):
        with pytest.raises(ValueError, match="through policy"):
            PrecisionCalculator(**{conflicting: "auto"})


def test_success_accounts_for_both_endpoints_and_owned_inputs(
    backend: SimpleNamespace,
) -> None:
    native_work = {
        "operator_work_counters_valid": True,
        "mixed_stage_fock_builds": 2,
        "strict_stage_fock_builds": 3,
        "post_scf_fock_builds": 1,
    }
    candidate = result(precision=native_work)
    backend.outputs = [candidate, result(-1.0 + 1e-9)]
    calculator = PrecisionCalculator(policy=PrecisionPolicy(target=ENERGY))
    observed = calculator.singlepoint(
        iter(ATOMS), properties=iter(("energy",)), charge=np.int64(0)
    )
    report = observed.precision_validation
    assert report.status == "observed_met" and report.solve_count == 2
    assert observed.energy == candidate.energy and observed.precision is native_work
    assert report.solves[0].fock_builds == 6 and report.solves[1].fock_builds is None
    assert all(stage.force_evaluations == 0 for stage in report.solves)
    assert report.endpoint_seconds >= sum(stage.seconds for stage in report.solves)
    assert backend.calls[0][1:] == backend.calls[1][1:]
    assert backend.live is None
    assert not calculator.capabilities.supports_batch
    assert not calculator.capabilities.supported_second_order
    assert json.loads(json.dumps(report.to_dict(), allow_nan=False))["solve_count"] == 2


def test_fallback_requires_its_own_verification(backend: SimpleNamespace) -> None:
    candidate, strict, tighter = result(force=1e-3), result(force=1e-8), result()
    backend.outputs = [candidate, strict, tighter]
    output = PrecisionCalculator().singlepoint(ATOMS)
    report = output.precision_validation
    np.testing.assert_array_equal(output.forces, strict.forces)
    assert report.solve_count == 3 and report.status == "observed_met"
    assert any(not error.met for error in report.rejected_errors)
    assert all(error.met for error in report.errors)
    assert "candidate_exceeded_observable_budget" in report.fallback_reasons
    assert report.solves[-1].role == "fp64_fallback_verification"
    for name in ("energy_tolerance", "density_tolerance", "screening_tolerance"):
        assert getattr(report.solves[-1], name) < getattr(report.solves[-2], name)


def test_failed_fallback_raises_with_observed_error_and_bounded_work(
    backend: SimpleNamespace,
) -> None:
    backend.outputs = [result(force=1e-2), result(force=1e-3), result()]
    with pytest.raises(PrecisionBudgetError) as failure:
        PrecisionCalculator().singlepoint(ATOMS)
    report = failure.value.validation
    assert report.status == "observed_unmet" and report.solve_count == 3
    assert any(not error.met for error in report.errors)
    assert backend.live is None and not backend.outputs


def test_rms_and_maximum_are_independent_requirements(backend: SimpleNamespace) -> None:
    # One nonzero component passes the maximum allowance but fails RMS.
    backend.outputs = [result(force=1e-4), result(), result()]
    report = (
        PrecisionCalculator(policy="audit_auto").singlepoint(ATOMS).precision_validation
    )
    errors = {
        error.norm: error
        for error in report.rejected_errors
        if error.observable == "forces"
    }
    assert errors["max_abs"].met and not errors["rms"].met
    assert errors["rms"].difference == pytest.approx(1e-4 / np.sqrt(6))


@pytest.mark.parametrize("reference,expected", [(0.0, False), (2.0, True)])
def test_relative_allowance_uses_reference_norm(
    backend: SimpleNamespace, reference: float, expected: bool
) -> None:
    target = TargetAccuracy(
        (ObservableTarget("energy", "absolute", "Eh", relative=1e-4),)
    )
    backend.outputs = [result(reference + 1e-5), result(reference), result(reference)]
    report = (
        PrecisionCalculator(policy=PrecisionPolicy(target=target))
        .singlepoint(ATOMS)
        .precision_validation
    )
    assert (report.solve_count == 2) is expected
    assert report.errors[0].allowance == abs(reference) * 1e-4


@pytest.mark.parametrize(
    "damaged",
    [
        result(energy=float("nan")),
        result(forces=np.zeros((1, 3))),
        result(converged=False),
    ],
)
def test_invalid_reference_never_passes(
    backend: SimpleNamespace, damaged: Result
) -> None:
    backend.outputs = [result(), damaged]
    with pytest.raises((ValueError, RuntimeError)):
        PrecisionCalculator().singlepoint(ATOMS)
    assert backend.live is None


@pytest.mark.parametrize(
    "profile,device,mode",
    [
        ("audit_auto", "cuda", "auto"),
        ("audit_auto", "cpu", "fp64"),
        ("balanced", "cuda", "fp64"),
        ("strict", "cuda", "fp64"),
    ],
)
def test_scf_risk_is_separate_from_gradient_admission(
    backend: SimpleNamespace, profile: str, device: str, mode: str
) -> None:
    backend_name = "cuda" if device == "cuda" else "cpu_reference"
    backend.outputs = [result(executed_backend=backend_name)] * 2
    report = (
        PrecisionCalculator(policy=profile, device=device, device_id=2)
        .singlepoint(ATOMS)
        .precision_validation
    )
    assert report.solves[0].requested_precision == mode
    assert report.profile == profile
    assert report.solves[1].requested_precision == "fp64"
    assert all(
        instance.options["device"] == device and instance.options["device_id"] == 2
        for instance in backend.instances
    )
    assert report.gradient_admission == "no-application-qualified-fp32-gradient-route"


def test_unavailable_auto_route_uses_fp64(backend: SimpleNamespace) -> None:
    backend.block_auto = True
    backend.outputs = [result(executed_backend="cuda")] * 2
    report = (
        PrecisionCalculator(policy="audit_auto", device="cuda")
        .singlepoint(ATOMS)
        .precision_validation
    )
    assert report.solves[0].requested_precision == "fp64"
    assert "auto_scf_unavailable_for_model" in report.fallback_reasons


@pytest.mark.parametrize(
    "outputs",
    [[RuntimeError("execution failed")], [result(), RuntimeError("execution failed")]],
)
def test_native_failures_release_owners_and_are_not_hidden(
    backend: SimpleNamespace, outputs: list
) -> None:
    backend.outputs = outputs
    with pytest.raises(RuntimeError, match="execution failed"):
        PrecisionCalculator().singlepoint(ATOMS)
    assert backend.live is None


def test_cuda_request_cannot_silently_execute_on_cpu(backend: SimpleNamespace) -> None:
    backend.outputs = [result()]
    with pytest.raises(RuntimeError, match="requested backend"):
        PrecisionCalculator(device="cuda").singlepoint(ATOMS)
    assert backend.live is None


def test_force_only_budget_still_rejects_nonfinite_energy(
    backend: SimpleNamespace,
) -> None:
    target = TargetAccuracy(
        (ObservableTarget("forces", "max_abs", "Eh/bohr", absolute=1e-5),)
    )
    backend.outputs = [result(energy=float("nan"))]
    with pytest.raises(ValueError, match="finite"):
        PrecisionCalculator(policy=PrecisionPolicy(target=target)).singlepoint(ATOMS)
    assert len(backend.calls) == 1 and backend.live is None


def test_force_shape_must_match_the_actual_molecule(backend: SimpleNamespace) -> None:
    backend.outputs = [result(forces=np.zeros((1, 3)))]
    with pytest.raises(ValueError, match="atom count"):
        PrecisionCalculator().singlepoint(ATOMS)
    assert len(backend.calls) == 1 and backend.live is None


@pytest.mark.parametrize("properties", [("energy",), ("energy", "hessian"), "energy"])
def test_incomplete_property_request_fails_before_solve(
    backend: SimpleNamespace, properties: typing.Any
) -> None:
    with pytest.raises((ValueError, TypeError)):
        PrecisionCalculator().singlepoint(ATOMS, properties=properties)
    assert not backend.calls


def test_mismatched_model_is_rejected_before_reference_execution(
    backend: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calculator = PrecisionCalculator()
    reference = calculator._reference()
    monkeypatch.setattr(
        reference,
        "resolved_model",
        lambda *args, **kwargs: ResolvedModel("rhf", "other-geometry", "orbital", 2),
    )
    monkeypatch.setattr(calculator, "_reference", lambda **kwargs: reference)
    backend.outputs = [result()]
    with pytest.raises(ValueError, match="scientific model"):
        calculator.singlepoint(ATOMS)
    assert len(backend.calls) == 1


@pytest.mark.parametrize("profile", ("fast", "audit_auto"))
def test_property_fallback_preserves_later_energy_only_auto(
    backend: SimpleNamespace, profile: str
) -> None:
    backend.auto_properties = frozenset({"energy"})
    backend.outputs = [result(executed_backend="cuda")] * 4
    calculator = PrecisionCalculator(
        method="pbe0-rks",
        device="cuda",
        policy=PrecisionPolicy(profile, target=ENERGY),
    )
    assert calculator.capabilities.supported_properties == {"energy", "forces"}
    forces = calculator.singlepoint(ATOMS, properties=("energy", "forces"))
    report = forces.precision_validation
    assert report.status == "observed_met" and report.solve_count == 2
    assert all(stage.requested_precision == "fp64" for stage in report.solves)
    assert "auto_properties_unavailable_for_model" in report.fallback_reasons
    energy = calculator.singlepoint(ATOMS, properties=("energy",))
    assert energy.precision_validation.solves[0].requested_precision == "auto"
    assert (
        "auto_properties_unavailable_for_model"
        not in energy.precision_validation.fallback_reasons
    )
    assert backend.live is None


@pytest.fixture
def cuda_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use actual scientific resolution and capability gates without a GPU."""
    import generativeqc.calculator as calculator_module
    from generativeqc import _native

    def available(method: int, pointer: object) -> int:
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int32))[0] = 1
        return _native.STATUS_SUCCESS

    library = SimpleNamespace(
        generativeqc_ks_options_version=lambda: 1,
        generativeqc_application_precision_version_v1=lambda: 1,
        generativeqc_method_available=available,
    )
    monkeypatch.setattr(_native, "load_library", lambda **kwargs: library)
    monkeypatch.setattr(
        calculator_module,
        "method_capabilities",
        lambda name: MethodCapabilities(
            name, "density_functional", True, True, frozenset({"energy"})
        ),
    )


@pytest.mark.parametrize("method", ("m06-2x-rks", "mn15-rks"))
@pytest.mark.parametrize("profile", ("fast", "audit_auto"))
def test_split_hybrid_fp64_force_capability_survives_auto_selection(
    cuda_constructor: None, method: str, profile: str
) -> None:
    calculator = PrecisionCalculator(
        method=method,
        device="cuda",
        initial_guess=None,
        ks_options=KsOptions(grid=GridSpec(16, 8, 16)),
        policy=profile,
    )
    assert calculator._candidate_precision == "auto"
    assert calculator._candidate.capabilities.supported_properties == {"energy"}
    assert calculator.capabilities.supported_properties == {"energy", "forces"}


@pytest.mark.parametrize("method", ("pbe-rks", "pbe0-rks", "m06-2x-rks"))
@pytest.mark.parametrize("profile", ("fast", "audit_auto"))
def test_density_fitted_force_context_keeps_fp64_fallback(
    cuda_constructor: None, method: str, profile: str
) -> None:
    calculator = PrecisionCalculator(
        method=method,
        device="cuda",
        density_fitting="cuda",
        initial_guess=None,
        ks_options=KsOptions(grid=GridSpec(16, 8, 16)),
        policy=profile,
    )
    assert calculator._candidate_precision == "fp64"
    assert calculator.capabilities.supported_properties == {"energy", "forces"}


def test_fast_keeps_resolved_custom_ks_composition(cuda_constructor: None) -> None:
    """Adding arithmetic controls cannot erase a resolved exact-exchange term."""
    from generativeqc_compiler.method import resolve_method

    pbe = resolve_method("PBE")
    custom = replace(
        pbe,
        identifier="PBE-plus-exchange",
        primitives=(*pbe.primitives, resolve_method("PBE0").primitives[1]),
    )
    original = Calculator(
        method="pbe-rks",
        device="cuda",
        initial_guess=None,
        ks_options=KsOptions(grid=GridSpec(16, 8, 16), composition=custom),
    )
    resolved = original.ks_options
    calculator = PrecisionCalculator(
        method="pbe-rks",
        device="cuda",
        initial_guess=None,
        ks_options=resolved,
        policy="fast",
    )
    assert calculator._candidate_precision == "auto"
    assert calculator._candidate.ks_options.method_ir.identity == custom.identity
    assert calculator._reference().ks_options.method_ir.identity == custom.identity
    assert calculator._candidate.ks_options.coefficients == (1.0, 1.0, -0.125)
    assert resolved._precision_budget is None


def test_fast_property_fallback_audits_the_implicit_force_budget(
    backend: SimpleNamespace,
) -> None:
    backend.auto_properties = frozenset({"energy"})
    backend.outputs = [result(executed_backend="cuda")] * 2
    output = PrecisionCalculator(
        method="pbe0-rks", device="cuda", policy="fast"
    ).singlepoint(ATOMS)
    report = output.precision_validation
    assert report.status == "observed_met" and report.solve_count == 2
    assert all(stage.requested_precision == "fp64" for stage in report.solves)
    assert {(error.observable, error.norm) for error in report.errors} == {
        ("energy", "absolute"),
        ("forces", "max_abs"),
        ("forces", "rms"),
    }
