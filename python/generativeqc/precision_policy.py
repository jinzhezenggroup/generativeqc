"""Budget-selected mixed arithmetic, strict final SCF and explicit FP64 audits.

Fast selects qualified XC arithmetic from TargetErrorBudget and still publishes
the incumbent FP64-converged state. Admitted stationary gradients also use
FP32 under the force budget, with bounded numerical FP64 fallback. Qualification does not
invent a per-call observable difference. TargetAccuracy remains an explicit
request for a separate relaxed endpoint audit, with its full cost recorded.
"""

from __future__ import annotations

import math
import threading
import typing
from dataclasses import asdict, dataclass, replace
from time import perf_counter

from ._api_types import Atom, MethodCapabilities, Result
from .accuracy import (
    ObservableTarget,
    TargetAccuracy,
    _measure_observables,
    _observable_array,
)
from .calculator import Calculator
from .elements import checked_integer
from .force_aware_numerics import TargetErrorBudget
from .profiles import canonical_hash

_DEFAULT_BUDGETS = {
    "audit_auto": (1e-5, 1e-4, 3e-5),
    "balanced": (1e-7, 1e-5, 3e-6),
    "strict": (1e-9, 1e-7, 3e-8),
}


def _budget_target(budget: TargetErrorBudget, *, forces: bool = True) -> TargetAccuracy:
    """Use the existing observable audit when qualification cannot admit work."""
    observables = [
        ObservableTarget("energy", "absolute", "Eh", absolute=budget.energy_abs)
    ]
    if forces:
        observables.append(
            ObservableTarget(
                "forces", "max_abs", "Eh/bohr", absolute=budget.force_max_abs
            )
        )
        if budget.force_rms is not None:
            observables.append(
                ObservableTarget("forces", "rms", "Eh/bohr", absolute=budget.force_rms)
            )
    return TargetAccuracy(tuple(observables))


@dataclass(frozen=True)
class PrecisionPolicy:
    """Select arithmetic by application budget, or request explicit audits.

    Fast's ``budget`` reuses the adaptive-policy allowance type and controls
    native FP32 admission. It is not a measured per-call error certificate.
    Explicit ``target`` always requests a relaxed audit, including with Fast.
    FP64 without ``target`` executes once in FP64; no observable error estimate
    is invented. Supplying a target opts into the full audit, including its cost.
    Other presets are audit allowances, not measured error ranges. Only
    relaxed-target audits are admitted.
    """

    profile: str = "balanced"
    target: TargetAccuracy | None = None
    budget: TargetErrorBudget | None = None

    def __post_init__(self) -> None:
        """Resolve preset allowances and reject incompatible audit contracts."""
        if isinstance(self.profile, str):
            object.__setattr__(self, "profile", self.profile.lower())
        if self.profile == "fast":
            if self.budget is None:
                object.__setattr__(self, "budget", TargetErrorBudget(1e-5, 1e-4, 3e-5))
            elif not isinstance(self.budget, TargetErrorBudget):
                raise TypeError("Fast requires a TargetErrorBudget")
        elif self.budget is not None:
            raise ValueError("arithmetic-selection budgets currently require Fast")
        if not isinstance(self.profile, str) or self.profile not in (
            "fast",
            "fp64",
            *_DEFAULT_BUDGETS,
        ):
            raise ValueError(
                "precision profile must be 'fast', 'fp64', 'audit_auto', 'balanced' or 'strict'"
            )
        if self.profile in ("fast", "fp64") and self.target is None:
            return
        if self.target is None:
            energy, maximum, rms = _DEFAULT_BUDGETS[self.profile]
            object.__setattr__(
                self,
                "target",
                TargetAccuracy(
                    (
                        ObservableTarget("energy", "absolute", "Eh", absolute=energy),
                        ObservableTarget(
                            "forces", "max_abs", "Eh/bohr", absolute=maximum
                        ),
                        ObservableTarget("forces", "rms", "Eh/bohr", absolute=rms),
                    )
                ),
            )
        if not isinstance(self.target, TargetAccuracy):
            raise TypeError("precision target must be a TargetAccuracy contract")
        if self.target.scope != "relaxed_target":
            raise ValueError("application precision requires a relaxed-target audit")

    def to_dict(self) -> dict:
        """Return portable settings without claiming numerical evidence."""
        payload = {
            "profile": self.profile,
            "target": self.target.to_dict() if self.target is not None else None,
        }
        if self.budget is not None:
            payload["budget"] = asdict(self.budget)
        return payload


@dataclass(frozen=True)
class PrecisionError:
    """One measured norm difference and its independently resolved allowance."""

    observable: str
    norm: str
    unit: str
    difference: float
    reference_norm: float
    allowance: float

    @property
    def met(self) -> bool:
        """Whether this measured difference fits the declared allowance."""
        return self.difference <= self.allowance


@dataclass(frozen=True)
class PrecisionSolve:
    """One complete solve, with execution-owned work counts when available."""

    role: str
    requested_precision: str
    seconds: float
    iterations: int
    force_evaluations: int
    fock_builds: int | None
    executed_backend: str
    energy_tolerance: float
    density_tolerance: float
    screening_tolerance: float


@dataclass(frozen=True)
class PrecisionValidation:
    """Endpoint evidence, rejected-candidate errors and complete-endpoint cost.

    ``observed_met`` requires all requested norms to agree with the stricter
    same-model solve. Basis/grid/model error, stability, reference uncertainty,
    force continuity and NVE conservation require separate qualification.
    """

    profile: str
    target: TargetAccuracy | None
    model_identity: str
    status: str
    fallback_reasons: tuple[str, ...]
    errors: tuple[PrecisionError, ...]
    rejected_errors: tuple[PrecisionError, ...]
    solves: tuple[PrecisionSolve, ...]
    endpoint_seconds: float
    reference: str = "same-backend-independent-relaxed-fp64-solve"
    gradient_admission: str = "no-application-qualified-fp32-gradient-route"
    scope: str = "relaxed_target"
    schema_version: int = 1
    budget: TargetErrorBudget | None = None

    @property
    def solve_count(self) -> int:
        """Count full candidate, audit and fallback-verification endpoints."""
        return len(self.solves)

    def to_dict(self) -> dict:
        """Return detached policy, error, provenance and timing records."""
        record = asdict(self)
        record["target"] = self.target.to_dict() if self.target is not None else None
        record["solve_count"] = self.solve_count
        return record


class PrecisionBudgetError(RuntimeError):
    """No endpoint passed the budget after bounded FP64 fallback verification."""

    def __init__(self, validation: PrecisionValidation) -> None:
        """Retain the failed audit report without exposing a partial Result."""
        super().__init__(
            "observable error budget unmet after FP64 fallback verification"
        )
        self.validation = validation


def _model_identity(
    calculator: Calculator, atoms: tuple[Atom, ...], charge: int, multiplicity: int
) -> str:
    """Bind actual geometry, Hamiltonian, grid and fitted model to this audit.

    The prepared-plan model_identity includes numerical controls and cannot be
    used here. Consume mathematical basis identities and resolved KS science
    instead. A changed grid or fitted metric is a different target.
    """
    metadata = calculator.basis_metadata(
        atoms, charge=charge, multiplicity=multiplicity
    )
    ks_options = calculator._effective_ks_options(
        (atoms,), charges=(charge,), multiplicities=(multiplicity,)
    )
    if ks_options is None:
        return calculator.resolved_model(
            atoms, charge=charge, multiplicity=multiplicity
        ).identity
    ks_payload = ks_options.to_payload()
    # These knobs schedule the same XC operator. The domain, functional, grid,
    # nonlocal density policy and complete method graph stay in the identity.
    for name in (
        "tile_points",
        "xc_schedule",
        "nonlocal_memory_budget_bytes",
        "application_precision_budget",
    ):
        ks_payload.pop(name, None)
    fitted = bool(calculator._density_fitting_mode)
    return canonical_hash(
        {
            "schema": "generativeqc.precision-target.v1",
            "basis": metadata["orbital"]["mathematical_identity"],
            "auxiliary_basis": (
                metadata.get("auxiliary", metadata["orbital"])["mathematical_identity"]
                if fitted
                else None
            ),
            "ks": ks_payload,
            "dispersion": (
                calculator._dispersion_method_ir.to_payload()
                if calculator._dispersion_method_ir is not None
                else None
            ),
            "geometry": [
                (
                    atom.atomic_number,
                    tuple(
                        value.hex() if value else "0x0.0p+0" for value in atom.position
                    ),
                )
                for atom in atoms
            ],
            "charge": charge,
            "multiplicity": multiplicity,
            "density_fitting": fitted,
            "metric_relative_threshold": (
                calculator._density_fitting_relative_threshold if fitted else None
            ),
        }
    )


def _errors(
    target: TargetAccuracy, candidate: Result, reference: Result
) -> tuple[PrecisionError, ...]:
    """Use public HF evidence's shape, finite-value and scaled-norm rules."""
    if not candidate.converged or not reference.converged:
        raise RuntimeError("application precision audit requires two converged solves")
    return tuple(
        PrecisionError(
            requirement.observable,
            requirement.norm,
            requirement.unit,
            difference,
            reference_norm,
            requirement.allowance(reference_norm),
        )
        for requirement, difference, reference_norm in _measure_observables(
            target,
            {"energy": candidate.energy, "forces": candidate.forces},
            {"energy": reference.energy, "forces": reference.forces},
        )
    )


class PrecisionCalculator:
    """Budget-selected arithmetic and explicit all-electron HF/KS precision audits.

    ``policy`` is a preset name or a PrecisionPolicy with an application budget.
    Other options have Calculator semantics except ``precision`` and
    ``target_accuracy``, which this facade owns. FP64 without an explicit target
    runs once in FP64; native exact source reuse also benefits ordinary Calculator.
    Fast selects provisional FP32 XC for admitted CUDA PBE-family RKS and
    stationary FP32 gradients for admitted PBE0/RKS force budgets. The final
    SCF state stays strict; unsupported budgets/domains use FP64 audits.
    Explicit targets/audit profiles cost two full endpoints, or three after
    rejection. FP64 does not imply FP32 arithmetic or a universal speed guarantee.
    Batches, correlated
    methods, Hessians and MD are not promoted. Calls on an instance serialize.
    """

    def __init__(
        self,
        method: typing.Any = "rhf",
        basis: typing.Any = None,
        device: str = "cpu",
        *,
        policy: PrecisionPolicy | str = "balanced",
        **options: typing.Any,
    ) -> None:
        """Snapshot one native model and select its admitted candidate precision."""
        if isinstance(policy, str):
            policy = PrecisionPolicy(policy)
        if not isinstance(policy, PrecisionPolicy):
            raise TypeError("policy must be a PrecisionPolicy or preset name")
        if {"precision", "target_accuracy"} & options.keys():
            raise ValueError(
                "set application requirements through policy, not precision/target_accuracy"
            )
        self._policy = policy
        self._lock = threading.RLock()
        self._options = {"method": method, "basis": basis, "device": device, **options}
        if policy.profile in ("fp64", "fast"):
            # A speed profile must not turn caller-supplied loose convergence
            # controls into an implicitly certified strict endpoint.
            for name, ceiling in (
                ("energy_tolerance", 1e-10),
                ("density_tolerance", 1e-8),
                ("screening_tolerance", 1e-12),
            ):
                requested = self._options.get(name)
                if requested is not None:
                    requested = float(requested)
                    if not math.isfinite(requested) or requested <= 0:
                        raise ValueError(f"{name} must be finite and positive")
                self._options[name] = min(
                    ceiling, ceiling if requested is None else requested
                )
        self._candidate = Calculator(**self._options, precision="fp64")
        self._fp64_properties = self._candidate.capabilities.supported_properties
        if self._candidate.capabilities.family not in (
            "hartree_fock",
            "density_functional",
        ):
            raise NotImplementedError(
                "application precision currently supports native HF and KS only"
            )
        # Snapshot file-backed bases. Keep the caller's immutable KS input:
        # resolved options also contain composition/functional outputs, which
        # named and MethodIR constructors reject as conflicting model owners.
        self._options.update(
            basis=self._candidate._basis,
            auxiliary_basis=self._candidate._auxiliary_basis,
        )
        self._candidate_precision = "fp64"
        self._fast_native = False
        self._route_reasons = ["fp32_gradient_not_qualified"]
        if policy.profile == "fast":
            from .ks import KsOptions

            if device == "cuda" and self._candidate._ks_options is not None:
                candidate_options = dict(self._options)
                ks_options = candidate_options.get("ks_options") or KsOptions()
                candidate_options["ks_options"] = replace(
                    ks_options,
                    _precision_budget=(
                        policy.budget.energy_abs,
                        policy.budget.force_max_abs,
                        policy.budget.force_rms or 0.0,
                    ),
                )
                # replace() resets init=False fields. Resolved options retain
                # their scientific composition there, including exact exchange.
                object.__setattr__(
                    candidate_options["ks_options"], "_method_ir", ks_options._method_ir
                )
                try:
                    self._candidate = Calculator(**candidate_options, precision="auto")
                    self._candidate_precision = "auto"
                    self._fast_native = True
                except NotImplementedError:
                    self._route_reasons.append(
                        "application_warmup_unavailable_for_model"
                    )
            else:
                self._route_reasons.append("application_warmup_unavailable_on_backend")
        if policy.profile == "audit_auto":
            if device != "cuda":
                self._route_reasons.append("auto_scf_unavailable_on_backend")
            else:
                try:
                    self._candidate = Calculator(**self._options, precision="auto")
                    self._candidate_precision = "auto"
                except NotImplementedError:
                    self._route_reasons.append("auto_scf_unavailable_for_model")

    @property
    def policy(self) -> PrecisionPolicy:
        """Return the immutable policy selected at construction."""
        return self._policy

    @property
    def capabilities(self) -> MethodCapabilities:
        """Expose only this facade's admitted single-point property boundary."""
        return replace(
            self._candidate.capabilities,
            supported_properties=(
                self._candidate.capabilities.supported_properties
                | self._fp64_properties
            ),
            supports_batch=False,
            supported_second_order=frozenset(),
        )

    def clear_cache(self) -> None:
        """Release retained candidate workspaces without changing the policy."""
        with self._lock:
            self._candidate.clear_cache()

    def _reference(self, *, tighter: bool = False) -> Calculator:
        """Tighten solver controls without changing the model or converting units.

        Force budgets are never mapped to SCF residuals. This fixed recipe is
        not an observable-error guarantee: the independent audit must pass.
        """
        options = dict(self._options)
        scale = 0.1 if tighter else 1.0
        for name, ceiling in (
            ("energy_tolerance", 1e-12),
            ("density_tolerance", 1e-10),
            ("screening_tolerance", 1e-14),
        ):
            requested = options.get(name)
            options[name] = (
                min(ceiling, ceiling if requested is None else requested) * scale
            )
            if options[name] <= 0:
                raise ValueError("precision reference tolerance underflowed")
        return Calculator(**options, precision="fp64")

    def singlepoint(
        self,
        atoms: typing.Iterable[Atom | tuple[str | int, typing.Sequence[float]]],
        *,
        charge: int = 0,
        multiplicity: int = 1,
        properties: typing.Iterable[str] | None = None,
    ) -> Result:
        """Publish qualified arithmetic, or measured budget evidence when requested.

        The reference uses the same backend and includes SCF relaxation and all
        requested force contributions. An over-budget candidate is replaced by
        the FP64 reference only after a third, tighter independent solve agrees.
        FP64 without a target reports ``strict_fp64``, not ``observed_met``;
        it retains native final-state checks but does not estimate observable
        errors. No geometry-independent calibration is presented as a bound.
        """
        with self._lock:
            started = perf_counter()
            native_atoms = tuple(Atom.from_value(atom) for atom in atoms)
            if not native_atoms:
                raise ValueError("at least one atom is required")
            charge = checked_integer(
                charge, "ionic charge", low=-(2**31), high=2**31 - 1
            )
            multiplicity = checked_integer(
                multiplicity, "multiplicity", low=1, high=2**31 - 1
            )
            target = self.policy.target
            required = (
                {requirement.observable for requirement in target.observables}
                | {"energy"}
                if target is not None
                else {"energy"}
            )
            if isinstance(properties, (str, bytes)):
                raise TypeError("properties must be an iterable of property names")
            default_properties = (
                required if target is not None else {"energy", "forces"}
            )
            requested = frozenset(
                default_properties if properties is None else properties
            )
            if not required <= requested:
                raise ValueError(
                    "properties must include every application-budget observable and energy"
                )
            if requested - self.capabilities.supported_properties:
                raise ValueError(
                    "requested properties are unavailable for this precision endpoint"
                )
            from .ecp import resolve_ecp

            candidate_calculator = self._candidate
            candidate_precision = self._candidate_precision
            route_reasons = list(self._route_reasons)
            if requested - candidate_calculator.capabilities.supported_properties:
                # AUTO can support energies while its force consumer is absent.
                # Choose FP64 for this request without disabling later AUTO calls.
                self._candidate.clear_cache()
                candidate_calculator = Calculator(**self._options, precision="fp64")
                candidate_precision = "fp64"
                route_reasons.append("auto_properties_unavailable_for_model")
            if any(resolve_ecp(candidate_calculator._basis, native_atoms)[0]):
                raise NotImplementedError(
                    "application precision ECP model identity is not implemented"
                )
            model = _model_identity(
                candidate_calculator, native_atoms, charge, multiplicity
            )
            solves = []

            def solve(calculator: Calculator, role: str, precision: str) -> Result:
                if calculator is not self._candidate and (
                    _model_identity(calculator, native_atoms, charge, multiplicity)
                    != model
                ):
                    raise ValueError(
                        "precision audit cannot change the scientific model"
                    )
                solve_started = perf_counter()
                try:
                    result = calculator.singlepoint(
                        native_atoms,
                        charge=charge,
                        multiplicity=multiplicity,
                        properties=requested,
                    )
                except BaseException:
                    calculator.clear_cache()
                    raise
                finally:
                    # Audit owners run sequentially within the same resource
                    # budget; a retained candidate must not double residency.
                    if not (
                        calculator is self._candidate
                        and self.policy.profile == "fast"
                        and target is None
                    ):
                        calculator.clear_cache()
                if not result.converged:
                    raise RuntimeError("precision endpoint did not converge")
                _observable_array("energy", result.energy)
                if "forces" in requested:
                    forces = _observable_array("forces", result.forces)
                    if forces.shape != (len(native_atoms), 3):
                        raise ValueError(
                            "force dimensions differ from the requested atom count"
                        )
                expected_backend = (
                    "cpu_reference" if self._options["device"] == "cpu" else "cuda"
                )
                if result.executed_backend != expected_backend:
                    raise RuntimeError(
                        "precision endpoint changed the requested backend"
                    )
                work = result.precision or {}
                valid = work.get("operator_work_counters_valid") is True
                counters = (
                    "mixed_stage_fock_builds",
                    "strict_stage_fock_builds",
                    "post_scf_fock_builds",
                )
                fock_builds = (
                    sum(work[name] for name in counters)
                    if valid and all(name in work for name in counters)
                    else None
                )
                solves.append(
                    PrecisionSolve(
                        role,
                        precision,
                        perf_counter() - solve_started,
                        result.iterations,
                        int("forces" in requested),
                        fock_builds,
                        result.executed_backend,
                        calculator._energy_tolerance,
                        calculator._density_tolerance,
                        calculator._screening_tolerance,
                    )
                )
                return result

            candidate = solve(candidate_calculator, "candidate", candidate_precision)
            fast_native = self._fast_native and candidate_calculator is self._candidate
            gradient = (candidate.precision or {}).get("gradient") or {}
            gradient_admission = "no-application-qualified-fp32-gradient-route"
            if gradient.get("complete") is True and gradient.get("schema_version") == 1:
                gradient_admission = "pbe0-rks-budget-fp32-v1"
                route_reasons.remove("fp32_gradient_not_qualified")
            if self.policy.profile == "fast" and target is None:
                work = candidate.precision or {}
                if (
                    fast_native
                    and "forces" in requested
                    and work.get("operator_work_counters_valid") is True
                    and work.get("complete") is True
                    and work.get("operator_inventory_complete") is True
                    and work.get("strict_stage_fock_builds", 0) > 0
                    and work.get("final_residual_audits", 0) > 0
                    and gradient.get("complete") is True
                    and gradient.get("schema_version") == 1
                    and gradient.get("attempted_fp32_geometry_points", 0) > 0
                    and gradient.get("returned_mode") in ("mixed", "fp64")
                ):
                    if gradient.get("fp64_gradient_retries", 0):
                        route_reasons.append("native_gradient_retried_fp64")
                    elif not gradient.get("attempted_fp32_grid_points", 0):
                        route_reasons.append("fp32_grid_storage_unavailable")
                    if work.get("execution_retries", 0):
                        route_reasons.append("native_mixed_execution_retried_fp64")
                    validation = PrecisionValidation(
                        self.policy.profile,
                        None,
                        model,
                        "qualified_mixed_gradient"
                        if gradient["returned_mode"] == "mixed"
                        else "qualified_strict_final",
                        tuple(route_reasons),
                        (),
                        (),
                        tuple(solves),
                        perf_counter() - started,
                        reference="pbe0-rks-gradient-v1/independent-qualification-and-native-fp64-scf",
                        gradient_admission=gradient_admission,
                        scope="qualification_not_per_call_observable_audit",
                        budget=self.policy.budget,
                    )
                    return replace(candidate, precision_validation=validation)
                if (
                    fast_native
                    and work.get("operator_work_counters_valid") is True
                    and work.get("complete") is True
                    and work.get("operator_inventory_complete") is True
                    and any(
                        operator.get("name") == "xc"
                        and operator.get("compute") == "fp32"
                        and operator.get("accumulation") == "fp32"
                        and operator.get("arithmetic_mode") == "mixed"
                        and operator.get("count", 0) > 0
                        for operator in work.get("operators", ())
                    )
                    and work.get("mixed_stage_fock_builds", 0) > 0
                    and work.get("strict_stage_fock_builds", 0) >= 2
                    and work.get("strict_refinement_applied") is True
                    and work.get("final_residual_audits", 0) > 0
                ):
                    if work.get("execution_retries", 0):
                        route_reasons.append("native_mixed_execution_retried_fp64")
                    validation = PrecisionValidation(
                        self.policy.profile,
                        None,
                        model,
                        "qualified_strict_final",
                        tuple(route_reasons),
                        (),
                        (),
                        tuple(solves),
                        perf_counter() - started,
                        reference="pbe-xc-warmup-v1/independent-qualification-and-native-fp64-final",
                        scope="qualification_not_per_call_observable_audit",
                        budget=self.policy.budget,
                    )
                    return replace(candidate, precision_validation=validation)
                target = _budget_target(
                    self.policy.budget, forces="forces" in requested
                )
                route_reasons.append("budget_or_domain_requires_fp64_audit")
            if target is None:
                validation = PrecisionValidation(
                    self.policy.profile,
                    None,
                    model,
                    "strict_fp64",
                    tuple(route_reasons),
                    (),
                    (),
                    tuple(solves),
                    perf_counter() - started,
                    reference="native-fp64-final-state-checks-no-observable-audit",
                    scope="native_fp64",
                )
                return replace(candidate, precision_validation=validation)
            self._candidate.clear_cache()
            reference_calculator = self._reference()
            reference = solve(reference_calculator, "fp64_audit", "fp64")
            errors = _errors(target, candidate, reference)
            rejected = ()
            reasons = route_reasons
            if not all(error.met for error in errors):
                rejected = errors
                reasons.append("candidate_exceeded_observable_budget")
                candidate = reference
                gradient_admission = "fp64-after-observable-budget-rejection"
                verification_calculator = self._reference(tighter=True)
                verification = solve(
                    verification_calculator, "fp64_fallback_verification", "fp64"
                )
                errors = _errors(target, candidate, verification)
            validation = PrecisionValidation(
                self.policy.profile,
                target,
                model,
                "observed_met"
                if all(error.met for error in errors)
                else "observed_unmet",
                tuple(reasons),
                errors,
                rejected,
                tuple(solves),
                perf_counter() - started,
                budget=self.policy.budget,
                gradient_admission=gradient_admission,
            )
            if validation.status != "observed_met":
                raise PrecisionBudgetError(validation)
            return replace(candidate, precision_validation=validation)
