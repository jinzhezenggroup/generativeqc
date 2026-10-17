"""Budget-selected shared modules, physical final SCF and explicit FP64 audits.

Fast selects supported Direct/XC arithmetic from TargetErrorBudget and publishes
a physically validated FP64 final state at the selected stationarity allowance.
Strict-density routes may also narrow stationary gradients, with bounded FP64
fallback. Qualification does not invent a per-call observable difference.
TargetAccuracy requests a separate endpoint audit with its full cost recorded.
"""

from __future__ import annotations

import math
import threading
import typing
from dataclasses import asdict, dataclass, replace
from time import perf_counter

from ._api_types import Atom, MethodCapabilities, Result
from ._composite_precision import COMPOSITE_GRADIENT_QUALIFICATIONS
from ._df_precision import DF_GRADIENT_QUALIFICATIONS
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


def _budget_fock_controls(
    calculator: Calculator, budget: TargetErrorBudget, requested: dict
) -> dict[str, float]:
    """Select controls for shared Direct modules, independently of XC identity.

    Native preparation checks the actual provider and basis/resource domain.
    Older libraries, unsupported providers and tighter budgets stay strict.
    Explicit caller tolerances are ceilings, never silently loosened.
    The density ceiling matches the native owner's maximum authenticated force
    stationarity allowance; it is not an observable-error estimate.
    """
    capability = getattr(
        getattr(calculator, "_library", None),
        "generativeqc_application_fock_precision_version_v2",
        None,
    )
    ks_options = getattr(calculator, "_ks_options", None)
    model = getattr(ks_options, "_method_ir", None)
    if (
        capability is None
        or capability() != 2
        or model is None
        or model.spin != "unpolarized"
        or not ks_options.execution_plan.self_consistent
        or ks_options.xc_schedule != "device_fused"
        or getattr(calculator, "_density_fitting_mode", None)
        or budget.energy_abs < 1e-5
        or budget.force_max_abs < 1e-4
        or (budget.force_rms is not None and budget.force_rms < 3e-5)
    ):
        return {}
    return {
        name: min(
            ceiling, ceiling if requested.get(name) is None else float(requested[name])
        )
        for name, ceiling in (
            ("energy_tolerance", 1e-6),
            ("density_tolerance", 1e-4),
            ("screening_tolerance", 1e-10),
        )
    }


def _budget_fock_work_admitted(calculator: Calculator, work: dict) -> bool:
    """Require executed mixed modules and physical closure, not a method label.

    Pure semilocal graphs have no K work to prove. Use the existing native
    full-range projection of the exchange graph, including SR/LR decompositions,
    without confusing separate range recurrence work with full-range K. Optional residency does not
    create recurrence work.
    """
    options = getattr(calculator, "_ks_options", None)
    if options is None or getattr(options, "_method_ir", None) is None:
        return False
    required = {"coulomb_j", "coulomb_recurrence"}
    if options.coefficients[2] != 0:
        required.add("exchange_k")
    observed = {
        operator.get("name")
        for operator in work.get("operators", ())
        if operator.get("compute") == "fp32"
        and operator.get("accumulation") == "fp64"
        and operator.get("arithmetic_mode") == "mixed"
        and operator.get("count", 0) > 0
    }
    return (
        work.get("operator_work_counters_valid") is True
        and work.get("complete") is True
        and work.get("operator_inventory_complete") is True
        and required <= observed
        and work.get("strict_stage_fock_builds", 0) >= 2
        and work.get("strict_refinement_applied") is True
        and work.get("final_residual_audits", 0) > 0
    )


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
    seconds: float | None
    iterations: int | None
    force_evaluations: int | None
    fock_builds: int | None
    executed_backend: str
    energy_tolerance: float
    density_tolerance: float
    screening_tolerance: float
    error: str | None = None


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
    endpoint_seconds: float | None
    reference: str = "same-backend-independent-relaxed-fp64-solve"
    gradient_admission: str = "no-application-qualified-fp32-gradient-route"
    scope: str = "relaxed_target"
    schema_version: int = 1
    budget: TargetErrorBudget | None = None
    timing_scope: str = "singlepoint_endpoint"

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
    ks_options = calculator._effective_ks_options(
        (atoms,), charges=(charge,), multiplicities=(multiplicity,)
    )
    if ks_options is None:
        return calculator.resolved_model(
            atoms, charge=charge, multiplicity=multiplicity
        ).identity
    metadata = calculator.basis_metadata(
        atoms, charge=charge, multiplicity=multiplicity
    )
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


def _complete_gradient_admission(
    calculator: Calculator, result: Result, electronic: str
) -> str | None:
    """Do not present an electronic proof as an arbitrary composite proof.

    Canonical 3c has independent endpoint qualification with the unchanged
    FP64 D4/native-CPU gCP owners. Its electronic precision record still describes
    only the electronic sources; the facade binds the complete model here.
    Other corrected graphs retain the explicit observable audit.
    """
    correction_model = getattr(calculator, "_dispersion_method_ir", None)
    if correction_model is None:
        return electronic
    from generativeqc_compiler.method import resolve_method

    from .dispersion import R2SCAN3CCorrectionResult

    correction = result.dispersion
    if (
        electronic in ("r2scan-rks-budget-fp32-v1", "r2scan-df-rks-budget-fp32-v1")
        and correction_model.identity
        == resolve_method("R2SCAN-3c", spin="unpolarized").identity
        and isinstance(correction, R2SCAN3CCorrectionResult)
        and correction.ok
        and correction.d4.ok
        and correction.gcp.ok
        and correction.backend == "cuda+cpu"
        and correction.gradient is not None
    ):
        return electronic.replace("r2scan-", "r2scan-3c-", 1)
    return None


def _solve_record(
    calculator: Calculator,
    result: typing.Any,
    *,
    role: str,
    precision: str,
    seconds: float | None,
    requested: frozenset[str],
    atom_count: int,
    backend: str,
) -> PrecisionSolve:
    """Check publication and retain only execution-owned semantic work counts.

    A fleet has no measured per-item time. Its native per-item Fock total is
    authoritative when present; scalar results use the existing complete census.
    """
    if not result.converged:
        raise RuntimeError("precision endpoint did not converge")
    _observable_array("energy", result.energy)
    if "forces" in requested:
        forces = _observable_array("forces", result.forces)
        if forces.shape != (atom_count, 3):
            raise ValueError("force dimensions differ from the requested atom count")
    if result.executed_backend != backend:
        raise RuntimeError("precision endpoint changed the requested backend")
    fock_builds = getattr(result, "fock_builds", None)
    if fock_builds is None:
        work = result.precision or {}
        counters = (
            "mixed_stage_fock_builds",
            "strict_stage_fock_builds",
            "post_scf_fock_builds",
        )
        if work.get("operator_work_counters_valid") is True and all(
            name in work for name in counters
        ):
            fock_builds = sum(work[name] for name in counters)
    return PrecisionSolve(
        role,
        precision,
        seconds,
        result.iterations,
        int("forces" in requested),
        fock_builds,
        result.executed_backend,
        calculator._energy_tolerance,
        calculator._density_tolerance,
        calculator._screening_tolerance,
    )


class PrecisionCalculator:
    """Budget-selected arithmetic and explicit all-electron HF/KS precision audits.

    ``policy`` is a preset name or a PrecisionPolicy with an application budget.
    Other options have Calculator semantics except ``precision`` and
    ``target_accuracy``, which this facade owns. FP64 without an explicit target
    runs once in FP64; native exact source reuse also benefits ordinary Calculator.
    Fast selects provisional FP32 J/full-range K through shared Direct-module
    capabilities, independently of XC identity; relaxed candidates keep FP64 forces.
    Other routes select provisional FP32 XC for admitted CUDA PBE-family RKS and
    stationary FP32 gradients for admitted RHF and PBE/PBE0/B3LYP/r2SCAN/M06-2X
    RKS force budgets. Canonical r2SCAN-3c retains its strict corrections. The final
    SCF state stays strict on those other routes; unsupported domains use FP64 audits.
    Qualified DF RKS narrows only semilocal geometry; DF energy-only requests
    still require the audit fallback and have no mixed-arithmetic acceleration.
    Explicit targets/audit profiles cost two full endpoints, or three after
    rejection. FP64 does not imply FP32 arithmetic or a universal speed guarantee.
    Prepared batches reuse native candidate plans and engine-local warm seeds.
    Audits release candidate GPU resources first and never seed candidates from
    reference densities. Correlated methods, Hessians and MD are not promoted.
    Calls on an instance serialize.
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
        if {
            "precision",
            "target_accuracy",
            "_application_precision_budget",
        } & options.keys():
            raise ValueError(
                "set application requirements through policy, not precision/target_accuracy"
            )
        if getattr(options.get("ks_options"), "_precision_budget", None) is not None:
            raise ValueError(
                "set application requirements through policy, not KS transport"
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
        self._fast_fock = False
        self._fast_transport_v2 = False
        self._route_reasons = ["fp32_gradient_not_qualified"]
        if policy.profile == "fast":
            from .ks import KsOptions

            method_transport = getattr(
                getattr(self._candidate, "_library", None),
                "generativeqc_application_precision_version_v2",
                None,
            )
            if device == "cuda" and (
                self._candidate._ks_options is not None or method_transport is not None
            ):
                candidate_options = dict(self._options)
                fock_controls = _budget_fock_controls(
                    self._candidate, policy.budget, options
                )
                candidate_options.update(fock_controls)
                if method_transport is not None:
                    candidate_options["_application_precision_budget"] = policy.budget
                else:
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
                        candidate_options["ks_options"],
                        "_method_ir",
                        ks_options._method_ir,
                    )
                resolved_ks = self._candidate._ks_options
                nonlocal_scf = (
                    resolved_ks is not None
                    and getattr(resolved_ks, "_method_ir", None) is not None
                    and resolved_ks.execution_plan.nonlocal_correlation is not None
                )
                precisions = (
                    ("auto", "fp64")
                    if resolved_ks is not None and (fock_controls or not nonlocal_scf)
                    else ("fp64",)
                )
                if method_transport is None:
                    precisions = ("auto",)
                # Transport does not confer arithmetic admission. In particular,
                # strict SCF may still own budget-qualified derivative consumers.
                for precision in precisions:
                    try:
                        candidate = Calculator(**candidate_options, precision=precision)
                    except NotImplementedError:
                        continue
                    self._candidate = candidate
                    self._candidate_precision = precision
                    self._fast_native = True
                    self._fast_fock = bool(fock_controls) and precision == "auto"
                    self._fast_transport_v2 = method_transport is not None
                    break
                else:
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
            supports_batch=self._candidate.capabilities.supports_batch,
            supported_second_order=frozenset(),
        )

    def clear_cache(self) -> None:
        """Release retained candidate workspaces without changing the policy."""
        with self._lock:
            self._candidate.clear_cache()

    def _reference(
        self, *, tighter: bool = False, _host_reservation: tuple | None = None
    ) -> Calculator:
        """Tighten solver controls without changing the model or converting units.

        Force budgets are never mapped to SCF residuals. This fixed recipe is
        not an observable-error guarantee: the independent audit must pass.
        """
        options = dict(self._options)
        if _host_reservation is not None:
            budget, reserved_bytes = _host_reservation
            options["resource_budget"] = (
                None
                if budget is None
                else replace(
                    budget,
                    host_reserve_bytes=budget.host_reserve_bytes + reserved_bytes,
                )
            )
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

    def prepare_batch(
        self,
        systems: typing.Any,
        *,
        charges: typing.Any = None,
        multiplicities: typing.Any = None,
        warm_start: bool = True,
        shell_class_profiling: bool = False,
        inactive_eigensolver_profiling: bool = False,
        resource_plan: typing.Any = None,
        maximum_seed_bytes: int = 256 << 20,
    ) -> typing.Any:
        """Prepare a reusable native fleet with per-item precision validation.

        Qualified Fast and untargeted FP64 retain the original native plan.
        Audits suspend that plan with bounded host seed storage; later execution
        restores only its own seeds, never an auditor's density. Candidate batch
        timing is shared, not invented per-item or divided equally among items.
        ``maximum_seed_bytes`` bounds numeric seed staging when an audit is needed.
        """
        from ._precision_batch import PreparedPrecisionBatch

        with self._lock:
            return PreparedPrecisionBatch(
                self,
                systems,
                charges=charges,
                multiplicities=multiplicities,
                warm_start=warm_start,
                shell_class_profiling=shell_class_profiling,
                inactive_eigensolver_profiling=inactive_eigensolver_profiling,
                resource_plan=resource_plan,
                maximum_seed_bytes=maximum_seed_bytes,
            )

    def _properties(self, properties: typing.Any) -> frozenset[str]:
        """Validate requested observables identically for scalar and fleet drivers."""
        target = self.policy.target
        required = (
            {requirement.observable for requirement in target.observables} | {"energy"}
            if target is not None
            else {"energy"}
        )
        if isinstance(properties, (str, bytes)):
            raise TypeError("properties must be an iterable of property names")
        default_properties = required if target is not None else {"energy", "forces"}
        requested = frozenset(default_properties if properties is None else properties)
        if not required <= requested:
            raise ValueError(
                "properties must include every application-budget observable and energy"
            )
        if requested - self.capabilities.supported_properties:
            raise ValueError(
                "requested properties are unavailable for this precision endpoint"
            )
        return requested

    def _candidate_for(self, requested: frozenset[str]) -> tuple:
        """Choose a capable candidate without changing later singlepoint requests."""
        candidate_calculator = self._candidate
        candidate_precision = self._candidate_precision
        route_reasons = list(self._route_reasons)
        if requested - candidate_calculator.capabilities.supported_properties:
            # AUTO can support energies while its force consumer is absent.
            # Choose FP64 for this request without disabling later AUTO calls.
            self._candidate.clear_cache()
            candidate_options = dict(self._options)
            if self._fast_transport_v2:
                candidate_options["_application_precision_budget"] = self.policy.budget
            candidate_calculator = Calculator(**candidate_options, precision="fp64")
            candidate_precision = "fp64"
            route_reasons.append("auto_properties_unavailable_for_model")
        return candidate_calculator, candidate_precision, route_reasons

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
            requested = self._properties(properties)
            from .ecp import resolve_ecp

            candidate_calculator, candidate_precision, route_reasons = (
                self._candidate_for(requested)
            )
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
                except BaseException as error:
                    calculator.clear_cache()
                    if self._fast_fock and calculator is self._candidate:
                        solves.append(
                            PrecisionSolve(
                                role,
                                precision,
                                perf_counter() - solve_started,
                                None,
                                None,
                                None,
                                "cuda",
                                calculator._energy_tolerance,
                                calculator._density_tolerance,
                                calculator._screening_tolerance,
                                f"{type(error).__name__}: {error}",
                            )
                        )
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
                solves.append(
                    _solve_record(
                        calculator,
                        result,
                        role=role,
                        precision=precision,
                        seconds=perf_counter() - solve_started,
                        requested=requested,
                        atom_count=len(native_atoms),
                        backend="cpu_reference"
                        if self._options["device"] == "cpu"
                        else "cuda",
                    )
                )
                return result

            return self._validate_endpoint(
                candidate_calculator=candidate_calculator,
                candidate_precision=candidate_precision,
                requested=requested,
                route_reasons=route_reasons,
                model=model,
                solves=solves,
                solve=solve,
                started=started,
            )

    def _validate_endpoint(
        self,
        *,
        candidate_calculator: Calculator,
        candidate_precision: str,
        requested: frozenset[str],
        route_reasons: list[str],
        model: str,
        solves: list[PrecisionSolve],
        solve: typing.Callable,
        started: float,
        reference_factory: typing.Callable | None = None,
    ) -> typing.Any:
        """Apply one policy to either a singlepoint or an input-indexed batch item.

        Drivers own execution, residency and timing. This shared controller owns
        complete-model qualification, explicit audits and bounded rejection.
        """
        target = self.policy.target
        if reference_factory is None:
            reference_factory = self._reference
        try:
            candidate = solve(candidate_calculator, "candidate", candidate_precision)
        except (RuntimeError, NotImplementedError):
            if self.policy.profile != "fast" or not self._fast_native:
                raise
            route_reasons.append(
                "budget_fock_failed_fp64_fallback"
                if self._fast_fock
                else "mixed_gradient_failed_fp64_fallback"
            )
            candidate_calculator = reference_factory()
            candidate_precision = "fp64"
            fast_native = False
            candidate = solve(candidate_calculator, "fp64_fallback", "fp64")
        fast_native = self._fast_native and (
            candidate_calculator is self._candidate or self._fast_transport_v2
        )
        gradient = (candidate.precision or {}).get("gradient") or {}
        gradient_admission = "no-application-qualified-fp32-gradient-route"
        qualification = gradient.get(
            "qualification",
            "pbe0-rks-budget-fp32-v1"
            if gradient.get("schema_version") == 1
            else "unknown",
        )
        gradient_requires_audit = gradient.get("requires_independent_audit") is True
        if gradient_requires_audit:
            # Beyond the frozen acceptance cohort and after a budget-Fock SCF,
            # a converged/finite FP32 gradient is not a measured force budget.
            # Keep the candidate, then compare complete E+F with fresh FP64.
            gradient_admission = qualification
            route_reasons.append("mixed_gradient_requires_independent_fp64_audit")
        qualified_gradient = (
            not gradient_requires_audit
            and gradient.get("complete") is True
            and (
                (
                    gradient.get("schema_version") in (1, 2, 3)
                    and qualification
                    in (
                        "pbe0-rks-budget-fp32-v1",
                        "pbe-rks-budget-fp32-v1",
                        "b3lyp-rks-budget-fp32-v1",
                        "r2scan-rks-budget-fp32-v1",
                        "rhf-budget-fp32-v1",
                        "rhf-df-shell-budget-fp32-v1",
                        "m06-2x-rks-budget-fp32-v1",
                    )
                )
                or (
                    gradient.get("schema_version") == 4
                    and (
                        (
                            gradient.get("owner") == "python-stationary-cuda"
                            and qualification in DF_GRADIENT_QUALIFICATIONS
                        )
                        or (
                            gradient.get("owner") == "python-composite-stationary-cuda"
                            and qualification in COMPOSITE_GRADIENT_QUALIFICATIONS
                        )
                    )
                )
            )
        )
        if qualified_gradient:
            admitted = _complete_gradient_admission(
                candidate_calculator, candidate, qualification
            )
            qualified_gradient = admitted is not None
            if qualified_gradient:
                gradient_admission = admitted
                route_reasons.remove("fp32_gradient_not_qualified")
                if gradient.get("capability_fallback"):
                    route_reasons.append("fp32_geometry_capability_unavailable")
            else:
                route_reasons.append("geometry_correction_not_qualified")
        if self.policy.profile == "fast" and target is None:
            work = candidate.precision or {}
            if (
                fast_native
                and self._fast_fock
                and not gradient_requires_audit
                and _budget_fock_work_admitted(candidate_calculator, work)
            ):
                validation = PrecisionValidation(
                    self.policy.profile,
                    None,
                    model,
                    "qualified_budget_fock",
                    tuple(route_reasons),
                    (),
                    (),
                    tuple(solves),
                    perf_counter() - started,
                    reference="direct-fock-budget-v2/shared-module-qualification",
                    gradient_admission=gradient_admission,
                    scope="qualification_not_per_call_observable_audit",
                    budget=self.policy.budget,
                )
                return replace(candidate, precision_validation=validation)
            if (
                fast_native
                and not self._fast_fock
                and "forces" in requested
                and qualified_gradient
                # V2 is emitted only after the complete gradient consumer
                # verifies its strict final-state token. Chunked SCF may
                # lack detailed operator rows; do not fabricate that census.
                and (
                    gradient.get("schema_version") in (2, 3, 4)
                    or (
                        work.get("operator_work_counters_valid") is True
                        and work.get("complete") is True
                        and work.get("operator_inventory_complete") is True
                        and work.get("strict_stage_fock_builds", 0) > 0
                        and work.get("final_residual_audits", 0) > 0
                    )
                )
                and (
                    gradient.get("attempted_fp32_geometry_points", 0) > 0
                    or (
                        gradient.get("schema_version") == 3
                        and gradient.get("weighted_task_counters_collected") is True
                        and (gradient.get("attempted_fp32_weighted_tasks") or 0) > 0
                        and (gradient.get("attempted_fp32_primitive_sources") or 0) > 0
                    )
                )
                and gradient.get("returned_mode") in ("mixed", "fp64")
            ):
                if gradient.get("fp64_gradient_retries", 0):
                    route_reasons.append("native_gradient_retried_fp64")
                elif (
                    gradient.get("schema_version") != 4
                    and qualification
                    not in ("rhf-budget-fp32-v1", "rhf-df-shell-budget-fp32-v1")
                    and not gradient.get("attempted_fp32_grid_points", 0)
                ):
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
                    reference=f"{gradient_admission}/independent-qualification-and-native-fp64-scf",
                    gradient_admission=gradient_admission,
                    scope="qualification_not_per_call_observable_audit",
                    budget=self.policy.budget,
                )
                return replace(candidate, precision_validation=validation)
            if (
                fast_native
                and not self._fast_fock
                and getattr(candidate_calculator, "_dispersion_method_ir", None) is None
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
            target = _budget_target(self.policy.budget, forces="forces" in requested)
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
        reference_calculator = reference_factory()
        reference = solve(reference_calculator, "fp64_audit", "fp64")
        errors = _errors(target, candidate, reference)
        rejected = ()
        reasons = route_reasons
        if not all(error.met for error in errors):
            rejected = errors
            reasons.append("candidate_exceeded_observable_budget")
            candidate = reference
            gradient_admission = "fp64-after-observable-budget-rejection"
            verification_calculator = reference_factory(tighter=True)
            verification = solve(
                verification_calculator, "fp64_fallback_verification", "fp64"
            )
            errors = _errors(target, candidate, verification)
        validation = PrecisionValidation(
            self.policy.profile,
            target,
            model,
            "observed_met" if all(error.met for error in errors) else "observed_unmet",
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
