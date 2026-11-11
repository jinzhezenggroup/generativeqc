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
from ._observable_comparison import (
    _observable_array,
    measure_observables,
    require_observable_comparison,
)
from ._precision_candidate import (
    REASONS,
    Disposition,
    classify_candidate,
    require_candidate_classification,
)
from ._precision_execution import (
    PrecisionEvent,
    PrecisionOutcome,
    require_precision_schedule,
    run_precision_schedule,
)
from ._scf_defaults_native import default_controls
from .accuracy import (
    ObservableTarget,
    TargetAccuracy,
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
    """Select controls for shared Direct modules or separately qualified DF XC.

    Direct admission is independent of XC identity; DF authenticates its local
    complete model and keeps J strict. Native preparation checks the actual
    provider and basis/resource domain.
    Unsupported providers and tighter budgets stay strict. The selected library
    must provide the shared policy rather than delegate its selection to Python.
    Explicit caller tolerances are ceilings, never silently loosened.
    The density ceiling matches the native owner's maximum authenticated force
    stationarity allowance; it is not an observable-error estimate.
    """
    from ._df_precision import fitted_scf_budget_admitted

    capability = getattr(
        getattr(calculator, "_library", None),
        "generativeqc_application_fock_precision_version_v2",
        None,
    )
    fitted = bool(getattr(calculator, "_density_fitting_mode", None))
    available = (
        fitted_scf_budget_admitted(calculator, budget)
        if fitted
        else capability is not None and capability() == 2
    )
    from ._precision_native import budget_controls

    return budget_controls(calculator, budget, requested, available=available)


def _budget_fock_work_admitted(calculator: Calculator, work: dict) -> bool:
    """Use the selected native owner's completed-work admission receipt."""
    from ._precision_native import admission

    return admission(calculator, work)[0]


def _budget_fock_strict_warm_admitted(
    calculator: Calculator, result: typing.Any, *, qualified_gradient: bool = False
) -> bool:
    """Require the shared native policy's exact returned-state physical proof."""
    from ._precision_native import admission

    decision = admission(
        calculator,
        result.precision or {},
        result=result,
        qualified_gradient=qualified_gradient,
    )
    return decision[1]


def _standard_fallback_admitted(
    calculator: Calculator,
    candidate: typing.Any,
    precision: str,
    budget: TargetErrorBudget,
) -> bool:
    """Query strict fallback without treating absent counts as measured zeros."""
    from ._precision_native import admission

    decision = admission(
        calculator,
        candidate.precision or {},
        result=candidate,
        precision=precision,
        budget=budget,
    )
    return decision[2]


@dataclass(frozen=True)
class PrecisionPolicy:
    """Select arithmetic by application budget, or request explicit audits.

    Fast's ``budget`` reuses the adaptive-policy allowance type and controls
    native FP32 admission. CCSDTCBSCalculator also uses this allowance for
    final FP64 convergence limits; its Standard uses the default Fast allowance
    and its CBS-only Tight preset retains strict convergence. These are not
    measured per-call error certificates.
    Explicit ``target`` always requests a relaxed audit, including with Fast.
    Standard is the default. Standard (or legacy FP64) without ``target`` executes once in FP64; no observable error estimate
    is invented. Supplying a target opts into the full audit, including its cost.
    Other presets are audit allowances, not measured error ranges. Only
    relaxed-target audits are admitted.
    """

    profile: str = "standard"
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
            "standard",
            "tight",
            "fp64",
            *_DEFAULT_BUDGETS,
        ):
            raise ValueError(
                "precision profile must be 'standard', 'tight', 'fast', 'fp64', 'audit_auto', 'balanced' or 'strict'"
            )
        if (
            self.profile in ("fast", "standard", "tight", "fp64")
            and self.target is None
        ):
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
    target: TargetAccuracy, candidate: Result, reference: Result, *, library: typing.Any
) -> tuple[PrecisionError, ...]:
    """Publish the selected native owner's complete norm/allowance receipt."""
    if not candidate.converged or not reference.converged:
        raise RuntimeError("application precision audit requires two converged solves")
    return tuple(
        PrecisionError(
            requirement.observable,
            requirement.norm,
            requirement.unit,
            difference,
            reference_norm,
            allowance,
        )
        for requirement, difference, reference_norm, allowance in measure_observables(
            target,
            {"energy": candidate.energy, "forces": candidate.forces},
            {"energy": reference.energy, "forces": reference.forces},
            library=library,
        )
    )


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

    ``policy`` defaults to Standard, the FP64 profile used by the README benchmark.
    Explicit audit profiles or targets retain their complete comparison work.
    ``policy`` is a preset name or a PrecisionPolicy with an application budget.
    Other options have Calculator semantics except ``precision`` and
    ``target_accuracy``, which this facade owns. Standard (legacy alias: fp64)
    without an explicit target
    runs once in FP64; native exact source reuse also benefits ordinary Calculator.
    Fast selects provisional FP32 J/full-range K through shared Direct-module
    capabilities, independently of XC identity; relaxed candidates keep FP64 forces.
    Other routes select provisional FP32 XC for admitted CUDA PBE-family RKS and
    stationary FP32 gradients for admitted RHF and PBE/PBE0/B3LYP/r2SCAN/M06-2X
    RKS force budgets. Canonical r2SCAN-3c retains its strict corrections. The final
    SCF state stays strict on those other routes; unsupported domains use FP64 audits.
    Qualified DF RKS may narrow semilocal geometry. Default-grid DF PBE/r2SCAN
    RKS may also use FP32 XC matrix products in bounded SCF proposals; DF J and
    the metric remain FP64. Other DF requests retain their strict fallback or
    audit path.
    Explicit targets/audit profiles cost two full endpoints, or three after
    rejection. FP64 does not imply FP32 arithmetic or a universal speed guarantee.
    Prepared batches reuse native candidate plans and engine-local warm seeds.
    Audits release candidate GPU resources first and never seed candidates from
    reference densities. Correlated methods, Hessians and MD are not promoted.
    Standard and Fast retain successful unaudited scalar candidates alike;
    explicit cache clearing, audit transitions and failures release residency.
    Calls on an instance serialize.
    """

    def __init__(
        self,
        method: typing.Any = "rhf",
        basis: typing.Any = None,
        device: str = "cpu",
        *,
        policy: PrecisionPolicy | str = "standard",
        **options: typing.Any,
    ) -> None:
        """Snapshot one native model and select its admitted candidate precision."""
        if isinstance(policy, str):
            policy = PrecisionPolicy(policy)
        if not isinstance(policy, PrecisionPolicy):
            raise TypeError("policy must be a PrecisionPolicy or preset name")
        if policy.profile == "tight":
            raise NotImplementedError(
                "the tight preset is specific to CCSDTCBSCalculator"
            )
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
        if policy.profile in ("standard", "fp64", "fast"):
            for name in (
                "energy_tolerance",
                "density_tolerance",
                "screening_tolerance",
            ):
                requested = self._options.get(name)
                if requested is not None:
                    requested = float(requested)
                    if not math.isfinite(requested) or requested <= 0:
                        raise ValueError(f"{name} must be finite and positive")
        self._candidate = Calculator(**self._options, precision="fp64")
        if policy.profile in ("standard", "fp64", "fast"):
            # Select the same qualified Standard controls as Calculator after
            # resolving the native library and complete model. Unsupported
            # native owners keep their strict defaults. Missing policy support
            # rejects the installation instead of choosing Python defaults. Tight
            # caller values survive both Standard and Fast fallback paths.
            for name, ceiling in default_controls(self._candidate).items():
                requested = self._options.get(name)
                value = min(ceiling, ceiling if requested is None else float(requested))
                self._options[name] = value
                setattr(self._candidate, "_" + name, value)
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
            from ._precision_native import require_admission
            from .ks import KsOptions

            require_admission(self._candidate)
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
        require_precision_schedule(self._candidate._library)
        require_observable_comparison(self._candidate._library)
        require_candidate_classification(self._candidate._library)

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
        Auditors may warm from their own same-geometry, same-control densities;
        every audit still recomputes its observables. ``maximum_seed_bytes``
        bounds combined candidate and auditor numeric seed staging. Optional
        audit history is discarded when that bound cannot accommodate it.
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
                    record = _solve_record(
                        calculator,
                        result,
                        role=role,
                        precision=precision,
                        seconds=None,
                        requested=requested,
                        atom_count=len(native_atoms),
                        backend="cpu_reference"
                        if self._options["device"] == "cpu"
                        else "cuda",
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
                    if not (calculator is self._candidate and target is None):
                        calculator.clear_cache()
                solves.append(replace(record, seconds=perf_counter() - solve_started))
                return result

            try:
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
            except BaseException:
                # Classification can fail after a successful solve. Neither
                # profile may retain an unpublished candidate's workspace.
                self._candidate.clear_cache()
                raise

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
        """Adapt endpoint owners to the selected native execution controller.

        Numerical qualification and observable measurement keep their existing
        owners. Native code alone chooses candidate retry, audit and bounded
        replacement verification; Python only executes each requested operation
        and publishes a complete detached validation after receipt checks.
        """
        target = self.policy.target
        if reference_factory is None:
            reference_factory = self._reference
        candidate = reference = publication = None
        errors = rejected = ()
        gradient_admission = "no-application-qualified-fp32-gradient-route"
        selected_library = candidate_calculator._library
        require_observable_comparison(selected_library)
        require_candidate_classification(selected_library)

        def operation(event: PrecisionEvent) -> int:
            nonlocal candidate_calculator, candidate_precision, candidate, reference
            nonlocal publication, target, errors, rejected, gradient_admission
            if event == PrecisionEvent.CANDIDATE:
                candidate = solve(
                    candidate_calculator, "candidate", candidate_precision
                )
                return 0
            if event == PrecisionEvent.FP64_FALLBACK:
                route_reasons.append("budget_fock_failed_fp64_fallback")
                candidate_calculator = reference_factory()
                candidate_precision = "fp64"
                candidate = solve(candidate_calculator, "fp64_fallback", "fp64")
                return 0
            if candidate is None:
                raise RuntimeError(
                    "native precision controller has no completed candidate"
                )
            if event == PrecisionEvent.CLASSIFY:
                publication, target, gradient_admission = self._classify_candidate(
                    candidate=candidate,
                    candidate_calculator=candidate_calculator,
                    candidate_precision=candidate_precision,
                    requested=requested,
                    route_reasons=route_reasons,
                    model=model,
                    solves=solves,
                    started=started,
                )
                return int(publication is not None)
            if event == PrecisionEvent.RELEASE:
                self._candidate.clear_cache()
                return 0
            if target is None:
                raise RuntimeError(
                    "native precision controller requested an undeclared audit"
                )
            if event == PrecisionEvent.AUDIT:
                reference = solve(reference_factory(), "fp64_audit", "fp64")
                errors = _errors(target, candidate, reference, library=selected_library)
                return int(all(error.met for error in errors))
            if event == PrecisionEvent.VERIFY:
                if reference is None:
                    raise RuntimeError(
                        "native precision controller has no independent auditor"
                    )
                rejected = errors
                route_reasons.append("candidate_exceeded_observable_budget")
                candidate = reference
                gradient_admission = "fp64-after-observable-budget-rejection"
                verification = solve(
                    reference_factory(tighter=True),
                    "fp64_fallback_verification",
                    "fp64",
                )
                errors = _errors(
                    target, candidate, verification, library=selected_library
                )
                return int(all(error.met for error in errors))
            raise RuntimeError(
                "native precision controller requested an unknown operation"
            )

        outcome = run_precision_schedule(
            selected_library, operation, retry_candidate=self._fast_fock
        )
        if outcome == PrecisionOutcome.QUALIFIED:
            if publication is None:
                raise RuntimeError(
                    "native precision controller omitted candidate publication"
                )
            return replace(
                candidate,
                precision_validation=replace(
                    publication, endpoint_seconds=perf_counter() - started
                ),
            )
        validation = PrecisionValidation(
            self.policy.profile,
            target,
            model,
            "observed_unmet"
            if outcome == PrecisionOutcome.REJECTED
            else "observed_met",
            tuple(route_reasons),
            errors,
            rejected,
            tuple(solves),
            perf_counter() - started,
            budget=self.policy.budget,
            gradient_admission=gradient_admission,
        )
        if outcome == PrecisionOutcome.REJECTED:
            raise PrecisionBudgetError(validation)
        return replace(candidate, precision_validation=validation)

    def _classify_candidate(
        self,
        *,
        candidate: Result,
        candidate_calculator: Calculator,
        candidate_precision: str,
        requested: frozenset[str],
        route_reasons: list[str],
        model: str,
        solves: list[PrecisionSolve],
        started: float,
    ) -> tuple[PrecisionValidation | None, TargetAccuracy | None, str]:
        """Present the selected native owner's route without selecting policy."""
        disposition, reasons, gradient_admission = classify_candidate(
            candidate_calculator,
            candidate,
            profile=self.policy.profile,
            target=self.policy.target,
            fast_native=self._fast_native
            and (candidate_calculator is self._candidate or self._fast_transport_v2),
            fast_fock=self._fast_fock,
            forces="forces" in requested,
            precision=candidate_precision,
            budget=self.policy.budget,
        )
        if reasons & 1:
            route_reasons.remove("fp32_gradient_not_qualified")
        route_reasons.extend(reason for bit, reason in REASONS if reasons & bit)
        target = self.policy.target
        if reasons & 128:
            target = _budget_target(self.policy.budget, forces="forces" in requested)
        if disposition == Disposition.AUDIT:
            return None, target, gradient_admission
        references = {
            Disposition.STRICT: "native-fp64-final-state-checks-no-observable-audit",
            Disposition.FOCK: (
                "df-xc-budget-v1/strict-j-shared-xc-qualification"
                if getattr(candidate_calculator, "_density_fitting_mode", None)
                else "direct-fock-budget-v2/shared-module-qualification"
            ),
            Disposition.STRICT_XC: "pbe-xc-warmup-v1/independent-qualification-and-native-fp64-final",
            Disposition.STANDARD: "native-standard-fp64-final-state-checks-no-observable-audit",
        }
        references[Disposition.WARM_FOCK] = references[Disposition.FOCK]
        references[Disposition.MIXED_GRADIENT] = references[
            Disposition.STRICT_GRADIENT
        ] = f"{gradient_admission}/independent-qualification-and-native-fp64-scf"
        statuses = {
            Disposition.STRICT: "strict_fp64",
            Disposition.FOCK: "qualified_budget_fock",
            Disposition.WARM_FOCK: "qualified_budget_fock_strict_warm",
            Disposition.MIXED_GRADIENT: "qualified_mixed_gradient",
            Disposition.STRICT_GRADIENT: "qualified_strict_final",
            Disposition.STRICT_XC: "qualified_strict_final",
            Disposition.STANDARD: "strict_standard_fallback",
        }
        qualification = disposition not in (Disposition.STRICT, Disposition.STANDARD)
        validation = PrecisionValidation(
            self.policy.profile,
            None,
            model,
            statuses[disposition],
            tuple(route_reasons),
            (),
            (),
            tuple(solves),
            perf_counter() - started,
            reference=references[disposition],
            gradient_admission=gradient_admission
            if qualification and disposition != Disposition.STRICT_XC
            else "no-application-qualified-fp32-gradient-route",
            scope="qualification_not_per_call_observable_audit"
            if qualification
            else "native_fp64",
            budget=self.policy.budget if disposition != Disposition.STRICT else None,
        )
        return validation, target, gradient_admission
