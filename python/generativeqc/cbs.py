"""MP2-assisted, all-electron canonical CCSD(T)/CBS energy and analytic forces.

Only orchestration and geometry-independent linear combinations live here.
Every electronic energy and force comes from the public native calculator;
reference packages and numerical differentiation are not production consumers.
"""

from __future__ import annotations

import math
import threading
import time
import typing
from dataclasses import asdict, dataclass, field
from types import MappingProxyType

import numpy as np

from ._api_types import Atom, Result
from ._model_resolution import snapshot_basis
from .basis import BasisSet
from .calculator import Calculator
from .elements import checked_integer, electron_state

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from .precision_policy import PrecisionPolicy


def _basis_snapshot(basis: BasisSet | str | Path) -> BasisSet:
    """Keep file identities and representations; named bases use spherical AOs."""
    representation = (
        "spherical" if isinstance(basis, str) and not basis.endswith(".json") else None
    )
    record = snapshot_basis(basis, representation)
    if not isinstance(record, BasisSet):
        raise TypeError("CBS requires a named basis, canonical basis file or BasisSet")
    if any(
        element.ecp_core_electrons
        or element.ecp_data is not None
        or element.nuclear_charge != element.atomic_number
        for element in record.elements
    ):
        raise NotImplementedError("CCSD(T)/CBS currently requires all-electron bases")
    return record


@dataclass(frozen=True)
class CBSBasis:
    """An owned basis record with an explicitly declared cardinal number.

    Declare 2/3/4 for a DZ/TZ/QZ member of one consistent basis family. The
    cardinal number is a scientific input, not inferred from the number of AOs
    or a filename. Arbitrary local records are allowed for qualification, but
    the caller remains responsible for choosing a meaningful extrapolation
    family. No basis download or reference-package lookup occurs.
    """

    basis: BasisSet | str | Path
    cardinal: int

    def __post_init__(self) -> None:
        """Validate the cardinal number and snapshot local all-electron data."""
        cardinal = checked_integer(
            self.cardinal, "CBS cardinal number", low=2, high=100
        )
        object.__setattr__(self, "cardinal", cardinal)
        object.__setattr__(self, "basis", _basis_snapshot(self.basis))


@dataclass(frozen=True)
class CBSComponent:
    """One completed native endpoint; equal method/basis roles execute once."""

    roles: tuple[str, ...]
    method: str
    basis_identity: str
    coefficient: float
    result: Result
    elapsed_seconds: float


@dataclass(frozen=True)
class CBSResult:
    """Complete focal-point output in Hartree and Hartree/Bohr.

    ``energy = hf_energy + mp2_correlation_energy + delta_ccsdt_energy``;
    forces use the identical, geometry-independent weights. Components retain
    native convergence/resource/response diagnostics after their owners close.
    This is an explicitly approximate CCSD(T)/CBS model, not direct CCSD(T)
    extrapolation or a certified basis-set error estimate. ``convergence``
    records selected stopping limits; it does not estimate observable errors.
    """

    energy: float
    forces: np.ndarray | None
    hf_energy: float
    mp2_correlation_energy: float
    delta_ccsdt_energy: float
    hf_forces: np.ndarray | None
    mp2_correlation_forces: np.ndarray | None
    delta_ccsdt_forces: np.ndarray | None
    hf_weights: tuple[float, float]
    mp2_weights: tuple[float, float]
    components: tuple[CBSComponent, ...]
    converged: bool
    executed_backend: str
    basis_metadata: dict[str, typing.Any]
    endpoint_seconds: float
    execution_groups: tuple[dict[str, typing.Any], ...]
    convergence: dict[str, typing.Any] = field(default_factory=dict)


def _limit_weights(log_ratio: float) -> tuple[float, float]:
    """Stable two-point limit for a decaying basis-error model."""
    denominator = -math.expm1(log_ratio)
    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError("CBS extrapolation parameters produce singular weights")
    high = 1.0 / denominator
    low = -math.exp(log_ratio) / denominator
    if not all(math.isfinite(value) for value in (low, high)):
        raise ValueError("CBS extrapolation parameters produce nonfinite weights")
    return low, high


class CCSDTCBSCalculator:
    """Basis-grouped native MP2-assisted CCSD(T)/CBS E+F workflow.

    The model is HF(limit) + MP2-correlation(limit) + [CCSD(T)-MP2](small).
    MP2 correlation uses ``X**(-correlation_exponent)``. HF uses a fixed
    ``exp(-hf_exponent * X)`` model, or the highest-basis HF when explicitly
    selecting ``hf_extrapolation="highest"``. Exponents are model inputs and
    are never fitted to the current geometry.

    All components use FP64, all-electron RHF references. Explicit auxiliary
    bases default to correlation-only DF-MP2 and DF-CCSD(T). ``hf_reference="df"``
    instead uses that same auxiliary basis for stationary RI-JK, its orbital
    response and its metric/center derivatives. Choose a basis suitable for
    both JK and correlation fitting; an MP2-fit label alone does not certify
    HF accuracy. This explicit model choice never changes on resource refusal.
    Open-shell, ECP and frozen-core variants are not silently substituted.
    Native backend/basis/force and complete memory-admission checks still apply.

    Same-basis components share one native RHF reference and correlation source.
    Native retained storage is released after each basis group, even on failure.
    Thus the workflow does not retain several basis-dependent CUDA owners at
    once. ``calculator_options`` supplies common SCF/CC/resource controls;
    budgets remain native endpoint budgets, not a total process-RSS bound.
    ``policy="standard"`` selects budget-scaled convergence, including force
    response. ``tight`` retains the original strict stopping limits; ``fp64``
    additionally keeps FP64 preconvergence. ``fast`` remains a compatible name
    for Standard's default allowance and accepts custom budgets.
    ``preconvergence`` independently selects bounded auxiliary
    warmup for CUDA DF-CCSD/Lambda. Final residual checks and response remain
    FP64. It preserves explicit tighter SCF/CC controls. The returned
    convergence metadata describes stopping limits, not measured error bounds.
    Calls on one workflow are serialized. Use separate workflows for concurrency.
    """

    def __init__(
        self,
        mp2_bases: Sequence[CBSBasis],
        *,
        correction_basis: BasisSet | str | Path | None = None,
        auxiliary_bases: Sequence[BasisSet | str | Path] | None = None,
        correction_auxiliary_basis: BasisSet | str | Path | None = None,
        hf_reference: str = "exact",
        device: str = "cpu",
        device_id: int = 0,
        correlation_exponent: float = 3.0,
        hf_extrapolation: str = "exponential",
        hf_exponent: float = 1.63,
        calculator_options: Mapping[str, typing.Any] | None = None,
        policy: PrecisionPolicy | str = "standard",
        preconvergence: str = "auto",
    ) -> None:
        """Bind basis/cardinal snapshots and the shared correlation Hamiltonian.

        Supplying two auxiliary bases selects CUDA correlation-only DF for
        both MP2 and CCSD(T), retaining conventional RHF references. A distinct
        correction basis requires its own explicit auxiliary basis. Equal
        orbital bases must use the same auxiliary snapshot within this model.
        ``hf_reference="df"`` selects full DF rather than a DF initial guess;
        the supplied auxiliary snapshots then serve both JK and correlation.
        ``preconvergence`` selects bounded DF-CCSD/Lambda auxiliary warmup
        independently of final stopping limits: ``mixed``, ``fp64`` or ``auto``.
        Auto uses native shape admission on supported fitted CUDA owners;
        older libraries retain their legacy schedule. The ``fp64`` policy
        keeps FP64 warmup.
        """
        from .precision_policy import PrecisionPolicy

        if isinstance(policy, str):
            policy = PrecisionPolicy(policy)
        if not isinstance(policy, PrecisionPolicy):
            raise TypeError("CBS policy must be a PrecisionPolicy or preset name")
        if (
            policy.profile not in ("standard", "tight", "fp64", "fast")
            or policy.target is not None
        ):
            raise NotImplementedError(
                "CBS supports standard/tight/fp64/fast without target audits"
            )
        self.policy = policy
        if preconvergence not in ("auto", "fp64", "mixed"):
            raise ValueError("CBS preconvergence must be auto/fp64/mixed")
        if policy.profile == "fp64" and preconvergence == "mixed":
            raise ValueError("fp64 policy cannot request mixed preconvergence")
        self.preconvergence = preconvergence
        bases = tuple(mp2_bases)
        if len(bases) != 2 or not all(isinstance(item, CBSBasis) for item in bases):
            raise ValueError("mp2_bases must contain two CBSBasis records")
        low, high = bases
        if low.cardinal >= high.cardinal:
            raise ValueError("CBS bases must have strictly increasing cardinal numbers")
        if low.basis.identity == high.basis.identity:
            raise ValueError("CBS extrapolation requires two distinct bases")
        small = (
            low.basis if correction_basis is None else _basis_snapshot(correction_basis)
        )
        if (
            len(
                {
                    low.basis.representation,
                    high.basis.representation,
                    small.representation,
                }
            )
            != 1
        ):
            raise ValueError("all CBS bases must use the same AO representation")
        if device not in ("cpu", "cuda"):
            raise ValueError("CBS device must be 'cpu' or 'cuda'")
        auxiliaries = {}
        if auxiliary_bases is not None:
            fitted = tuple(_basis_snapshot(basis) for basis in auxiliary_bases)
            if len(fitted) != 2:
                raise ValueError("DF-CBS requires two explicit auxiliary bases")
            if device != "cuda":
                raise NotImplementedError(
                    "DF-CBS analytic endpoints currently require CUDA"
                )
            auxiliaries = {
                low.basis.identity: fitted[0],
                high.basis.identity: fitted[1],
            }
            if correction_auxiliary_basis is not None:
                correction_auxiliary = _basis_snapshot(correction_auxiliary_basis)
                if (
                    small.identity in auxiliaries
                    and auxiliaries[small.identity].identity
                    != correction_auxiliary.identity
                ):
                    raise ValueError(
                        "equal CBS orbital bases must share their auxiliary basis"
                    )
                auxiliaries[small.identity] = correction_auxiliary
            if small.identity not in auxiliaries:
                raise ValueError(
                    "a distinct DF correction basis requires correction_auxiliary_basis"
                )
        elif correction_auxiliary_basis is not None:
            raise ValueError("correction_auxiliary_basis requires auxiliary_bases")
        self._auxiliary_bases = MappingProxyType(auxiliaries)
        if preconvergence == "mixed" and not auxiliaries:
            raise NotImplementedError("mixed CBS preconvergence requires CUDA DF bases")
        if hf_reference not in ("exact", "df"):
            raise ValueError("hf_reference must be 'exact' or 'df'")
        if hf_reference == "df" and not auxiliaries:
            raise ValueError("DF-HF CBS requires explicit auxiliary_bases")
        self.hf_reference = hf_reference
        checked_integer(device_id, "CBS device ID", high=2**31 - 1)
        if hf_extrapolation not in ("exponential", "highest"):
            raise ValueError("hf_extrapolation must be 'exponential' or 'highest'")
        for name, value in (
            ("correlation_exponent", correlation_exponent),
            ("hf_exponent", hf_exponent),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        options = dict(calculator_options or {})
        allowed = {
            "correlation_memory_budget_bytes",
            "mp2_denominator_threshold",
            "ccsd_max_iterations",
            "ccsd_diis_history",
            "ccsd_energy_tolerance",
            "ccsd_residual_tolerance",
            "ccsd_denominator_threshold",
            "ccsd_damping",
            "ccsd_level_shift",
            "max_iterations",
            "energy_tolerance",
            "density_tolerance",
            "diis_history",
            "screening_tolerance",
            "resource_budget",
            "initial_guess",
        }
        unknown = options.keys() - allowed
        if unknown:
            raise ValueError(
                f"unsupported CBS calculator options: {', '.join(sorted(unknown))}"
            )
        requested_options = options
        options = {
            "max_iterations": 200,
            "energy_tolerance": 1e-12,
            "density_tolerance": 1e-10,
            "ccsd_max_iterations": 150,
            "ccsd_energy_tolerance": 1e-12,
            "ccsd_residual_tolerance": 1e-10,
            **options,
        }
        self.mp2_bases = bases
        self.correction_basis = small
        self.device = device
        self.device_id = device_id
        self.correlation_exponent = float(correlation_exponent)
        self.hf_extrapolation = hf_extrapolation
        self.hf_exponent = float(hf_exponent)
        self._options = options
        self._lock = threading.Lock()
        self.mp2_weights = _limit_weights(
            -self.correlation_exponent * math.log(high.cardinal / low.cardinal)
        )
        self.hf_weights = (
            _limit_weights(-self.hf_exponent * (high.cardinal - low.cardinal))
            if hf_extrapolation == "exponential"
            else (0.0, 1.0)
        )

        from ._correlation_convergence import convergence_policy
        from .force_aware_numerics import TargetErrorBudget

        coefficients: dict[tuple[str, str], float] = {}
        for _, method, basis, coefficient in self._terms():
            key = (method, basis.identity)
            coefficients[key] = coefficients.get(key, 0.0) + coefficient
        amplification = math.fsum(abs(value) for value in coefficients.values())
        # CBS owns its routine convergence contract. Do not change the shared
        # HF/DFT Standard preset when promoting the previously explicit allowance.
        # Keep Fast's custom-budget transport and the strict Tight/FP64 escapes.
        budget = (
            TargetErrorBudget(1e-5, 1e-4, 3e-5)
            if policy.profile == "standard"
            else policy.budget
        )
        component_budget = (
            TargetErrorBudget(
                budget.energy_abs / amplification,
                budget.force_max_abs / amplification,
                budget.force_rms / amplification
                if budget.force_rms is not None
                else None,
            )
            if budget is not None
            else None
        )
        scale, limits = convergence_policy(component_budget)
        if scale:
            for name in (
                "energy_tolerance",
                "density_tolerance",
                "ccsd_energy_tolerance",
                "ccsd_residual_tolerance",
            ):
                options[name] = min(
                    requested_options.get(name, limits[name]), limits[name]
                )
            options["_correlation_convergence_budget"] = component_budget
        # A post-HF reference must satisfy its canonical-frame export contract.
        # Relaxing this density gate can terminate SCF before a valid reference
        # exists (notably stretched H2), even when the energy has stabilized.
        # Keep the same ceiling for both routine names and preserve tighter input.
        options["density_tolerance"] = min(options["density_tolerance"], 1e-11)
        # An energy-change plateau can coexist with observable error on a
        # poorly conditioned reference branch. Retain a physical residual cap
        # independently of the budget scale, while honoring tighter controls.
        options["ccsd_residual_tolerance"] = min(
            options["ccsd_residual_tolerance"], 1e-7
        )
        # Publish configured limits, not an invented observable error estimate.
        self._convergence = {
            "requested_policy": policy.profile,
            "preconvergence": preconvergence,
            "selected": "relaxed" if scale else "strict_fp64",
            "arithmetic_request": "df_ccsd_fp32_preconvergence"
            if scale and self._auxiliary_bases
            else "fp64",
            "reason": "budget_scaled_convergence"
            if scale
            else (
                "component_budget_requires_strict" if budget else "tight_convergence"
            ),
            "budget": asdict(budget) if budget is not None else None,
            "component_budget": asdict(component_budget)
            if component_budget is not None
            else None,
            "coefficient_l1": amplification,
            "scale": scale,
            "limits": {
                **limits,
                **{
                    name: min(options[name], limits[name])
                    for name in (
                        "energy_tolerance",
                        "density_tolerance",
                        "ccsd_energy_tolerance",
                        "ccsd_residual_tolerance",
                    )
                },
            },
        }

    def _terms(self) -> tuple:
        """Use one set of component weights for composition and error allocation."""
        low, high = self.mp2_bases
        return (
            ("hf_low", "rhf", low.basis, self.hf_weights[0] - self.mp2_weights[0]),
            (
                "hf_high",
                "rhf",
                high.basis,
                self.hf_weights[1] - self.mp2_weights[1],
            ),
            ("mp2_low", "mp2", low.basis, self.mp2_weights[0]),
            ("mp2_high", "mp2", high.basis, self.mp2_weights[1]),
            (
                "correction_ccsdt",
                "df-ccsd(t)" if self._auxiliary_bases else "ccsd(t)",
                self.correction_basis,
                1.0,
            ),
            ("correction_mp2", "mp2", self.correction_basis, -1.0),
        )

    def singlepoint(
        self,
        atoms: Iterable[Atom | tuple[str | int, Sequence[float]]],
        *,
        charge: int = 0,
        multiplicity: int = 1,
        properties: Iterable[str] = ("energy", "forces"),
    ) -> CBSResult:
        """Return a complete composite, or raise without publishing partial data.

        Coordinates are Bohr. Energy is always computed; explicit energy-only
        requests omit every analytic-force endpoint and all returned forces.
        HF and correction-basis MP2 companions are produced by the same native
        reference owner, without independent same-basis SCF endpoints.
        """
        with self._lock:
            started = time.perf_counter()
            geometry = tuple(Atom.from_value(atom) for atom in atoms)
            checked_integer(multiplicity, "multiplicity", low=1, high=2**31 - 1)
            if multiplicity != 1:
                raise NotImplementedError(
                    "CCSD(T)/CBS currently requires a closed-shell singlet"
                )
            if isinstance(properties, (str, bytes)):
                raise TypeError("properties must be an iterable of property names")
            requested = frozenset(properties)
            if "energy" not in requested or requested - {"energy", "forces"}:
                raise ValueError(
                    "CBS properties must include energy and optionally forces"
                )
            forces_requested = "forces" in requested
            expected_backend = "cuda" if self.device == "cuda" else "cpu_reference"
            native_properties = (
                ("energy", "forces") if forces_requested else ("energy",)
            )
            low, high = self.mp2_bases
            for record in (low.basis, high.basis, self.correction_basis):
                electron_state(
                    geometry,
                    charge=charge,
                    multiplicity=multiplicity,
                    element_metadata=record.by_element,
                )
                record.shells_for(geometry)

            terms = self._terms()
            completed: dict[tuple[str, str], tuple[Result, float]] = {}
            by_role: dict[str, Result] = {}
            groups = {basis.identity: basis for _, _, basis, _ in terms}
            execution_groups = []
            warmup_requested = False
            automatic_warmup = False
            warmup_fallback = None
            for identity, basis in groups.items():
                component_started = time.perf_counter()
                primary = (
                    ("df-ccsd(t)" if self._auxiliary_bases else "ccsd(t)")
                    if identity == self.correction_basis.identity
                    else "mp2"
                )
                requested = {
                    method
                    for _, method, record, _ in terms
                    if record.identity == identity
                }
                companions = tuple(sorted(requested - {primary}))
                calculator = Calculator(
                    method=primary,
                    basis=basis,
                    device=self.device,
                    device_id=self.device_id,
                    precision="fp64",
                    **(
                        {"_correlation_preconvergence": self.preconvergence}
                        if primary == "df-ccsd(t)" and self.policy.profile != "fp64"
                        else {}
                    ),
                    **(
                        {
                            "density_fitting": (
                                "full_cuda"
                                if self.hf_reference == "df"
                                else "cuda"
                                if primary == "df-ccsd(t)"
                                else "correlation_cuda"
                            ),
                            "auxiliary_basis": self._auxiliary_bases[identity],
                        }
                        if self._auxiliary_bases
                        else {}
                    ),
                    **self._options,
                )
                if primary == "df-ccsd(t)":
                    selected_warmup = getattr(
                        calculator, "_correlation_preconvergence", None
                    )
                    automatic_warmup = selected_warmup == "auto"
                    warmup_requested = selected_warmup == "mixed" or (
                        selected_warmup is None and self._convergence["scale"] > 0
                    )
                    warmup_fallback = getattr(
                        calculator, "_correlation_preconvergence_fallback", None
                    )
                try:
                    results = calculator.singlepoint_components(
                        geometry,
                        companions=companions,
                        charge=charge,
                        multiplicity=multiplicity,
                        properties=native_properties,
                    )
                    if set(results) != requested:
                        raise RuntimeError(
                            "CBS native group returned incomplete components"
                        )
                    for method, result in results.items():
                        role = next(
                            role
                            for role, candidate, record, _ in terms
                            if candidate == method and record.identity == identity
                        )
                        if (
                            not result.converged
                            or result.executed_backend != expected_backend
                        ):
                            raise RuntimeError(
                                f"CBS {role} did not complete on {self.device}"
                            )
                        if not math.isfinite(result.energy):
                            raise RuntimeError(
                                f"CBS {role} returned a nonfinite energy"
                            )
                        if forces_requested and (
                            result.forces is None
                            or result.forces.shape != (len(geometry), 3)
                            or not np.all(np.isfinite(result.forces))
                        ):
                            raise RuntimeError(
                                f"CBS {role} returned invalid analytic forces"
                            )
                finally:
                    calculator.clear_cache()
                elapsed = time.perf_counter() - component_started
                execution_groups.append(
                    {
                        "basis_identity": identity,
                        "primary": primary,
                        "methods": tuple(sorted(requested)),
                        "elapsed_seconds": elapsed,
                    }
                )
                for method, result in results.items():
                    completed[(method, identity)] = result, elapsed
            for role, method, basis, _ in terms:
                by_role[role] = completed[(method, basis.identity)][0]

            components = tuple(
                CBSComponent(
                    roles=tuple(
                        role
                        for role, method, basis, _ in terms
                        if (method, basis.identity) == key
                    ),
                    method=key[0],
                    basis_identity=key[1],
                    coefficient=math.fsum(
                        weight
                        for _, method, basis, weight in terms
                        if (method, basis.identity) == key
                    ),
                    result=result,
                    elapsed_seconds=elapsed,
                )
                for key, (result, elapsed) in completed.items()
            )
            hf_energy = math.fsum(
                weight * by_role[f"hf_{label}"].energy
                for label, weight in zip(("low", "high"), self.hf_weights, strict=True)
            )
            mp2_energy = math.fsum(
                weight
                * (by_role[f"mp2_{label}"].energy - by_role[f"hf_{label}"].energy)
                for label, weight in zip(("low", "high"), self.mp2_weights, strict=True)
            )
            delta_energy = (
                by_role["correction_ccsdt"].energy - by_role["correction_mp2"].energy
            )
            energy = math.fsum((hf_energy, mp2_energy, delta_energy))
            hf_forces = mp2_forces = delta_forces = forces = None
            if forces_requested:
                hf_forces = sum(
                    weight * by_role[f"hf_{label}"].forces
                    for label, weight in zip(
                        ("low", "high"), self.hf_weights, strict=True
                    )
                )
                mp2_forces = sum(
                    weight
                    * (by_role[f"mp2_{label}"].forces - by_role[f"hf_{label}"].forces)
                    for label, weight in zip(
                        ("low", "high"), self.mp2_weights, strict=True
                    )
                )
                delta_forces = (
                    by_role["correction_ccsdt"].forces
                    - by_role["correction_mp2"].forces
                )
                forces = hf_forces + mp2_forces + delta_forces
            if not math.isfinite(energy) or (
                forces is not None and not np.all(np.isfinite(forces))
            ):
                raise RuntimeError(
                    "CBS composition produced nonfinite energy or forces"
                )
            # Report actual arithmetic from the copied native counters. A
            # resource-refused FP32 request must not be relabelled as mixed.
            arithmetic = "fp64"
            lambda_request = "fp64" if forces is not None else "not_requested"
            lambda_arithmetic = lambda_request
            if automatic_warmup:
                # Native dimensions/resource ownership decide admission. Do
                # not infer a mixed request merely from Python's auto setting.
                performance = next(
                    component.result.cc_performance
                    for component in components
                    if component.method == "df-ccsd(t)"
                )
                if performance is None or performance.ccsd_fp32_requested is None:
                    raise RuntimeError("CBS lacks native FP32 admission telemetry")
                warmup_requested = performance.ccsd_fp32_requested
            if warmup_requested:
                performance = next(
                    component.result.cc_performance
                    for component in components
                    if component.method == "df-ccsd(t)"
                )
                if performance is None or performance.ccsd_fp32_requested is not True:
                    raise RuntimeError("CBS lacks native FP32 admission telemetry")
                if performance.ccsd_fp32_executed and performance.ccsd_fp32_gemm_calls:
                    arithmetic = "mixed"
                if forces is not None:
                    lambda_request = "df_lambda_fp32_preconvergence"
                    if performance.lambda_fp32_requested is not True:
                        raise RuntimeError(
                            "CBS lacks native Lambda FP32 admission telemetry"
                        )
                    if (
                        performance.lambda_fp32_executed
                        and performance.lambda_fp32_gemm_calls
                    ):
                        lambda_arithmetic = arithmetic = "mixed"
            return CBSResult(
                energy,
                forces,
                hf_energy,
                mp2_energy,
                delta_energy,
                hf_forces,
                mp2_forces,
                delta_forces,
                self.hf_weights,
                self.mp2_weights,
                components,
                True,
                expected_backend,
                {
                    "mp2": [
                        {
                            "identity": item.basis.identity,
                            "name": item.basis.name,
                            "cardinal": item.cardinal,
                        }
                        for item in self.mp2_bases
                    ],
                    "correction": {
                        "identity": self.correction_basis.identity,
                        "name": self.correction_basis.name,
                    },
                    "representation": low.basis.representation,
                    "hf_extrapolation": self.hf_extrapolation,
                    "hf_exponent": self.hf_exponent,
                    "correlation_exponent": self.correlation_exponent,
                    "frozen_core": 0,
                    "density_fitting": bool(self._auxiliary_bases),
                    "hf_reference": self.hf_reference,
                    "auxiliary_basis_identities": {
                        identity: basis.identity
                        for identity, basis in self._auxiliary_bases.items()
                    },
                },
                time.perf_counter() - started,
                tuple(execution_groups),
                {
                    **self._convergence,
                    "selected": "strict"
                    if warmup_requested and not self._convergence["scale"]
                    else self._convergence["selected"],
                    "arithmetic_request": "df_ccsd_fp32_preconvergence"
                    if warmup_requested
                    else "fp64",
                    "preconvergence_fallback": warmup_fallback,
                    "preconvergence_selection": "mixed" if warmup_requested else "fp64",
                    "arithmetic": arithmetic,
                    "lambda_arithmetic_request": lambda_request,
                    "lambda_arithmetic": lambda_arithmetic,
                    "limits": dict(self._convergence["limits"]),
                    "budget": dict(self._convergence["budget"])
                    if self._convergence["budget"]
                    else None,
                    "component_budget": dict(self._convergence["component_budget"])
                    if self._convergence["component_budget"]
                    else None,
                },
            )


__all__ = ["CBSBasis", "CBSComponent", "CBSResult", "CCSDTCBSCalculator"]
