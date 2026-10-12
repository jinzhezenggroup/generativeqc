"""Corrected UCCSD(T) Lambda on shared native adjoint/Krylov infrastructure.

The corrected record is deliberately not a standard UCCSD Lambda subtype.
Triples bars are freshly replayed on the same backend and enter both the RHS
and actual physical amplitude stationarity. This module owns no CC algebra,
dense production Jacobian, interpreter/oracle or separate SCF response.
"""

from __future__ import annotations

import math
import time
from dataclasses import InitVar, asdict, dataclass, replace
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from generativeqc.response_solver import _vector_norm
from generativeqc_compiler.cc.uccsd_layout import BLOCK_NAMES
from generativeqc_compiler.cc.uccsd_response import build_parameter_vjp
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.evidence import canonical_hash
from generativeqc_compiler.common.resources import checked_bytes

from tools.generativeqc_response.implicit import (
    ImplicitSolveError,
    checked_transpose_solve,
)

from .uccsd_lambda import (
    PreparedUCCSDLambda,
    UCCSDParameterCotangent,
    _Counter,
    _freeze,
    _validate_physical_multipliers,
)
from .uccsd_triples_response import (
    PreparedUCCSDTriplesResponse,
    UCCSDTriplesResponseResult,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self


@dataclass(frozen=True)
class UCCSDTLambdaOptions:
    """Aggregate logical numeric/work admission and fresh correction drift gate.

    CC, Lambda, solver and operator-action controls belong to the borrowed
    standard adjoint owner; this consumer neither relaxes nor replaces them.
    """

    max_bytes: int = 512 << 20
    max_total_work: int = 1_000_000_000_000
    correction_tolerance: float = 1e-11

    def __post_init__(self) -> None:
        for name in ("max_bytes", "max_total_work"):
            checked_bytes(getattr(self, name), name)
            if not getattr(self, name):
                raise ValueError(f"{name} must be positive")
        if (
            isinstance(self.correction_tolerance, bool)
            or not math.isfinite(self.correction_tolerance)
            or not 0 < self.correction_tolerance <= 1e-10
        ):
            raise ValueError("correction_tolerance must be positive and at most 1e-10")


@dataclass(frozen=True)
class UCCSDTLambdaResult:
    """Immutable corrected excitation dual with a separately owned (T) response.

    Multipliers retain standard physical U/U* coordinates and residual seed
    factors, but solve grad(E_CCSD+E_T)+J*lambda=0. This is not interchangeable
    with UCCSDLambdaResult, even in a zero-triples limit.
    """

    packed_multipliers: np.ndarray
    multipliers: Mapping[str, np.ndarray]
    state_identity: str
    lambda_identity: str
    correction_identity: str
    triples_response: UCCSDTriplesResponseResult
    primal_residual_max: float
    energy_drift: float
    solver_residual_norm: float
    stationarity_norm: float
    stationarity_max: float
    iterations: int
    operator_actions: int
    provenance: Mapping[str, Any]
    _endpoint_started: InitVar[float | None] = None
    _preparation_seconds: InitVar[float] = 0.0

    def __post_init__(
        self, _endpoint_started: float | None, _preparation_seconds: float
    ) -> None:
        packed = _validate_physical_multipliers(
            self.packed_multipliers, self.multipliers
        )
        if (
            not isinstance(self.triples_response, UCCSDTriplesResponseResult)
            or not self.triples_response.converged
            or self.triples_response.energy.provenance.get("state_identity")
            != self.state_identity
        ):
            raise ValueError(
                "corrected Lambda requires a matching complete triples response"
            )
        for name in (
            "primal_residual_max",
            "energy_drift",
            "solver_residual_norm",
            "stationarity_norm",
            "stationarity_max",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(
                    "corrected Lambda diagnostics must be finite and nonnegative"
                )
        for name in ("iterations", "operator_actions"):
            checked_bytes(getattr(self, name), name)
        for name in ("state_identity", "lambda_identity", "correction_identity"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(
                    "corrected Lambda requires explicit state/equation identities"
                )
        energy = self.triples_response.energy
        owned_response = replace(
            self.triples_response,
            energy=replace(
                energy,
                channel_energies=_freeze(energy.channel_energies),
                provenance=_freeze(energy.provenance),
            ),
        )
        object.__setattr__(self, "triples_response", owned_response)
        object.__setattr__(self, "packed_multipliers", immutable(packed))
        object.__setattr__(
            self,
            "multipliers",
            MappingProxyType(
                {name: immutable(value) for name, value in self.multipliers.items()}
            ),
        )
        provenance = dict(self.provenance)
        if _endpoint_started is not None:
            provenance["corrected_lambda_endpoint_seconds"] = (
                _preparation_seconds + time.perf_counter() - _endpoint_started
            )
        object.__setattr__(self, "provenance", _freeze(provenance))

    @property
    def converged(self) -> bool:
        """Only freshly qualified corrected results are published."""
        return True


class PreparedUCCSDTLambda:
    """Borrow matching standard adjoint and bounded triples owners, never bypass them.

    Standard qualification remains standard. The native energy/J* programs,
    physical coordinate maps and checked GMRES are shared, while corrected
    records and stationarity are explicit. Closing this owner closes neither
    borrowed owner, parent, backend nor solver.
    """

    def __init__(
        self,
        lambdas: PreparedUCCSDLambda,
        triples: PreparedUCCSDTriplesResponse,
        *,
        options: UCCSDTLambdaOptions | None = None,
    ) -> None:
        started = time.perf_counter()
        if not isinstance(lambdas, PreparedUCCSDLambda) or not isinstance(
            triples, PreparedUCCSDTriplesResponse
        ):
            raise TypeError(
                "corrected Lambda requires live standard adjoint and triples owners"
            )
        lambdas._check_open()
        triples._check_open()
        if (
            triples.prepared is not lambdas.prepared
            or triples.state_identity != lambdas.state_identity
            or not np.array_equal(triples.packed_amplitudes, lambdas.packed_amplitudes)
        ):
            raise ValueError(
                "corrected Lambda owners differ in physical state/parent/backend"
            )
        self.standard, self.triples = lambdas, triples
        self.limits = UCCSDTLambdaOptions() if options is None else options
        if not isinstance(self.limits, UCCSDTLambdaOptions):
            raise TypeError("corrected Lambda requires UCCSDTLambdaOptions")
        self.prepared, self.layout, self.options = (
            lambdas.prepared,
            lambdas.layout,
            lambdas.options,
        )
        self.programs, self.amplitudes = lambdas.programs, lambdas.amplitudes
        self.packed_amplitudes, self.state_identity = (
            lambdas.packed_amplitudes,
            lambdas.state_identity,
        )
        self.correlation_energy = lambdas.correlation_energy
        self._resources = lambdas._resources
        source_bytes = sum(value.nbytes for value in triples.sources.values())
        self.peak_bytes = checked_bytes(
            lambdas.peak_bytes
            + triples.peak_bytes
            + 4 * source_bytes
            + 32 * self.layout.packed_size,
            "corrected Lambda aggregate numeric storage",
        )
        self.planned_scalar_work = checked_bytes(
            lambdas.planned_scalar_work + triples.total_scalar_work,
            "corrected Lambda aggregate solve work",
        )
        if self.peak_bytes > self.limits.max_bytes:
            raise MemoryError("corrected Lambda aggregate numeric budget exceeded")
        if (
            max(self.planned_scalar_work, self.qualification_scalar_work)
            > self.limits.max_total_work
        ):
            raise MemoryError(
                "corrected Lambda aggregate semantic-work budget exceeded"
            )
        self.correction_identity = canonical_hash(
            {
                "method": "canonical-real-UCCSD(T)",
                "state": self.state_identity,
                "templates": tuple(
                    template.binding_identity for template in triples.templates.values()
                ),
                "tensor_backend": self.prepared.executor.backend,
                "denominator_chain": "spin-owned occupied/virtual Fock diagonal v1",
            }
        )
        self._parameters = {}
        self._closed = False
        self._binding = self._current_binding()
        self.preparation_seconds = time.perf_counter() - started

    def _current_binding(self) -> tuple:
        return (
            self.standard,
            self.triples,
            self.limits,
            self.prepared,
            self.layout,
            self.options,
            self.programs,
            self.amplitudes,
            self.packed_amplitudes,
            self.state_identity,
            self.correlation_energy,
            self.correction_identity,
            self._resources,
        )

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("corrected UCCSD(T) Lambda is closed")
        if any(
            actual is not expected
            for actual, expected in zip(
                self._current_binding(), self._binding, strict=True
            )
        ):
            raise RuntimeError("corrected Lambda admitted configuration changed")
        self.standard._check_open()
        self.triples._check_open()

    @property
    def qualification_scalar_work(self) -> int:
        """Fresh triples replay plus parent and corrected stationarity action work."""
        return checked_bytes(
            self.standard.qualification_scalar_work + self.triples.total_scalar_work,
            "corrected Lambda qualification work",
        )

    def _resource(self, program: Any) -> Any:
        return self.standard._resource(program)

    def _seeds(self, packed: np.ndarray) -> dict[str, np.ndarray]:
        return self.standard._seeds(packed)

    def _admit_action(self, program: Any, counter: _Counter) -> None:
        self._check_open()
        if (
            self.triples.total_scalar_work
            + counter.scalar_work
            + self._resources[program.logical_hash].scalar_work
            > self.limits.max_total_work
        ):
            raise MemoryError("corrected Lambda aggregate execution work exhausted")

    def _execute(self, program: Any, extra: Mapping, counter: _Counter) -> dict:
        self._admit_action(program, counter)
        return self.standard._execute(program, extra, counter)

    def _fresh_correction(self) -> UCCSDTriplesResponseResult:
        self._check_open()
        correction = self.triples.response()
        if not correction.converged:
            raise ImplicitSolveError(
                f"corrected Lambda triples replay failed: {correction.energy.reason}"
            )
        self._check_open()
        return correction

    def _stationarity(
        self,
        packed: np.ndarray,
        correction: UCCSDTriplesResponseResult,
        counter: _Counter,
    ) -> tuple[float, float]:
        values = self._execute(
            self.programs.stationarity_vjp.program,
            {**self._seeds(packed), "bar_correlation_energy": np.array(1.0)},
            counter,
        )
        residual = self.layout.pack_dual(
            {
                name: np.asarray(values[f"bar_{name}"]) + correction.cotangents[name]
                for name in BLOCK_NAMES
            }
        )
        norm, maximum = (
            _vector_norm(residual),
            float(np.max(np.abs(residual), initial=0)),
        )
        if max(norm, maximum) > self.options.lambda_tolerance:
            raise ImplicitSolveError(
                "UCCSD(T) fresh corrected amplitude stationarity failed"
            )
        return norm, maximum

    def _lambda_identity(self, packed: np.ndarray) -> str:
        return canonical_hash(
            {
                "state": self.state_identity,
                "correction": self.correction_identity,
                "stationarity_graph": self.programs.stationarity_vjp.derivative_hash,
                "packed_physical_dual": sha256(
                    packed.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
            }
        )

    def _diagnostics(
        self,
        correction: UCCSDTriplesResponseResult,
        counter: _Counter,
        residual: float,
        drift: float,
        norm: float,
        maximum: float,
    ) -> dict:
        return {
            "method": "UCCSD(T)",
            "correction_identity": self.correction_identity,
            "fresh_triples_response_identity": correction.response_identity,
            "primal_residual_max": residual,
            "energy_drift": drift,
            "stationarity_norm": norm,
            "stationarity_max": maximum,
            "executed_semantic_scalar_work": counter.scalar_work
            + correction.provenance["semantic_scalar_work"],
            "cc_semantic_scalar_work": counter.scalar_work,
            "native_calls": counter.native_calls
            + correction.provenance["completed_tiles"]
            + 1,
            "cc_native_calls": counter.native_calls,
        }

    def solve(self) -> UCCSDTLambdaResult:
        """Solve the corrected RHS and freshly check the actual corrected residual."""
        self._check_open()
        started = time.perf_counter()
        correction = self._fresh_correction()
        counter = _Counter()
        self._admit_action(self.prepared.program, counter)
        primal_residual, drift = self.standard._qualify_parent(counter)
        dimension = self.layout.packed_size
        if dimension:
            rhs = (
                self.standard._pullback(
                    self._execute(
                        self.programs.energy_vjp.program,
                        {"bar_correlation_energy": np.array(-1.0)},
                        counter,
                    )
                )
                - correction.packed_amplitude_cotangent
            )
            owner = self

            class Operator:
                def __init__(self) -> None:
                    self.dimension = dimension

                def apply(self, vector: np.ndarray) -> np.ndarray:
                    owner._admit_action(owner.programs.residual_vjp.program, counter)
                    return owner.standard._transpose(vector, counter)

            checked = checked_transpose_solve(
                Operator(),
                rhs,
                solver=self.standard.solver,
                assert_current=self._check_open,
            )
            solution, solver_residual, iterations = (
                checked.solution,
                checked.residual_norm,
                checked.iterations,
            )
        else:
            solution, solver_residual, iterations = (
                np.empty(0, dtype=np.float64),
                0.0,
                0,
            )
        norm, maximum = self._stationarity(solution, correction, counter)
        self._check_open()
        return UCCSDTLambdaResult(
            solution,
            {
                name.replace("t", "l", 1): value
                for name, value in self.layout.unpack(solution).items()
            },
            self.state_identity,
            self._lambda_identity(solution),
            self.correction_identity,
            correction,
            primal_residual,
            drift,
            solver_residual,
            norm,
            maximum,
            iterations,
            counter.operator_actions,
            {
                **self._diagnostics(
                    correction, counter, primal_residual, drift, norm, maximum
                ),
                "tensor_backend": self.prepared.executor.backend,
                "integral_backend": self.prepared.integral_backend,
                "numeric_bound_bytes": self.peak_bytes,
                "planned_max_semantic_scalar_work": self.planned_scalar_work,
                "lagrangian": "E_CCSD + E_T + physical_lambda dot physical_residual",
                "amplitude_gradient_map": "U*; signed same-spin orbit sum without an additional quarter",
                "same_spin_full_residual_seed_factor": 0.25,
                "limits": asdict(self.limits),
                "triples": correction.provenance,
                "scope": "corrected amplitude response; orbital/nuclear chains remain separate",
            },
            _endpoint_started=started,
            _preparation_seconds=self.preparation_seconds,
        )

    def qualify_with_correction(
        self, result: UCCSDTLambdaResult
    ) -> tuple[np.ndarray, UCCSDTriplesResponseResult, Mapping[str, Any]]:
        """Replay all correction bars and corrected stationarity, not stored diagnostics.

        Numeric correction values have an explicit drift gate; their byte-level
        result hashes are not assumed deterministic across valid native replays.
        Physical state, equation, template and backend bindings remain exact.
        """
        self._check_open()
        if (
            not isinstance(result, UCCSDTLambdaResult)
            or result.state_identity != self.state_identity
            or result.correction_identity != self.correction_identity
        ):
            raise ValueError(
                "corrected response requires this owner's distinct UCCSD(T) Lambda state"
            )
        packed = _validate_physical_multipliers(
            result.packed_multipliers, result.multipliers
        )
        owner_packed = self.layout.pack(
            {
                name: result.multipliers[name.replace("t", "l", 1)]
                for name in BLOCK_NAMES
            }
        )
        if not np.array_equal(owner_packed, packed):
            raise ValueError(
                "corrected Lambda multiplier populations differ from owner"
            )
        if self._lambda_identity(packed) != result.lambda_identity:
            raise ValueError("corrected Lambda identity differs")
        correction = self._fresh_correction()
        previous = result.triples_response
        energy = previous.energy
        stored_triples_energy = energy.triples_energy
        fresh_triples_energy = correction.energy.triples_energy
        stored_correlation_energy = energy.correlation_energy
        if (
            stored_triples_energy is None
            or fresh_triples_energy is None
            or stored_correlation_energy is None
        ):
            raise ValueError("corrected Lambda triples energies are incomplete")
        if (
            energy.ccsd_correlation_energy != self.correlation_energy
            or stored_correlation_energy
            != self.correlation_energy + stored_triples_energy
            or energy.total_energy
            != self.prepared.reference.reference_energy + stored_correlation_energy
            or energy.channel_energies.keys()
            != correction.energy.channel_energies.keys()
            or not math.isfinite(stored_triples_energy)
            or any(
                not math.isfinite(value) for value in energy.channel_energies.values()
            )
            or abs(math.fsum(energy.channel_energies.values()) - stored_triples_energy)
            > self.limits.correction_tolerance
        ):
            raise ValueError("corrected Lambda triples energies are inconsistent")
        if (
            any(
                previous.energy.provenance.get(name)
                != correction.energy.provenance.get(name)
                for name in (
                    "reference_id",
                    "hamiltonian_id",
                    "integral_hash",
                    "state_identity",
                    "template_identities",
                    "tensor_backend",
                    "integral_backend",
                    "source_backend",
                )
            )
            or previous.cotangents.keys() != correction.cotangents.keys()
        ):
            raise ValueError(
                "corrected Lambda triples equation/backend binding differs"
            )
        differences = [abs(stored_triples_energy - fresh_triples_energy)]
        differences.extend(
            abs(value - correction.energy.channel_energies[name])
            for name, value in previous.energy.channel_energies.items()
        )
        for name, values in correction.cotangents.items():
            earlier = previous.cotangents[name]
            if earlier.shape != values.shape:
                raise ValueError("corrected Lambda triples source populations differ")
            differences.append(float(np.max(np.abs(earlier - values), initial=0)))
        correction_drift = max(differences)
        if correction_drift > self.limits.correction_tolerance:
            raise ImplicitSolveError(
                "corrected Lambda fresh triples cotangent/energy drift failed"
            )
        counter = _Counter()
        self._admit_action(self.prepared.program, counter)
        residual, drift = self.standard._qualify_parent(counter)
        norm, maximum = self._stationarity(packed, correction, counter)
        self._check_open()
        return (
            immutable(packed),
            correction,
            _freeze(
                {
                    **self._diagnostics(
                        correction, counter, residual, drift, norm, maximum
                    ),
                    "correction_drift": correction_drift,
                }
            ),
        )

    def qualify(
        self, result: UCCSDTLambdaResult
    ) -> tuple[np.ndarray, Mapping[str, Any]]:
        """Qualify only corrected Lambda; standard qualification is never bypassed."""
        packed, _, diagnostics = self.qualify_with_correction(result)
        return packed, diagnostics

    def parameter_cotangent(
        self, result: UCCSDTLambdaResult, parameter: str
    ) -> UCCSDParameterCotangent:
        """Compose corrected CC Lagrangian and explicit denominator-bound (T) bars."""
        self._check_open()
        started = time.perf_counter()
        action = self._parameters.get(parameter)
        if action is None:
            action = build_parameter_vjp(
                self.programs, parameter, max_elements=self.limits.max_bytes // 8
            )
            resource = self._resource(action.program)
            self._parameters[parameter] = action
            self._resources[action.derivative_hash] = resource
        resource = self._resources[action.derivative_hash]
        if (
            self.qualification_scalar_work + resource.scalar_work
            > self.limits.max_total_work
        ):
            raise MemoryError(
                "corrected Lambda parameter semantic-work budget exceeded"
            )
        peak = checked_bytes(
            self.peak_bytes + resource.native_bytes,
            "corrected Lambda parameter numeric storage",
        )
        if peak > self.limits.max_bytes:
            raise MemoryError("corrected Lambda parameter numeric budget exceeded")
        packed, correction, diagnostics = self.qualify_with_correction(result)
        counter = _Counter(
            scalar_work=diagnostics["cc_semantic_scalar_work"],
            native_calls=diagnostics["cc_native_calls"],
        )
        outputs = self._execute(
            action.program,
            {**self._seeds(packed), "bar_correlation_energy": np.array(1.0)},
            counter,
        )
        values = (
            np.asarray(outputs[f"bar_{parameter}"])
            + correction.hamiltonian_cotangents()[parameter]
        )
        self._check_open()
        return UCCSDParameterCotangent(
            parameter,
            values,
            self.state_identity,
            result.lambda_identity,
            canonical_hash(
                {
                    "cc_parameter_graph": action.derivative_hash,
                    "correction_binding": self.correction_identity,
                }
            ),
            {
                **diagnostics,
                "method": "UCCSD(T)",
                "numeric_bound_bytes": peak,
                "executed_semantic_scalar_work": counter.scalar_work
                + correction.provenance["semantic_scalar_work"],
                "denominator_chain": "explicit spin Fock diagonal",
                "scope": "fixed-orbital corrected correlation Lagrangian",
            },
            _endpoint_started=started,
        )

    def close(self) -> None:
        """Release owned plans, preserving all borrowed owners and solver resources."""
        self._parameters.clear()
        self._closed = True

    def __enter__(self) -> Self:
        self._check_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
