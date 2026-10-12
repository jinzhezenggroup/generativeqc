"""Shared native CPU/CUDA UCCSD Lambda and fixed-orbital Hamiltonian VJPs.

Krylov control and physical excitation-coordinate maps are explicitly host
owned. All physical equations and derivative contractions use the borrowed
parent's native executor; no interpreter, PySCF or restricted-response fallback
belongs to this owner. Orbital relaxation and nuclear forces are separate.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import InitVar, asdict, dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from generativeqc.response_solver import GMRESOptions, _vector_norm
from generativeqc_compiler.cc.uccsd_layout import UCCSDLayout
from generativeqc_compiler.cc.uccsd_response import (
    build_parameter_vjp,
    build_response_programs,
)
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.evidence import canonical_hash
from generativeqc_compiler.common.resources import MAX_BYTES, checked_bytes
from generativeqc_compiler.tensor.cpu import emit_cpu
from generativeqc_compiler.tensor.cuda_plan import plan_cuda

from tools.generativeqc_response.implicit import (
    ImplicitSolveError,
    ResponseGMRES,
    checked_transpose_solve,
    transpose_solver_contract,
)

from .native_tensor_cuda import CudaCCTensorExecutor
from .uccsd_solver import PreparedUCCSD, UCCSDResult

if TYPE_CHECKING:
    from typing_extensions import Self

    from tools.generativeqc_response.implicit import TransposeSolver


def _freeze(value: Any) -> Any:
    """Freeze nested JSON-like diagnostics, including GMRES option mappings."""
    if isinstance(value, Mapping):
        return MappingProxyType({name: _freeze(item) for name, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("UCCSD response provenance must contain finite JSON-like values")


@dataclass(frozen=True)
class UCCSDLambdaOptions:
    """Fresh physical gates and complete-solve logical numeric/work limits."""

    cc_tolerance: float = 1e-9
    lambda_tolerance: float = 1e-9
    max_bytes: int = 512 << 20
    max_operator_actions: int = 2048
    max_total_work: int = 1_000_000_000_000
    gmres: GMRESOptions = field(
        default_factory=lambda: GMRESOptions(rtol=0.0, atol=1e-11, max_iterations=400)
    )

    def __post_init__(self) -> None:
        for name in ("cc_tolerance", "lambda_tolerance"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 < value <= 1e-9
            ):
                raise ValueError(f"{name} must be positive, finite and at most 1e-9")
        for name in ("max_bytes", "max_operator_actions", "max_total_work"):
            if checked_bytes(getattr(self, name), name) == 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.gmres, GMRESOptions):
            raise TypeError("UCCSD Lambda requires GMRESOptions")


def _validate_physical_multipliers(
    packed_multipliers: np.ndarray, multipliers: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Validate the shared unweighted excitation dual, without method qualification."""
    names = ("l1a", "l1b", "l2aa", "l2ab", "l2bb")
    if set(multipliers) != set(names):
        raise ValueError("UCCSD Lambda requires exactly five multiplier blocks")
    singles_alpha, singles_beta = multipliers["l1a"], multipliers["l1b"]
    if (
        type(singles_alpha) is not np.ndarray
        or type(singles_beta) is not np.ndarray
        or singles_alpha.ndim != 2
        or singles_beta.ndim != 2
    ):
        raise ValueError("UCCSD Lambda singles must have rank two")
    layout = UCCSDLayout(
        singles_alpha.shape[0],
        singles_beta.shape[0],
        singles_alpha.shape[1],
        singles_beta.shape[1],
    )
    packed = layout.pack(
        {name.replace("l", "t", 1): multipliers[name] for name in names}
    )
    if (
        type(packed_multipliers) is not np.ndarray
        or packed_multipliers.dtype != np.float64
        or not np.array_equal(packed, packed_multipliers)
    ):
        raise ValueError("UCCSD Lambda full and packed multipliers disagree")
    return packed


@dataclass(frozen=True)
class UCCSDLambdaResult:
    """Immutable physical excitation dual, published only after stationarity.

    Multipliers use l1a/l1b/l2aa/l2ab/l2bb, with ordinary signed expansion of
    the unweighted packed dual. Same-spin full arrays are physical Lambda,
    NOT full Frobenius residual seeds: those are smaller by a factor of four.
    """

    packed_multipliers: np.ndarray
    multipliers: Mapping[str, np.ndarray]
    state_identity: str
    lambda_identity: str
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
        """Own immutable, non-aliasing arrays even for explicitly constructed records."""
        packed = _validate_physical_multipliers(
            self.packed_multipliers, self.multipliers
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
                    "UCCSD Lambda diagnostics must be finite and nonnegative"
                )
        for name in ("iterations", "operator_actions"):
            checked_bytes(getattr(self, name), name)
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
            provenance["lambda_endpoint_seconds"] = (
                _preparation_seconds + time.perf_counter() - _endpoint_started
            )
        object.__setattr__(self, "provenance", _freeze(provenance))

    @property
    def converged(self) -> bool:
        """Only qualified results are constructed; failure raises before publication."""
        return True


@dataclass(frozen=True)
class UCCSDParameterCotangent:
    """One immutable independent MO-input cotangent, not a relaxed RDM or force."""

    parameter: str
    values: np.ndarray
    state_identity: str
    lambda_identity: str
    derivative_identity: str
    provenance: Mapping[str, Any]
    _endpoint_started: InitVar[float | None] = None

    def __post_init__(self, _endpoint_started: float | None) -> None:
        """Own finite FP64 values and immutable nested diagnostics."""
        if (
            type(self.values) is not np.ndarray
            or self.values.dtype != np.float64
            or not np.isfinite(self.values).all()
        ):
            raise ValueError(
                "UCCSD parameter cotangent must contain finite FP64 values"
            )
        object.__setattr__(self, "values", immutable(self.values))
        provenance = dict(self.provenance)
        if _endpoint_started is not None:
            provenance["parameter_endpoint_seconds"] = (
                time.perf_counter() - _endpoint_started
            )
        object.__setattr__(self, "provenance", _freeze(provenance))

    def contract(self, direction: np.ndarray) -> float:
        """Contract this input direction; physical changes may affect many inputs."""
        if (
            type(direction) is not np.ndarray
            or direction.dtype != np.float64
            or direction.shape != self.values.shape
            or not np.isfinite(direction).all()
        ):
            raise ValueError("UCCSD parameter direction must be finite matching FP64")
        with np.errstate(over="raise", invalid="raise"):
            value = float(np.sum(self.values * direction))
        if not math.isfinite(value):
            raise FloatingPointError("nonfinite UCCSD parameter contraction")
        return value


@dataclass(frozen=True)
class _Resource:
    native_bytes: int
    scalar_work: int


@dataclass
class _Counter:
    scalar_work: int = 0
    native_calls: int = 0
    operator_actions: int = 0


class PreparedUCCSDLambda:
    """Borrow one live UCCSD owner and solve in physical excitation coordinates.

    The parent/reference/feeds/options/executor must stay open and unchanged.
    Providers and AO sources can close after parent preparation. This owner
    owns immutable amplitude copies, never closes borrowed objects, and admits
    metadata, worst-case native work and Krylov storage before state copies or
    native execution. No dense amplitude Jacobian is built in production.
    """

    def __init__(
        self,
        prepared: PreparedUCCSD,
        state: UCCSDResult,
        *,
        options: UCCSDLambdaOptions | None = None,
        solver: Any = None,
    ) -> None:
        started = time.perf_counter()
        if not isinstance(prepared, PreparedUCCSD) or prepared._closed:
            raise TypeError("UCCSD Lambda requires a live PreparedUCCSD")
        if prepared._execution_configuration() != prepared._executor_configuration:
            raise RuntimeError("UCCSD Lambda parent executor configuration changed")
        if not isinstance(state, UCCSDResult) or not state.converged:
            raise ValueError("UCCSD Lambda requires a converged UCCSD state")
        options = UCCSDLambdaOptions() if options is None else options
        if not isinstance(options, UCCSDLambdaOptions):
            raise TypeError("UCCSD Lambda requires UCCSDLambdaOptions")
        identities = {
            "reference_id": prepared.reference.identity,
            "hamiltonian_id": prepared.reference.hamiltonian_id,
            "integral_hash": prepared.integral_hash,
            "equation_hash": prepared.program.logical_hash,
        }
        if any(
            state.provenance.get(name) != value for name, value in identities.items()
        ):
            raise ValueError("UCCSD Lambda parent state identities differ")
        if (
            state.correlation_energy is None
            or not math.isfinite(state.correlation_energy)
            or state.total_energy is None
            or not math.isfinite(state.total_energy)
            or state.total_energy
            != prepared.reference.reference_energy + state.correlation_energy
        ):
            raise ValueError("UCCSD Lambda requires finite consistent parent energies")
        self.prepared, self.options, self.layout = prepared, options, prepared.layout
        self.correlation_energy = float(state.correlation_energy)
        self._parent_binding = (
            prepared.reference,
            prepared.layout,
            prepared.program,
            prepared.feeds,
            prepared.options,
            prepared.executor,
        )
        self._executor_configuration = prepared._execution_configuration()
        self._closed = False
        full_bytes = sum(
            block.full_layout.storage_bytes(8) for block in self.layout.blocks
        )
        packed_bytes = self.layout.packed_size * 8
        self.base_bytes = checked_bytes(
            prepared.reference.numeric_bytes
            + sum(value.nbytes for value in prepared.feeds.values())
            + prepared.denominators.nbytes
            + 12 * full_bytes
            + 8 * packed_bytes,
            "UCCSD Lambda state/map storage",
        )
        if self.base_bytes > options.max_bytes:
            raise MemoryError("UCCSD Lambda persistent/state-map budget exceeded")
        self.programs = build_response_programs(
            self.layout, max_elements=options.max_bytes // 8
        )
        if self.programs.primal.logical_hash != prepared.program.logical_hash:
            raise ValueError("UCCSD Lambda physical equation identity differs")
        dimension = self.layout.packed_size
        if not dimension and solver is not None:
            raise ValueError("empty UCCSD excitation space needs no transpose solver")
        self.solver = (
            solver
            if solver is not None
            else (ResponseGMRES(dimension, options.gmres) if dimension else None)
        )
        if (
            isinstance(self.solver, ResponseGMRES)
            and self.solver.dimension != dimension
        ):
            raise ValueError("UCCSD Lambda transpose solver dimension mismatch")
        self._solver_contract = (
            transpose_solver_contract(cast("TransposeSolver", self.solver))
            if self.solver is not None
            else None
        )
        self._resources = {}
        for program in (
            prepared.program,
            self.programs.energy_vjp.program,
            self.programs.residual_vjp.program,
            self.programs.stationarity_vjp.program,
        ):
            self._resources[program.logical_hash] = self._resource(program)
        krylov_bytes = (
            0
            if self._solver_contract is None
            else self._solver_contract["workspace_bytes"]
        )
        if (
            isinstance(self.solver, ResponseGMRES)
            and krylov_bytes > self.solver.options.max_workspace_bytes
        ):
            raise MemoryError("UCCSD Lambda Krylov workspace budget exceeded")
        self.peak_bytes = checked_bytes(
            self.base_bytes
            + krylov_bytes
            + max(resource.native_bytes for resource in self._resources.values()),
            "UCCSD Lambda simultaneous numeric storage",
        )
        if self.peak_bytes > options.max_bytes:
            raise MemoryError(
                "UCCSD Lambda state, maps, native and Krylov budget exceeded"
            )
        planned_work = self._resources[prepared.program.logical_hash].scalar_work
        if dimension:
            planned_work += (
                self._resources[self.programs.energy_vjp.derivative_hash].scalar_work
                + options.max_operator_actions
                * self._resources[
                    self.programs.residual_vjp.derivative_hash
                ].scalar_work
                + self._resources[
                    self.programs.stationarity_vjp.derivative_hash
                ].scalar_work
            )
        self.planned_scalar_work = checked_bytes(
            planned_work, "complete UCCSD Lambda work"
        )
        if self.planned_scalar_work > options.max_total_work:
            raise MemoryError("UCCSD Lambda complete semantic-work budget exceeded")
        packed = self.layout.pack(state.amplitudes)
        if not np.array_equal(packed, state.packed_amplitudes):
            raise ValueError("UCCSD Lambda full and packed parent amplitudes disagree")
        self.packed_amplitudes = immutable(packed)
        self.amplitudes = MappingProxyType(
            {
                name: immutable(value)
                for name, value in self.layout.unpack(self.packed_amplitudes).items()
            }
        )
        self.feeds = MappingProxyType({**prepared.feeds, **self.amplitudes})
        self.state_identity = canonical_hash(
            {
                **identities,
                "packed_amplitudes": sha256(
                    self.packed_amplitudes.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "correlation_energy": self.correlation_energy,
            }
        )
        self._parameter_programs = {}
        self._binding = (
            self.prepared,
            self.options,
            self.layout,
            self.programs,
            self.solver,
            self.packed_amplitudes,
            self.amplitudes,
            self.feeds,
        )
        self.preparation_seconds = time.perf_counter() - started

    def _resource(self, program: Any) -> _Resource:
        """Model scalar work on both backends; never execute a CPU oracle on CUDA."""
        executor = self.prepared.executor
        _, resources = emit_cpu(
            program,
            max_bytes=MAX_BYTES,
            max_work=MAX_BYTES,
            max_nodes=getattr(executor, "max_nodes", 16384),
        )
        if isinstance(executor, CudaCCTensorExecutor):
            try:
                native_bytes = plan_cuda(
                    program, executor.compiler.target, max_bytes=executor.max_bytes
                ).peak_bytes
            except ValueError as error:
                if "infeasible tensor byte budget" not in str(error):
                    raise
                raise MemoryError(
                    "UCCSD derivative exceeds CUDA executor capacity"
                ) from error
        else:
            native_bytes = resources["required_bytes"]
            if (
                native_bytes > executor.max_bytes
                or resources["scalar_work"] > executor.max_work
            ):
                raise MemoryError("UCCSD derivative exceeds CPU executor capacity")
        return _Resource(native_bytes, resources["scalar_work"])

    @property
    def qualification_scalar_work(self) -> int:
        """Complete fresh parent plus standard Lambda stationarity replay work."""
        return checked_bytes(
            self._resources[self.prepared.program.logical_hash].scalar_work
            + self._resources[
                self.programs.stationarity_vjp.derivative_hash
            ].scalar_work,
            "UCCSD Lambda qualification work",
        )

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("prepared UCCSD Lambda is closed")
        current = (
            self.prepared,
            self.options,
            self.layout,
            self.programs,
            self.solver,
            self.packed_amplitudes,
            self.amplitudes,
            self.feeds,
        )
        if any(
            actual is not expected
            for actual, expected in zip(current, self._binding, strict=True)
        ):
            raise RuntimeError("UCCSD Lambda admitted configuration changed")
        parent = self.prepared
        current_parent = (
            parent.reference,
            parent.layout,
            parent.program,
            parent.feeds,
            parent.options,
            parent.executor,
        )
        if parent._closed or any(
            actual is not expected
            for actual, expected in zip(
                current_parent, self._parent_binding, strict=True
            )
        ):
            raise RuntimeError("UCCSD Lambda borrowed parent changed or closed")
        if parent._execution_configuration() != self._executor_configuration:
            raise RuntimeError("UCCSD Lambda borrowed executor configuration changed")
        if (
            self.solver is not None
            and transpose_solver_contract(cast("TransposeSolver", self.solver))
            != self._solver_contract
        ):
            raise RuntimeError("UCCSD Lambda borrowed solver contract changed")

    def _charge(self, program: Any, counter: _Counter) -> None:
        self._check_open()
        counter.scalar_work += self._resources[program.logical_hash].scalar_work
        if counter.scalar_work > self.options.max_total_work:
            raise MemoryError("UCCSD response semantic-work budget exhausted")
        counter.native_calls += 1

    def _execute(
        self, program: Any, extra: Mapping[str, np.ndarray], counter: _Counter
    ) -> dict:
        self._charge(program, counter)
        outputs = self.prepared.executor.execute(program, {**self.feeds, **extra})
        if outputs.keys() != program.outputs.keys():
            raise ValueError("UCCSD derivative executor returned incomplete outputs")
        for name, node in program.outputs.items():
            value = np.asarray(outputs[name])
            if value.dtype != np.float64 or value.shape != node.spec.shape:
                raise ValueError(
                    "UCCSD derivative executor violated the FP64 shape contract"
                )
            if not np.isfinite(value).all():
                raise FloatingPointError("nonfinite UCCSD derivative output")
        self._check_open()
        return outputs

    def _qualify_parent(self, counter: _Counter) -> tuple[float, float]:
        self._charge(self.prepared.program, counter)
        outputs = self.prepared.evaluate(self.packed_amplitudes)
        energy_drift = abs(
            float(outputs["correlation_energy"]) - self.correlation_energy
        )
        residual = max(
            float(np.max(np.abs(value), initial=0))
            for name, value in outputs.items()
            if name != "correlation_energy"
        )
        if (
            energy_drift > min(1e-10, self.prepared.options.energy_tolerance)
            or residual > self.options.cc_tolerance
        ):
            raise ImplicitSolveError(
                "UCCSD Lambda parent failed fresh physical qualification"
            )
        return residual, energy_drift

    def _seeds(self, packed: np.ndarray) -> dict[str, np.ndarray]:
        return {
            f"bar_{name.replace('t', 'r', 1)}": value
            for name, value in self.layout.unpack_dual(packed).items()
        }

    def _pullback(self, outputs: Mapping[str, np.ndarray]) -> np.ndarray:
        return self.layout.pack_dual(
            {block.name: outputs[f"bar_{block.name}"] for block in self.layout.blocks}
        )

    def _transpose(self, packed: np.ndarray, counter: _Counter) -> np.ndarray:
        if counter.operator_actions >= self.options.max_operator_actions:
            raise MemoryError("UCCSD Lambda operator-action budget exhausted")
        counter.operator_actions += 1
        return self._pullback(
            self._execute(
                self.programs.residual_vjp.program, self._seeds(packed), counter
            )
        )

    def _stationarity(
        self, packed: np.ndarray, counter: _Counter
    ) -> tuple[float, float]:
        outputs = self._execute(
            self.programs.stationarity_vjp.program,
            {**self._seeds(packed), "bar_correlation_energy": np.array(1.0)},
            counter,
        )
        residual_norm = _vector_norm(self._pullback(outputs))
        residual_max = max(
            block.full_metric_weight
            * float(np.max(np.abs(outputs[f"bar_{block.name}"]), initial=0))
            for block in self.layout.blocks
        )
        if max(residual_norm, residual_max) > self.options.lambda_tolerance:
            raise ImplicitSolveError(
                "UCCSD Lambda fresh Lagrangian stationarity failed"
            )
        return residual_norm, residual_max

    def _lambda_identity(self, packed: np.ndarray) -> str:
        return canonical_hash(
            {
                "state_identity": self.state_identity,
                "stationarity_graph": self.programs.stationarity_vjp.derivative_hash,
                "packed_physical_dual": sha256(
                    packed.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
            }
        )

    def solve(self) -> UCCSDLambdaResult:
        """Solve J*lambda=-grad(E), then freshly recheck the full Lagrangian VJP."""
        self._check_open()
        started = time.perf_counter()
        counter = _Counter()
        primal_residual, energy_drift = self._qualify_parent(counter)
        dimension = self.layout.packed_size
        if dimension:
            rhs = self._pullback(
                self._execute(
                    self.programs.energy_vjp.program,
                    {"bar_correlation_energy": np.array(-1.0)},
                    counter,
                )
            )
            owner = self

            class Operator:
                def __init__(self) -> None:
                    self.dimension = dimension

                def apply(self, vector: np.ndarray) -> np.ndarray:
                    return owner._transpose(vector, counter)

            checked = checked_transpose_solve(
                Operator(), rhs, solver=self.solver, assert_current=self._check_open
            )
            solution = checked.solution
            residual_norm, iterations = checked.residual_norm, checked.iterations
            stationary_norm, stationary_max = self._stationarity(solution, counter)
        else:
            solution = np.empty(0, dtype=np.float64)
            residual_norm = stationary_norm = stationary_max = 0.0
            iterations = 0
        self._check_open()
        packed = immutable(solution)
        multipliers = MappingProxyType(
            {
                name.replace("t", "l", 1): immutable(value)
                for name, value in self.layout.unpack(packed).items()
            }
        )
        return UCCSDLambdaResult(
            packed,
            multipliers,
            self.state_identity,
            self._lambda_identity(packed),
            primal_residual,
            energy_drift,
            residual_norm,
            stationary_norm,
            stationary_max,
            iterations,
            counter.operator_actions,
            MappingProxyType(
                {
                    "reference_id": self.prepared.reference.identity,
                    "hamiltonian_id": self.prepared.reference.hamiltonian_id,
                    "integral_hash": self.prepared.integral_hash,
                    "tensor_backend": self.prepared.executor.backend,
                    "integral_backend": self.prepared.integral_backend,
                    "source_backend": self.prepared.source_backend,
                    "solver_backend": "empty-excitation-space"
                    if self._solver_contract is None
                    else self._solver_contract["backend"],
                    "coordinate_backend": "host-fp64-unweighted-physical-excitation-duals",
                    "lagrangian": "E_corr + physical_lambda dot physical_residual",
                    "same_spin_full_residual_seed_factor": 0.25,
                    "numeric_bound_bytes": self.peak_bytes,
                    "planned_max_semantic_scalar_work": self.planned_scalar_work,
                    "executed_semantic_scalar_work": counter.scalar_work,
                    "work_model": "generated CPU scalar-loop model for physical replay and all derivative actions; not hardware FLOPs or host Krylov/coordinate work",
                    "native_calls": counter.native_calls,
                    "max_operator_actions": self.options.max_operator_actions,
                    "rhs_hash": self.programs.energy_vjp.derivative_hash,
                    "transpose_hash": self.programs.residual_vjp.derivative_hash,
                    "stationarity_hash": self.programs.stationarity_vjp.derivative_hash,
                    "options": MappingProxyType(asdict(self.options)),
                    "preparation_seconds": self.preparation_seconds,
                    "scope": "supplied converged UCCSD amplitude response; no orbital/nuclear relaxation",
                }
            ),
            _endpoint_started=started,
            _preparation_seconds=self.preparation_seconds,
        )

    def _packed_result(self, result: UCCSDLambdaResult) -> np.ndarray:
        """Validate physical arrays and identities without trusting diagnostics."""
        if (
            not isinstance(result, UCCSDLambdaResult)
            or result.state_identity != self.state_identity
        ):
            raise ValueError(
                "UCCSD parameter response requires this owner's Lambda state"
            )
        packed = self.layout.pack(
            {
                block.name: result.multipliers[block.name.replace("t", "l", 1)]
                for block in self.layout.blocks
            }
        )
        if (
            not np.array_equal(packed, result.packed_multipliers)
            or self._lambda_identity(packed) != result.lambda_identity
        ):
            raise ValueError("UCCSD Lambda packed/full or identity disagreement")
        return immutable(packed)

    def qualify(
        self, result: UCCSDLambdaResult
    ) -> tuple[np.ndarray, Mapping[str, Any]]:
        """Freshly qualify standard CCSD Lambda for a composed physical consumer.

        Returned packed duals own immutable storage. Parent and actual amplitude
        stationarity are replayed through this owner's native executor, with
        the same admitted numeric/work bounds as per-input requests. This is
        not a bypass for corrected-(T) or forged multipliers.
        """
        self._check_open()
        packed = self._packed_result(result)
        counter = _Counter()
        primal_residual, energy_drift = self._qualify_parent(counter)
        norm, maximum = self._stationarity(packed, counter)
        self._check_open()
        return packed, _freeze(
            {
                "primal_residual_max": primal_residual,
                "energy_drift": energy_drift,
                "stationarity_norm": norm,
                "stationarity_max": maximum,
                "executed_semantic_scalar_work": counter.scalar_work,
                "native_calls": counter.native_calls,
            }
        )

    def parameter_cotangent(
        self, result: UCCSDLambdaResult, parameter: str
    ) -> UCCSDParameterCotangent:
        """Requalify Lambda and return one bounded canonical MO-input cotangent."""
        self._check_open()
        started = time.perf_counter()
        packed = self._packed_result(result)
        action = self._parameter_programs.get(parameter)
        if action is None:
            action = build_parameter_vjp(
                self.programs, parameter, max_elements=self.options.max_bytes // 8
            )
            resource = self._resource(action.program)
            self._parameter_programs[parameter] = action
            self._resources[action.derivative_hash] = resource
        resource = self._resources[action.derivative_hash]
        parameter_bytes = action.program.outputs[f"bar_{parameter}"].spec.size * 8
        peak = checked_bytes(
            max(
                self.peak_bytes,
                self.base_bytes + 4 * parameter_bytes + resource.native_bytes,
            ),
            "UCCSD parameter cotangent numeric storage",
        )
        work = sum(
            self._resources[program.logical_hash].scalar_work
            for program in (
                self.prepared.program,
                self.programs.stationarity_vjp.program,
                action.program,
            )
        )
        if peak > self.options.max_bytes or work > self.options.max_total_work:
            raise MemoryError(
                "UCCSD parameter response numeric/semantic-work budget exceeded"
            )
        packed = immutable(packed)
        counter = _Counter()
        self._qualify_parent(counter)
        stationary_norm, stationary_max = self._stationarity(packed, counter)
        outputs = self._execute(
            action.program,
            {**self._seeds(packed), "bar_correlation_energy": np.array(1.0)},
            counter,
        )
        self._check_open()
        return UCCSDParameterCotangent(
            parameter,
            immutable(outputs[f"bar_{parameter}"]),
            self.state_identity,
            result.lambda_identity,
            action.derivative_hash,
            MappingProxyType(
                {
                    "tensor_backend": self.prepared.executor.backend,
                    "numeric_bound_bytes": peak,
                    "executed_semantic_scalar_work": counter.scalar_work,
                    "stationarity_norm": stationary_norm,
                    "stationarity_max": stationary_max,
                    "scope": "one independent fixed-orbital correlation-Lagrangian input; not a relaxed RDM or force",
                }
            ),
            _endpoint_started=started,
        )

    def close(self) -> None:
        """Release owned state/action metadata without closing the borrowed parent."""
        self._binding = self._parent_binding = ()
        self.feeds = self.amplitudes = MappingProxyType({})
        self.packed_amplitudes = np.empty(0, dtype=np.float64)
        self._parameter_programs.clear()
        self._resources.clear()
        self._closed = True

    def __enter__(self) -> Self:
        self._check_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
