"""Native physical UCCSD Hamiltonian pullback with the shared UHF adjoint.

This owner composes the original CC equations with raw normal ordering before
AD, rather than replaying the complete CC Lagrangian for every canonical input.
It publishes full-Frobenius relaxed raw-MO and overlap cotangents, not a packed
RDM convention or a nuclear force. Nuclear consumers must bind the actual
four spin-owned coefficient frames and qualified shell derivatives explicitly.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from generativeqc.response_solver import GMRESOptions, _vector_norm
from generativeqc_compiler.cc.uccsd_hamiltonian import (
    build_lagrangian_programs,
    build_raw_hamiltonian,
)
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.evidence import canonical_hash
from generativeqc_compiler.common.resources import checked_bytes

from tools.generativeqc_posthf.conventions import SpinMOBlock
from tools.generativeqc_posthf.providers import ConventionalProvider
from tools.generativeqc_posthf.sources import NativeSource
from tools.generativeqc_response.backends import NativeJKBackend
from tools.generativeqc_response.implicit import (
    ImplicitSolveError,
    ResponseGMRES,
    checked_transpose_solve,
    transpose_solver_contract,
)
from tools.generativeqc_response.spin_cuda import CudaSpinJKBackend
from tools.generativeqc_response.uhf import UHFResponseOperator

from .complete_gradient import _nuclear_energy
from .uccsd_lambda import PreparedUCCSDLambda, UCCSDLambdaResult, _freeze

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self

    _RelaxedArguments = tuple[
        Mapping[str, np.ndarray],
        np.ndarray,
        str,
        str,
        str,
        float,
        float,
        Mapping[str, Any],
    ]


@dataclass(frozen=True)
class UCCSDPhysicalResponseOptions:
    """Complete physical solve bounds; stationary UHF branches need not be SPD."""

    max_bytes: int = 512 << 20
    max_total_work: int = 1_000_000_000_000
    max_operator_actions: int = 1024
    max_producer_tiles: int = 1_000_000_000
    max_primitive_terms: int = 1_000_000_000_000
    binding_tolerance: float = 1e-10
    orbital_tolerance: float = 1e-8
    gmres: GMRESOptions = field(
        default_factory=lambda: GMRESOptions(rtol=0.0, atol=1e-11, max_iterations=400)
    )

    def __post_init__(self) -> None:
        for name in (
            "max_bytes",
            "max_total_work",
            "max_operator_actions",
            "max_producer_tiles",
            "max_primitive_terms",
        ):
            checked_bytes(getattr(self, name), name)
            if not getattr(self, name):
                raise ValueError(f"{name} must be positive")
        for name, ceiling in (
            ("binding_tolerance", 1e-10),
            ("orbital_tolerance", 1e-8),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 < value <= ceiling
            ):
                raise ValueError(
                    f"{name} must be finite, positive and at most {ceiling}"
                )
        if not isinstance(self.gmres, GMRESOptions):
            raise TypeError("physical response requires GMRESOptions")


@dataclass(frozen=True)
class _UCCSDRelaxedCotangents:
    """Detached full-Frobenius raw-MO/overlap weights with fresh stationarity.

    g_ab is alpha-alpha/beta-beta chemists' order; g_ba is not another input.
    h_a/h_b and overlap_a/overlap_b must accumulate into the same physical AO
    matrices with their respective spin coefficients. No nuclear result is
    implied until every integral/overlap and nuclear-repulsion chain is bound.
    """

    weights: Mapping[str, np.ndarray]
    orbital_adjoint: np.ndarray
    state_identity: str
    lambda_identity: str
    response_identity: str
    orbital_residual_norm: float
    orbital_stationarity_max: float
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected = {"h_a", "h_b", "g_aa", "g_ab", "g_bb", "overlap_a", "overlap_b"}
        if set(self.weights) != expected:
            raise ValueError("relaxed UCCSD requires all seven physical weight blocks")
        for value in (*self.weights.values(), self.orbital_adjoint):
            if (
                type(value) is not np.ndarray
                or value.dtype != np.float64
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    "relaxed UCCSD weights must be finite base FP64 arrays"
                )
        for value in (self.orbital_residual_norm, self.orbital_stationarity_max):
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid relaxed UCCSD residual diagnostics")
        object.__setattr__(
            self,
            "weights",
            MappingProxyType(
                {name: immutable(value) for name, value in self.weights.items()}
            ),
        )
        object.__setattr__(self, "orbital_adjoint", immutable(self.orbital_adjoint))
        object.__setattr__(self, "provenance", _freeze(self.provenance))


@dataclass(frozen=True)
class UCCSDRelaxedCotangents(_UCCSDRelaxedCotangents):
    """Standard CCSD relaxed weights; corrected records are not this subtype."""


class _PreparedUCCSDPhysicalResponse:
    """Borrow live Lambda/provider and use one shared UHF response contract.

    The source/provider/reference, native executor and admitted solver/backend
    must remain live and unchanged. Preparation admits state, raw integrals,
    native actions and Krylov storage before producer reads or state copies.
    No SCF-response mathematics or dense response Jacobian is introduced here.
    """

    def __init__(
        self,
        lambdas: Any,
        result: Any,
        provider: ConventionalProvider,
        *,
        options: UCCSDPhysicalResponseOptions | None = None,
        response_backend: Any = None,
        solver: Any = None,
        nuclear_repulsion: float | None = None,
    ) -> None:
        started = time.perf_counter()
        self._validate_lambda(lambdas, result)
        parent = lambdas.prepared
        reference = parent.reference
        if (
            not isinstance(provider, ConventionalProvider)
            or provider._closed
            or provider.snapshot.identity != reference.identity
        ):
            raise ValueError("physical UCCSD requires a live identity-matched provider")
        provider.source._check_open()
        if provider.backend != parent.integral_backend:
            raise ValueError("physical UCCSD provider/backend differs from parent")
        if options is None:
            options = UCCSDPhysicalResponseOptions()
        if not isinstance(options, UCCSDPhysicalResponseOptions):
            raise TypeError("physical UCCSD requires response options")
        self.lambdas, self.result, self.provider = lambdas, result, provider
        self.reference, self.options, self.source = reference, options, provider.source
        self.source_identity = self.source.identity
        self._closed = False
        self.nmo = reference.nmo
        self.raw_bytes = (3 * self.nmo**4 + 4 * self.nmo**2) * 8
        preliminary = checked_bytes(
            lambdas.peak_bytes
            + 12 * self.raw_bytes
            + provider.budget_bytes
            + self._additional_numeric_bytes(),
            "physical UCCSD persistent/state storage",
        )
        if preliminary > options.max_bytes:
            raise MemoryError("physical UCCSD persistent/state budget exceeded")
        self.hamiltonian = build_raw_hamiltonian(
            parent.layout, max_elements=options.max_bytes // 8
        )
        self.programs = build_lagrangian_programs(
            parent.layout, self.hamiltonian, max_elements=options.max_bytes // 8
        )
        if self.programs.equation_hash != parent.program.logical_hash:
            raise ValueError("physical UCCSD CC equation identity differs")
        if response_backend is None:
            if provider.backend != "cpu":
                raise ValueError(
                    "CUDA physical UCCSD requires an explicit CUDA spin J/K backend"
                )
            response_backend = NativeJKBackend(
                self.source, budget_bytes=provider.budget_bytes
            )
        response_backend.validate_reference(reference)
        self.backend = response_backend
        if provider.backend == "cuda" and (
            not isinstance(response_backend, CudaSpinJKBackend)
            or response_backend.approximation != "exact"
            or response_backend.device_id != parent.executor.device
        ):
            raise ValueError(
                "CUDA physical UCCSD requires matching exact CUDA spin J/K; no CPU fallback"
            )
        if provider.backend == "cpu" and (
            not isinstance(response_backend, NativeJKBackend)
            or response_backend.source is not self.source
        ):
            raise ValueError(
                "CPU physical UCCSD requires the same native shell-tile source"
            )
        self.operator = UHFResponseOperator(
            UHFResponseOperator.build_problem(reference, response_backend),
            response_backend,
        )
        self.layout = self.operator.problem.layout
        self.solver: Any = (
            solver
            if solver is not None
            else (
                ResponseGMRES(self.operator.dimension, options.gmres)
                if self.operator.dimension
                else None
            )
        )
        if not self.operator.dimension and solver is not None:
            raise ValueError("empty UHF response space needs no solver")
        if (
            isinstance(self.solver, ResponseGMRES)
            and self.solver.dimension != self.operator.dimension
        ):
            raise ValueError("physical UCCSD orbital solver dimension mismatch")
        self._solver_contract = (
            None if self.solver is None else transpose_solver_contract(self.solver)
        )
        krylov = (
            0
            if self._solver_contract is None
            else self._solver_contract["workspace_bytes"]
        )
        if (
            isinstance(self.solver, ResponseGMRES)
            and krylov > self.solver.options.max_workspace_bytes
        ):
            raise MemoryError("physical UCCSD Krylov budget exceeded")
        self._resources = {
            program.logical_hash: lambdas._resource(program)
            for program in (
                self.hamiltonian.primal,
                self.hamiltonian.orbital_jvp.program,
                self.programs.primal,
                self.programs.weights,
                *self._additional_programs(),
            )
        }
        backend_bytes = checked_bytes(
            getattr(response_backend, "host_workspace_bytes", 0)
            + getattr(response_backend, "device_workspace_bytes", 0)
            + getattr(response_backend, "device_resident_bytes", 0)
            + 20 * reference.nbf**2 * 8,
            "shared UHF workspace",
        )
        self.peak_bytes = checked_bytes(
            preliminary
            + krylov
            + backend_bytes
            + max(value.native_bytes for value in self._resources.values()),
            "complete physical UCCSD numeric storage",
        )
        if self.peak_bytes > options.max_bytes:
            raise MemoryError("physical UCCSD native/Krylov/state budget exceeded")
        self.operator_work = 12 * self.nmo**4 + 32 * self.nmo**3 + 128 * self.nmo**2
        self.planned_work = checked_bytes(
            2 * self._resources[self.hamiltonian.primal.logical_hash].scalar_work
            + self._resources[self.programs.primal.logical_hash].scalar_work
            + 2 * self._resources[self.programs.weights.logical_hash].scalar_work
            + 2
            * self._resources[self.hamiltonian.orbital_jvp.derivative_hash].scalar_work
            + options.max_operator_actions * self.operator_work
            + lambdas.qualification_scalar_work
            + self._additional_scalar_work(),
            "complete physical UCCSD semantic work",
        )
        if self.planned_work > options.max_total_work:
            raise MemoryError("physical UCCSD complete work budget exceeded")
        # Admission counts complete source traversals even when a cache hit or
        # native symmetry schedule later avoids them. Primitive tuples are not
        # a claim about scalar recurrence FLOPs or host Krylov implementation.
        provider_tiles = sum(
            (size + provider.axis_tile - 1) // provider.axis_tile
            for size in self.source.shell_sizes
        )
        self.producer_transform_work = checked_bytes(
            6
            * sum(
                self.nmo ** (stage + 1)
                * self.reference.nbf ** (4 - stage)
                * provider_tiles**stage
                for stage in range(4)
            ),
            "physical UCCSD raw-block transform work",
        )
        self.planned_work = checked_bytes(
            self.planned_work + self.producer_transform_work,
            "physical UCCSD work including raw transforms",
        )
        if self.planned_work > options.max_total_work:
            raise MemoryError("physical UCCSD raw-transform work budget exceeded")
        response_tile = getattr(
            response_backend, "axis_tile", max(self.source.shell_sizes)
        )
        response_tiles = sum(
            (size + response_tile - 1) // response_tile
            for size in self.source.shell_sizes
        )
        self.planned_producer_tiles = checked_bytes(
            3 * provider_tiles**4
            + 3 * options.max_operator_actions * response_tiles**4
            + (2 if isinstance(self.source, NativeSource) else 0),
            "physical UCCSD producer tasks",
        )
        terms = (
            sum(
                size
                * len(shell.primitives)
                * (
                    (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
                    if self.source.representation == "real_spherical"
                    else 1
                )
                for size, shell in zip(
                    self.source.shell_sizes, self.source.shells, strict=True
                )
            )
            if isinstance(self.source, NativeSource)
            else 0
        )
        self.planned_primitive_terms = checked_bytes(
            (3 + 3 * options.max_operator_actions) * terms**4
            + (2 + len(self.source.atoms)) * terms**2
            if isinstance(self.source, NativeSource)
            else 0,
            "physical UCCSD primitive source terms",
        )
        if self.planned_producer_tiles > options.max_producer_tiles:
            raise MemoryError("physical UCCSD producer-task budget exceeded")
        if self.planned_primitive_terms > options.max_primitive_terms:
            raise MemoryError("physical UCCSD primitive-source budget exceeded")
        if isinstance(self.source, NativeSource):
            actual_nuclear = _nuclear_energy(self.source)
            if (
                nuclear_repulsion is not None
                and abs(nuclear_repulsion - actual_nuclear) > 1e-10
            ):
                raise ValueError(
                    "physical UCCSD nuclear-energy binding differs from native source"
                )
            nuclear_repulsion = actual_nuclear
        if (
            nuclear_repulsion is None
            or isinstance(nuclear_repulsion, bool)
            or not math.isfinite(nuclear_repulsion)
        ):
            raise ValueError(
                "non-native physical provider requires an explicit finite nuclear-repulsion binding"
            )
        self.nuclear_repulsion = float(nuclear_repulsion)
        self._nuclear_binding = self.nuclear_repulsion
        slots = tuple(range(self.nmo))
        self.requests = {
            pair: SpinMOBlock(
                (slots,) * 4,
                ("alpha", "alpha", "beta", "beta")
                if pair == "ab"
                else ("alpha" if pair == "aa" else "beta",) * 4,
            )
            for pair in ("aa", "ab", "bb")
        }
        provider.admit(tuple(self.requests.values()))
        self._consumer_binding = self._consumer_contract()
        # A supplied reference can be self-consistent yet describe a different
        # one-electron Hamiltonian. Native forces must differentiate the actual
        # source S/h, not silently bind native derivatives to arbitrary inputs.
        if isinstance(self.source, NativeSource):
            native_overlap, native_hcore = self.source.one_electron()
            for name, actual, expected in (
                ("overlap", native_overlap, reference.overlap),
                ("hcore", native_hcore, reference.hcore),
            ):
                if (
                    type(actual) is not np.ndarray
                    or actual.dtype != np.float64
                    or actual.shape != expected.shape
                    or not np.isfinite(actual).all()
                    or float(np.max(np.abs(actual - expected), initial=0))
                    > options.binding_tolerance
                ):
                    raise ValueError(
                        f"native UCCSD {name} binding differs from reference"
                    )
        self._binding = (
            lambdas,
            result,
            provider,
            self.source,
            reference,
            options,
            self.hamiltonian,
            self.programs,
            self.backend,
            self.operator,
            self.solver,
            *self._additional_binding(),
        )
        values = {}
        for spin, name in (("alpha", "a"), ("beta", "b")):
            coefficients = getattr(reference, f"coefficients_{spin}")
            values[f"h_{name}"] = immutable(
                coefficients.T @ reference.hcore @ coefficients
            )
            values[f"rotation_{name}"] = immutable(np.eye(self.nmo))
        for pair, block in self.requests.items():
            value = provider.get(block)
            if (
                value.reference_id != reference.identity
                or value.hamiltonian_id != reference.hamiltonian_id
            ):
                raise ValueError("physical UCCSD producer identity differs")
            values[f"g_{pair}"] = immutable(value.to_host())
        self.feeds = MappingProxyType(values)
        self._feed_binding = self.feeds
        self._assert_current()
        self.preparation_seconds = time.perf_counter() - started

    def _validate_lambda(self, lambdas: Any, result: Any) -> None:
        if not isinstance(lambdas, PreparedUCCSDLambda):
            raise TypeError("physical UCCSD requires PreparedUCCSDLambda")
        lambdas._check_open()
        if (
            not isinstance(result, UCCSDLambdaResult)
            or result.state_identity != lambdas.state_identity
        ):
            raise ValueError(
                "physical UCCSD requires the matching standard Lambda state"
            )

    def _additional_numeric_bytes(self) -> int:
        return 0

    def _additional_programs(self) -> tuple:
        return ()

    def _additional_scalar_work(self) -> int:
        return 0

    def _additional_binding(self) -> tuple:
        return ()

    def _qualify_lambda(self) -> tuple:
        packed, diagnostics = self.lambdas.qualify(self.result)
        return packed, diagnostics, None

    def _correction_weights(self, fresh: Any, counter: dict) -> Any:
        return None

    def _combine_weights(self, weights: dict, correction: Any) -> dict:
        if correction is None:
            return weights
        additional = correction["weights"]
        if weights.keys() != additional.keys():
            raise ValueError("corrected physical weight inventory differs")
        with np.errstate(over="raise", invalid="raise"):
            combined = {
                name: values + additional[name] for name, values in weights.items()
            }
        if any(not np.isfinite(values).all() for values in combined.values()):
            raise FloatingPointError("nonfinite corrected physical weights")
        return combined

    def _additional_identity(self, correction: Any) -> dict:
        return {}

    def _additional_provenance(self, correction: Any) -> dict:
        return {}

    def _make_result(
        self, arguments: _RelaxedArguments, correction: Any
    ) -> _UCCSDRelaxedCotangents:
        return UCCSDRelaxedCotangents(*arguments)

    def _assert_current(self) -> None:
        if self._closed:
            raise RuntimeError("physical UCCSD response is closed")
        self.lambdas._check_open()
        current = (
            self.lambdas,
            self.result,
            self.provider,
            self.source,
            self.reference,
            self.options,
            self.hamiltonian,
            self.programs,
            self.backend,
            self.operator,
            self.solver,
            *self._additional_binding(),
        )
        if any(
            actual is not expected
            for actual, expected in zip(current, self._binding, strict=True)
        ):
            raise RuntimeError("physical UCCSD admitted configuration changed")
        if (
            self.provider._closed
            or self.provider.source is not self.source
            or self.source.identity != self.source_identity
        ):
            raise RuntimeError(
                "physical UCCSD borrowed provider/source changed or closed"
            )
        self.source._check_open()
        self.operator.validate_current()
        if self.feeds is not self._feed_binding:
            raise RuntimeError("physical UCCSD raw feeds changed")
        if self.nuclear_repulsion != self._nuclear_binding:
            raise RuntimeError("physical UCCSD nuclear-energy configuration changed")
        self.backend.validate_reference(self.reference)
        if self._consumer_contract() != self._consumer_binding:
            raise RuntimeError("physical UCCSD provider/backend configuration changed")
        if (
            self.solver is not None
            and transpose_solver_contract(self.solver) != self._solver_contract
        ):
            raise RuntimeError("physical UCCSD solver contract changed")

    def _consumer_contract(self) -> tuple:
        """Bind mutable borrowed producer/backend admission controls, not statistics."""
        return (
            self.provider.snapshot.identity,
            self.provider.backend,
            self.provider.axis_tile,
            self.provider.budget_bytes,
            self.provider.cache_policy,
            self.provider.device_id,
            self.provider.cuda_artifact,
            self.backend.identity,
            self.backend.source,
            getattr(self.backend, "axis_tile", None),
            getattr(self.backend, "budget_bytes", None),
            getattr(self.backend, "device_id", None),
            getattr(self.backend, "approximation", None),
        )

    def _run(
        self, program: Any, extra: Mapping[str, np.ndarray], counter: dict
    ) -> dict:
        self._assert_current()
        counter["work"] += self._resources[program.logical_hash].scalar_work
        if counter["work"] > self.options.max_total_work:
            raise MemoryError("physical UCCSD work budget exhausted")
        values = self.lambdas.prepared.executor.execute(
            program, {**self.feeds, **extra}
        )
        if set(values) != set(program.outputs):
            raise ValueError("physical UCCSD native outputs are incomplete")
        for name, node in program.outputs.items():
            value = values[name]
            if (
                type(value) is not np.ndarray
                or value.dtype != np.float64
                or value.shape != node.spec.shape
            ):
                raise ValueError("physical UCCSD violated native FP64/shape contract")
            if not np.isfinite(value).all():
                raise FloatingPointError("nonfinite physical UCCSD native output")
        counter["native_calls"] += 1
        self._assert_current()
        return values

    def solve(self) -> _UCCSDRelaxedCotangents:
        """Requalify CC/Lambda, solve shared UHF adjoint, and freshly check all rotations."""
        self._assert_current()
        started = time.perf_counter()
        packed, lambda_diagnostics, fresh = self._qualify_lambda()
        counter = {
            "work": lambda_diagnostics["executed_semantic_scalar_work"],
            "native_calls": lambda_diagnostics["native_calls"],
            "operator_actions": 0,
        }
        correction = self._correction_weights(fresh, counter)
        fields = self._run(self.hamiltonian.primal, {}, counter)
        drift = max(
            float(
                np.max(
                    np.abs(fields[name] - self.lambdas.prepared.feeds[name]), initial=0
                )
            )
            for name in self.hamiltonian.parameter_names
        )
        if drift > self.options.binding_tolerance:
            raise ImplicitSolveError(
                "physical UCCSD normal-ordering binding differs from parent"
            )
        hf_drift = abs(
            float(fields["reference_electronic_energy"])
            + self.nuclear_repulsion
            - self.reference.reference_energy
        )
        if hf_drift > 1e-8:
            raise ImplicitSolveError(
                "physical UCCSD fresh raw HF energy differs from reference"
            )
        extra = {
            **self.lambdas.amplitudes,
            **self.lambdas._seeds(packed),
            "bar_correlation_energy": np.array(1.0),
            "bar_reference_electronic_energy": np.array(0.0),
            "bar_fov_aa": np.zeros(
                (self.reference.nocc("alpha"), self.nmo - self.reference.nocc("alpha"))
            ),
            "bar_fov_bb": np.zeros(
                (self.reference.nocc("beta"), self.nmo - self.reference.nocc("beta"))
            ),
        }
        primal = self._run(self.programs.primal, self.lambdas.amplitudes, counter)
        if (
            abs(float(primal["correlation_energy"]) - self.lambdas.correlation_energy)
            > 1e-10
        ):
            raise ImplicitSolveError("physical UCCSD fresh correlation energy differs")
        residual = max(
            float(np.max(np.abs(primal[name]), initial=0))
            for name in ("r1a", "r1b", "r2aa", "r2ab", "r2bb")
        )
        if residual > self.lambdas.options.cc_tolerance:
            raise ImplicitSolveError("physical UCCSD fresh CC stationarity failed")
        correlation = self._combine_weights(
            self._run(self.programs.weights, extra, counter), correction
        )
        rhs = self.layout.pack(
            correlation["orbital_rhs_a"], correlation["orbital_rhs_b"]
        )
        owner = self

        class Operator:
            def __init__(self) -> None:
                self.dimension = owner.operator.dimension

            def apply(self, vector: np.ndarray) -> np.ndarray:
                owner._assert_current()
                if counter["operator_actions"] >= owner.options.max_operator_actions:
                    raise MemoryError("physical UCCSD operator-action budget exhausted")
                counter["operator_actions"] += 1
                counter["work"] += owner.operator_work
                if counter["work"] > owner.options.max_total_work:
                    raise MemoryError("physical UCCSD work budget exhausted")
                value = owner.operator.apply_transpose(vector)
                owner._assert_current()
                return value

        for candidate in (
            np.ones(self.operator.dimension),
            np.arange(1, self.operator.dimension + 1, dtype=np.float64),
        ):
            if not candidate.size:
                break
            direction = candidate / _vector_norm(candidate)
            generated = self._run(
                self.hamiltonian.orbital_jvp.program,
                {
                    f"d_rotation_{suffix}": -self.layout.generator_matrix(
                        spin, direction
                    )
                    for spin, suffix in (("alpha", "a"), ("beta", "b"))
                },
                counter,
            )
            forward = self.layout.pack(generated["d_fov_aa"], generated["d_fov_bb"])
            if (
                _vector_norm(Operator().apply(direction) - forward)
                > self.options.orbital_tolerance
            ):
                raise ImplicitSolveError(
                    "shared UHF response differs from generated physical Fock JVP"
                )
        if self.operator.dimension:
            checked = checked_transpose_solve(
                Operator(), rhs, solver=self.solver, assert_current=self._assert_current
            )
            solution, response_residual = checked.solution, checked.residual_norm
            iterations = checked.iterations
        else:
            solution, response_residual, iterations = np.empty(0), 0.0, 0
        alpha, beta = self.layout.split(solution)
        weights = self._combine_weights(
            self._run(
                self.programs.weights,
                {
                    **extra,
                    "bar_reference_electronic_energy": np.array(1.0),
                    "bar_fov_aa": alpha,
                    "bar_fov_bb": beta,
                },
                counter,
            ),
            correction,
        )
        stationarity = max(
            float(np.max(np.abs(weights[f"stationarity_{spin}"]), initial=0))
            for spin in "ab"
        )
        if max(response_residual, stationarity) > self.options.orbital_tolerance:
            raise ImplicitSolveError(
                "physical UCCSD fresh full orbital stationarity failed"
            )
        self._assert_current()
        published = {
            name: weights[name]
            for name in ("h_a", "h_b", "g_aa", "g_ab", "g_bb", "overlap_a", "overlap_b")
        }
        identity = canonical_hash(
            {
                "lambda": self.result.lambda_identity,
                "binding": self.programs.binding_hash,
                "physical_lagrangian": self.programs.pullback.derivative_hash,
                "operator": self.operator.identity,
                **self._additional_identity(correction),
            }
        )
        return self._make_result(
            (
                published,
                solution,
                self.lambdas.state_identity,
                self.result.lambda_identity,
                identity,
                response_residual,
                stationarity,
                {
                    "tensor_backend": self.lambdas.prepared.executor.backend,
                    "source_backend": self.source.backend,
                    "orbital_backend": self.provider.backend,
                    "numeric_bound_bytes": self.peak_bytes,
                    "planned_semantic_scalar_work": self.planned_work,
                    "executed_semantic_scalar_work": counter["work"],
                    "work_model": "generated tensor scalar-loop work plus shared UHF matrix arithmetic; producer recurrence and host Krylov work excluded",
                    "planned_producer_tiles": self.planned_producer_tiles,
                    "planned_raw_transform_scalar_work": self.producer_transform_work,
                    "planned_primitive_source_terms": self.planned_primitive_terms,
                    "producer_work_model": "conservative full-traversal primitive tuples, including public-basis expansion; recurrence FLOPs and external Krylov work are not scalar-counted",
                    "native_calls": counter["native_calls"],
                    "operator_actions": counter["operator_actions"],
                    "iterations": iterations,
                    "binding_drift": drift,
                    "fresh_cc_residual_max": residual,
                    "raw_hf_energy_drift": hf_drift,
                    "lambda_qualification": lambda_diagnostics,
                    "physical_response_endpoint_seconds": self.preparation_seconds
                    + time.perf_counter()
                    - started,
                    "scope": "relaxed raw-MO full-Frobenius cotangents; AO/nuclear bindings remain explicit",
                    **self._additional_provenance(correction),
                },
            ),
            correction,
        )

    def close(self) -> None:
        """Release owned raw integrals/program metadata, not borrowed consumers."""
        self.feeds = MappingProxyType({})
        self._feed_binding = self.feeds
        self._resources.clear()
        self._closed = True

    def __enter__(self) -> Self:
        self._assert_current()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PreparedUCCSDPhysicalResponse(_PreparedUCCSDPhysicalResponse):
    """Standard CCSD physical response with the unchanged standard Lambda gate."""

    def __init__(
        self,
        lambdas: PreparedUCCSDLambda,
        result: UCCSDLambdaResult,
        provider: ConventionalProvider,
        *,
        options: UCCSDPhysicalResponseOptions | None = None,
        response_backend: Any = None,
        solver: Any = None,
        nuclear_repulsion: float | None = None,
    ) -> None:
        super().__init__(
            lambdas,
            result,
            provider,
            options=options,
            response_backend=response_backend,
            solver=solver,
            nuclear_repulsion=nuclear_repulsion,
        )

    def solve(self) -> UCCSDRelaxedCotangents:
        """Freshly qualify standard CCSD/Lambda and every orbital rotation."""
        return cast("UCCSDRelaxedCotangents", super().solve())
