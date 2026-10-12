"""Complete conventional UCCSD forces from shared relaxed physical weights.

No dense AO ERI cotangent or coordinate-major integral derivative is formed.
CPU ERI derivatives use packaged kernels; CPU one-electron derivatives use the
explicit native compatibility recurrence. CUDA uses the existing generated
one-electron and shell-ERI consumers without a CPU derivative fallback.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
from generativeqc_compiler.cc.gradient_equations import (
    build_ao_eri_weight_block_program,
    build_ao_one_electron_weight_program,
)
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.resources import checked_bytes

from tools.generativeqc_posthf.sources import NativeSource

from .complete_gradient import _nuclear_gradient
from .uccsd_lambda import _freeze
from .uccsd_physical_response import PreparedUCCSDPhysicalResponse

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self


@dataclass(frozen=True)
class UCCSDGradientOptions:
    """Explicit derivative policy and finite complete contraction admission.

    The byte model includes a CPU compatibility scratch allowance, not a hard
    allocator cap on the legacy recurrence. CUDA stage storage is enforced by
    its native consumer. Interpreter/compiler/CUDA-context storage is excluded.
    """

    max_bytes: int = 1 << 30
    max_total_work: int = 1_000_000_000_000
    max_shell_quartets: int = 1_000_000
    max_primitive_terms: int = 1_000_000_000_000
    derivative_stage_budget_bytes: int = 64 << 20
    cpu_one_electron_policy: str = "native-compatibility"

    def __post_init__(self) -> None:
        for name in (
            "max_bytes",
            "max_total_work",
            "max_shell_quartets",
            "max_primitive_terms",
            "derivative_stage_budget_bytes",
        ):
            if not checked_bytes(getattr(self, name), name):
                raise ValueError(f"{name} must be positive")
        if self.cpu_one_electron_policy != "native-compatibility":
            raise ValueError(
                "CPU one-electron derivatives require explicit native-compatibility policy"
            )


@dataclass(frozen=True)
class UCCSDGradientResult:
    """Detached total energy, positive nuclear gradient, and negative forces."""

    total_energy: float
    gradient: np.ndarray
    forces: np.ndarray
    response_identity: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not math.isfinite(self.total_energy):
            raise ValueError("nonfinite UCCSD total energy")
        for value in (self.gradient, self.forces):
            if (
                type(value) is not np.ndarray
                or value.dtype != np.float64
                or value.ndim != 2
                or value.shape[1] != 3
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    "UCCSD gradient requires finite FP64 [atom,xyz] arrays"
                )
        if self.gradient.shape != self.forces.shape or not np.array_equal(
            self.forces, -self.gradient
        ):
            raise ValueError("UCCSD force must be exactly the negative gradient")
        object.__setattr__(self, "gradient", immutable(self.gradient))
        object.__setattr__(self, "forces", immutable(self.forces))
        object.__setattr__(self, "provenance", _freeze(self.provenance))


class PreparedUCCSDGradient:
    """Borrow a physical owner and bind every AO derivative and nuclear term.

    Each solve freshly qualifies CC, standard Lambda, and full orbital
    stationarity before contracting shell derivatives. Three raw-MO spin
    Coulomb cotangents remain explicit; g_ab is not doubled or pair-folded.
    The derivative scope is all-electron conventional s/p/d, without frozen
    core or ECP. This internal owner does not register Calculator methods.
    """

    def __init__(
        self,
        physical: PreparedUCCSDPhysicalResponse,
        *,
        options: UCCSDGradientOptions | None = None,
    ) -> None:
        started = time.perf_counter()
        if not isinstance(physical, PreparedUCCSDPhysicalResponse):
            raise TypeError("UCCSD gradient requires a physical response owner")
        options = UCCSDGradientOptions() if options is None else options
        if not isinstance(options, UCCSDGradientOptions):
            raise TypeError("UCCSD gradient requires gradient options")
        physical._assert_current()
        if not isinstance(physical.source, NativeSource):
            raise TypeError("complete UCCSD gradient requires a native shell source")
        if any(shell.angular_momentum > 2 for shell in physical.source.shells):
            raise NotImplementedError(
                "UCCSD shell derivatives currently support s/p/d only"
            )
        self.physical, self.options = physical, options
        self.source, self.reference = physical.source, physical.reference
        self._closed = False
        self.backend = physical.provider.backend
        self.nmo = self.reference.nmo
        self.planned_shell_quartets = checked_bytes(
            len(self.source.shells) ** 4, "UCCSD derivative shell tasks"
        )
        if self.planned_shell_quartets > options.max_shell_quartets:
            raise MemoryError("UCCSD derivative shell-task budget exceeded")
        terms = sum(
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
        self.planned_primitive_terms = checked_bytes(
            terms**4 + (2 + len(self.source.atoms)) * terms**2,
            "UCCSD derivative primitive tuples",
        )
        if self.planned_primitive_terms > options.max_primitive_terms:
            raise MemoryError("UCCSD derivative primitive-task budget exceeded")
        preliminary = checked_bytes(
            physical.peak_bytes
            + 4 * physical.raw_bytes
            + options.derivative_stage_budget_bytes
            + 12 * self.nmo**2 * 8
            + 12 * len(self.source.atoms) * 8,
            "UCCSD gradient logical persistent storage",
        )
        if preliminary > options.max_bytes:
            raise MemoryError("UCCSD gradient persistent byte budget exceeded")
        self.one_program = build_ao_one_electron_weight_program(
            self.nmo,
            max_elements=options.max_bytes // 8,
        )
        sizes = Counter(self.source.shell_sizes)
        self.eri_programs = {
            shape: build_ao_eri_weight_block_program(
                self.nmo,
                shape,
                max_elements=options.max_bytes // 8,
            )
            for shape in product(sorted(sizes), repeat=4)
        }
        self.resources = {
            program.logical_hash: physical.lambdas._resource(program)
            for program in (self.one_program, *self.eri_programs.values())
        }
        self.peak_bytes = checked_bytes(
            preliminary + max(value.native_bytes for value in self.resources.values()),
            "UCCSD gradient native logical storage",
        )
        self.planned_work = checked_bytes(
            physical.planned_work
            + 2 * self.resources[self.one_program.logical_hash].scalar_work
            + sum(
                3
                * math.prod(sizes[extent] for extent in shape)
                * self.resources[program.logical_hash].scalar_work
                for shape, program in self.eri_programs.items()
            ),
            "UCCSD gradient generated scalar work",
        )
        if self.peak_bytes > options.max_bytes:
            raise MemoryError("UCCSD gradient native byte budget exceeded")
        if self.planned_work > options.max_total_work:
            raise MemoryError("UCCSD gradient scalar-work budget exceeded")
        self._binding = (
            physical,
            options,
            self.source,
            self.reference,
            self.one_program,
            self.eri_programs,
            self.resources,
            self.backend,
        )
        self.preparation_seconds = time.perf_counter() - started

    def _assert_current(self) -> None:
        if self._closed:
            raise RuntimeError("prepared UCCSD gradient is closed")
        self.physical._assert_current()
        current = (
            self.physical,
            self.options,
            self.source,
            self.reference,
            self.one_program,
            self.eri_programs,
            self.resources,
            self.backend,
        )
        if any(
            actual is not expected
            for actual, expected in zip(current, self._binding, strict=True)
        ):
            raise RuntimeError("UCCSD gradient admitted configuration changed")

    def _run(
        self, program: Any, feeds: Mapping[str, np.ndarray], counter: dict
    ) -> dict:
        self._assert_current()
        counter["work"] += self.resources[program.logical_hash].scalar_work
        if counter["work"] > self.options.max_total_work:
            raise MemoryError("UCCSD gradient scalar-work budget exhausted")
        outputs = self.physical.lambdas.prepared.executor.execute(program, feeds)
        if set(outputs) != set(program.outputs):
            raise ValueError("UCCSD AO native transform returned incomplete outputs")
        for name, node in program.outputs.items():
            value = outputs[name]
            if (
                type(value) is not np.ndarray
                or value.dtype != np.float64
                or value.shape != node.spec.shape
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    "UCCSD AO native transform violated finite FP64 shape contract"
                )
        counter["transform_calls"] += 1
        self._assert_current()
        return outputs

    def solve(self) -> UCCSDGradientResult:
        """Publish only after all fresh response and native derivative gates pass."""
        self._assert_current()
        started = time.perf_counter()
        relaxed = self.physical.solve()
        counter = {
            "work": relaxed.provenance["executed_semantic_scalar_work"],
            "transform_calls": 0,
        }
        one = {name: np.zeros((self.nmo, self.nmo)) for name in ("hcore", "overlap")}
        coefficients = {
            suffix: getattr(self.reference, f"coefficients_{spin}")
            for spin, suffix in (("alpha", "a"), ("beta", "b"))
        }
        for spin in "ab":
            values = self._run(
                self.one_program,
                {
                    "coefficients": coefficients[spin],
                    "hcore": relaxed.weights[f"h_{spin}"],
                    "overlap": relaxed.weights[f"overlap_{spin}"],
                },
                counter,
            )
            for name in one:
                one[name] += values[name]
        if self.backend == "cuda":
            gradient, one_resources = self.source.one_electron_gradient_cuda(
                overlap_weights=one["overlap"],
                kinetic_weights=one["hcore"],
                attraction_weights=one["hcore"],
                device_id=self.physical.backend.device_id,
                stage_budget_bytes=self.options.derivative_stage_budget_bytes,
            )
            one_policy = "generated-cuda"
        else:
            gradient = self.source.one_electron_gradient_cpu(
                overlap_weights=one["overlap"],
                hcore_weights=one["hcore"],
            )
            one_resources, one_policy = {}, "explicit-native-compatibility"
        self._assert_current()
        if (
            type(gradient) is not np.ndarray
            or gradient.dtype != np.float64
            or gradient.shape != (len(self.source.atoms), 3)
            or not np.isfinite(gradient).all()
        ):
            raise ValueError(
                "native UCCSD one-electron derivative returned invalid gradients"
            )
        gradient = np.array(gradient, copy=True)
        offsets = np.cumsum((0, *self.source.shell_sizes))
        primitive_evaluations, calls = 0, 0
        for indices in product(range(len(self.source.shells)), repeat=4):
            shape = tuple(self.source.shell_sizes[index] for index in indices)
            program = self.eri_programs[shape]
            block = np.zeros(shape)
            for pair in ("aa", "ab", "bb"):
                feeds = {
                    f"coefficients_{slot}": coefficients[spin][
                        offsets[index] : offsets[index + 1]
                    ]
                    for slot, (spin, index) in enumerate(
                        zip(pair[0] * 2 + pair[1] * 2, indices, strict=True)
                    )
                }
                block += self._run(
                    program, {**feeds, "eri": relaxed.weights[f"g_{pair}"]}, counter
                )["eri"]
            self._assert_current()
            if self.backend == "cuda":
                local = self.source.weighted_eri_shell_gradient_cuda(
                    indices,
                    block,
                    device_id=self.physical.backend.device_id,
                    stage_budget_bytes=self.options.derivative_stage_budget_bytes,
                )
            else:
                local, evaluations = self.source.weighted_eri_shell_gradient_cpu(
                    indices, block
                )
                primitive_evaluations += evaluations
            if (
                type(local) is not np.ndarray
                or local.dtype != np.float64
                or local.shape != (4, 3)
                or not np.isfinite(local).all()
            ):
                raise ValueError(
                    "native UCCSD shell derivative returned invalid slot gradients"
                )
            for slot, index in enumerate(indices):
                gradient[self.source.shells[index].atom_index] += local[slot]
            calls += 1
            self._assert_current()
        gradient += _nuclear_gradient(self.source)
        if not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite complete UCCSD gradient")
        if primitive_evaluations > self.planned_primitive_terms:
            raise MemoryError("UCCSD derivative primitive count exceeded admission")
        self._assert_current()
        return UCCSDGradientResult(
            self.reference.reference_energy + self.physical.lambdas.correlation_energy,
            gradient,
            -gradient,
            relaxed.response_identity,
            {
                "physical_response": relaxed.provenance,
                "derivative_backend": self.backend,
                "one_electron_policy": one_policy,
                "eri_policy": "generated-cuda"
                if self.backend == "cuda"
                else "packaged-cpu",
                "one_electron_resources": one_resources,
                "logical_numeric_bound_bytes": self.peak_bytes,
                "memory_scope": "logical state/native transform and derivative stage allowance; CPU compatibility recurrence scratch is not allocator-capped; excludes interpreter/compiler/context",
                "planned_generated_scalar_work": self.planned_work,
                "executed_generated_scalar_work": counter["work"],
                "planned_derivative_primitive_terms": self.planned_primitive_terms,
                "actual_cpu_eri_primitive_evaluations": primitive_evaluations,
                "shell_quartets": calls,
                "ao_transform_calls": counter["transform_calls"],
                "maximum_ao_eri_weight_block_elements": max(
                    math.prod(shape) for shape in self.eri_programs
                ),
                "nuclear_repulsion_terms": 1,
                "complete_gradient_endpoint_seconds": self.preparation_seconds
                + self.physical.preparation_seconds
                + time.perf_counter()
                - started,
            },
        )

    def close(self) -> None:
        """Close this owner without closing any borrowed physical consumer."""
        self._closed = True

    def __enter__(self) -> Self:
        self._assert_current()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
