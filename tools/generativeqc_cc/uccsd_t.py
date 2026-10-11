"""Complete conventional UCCSD(T) energy from a qualified UCCSD state.

One host-controlled owner admits the complete ordered tile grid before any
native execution, then rechecks the physical UCCSD state and accumulates all
four spin channels through the selected CPU/CUDA TensorIR executor. Scientific
equations remain in the compiler; host binding/accumulation is explicit and
never substitutes CPU contractions or a reference oracle for CUDA execution.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from generativeqc_compiler.cc.uccsd_triples import (
    CHANNELS,
    TriplesTileProgram,
    build_tile_program,
    channel_possible,
    channel_shape,
)
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.evidence import canonical_hash
from generativeqc_compiler.common.resources import MAX_BYTES, checked_bytes
from generativeqc_compiler.tensor.cpu import emit_cpu
from generativeqc_compiler.tensor.cuda_plan import plan_cuda
from typing_extensions import Self

from .native_tensor_cuda import CudaCCTensorExecutor
from .uccsd_solver import PreparedUCCSD, UCCSDResult

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class UCCSDTOptions:
    """Explicit logical numeric and complete-grid work limits, not RSS limits.

    A larger requested tile may shrink deterministically when numeric or
    per-call executor capacity requires it. Neither the total tile nor the
    total semantic-work limit is bypassed by that memory fallback.
    """

    tile_shape: tuple[int, ...] = (4, 4, 4, 2, 2, 2)
    max_bytes: int = 512 << 20
    max_tiles: int = 100_000
    max_total_work: int = 100_000_000_000
    denominator_threshold: float = 1e-10

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tile_shape, tuple)
            or len(self.tile_shape) != 6
            or any(
                type(size) is not int or not 0 < size <= MAX_BYTES
                for size in self.tile_shape
            )
        ):
            raise ValueError("UCCSD(T) tile shape requires six positive int64 integers")
        for name in ("max_bytes", "max_tiles", "max_total_work"):
            if checked_bytes(getattr(self, name), name) == 0:
                raise ValueError(f"UCCSD(T) {name} must be positive")
        if (
            isinstance(self.denominator_threshold, bool)
            or not math.isfinite(self.denominator_threshold)
            or self.denominator_threshold <= 0
        ):
            raise ValueError(
                "UCCSD(T) denominator threshold must be positive and finite"
            )


@dataclass(frozen=True)
class UCCSDTResult:
    """Immutable complete energies, or an explicit failure with no partial total."""

    status: str
    reason: str
    ccsd_correlation_energy: float
    triples_energy: float | None
    channel_energies: Mapping[str, float]
    correlation_energy: float | None
    total_energy: float | None
    provenance: Mapping[str, Any]

    @property
    def converged(self) -> bool:
        """Whether the rechecked parent and every admitted triples tile succeeded."""
        return self.status == "converged"


@dataclass(frozen=True)
class _Template:
    tile: TriplesTileProgram
    multiplicity: int
    peak_bytes: int
    scalar_work: int
    binding_identity: str


def _axis_variants(size: int, limit: int) -> tuple[tuple[int, int], ...]:
    whole, remainder = divmod(size, min(size, limit))
    return tuple(
        (length, count)
        for length, count in ((min(size, limit), whole), (remainder, 1))
        if length and count
    )


class PreparedUCCSDT:
    """Borrow a live UCCSD owner and retain an owned immutable excitation state.

    The parent and its executor must remain open and configuration-stable;
    provider/source closure is allowed because the parent owns its MO feeds.
    Closing this owner never closes borrowed objects. Preparation performs only
    metadata/host admission; energy() does the fresh selected-backend replay.
    """

    def __init__(
        self,
        prepared: PreparedUCCSD,
        state: UCCSDResult,
        *,
        options: UCCSDTOptions | None = None,
    ) -> None:
        started = time.perf_counter()
        if not isinstance(prepared, PreparedUCCSD) or prepared._closed:
            raise TypeError("UCCSD(T) requires a live PreparedUCCSD owner")
        if prepared._execution_configuration() != prepared._executor_configuration:
            raise RuntimeError("UCCSD(T) parent executor configuration changed")
        if not isinstance(state, UCCSDResult) or not state.converged:
            raise ValueError("UCCSD(T) requires a converged UCCSD state")
        options = UCCSDTOptions() if options is None else options
        if not isinstance(options, UCCSDTOptions):
            raise TypeError("UCCSD(T) options must be UCCSDTOptions")
        identities = {
            "reference_id": prepared.reference.identity,
            "hamiltonian_id": prepared.reference.hamiltonian_id,
            "integral_hash": prepared.integral_hash,
            "equation_hash": prepared.program.logical_hash,
        }
        if any(
            state.provenance.get(name) != value for name, value in identities.items()
        ):
            raise ValueError("UCCSD(T) state identities differ from the bound parent")
        if state.correlation_energy is None or not math.isfinite(
            state.correlation_energy
        ):
            raise ValueError("UCCSD(T) requires finite accepted CCSD energy")
        expected_total = prepared.reference.reference_energy + state.correlation_energy
        if (
            state.total_energy is None
            or not math.isfinite(expected_total)
            or state.total_energy != expected_total
        ):
            raise ValueError(
                "UCCSD(T) parent total energy is inconsistent or nonfinite"
            )
        self.prepared, self.options, self.layout = prepared, options, prepared.layout
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
        self.ccsd_correlation_energy = float(state.correlation_energy)
        self.channels = tuple(
            channel for channel in CHANNELS if channel_possible(self.layout, channel)
        )
        self.minimum_absolute_denominator = self._denominator_admission()
        full_bytes = sum(
            block.full_layout.storage_bytes(8) for block in self.layout.blocks
        )
        packed_bytes = 8 * self.layout.packed_size
        self.base_bytes = checked_bytes(
            prepared.reference.numeric_bytes
            + sum(value.nbytes for value in prepared.feeds.values())
            + prepared.denominators.nbytes
            + 3 * full_bytes
            + 4 * packed_bytes,
            "UCCSD(T) persistent/state admission bytes",
        )
        self.replay_peak_bytes = checked_bytes(
            self.base_bytes + 4 * full_bytes + prepared.execution_peak_bytes,
            "UCCSD(T) physical replay bytes",
        )
        if self.replay_peak_bytes > options.max_bytes:
            raise MemoryError("UCCSD(T) budget cannot admit the physical parent replay")
        _, replay_resources = emit_cpu(
            prepared.program,
            max_bytes=MAX_BYTES,
            max_work=MAX_BYTES,
            max_nodes=getattr(prepared.executor, "max_nodes", 16384),
        )
        self.replay_scalar_work = replay_resources["scalar_work"]
        self.tile_shape, self.templates, self.fallback_steps = self._admit_templates()
        self.tile_count = checked_bytes(
            sum(template.multiplicity for template in self.templates.values()),
            "UCCSD(T) complete tile count",
        )
        self.triples_scalar_work = checked_bytes(
            sum(
                template.multiplicity * template.scalar_work
                for template in self.templates.values()
            ),
            "UCCSD(T) complete semantic work",
        )
        self.total_scalar_work = checked_bytes(
            self.triples_scalar_work + self.replay_scalar_work,
            "UCCSD(T) replay plus complete-grid work",
        )
        if self.tile_count > options.max_tiles:
            raise MemoryError("UCCSD(T) complete grid exceeds the tile-count limit")
        if self.total_scalar_work > options.max_total_work:
            raise MemoryError("UCCSD(T) complete grid exceeds the semantic-work limit")
        self.peak_bytes = max(
            self.replay_peak_bytes,
            max(
                (template.peak_bytes for template in self.templates.values()), default=0
            ),
        )
        packed = self.layout.pack(state.amplitudes)
        if not np.array_equal(packed, state.packed_amplitudes):
            raise ValueError("UCCSD(T) full and packed parent amplitudes disagree")
        self.packed_amplitudes = immutable(packed)
        self.amplitudes = MappingProxyType(
            {
                name: immutable(value)
                for name, value in self.layout.unpack(self.packed_amplitudes).items()
            }
        )
        sources = {**prepared.feeds, **self.amplitudes}
        for spin, suffix in (("alpha", "a"), ("beta", "b")):
            energies = getattr(prepared.reference, f"orbital_energies_{spin}")
            occupied = prepared.reference.nocc(spin)
            sources[f"eps_o_{suffix}"] = energies[:occupied]
            sources[f"eps_v_{suffix}"] = energies[occupied:]
        self.sources = MappingProxyType(sources)
        if any(
            binding.source not in self.sources
            for template in self.templates.values()
            for binding in template.tile.operands
        ):
            raise ValueError("UCCSD(T) parent lacks a required physical source feed")
        self.state_identity = canonical_hash(
            {
                **identities,
                "packed_amplitudes": sha256(
                    self.packed_amplitudes.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "correlation_energy": self.ccsd_correlation_energy,
            }
        )
        self.preparation_seconds = time.perf_counter() - started
        self._admission_binding = (
            self.prepared,
            self.options,
            self.layout,
            self.templates,
            self.sources,
            self.packed_amplitudes,
            self.amplitudes,
            self.tile_shape,
        )

    def _denominator_admission(self) -> float | None:
        """Guard every executed ordered coordinate, including repeated indices.

        Repeated same-spin coordinates have zero exact moments, but admitting
        a tiny denominator there could amplify cancellation roundoff. No level
        shift, regularization or distinct-index-only shortcut is applied.
        """
        minima = []
        for channel in self.channels:
            try:
                with np.errstate(over="raise", invalid="raise"):
                    prefix_minimum = prefix_maximum = 0.0
                    for position, suffix in enumerate(channel * 2):
                        spin = "alpha" if suffix == "a" else "beta"
                        energies = getattr(
                            self.prepared.reference, f"orbital_energies_{spin}"
                        )
                        occupied = self.prepared.reference.nocc(spin)
                        if position < 3:
                            prefix_minimum += np.min(energies[:occupied])
                            prefix_maximum += np.max(energies[:occupied])
                        else:
                            prefix_minimum -= np.max(energies[occupied:])
                            prefix_maximum -= np.min(energies[occupied:])
                        if not np.isfinite((prefix_minimum, prefix_maximum)).all():
                            raise ValueError("nonfinite UCCSD(T) denominator prefix")
            except FloatingPointError as error:
                raise ValueError("nonfinite UCCSD(T) denominator extrema") from error
            # The graph sums occupied energies before subtracting virtual ones.
            # Gap-times-multiplicity can miss FP64 cancellation in that order;
            # extrema also cannot assume sorting tighter than the reference gate.
            largest = prefix_maximum
            if largest >= 0 or -largest <= self.options.denominator_threshold:
                raise ValueError(
                    "near-zero or non-Aufbau physical UCCSD(T) denominator"
                )
            minima.append(float(-largest))
        return min(minima) if minima else None

    def _template(
        self, channel: str, shape: tuple[int, ...], multiplicity: int
    ) -> _Template:
        tile = build_tile_program(self.layout, channel, shape)
        executor = self.prepared.executor
        _, resources = emit_cpu(
            tile.program,
            max_bytes=MAX_BYTES,
            max_work=MAX_BYTES,
            max_nodes=getattr(executor, "max_nodes", 16384),
        )
        if isinstance(executor, CudaCCTensorExecutor):
            try:
                native_peak = plan_cuda(
                    tile.program, executor.compiler.target, max_bytes=executor.max_bytes
                ).peak_bytes
            except ValueError as error:
                if "infeasible tensor byte budget" not in str(error):
                    raise
                raise MemoryError(
                    "UCCSD(T) tile exceeds the CUDA executor budget"
                ) from error
        else:
            native_peak = resources["required_bytes"]
            if (
                native_peak > executor.max_bytes
                or resources["scalar_work"] > executor.max_work
            ):
                raise MemoryError("UCCSD(T) tile exceeds the CPU executor budget")
        input_bytes = sum(
            node.spec.size * 8 for node in tile.program.live_nodes if node.op == "input"
        )
        peak = checked_bytes(
            self.base_bytes
            + native_peak
            + 2 * input_bytes
            + 16 * (6 * self.prepared.reference.nmo + sum(shape)),
            "UCCSD(T) bound/execution numeric bytes",
        )
        if peak > self.options.max_bytes:
            raise MemoryError("UCCSD(T) tile exceeds the complete numeric budget")
        return _Template(
            tile,
            multiplicity,
            peak,
            resources["scalar_work"],
            canonical_hash(
                {
                    "equation": tile.program.logical_hash,
                    "operands": [asdict(operand) for operand in tile.operands],
                }
            ),
        )

    def _admit_templates(self) -> tuple[tuple[int, ...], Mapping, int]:
        """Enumerate edge shapes/multiplicities without materializing the grid."""
        if not self.channels:
            return self.options.tile_shape, MappingProxyType({}), 0
        shapes = {
            channel: channel_shape(self.layout, channel) for channel in self.channels
        }
        limit = tuple(
            max(
                min(shape[axis], self.options.tile_shape[axis])
                for shape in shapes.values()
            )
            for axis in range(6)
        )
        fallback_steps = 0
        while True:
            templates = {}
            try:
                for channel, full_shape in shapes.items():
                    for variants in product(
                        *(
                            _axis_variants(size, bound)
                            for size, bound in zip(full_shape, limit, strict=True)
                        )
                    ):
                        shape = tuple(length for length, _ in variants)
                        count = checked_bytes(
                            math.prod(count for _, count in variants),
                            "UCCSD(T) shape multiplicity",
                        )
                        templates[channel, shape] = self._template(
                            channel, shape, count
                        )
                return limit, MappingProxyType(templates), fallback_steps
            except MemoryError:
                if max(limit) == 1:
                    raise MemoryError(
                        "UCCSD(T) minimum tile cannot satisfy the explicit budgets"
                    ) from None
                axis = max(range(6), key=lambda position: limit[position])
                limit = tuple(
                    max(1, size // 2) if position == axis else size
                    for position, size in enumerate(limit)
                )
                fallback_steps += 1

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("prepared UCCSD(T) is closed")
        admitted = (
            self.prepared,
            self.options,
            self.layout,
            self.templates,
            self.sources,
            self.packed_amplitudes,
            self.amplitudes,
            self.tile_shape,
        )
        if any(
            actual is not expected
            for actual, expected in zip(admitted, self._admission_binding, strict=True)
        ):
            raise RuntimeError("UCCSD(T) admitted owner configuration changed")
        prepared = self.prepared
        current = (
            prepared.reference,
            prepared.layout,
            prepared.program,
            prepared.feeds,
            prepared.options,
            prepared.executor,
        )
        if prepared._closed or any(
            actual is not expected
            for actual, expected in zip(current, self._parent_binding, strict=True)
        ):
            raise RuntimeError("UCCSD(T) borrowed parent binding changed or closed")
        if prepared._execution_configuration() != self._executor_configuration:
            raise RuntimeError("UCCSD(T) borrowed executor configuration changed")

    def _bind(
        self, tile: TriplesTileProgram, coordinates: tuple[tuple[int, ...], ...]
    ) -> dict:
        """Gather bounded host inputs according to compiler-owned bindings only."""
        positions = dict(zip("ijkabc", coordinates, strict=True))
        feeds = {}
        for binding in tile.operands:
            values = self.sources[binding.source].transpose(binding.axes)
            selected = tuple(
                positions.get(label, tuple(range(size)))
                for label, size in zip(binding.labels, values.shape, strict=True)
            )
            feeds[binding.name] = np.ascontiguousarray(values[np.ix_(*selected)])
        return feeds

    def energy(self) -> UCCSDTResult:
        """Recheck the parent, then accumulate all admitted tiles transactionally."""
        self._check_open()
        started = time.perf_counter()
        outputs = self.prepared.evaluate(self.packed_amplitudes)
        energy_drift = abs(
            float(outputs["correlation_energy"]) - self.ccsd_correlation_energy
        )
        residual_max = max(
            float(np.max(np.abs(value), initial=0))
            for name, value in outputs.items()
            if name != "correlation_energy"
        )
        if (
            energy_drift > self.prepared.options.energy_tolerance
            or residual_max > self.prepared.options.residual_tolerance
        ):
            raise ValueError(
                "UCCSD(T) parent failed fresh physical residual/energy qualification"
            )
        del outputs
        self._check_open()
        channel_energies = {channel: 0.0 for channel in CHANNELS}
        completed = 0
        status, reason = (
            "converged",
            "fresh parent and all four complete triples channels passed",
        )
        try:
            for channel in self.channels:
                full_shape = channel_shape(self.layout, channel)
                ranges = tuple(
                    range(0, size, bound)
                    for size, bound in zip(full_shape, self.tile_shape, strict=True)
                )
                total = compensation = 0.0
                for starts in product(*ranges):
                    coordinates = tuple(
                        tuple(range(start, min(start + bound, size)))
                        for start, bound, size in zip(
                            starts, self.tile_shape, full_shape, strict=True
                        )
                    )
                    template = self.templates[
                        channel, tuple(len(values) for values in coordinates)
                    ]
                    self._check_open()
                    result = self.prepared.executor.execute(
                        template.tile.program, self._bind(template.tile, coordinates)
                    )
                    if result.keys() != {"triples_energy"}:
                        raise ValueError(
                            "UCCSD(T) executor returned incomplete tile outputs"
                        )
                    value = np.asarray(result["triples_energy"])
                    if value.dtype != np.float64 or value.shape != ():
                        raise ValueError(
                            "UCCSD(T) executor violated the FP64 scalar contract"
                        )
                    if not np.isfinite(value):
                        raise FloatingPointError("nonfinite UCCSD(T) tile energy")
                    correction = float(value) - compensation
                    next_total = total + correction
                    compensation = (next_total - total) - correction
                    total = next_total
                    if not math.isfinite(total) or not math.isfinite(compensation):
                        raise FloatingPointError(
                            "nonfinite UCCSD(T) channel accumulation"
                        )
                    completed += 1
                channel_energies[channel] = total
            self._check_open()
            if completed != self.tile_count:
                raise RuntimeError("UCCSD(T) incomplete admitted tile coverage")
            triples = math.fsum(channel_energies.values())
            correlation = self.ccsd_correlation_energy + triples
            total_energy = self.prepared.reference.reference_energy + correlation
            if not all(
                math.isfinite(value) for value in (triples, correlation, total_energy)
            ):
                raise FloatingPointError("nonfinite complete UCCSD(T) energy")
        except (FloatingPointError, OverflowError) as error:
            status, reason = "nonfinite", str(error)
            triples = correlation = total_energy = None
            channel_energies = {}
        except (ValueError, RuntimeError, MemoryError) as error:
            status, reason = "execution_error", str(error)
            triples = correlation = total_energy = None
            channel_energies = {}
        elapsed = time.perf_counter() - started
        provenance = MappingProxyType(
            {
                "reference_id": self.prepared.reference.identity,
                "hamiltonian_id": self.prepared.reference.hamiltonian_id,
                "integral_hash": self.prepared.integral_hash,
                "state_identity": self.state_identity,
                "tensor_backend": self.prepared.executor.backend,
                "integral_backend": self.prepared.integral_backend,
                "source_backend": self.prepared.source_backend,
                "iteration_backend": "host-packed-excitation-diis",
                "triples_binding_backend": "host-fp64-bounded-gather-and-compensated-sum",
                "minimum_absolute_executed_denominator": self.minimum_absolute_denominator,
                "fresh_parent_residual_max": residual_max,
                "fresh_parent_energy_drift": energy_drift,
                "numeric_bound_bytes": self.peak_bytes,
                "tile_shape": self.tile_shape,
                "fallback_steps": self.fallback_steps,
                "planned_tiles": self.tile_count,
                "completed_tiles": completed,
                "semantic_scalar_work": self.total_scalar_work,
                "triples_semantic_scalar_work": self.triples_scalar_work,
                "replay_semantic_scalar_work": self.replay_scalar_work,
                "channel_tiles": MappingProxyType(
                    {
                        channel: sum(
                            template.multiplicity
                            for template in self.templates.values()
                            if template.tile.channel == channel
                        )
                        for channel in CHANNELS
                    }
                ),
                "work_model": "sum of exact admitted CPU TensorIR scalar-loop work, not hardware FLOPs",
                "template_identities": tuple(
                    template.binding_identity for template in self.templates.values()
                ),
                "options": MappingProxyType(asdict(self.options)),
                "preparation_seconds": self.preparation_seconds,
                "triples_endpoint_seconds": self.preparation_seconds + elapsed,
            }
        )
        return UCCSDTResult(
            status,
            reason,
            self.ccsd_correlation_energy,
            triples,
            MappingProxyType(channel_energies),
            correlation,
            total_energy,
            provenance,
        )

    def close(self) -> None:
        """Release owned states/templates, without closing any borrowed object."""
        self._admission_binding = ()
        self._parent_binding = ()
        self.sources = MappingProxyType({})
        self.amplitudes = MappingProxyType({})
        self.packed_amplitudes = np.empty(0)
        self.templates = MappingProxyType({})
        self._closed = True

    def __enter__(self) -> Self:
        self._check_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
