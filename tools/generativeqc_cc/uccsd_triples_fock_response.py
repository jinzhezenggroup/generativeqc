"""Complete bounded same-space unrestricted triples Fock resolvent response.

All vector and paired contractions use the borrowed CPU/CUDA executor. Host
work is limited to admission, shared operand binding, compensated scatter and
qualification. This is a covariant canonical-reference Hamiltonian cotangent,
not a corrected physical/orbital/nuclear response owner.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from generativeqc_compiler.cc.uccsd_layout import BLOCK_NAMES
from generativeqc_compiler.cc.uccsd_triples import channel_shape
from generativeqc_compiler.cc.uccsd_triples_fock import (
    build_axis_moment_program,
    build_tile_resolvent_program,
)
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.evidence import canonical_hash
from generativeqc_compiler.common.resources import MAX_BYTES, checked_bytes
from generativeqc_compiler.tensor.cpu import emit_cpu
from generativeqc_compiler.tensor.cuda_plan import plan_cuda

from .native_tensor_cuda import CudaCCTensorExecutor
from .uccsd_lambda import _freeze
from .uccsd_t import PreparedUCCSDT, _axis_variants
from .uccsd_triples_response import (
    PreparedUCCSDTriplesResponse,
    UCCSDTriplesResponseResult,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from generativeqc_compiler.tensor import Program
    from typing_extensions import Self


_FOCK_NAMES = ("foo_aa", "foo_bb", "fvv_aa", "fvv_bb")


@dataclass(frozen=True)
class UCCSDTriplesFockOptions:
    """Aggregate logical numeric and complete paired/recomputation work limits.

    Tile requests start with the borrowed response owner's admitted shape and
    may shrink only for numeric/executor capacity. Pair and total-work bounds
    remain explicit; a smaller resident page cannot evade either bound.
    """

    max_bytes: int = 512 << 20
    max_pairs: int = 1_000_000
    max_total_work: int = 1_000_000_000_000
    diagonal_tolerance: float = 1e-11

    def __post_init__(self) -> None:
        for name in ("max_bytes", "max_pairs", "max_total_work"):
            if not checked_bytes(getattr(self, name), name):
                raise ValueError(f"{name} must be positive")
        if (
            isinstance(self.diagonal_tolerance, bool)
            or not math.isfinite(self.diagonal_tolerance)
            or not 0 < self.diagonal_tolerance <= 1e-10
        ):
            raise ValueError("diagonal_tolerance must be positive and at most 1e-10")


@dataclass(frozen=True)
class UCCSDTriplesFockResponseResult:
    """Immutable complete source bars plus full oo/vv denominator moments.

    The four moments replace the diagonal-only eps pullback. Failed endpoints
    expose neither prefix correction energies/bars nor partial Fock moments.
    A detached record is not a substitute for a fresh physical consumer replay.
    """

    triples_response: UCCSDTriplesResponseResult
    fock_moments: Mapping[str, np.ndarray]
    response_identity: str | None
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        correction = self.triples_response
        if not isinstance(correction, UCCSDTriplesResponseResult):
            raise TypeError("triples Fock response requires a triples response record")
        if not correction.converged:
            if self.fock_moments or self.response_identity is not None:
                raise ValueError("failed Fock response cannot publish partial moments")
        else:
            if set(self.fock_moments) != set(_FOCK_NAMES):
                raise ValueError("complete Fock response requires all four moments")
            if (
                not isinstance(self.response_identity, str)
                or not self.response_identity
            ):
                raise ValueError("complete Fock response requires an identity")
            for name, values in self.fock_moments.items():
                domain, spin = name[1], name[-1]
                size = correction.cotangents[f"eps_{domain}_{spin}"].size
                if (
                    type(values) is not np.ndarray
                    or values.dtype != np.float64
                    or values.shape != (size, size)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(
                        "Fock moments require finite spin-owned FP64 matrices"
                    )
        owned = replace(
            correction,
            energy=replace(
                correction.energy,
                channel_energies=_freeze(correction.energy.channel_energies),
                provenance=_freeze(correction.energy.provenance),
            ),
        )
        object.__setattr__(self, "triples_response", owned)
        object.__setattr__(
            self,
            "fock_moments",
            MappingProxyType(
                {name: immutable(values) for name, values in self.fock_moments.items()}
            ),
        )
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    @property
    def converged(self) -> bool:
        """Whether all source bars and the complete paired Fock grid succeeded."""
        return self.triples_response.converged

    def hamiltonian_cotangents(self) -> Mapping[str, np.ndarray]:
        """Replace the diagonal denominator chain by complete same-space moments.

        Non-denominator amplitude/Fock/integral bars keep their existing spin
        ownership. This is not normal ordering, orbital relaxation or a force.
        """
        if not self.converged:
            raise ValueError("failed triples Fock response has no Hamiltonian bars")
        weights = {
            name: values.copy()
            for name, values in self.triples_response.cotangents.items()
            if name not in BLOCK_NAMES and not name.startswith("eps_")
        }
        with np.errstate(over="raise", invalid="raise"):
            for name, moment in self.fock_moments.items():
                if name not in weights and not np.any(moment):
                    continue
                if name not in weights or weights[name].shape != moment.shape:
                    raise ValueError(
                        "triples Fock moment populations differ from parent"
                    )
                weights[name] += moment
                if not np.isfinite(weights[name]).all():
                    raise FloatingPointError("nonfinite complete triples Fock pullback")
        return MappingProxyType(
            {name: immutable(values) for name, values in weights.items()}
        )


@dataclass(frozen=True)
class _Bound:
    native_bytes: int
    input_bytes: int
    scalar_work: int


class PreparedUCCSDTriplesFockResponse:
    """Borrow a live triples owner and admit the entire six-axis paired grid.

    Five common coordinates are contracted in bounded tiles. The remaining
    axis uses triangular page pairs and two resident X/Y pages. Right pages
    are recomputed, not retained for the whole axis; that work is admitted in
    advance. Closing this owner closes none of its borrowed resources.
    """

    def __init__(
        self,
        triples: PreparedUCCSDTriplesResponse,
        *,
        options: UCCSDTriplesFockOptions | None = None,
    ) -> None:
        started = time.perf_counter()
        if not isinstance(triples, PreparedUCCSDTriplesResponse):
            raise TypeError("Fock response requires a live triples-response owner")
        triples._check_open()
        self.triples, self.prepared, self.layout = (
            triples,
            triples.prepared,
            triples.layout,
        )
        self.options = UCCSDTriplesFockOptions() if options is None else options
        if not isinstance(self.options, UCCSDTriplesFockOptions):
            raise TypeError("Fock response requires UCCSDTriplesFockOptions")
        self.state_identity = triples.state_identity
        source_bytes = sum(values.nbytes for values in triples.sources.values())
        matrix_bytes = 8 * sum(
            getattr(self.layout, domain + spin) ** 2 for domain in "ov" for spin in "ab"
        )
        self.base_bytes = checked_bytes(
            triples.peak_bytes + 4 * source_bytes + 6 * matrix_bytes,
            "triples Fock persistent and publication bytes",
        )
        if self.base_bytes > self.options.max_bytes:
            raise MemoryError("triples Fock persistent numeric budget exceeded")
        self._closed = False
        self._admit_grid()
        triples._check_open()
        self.binding_identity = canonical_hash(
            {
                "state": self.state_identity,
                "vectors": tuple(
                    (key, tile.program.logical_hash)
                    for key, tile in self.vector_templates.items()
                ),
                "moments": tuple(
                    (key, program.logical_hash)
                    for key, program in self.moment_templates.items()
                ),
                "backend": self.prepared.executor.backend,
                "contract": "full spin-owned oo/vv separable resolvent, no internal gaps",
            }
        )
        self._binding = self._current_binding()
        self.preparation_seconds = time.perf_counter() - started

    def _current_binding(self) -> tuple:
        return (
            self.triples,
            self.prepared,
            self.layout,
            self.options,
            self.state_identity,
            self.tile_shape,
            self.vector_templates,
            self.moment_templates,
            self.binding_identity,
        )

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("triples Fock response is closed")
        if any(
            actual is not expected
            for actual, expected in zip(
                self._current_binding(), self._binding, strict=True
            )
        ):
            raise RuntimeError("triples Fock admitted configuration changed")
        self.triples._check_open()

    def _bound(self, program: Program, retained_bytes: int) -> tuple[_Bound, int]:
        executor = self.prepared.executor
        _, resources = emit_cpu(
            program,
            max_bytes=MAX_BYTES,
            max_work=MAX_BYTES,
            max_nodes=getattr(executor, "max_nodes", 16384),
        )
        if isinstance(executor, CudaCCTensorExecutor):
            try:
                native = plan_cuda(
                    program, executor.compiler.target, max_bytes=executor.max_bytes
                ).peak_bytes
            except ValueError as error:
                if "infeasible tensor byte budget" not in str(error):
                    raise
                raise MemoryError(
                    "triples Fock CUDA page exceeds executor capacity"
                ) from error
        else:
            native = resources["required_bytes"]
            if (
                native > executor.max_bytes
                or resources["scalar_work"] > executor.max_work
            ):
                raise MemoryError("triples Fock CPU page exceeds executor capacity")
        input_bytes = 8 * sum(
            node.spec.size for node in program.live_nodes if node.op == "input"
        )
        peak = checked_bytes(
            self.base_bytes + retained_bytes + native + 2 * input_bytes,
            "triples Fock native and two-page numeric bytes",
        )
        if peak > self.options.max_bytes:
            raise MemoryError("triples Fock page exceeds aggregate numeric budget")
        return _Bound(native, input_bytes, resources["scalar_work"]), peak

    def _grid_counts(self, limit: tuple[int, ...]) -> tuple[dict, dict]:
        vectors, moments = {}, {}
        pair_count = 0
        for channel in self.triples.channels:
            full_shape = channel_shape(self.layout, channel)
            variants = tuple(
                _axis_variants(size, bound)
                for size, bound in zip(full_shape, limit, strict=True)
            )
            for axis in range(6):
                page_count = sum(count for _, count in variants[axis])
                common_count = math.prod(
                    sum(count for _, count in options)
                    for position, options in enumerate(variants)
                    if position != axis
                )
                pair_count = checked_bytes(
                    pair_count + common_count * page_count * (page_count + 1) // 2,
                    "triples Fock complete paired-grid count",
                )
                if pair_count > self.options.max_pairs:
                    raise MemoryError(
                        "triples Fock complete pair-count budget exceeded"
                    )
                common_variants = tuple(
                    ((0, 1),) if position == axis else options
                    for position, options in enumerate(variants)
                )
                for common in product(*common_variants):
                    shape = tuple(length for length, _ in common)
                    multiplicity = math.prod(count for _, count in common)
                    earlier_pages = 0
                    for left_position, (left_length, left_count) in enumerate(
                        variants[axis]
                    ):
                        left_shape = shape[:axis] + (left_length,) + shape[axis + 1 :]
                        vector_count = (
                            left_count * (earlier_pages + 1)
                            + left_count * (left_count - 1) // 2
                        )
                        key = channel, left_shape
                        vectors[key] = checked_bytes(
                            vectors.get(key, 0) + multiplicity * vector_count,
                            "triples Fock complete vector recomputation count",
                        )
                        for right_position in range(left_position, len(variants[axis])):
                            right_length, right_count = variants[axis][right_position]
                            right_shape = (
                                shape[:axis] + (right_length,) + shape[axis + 1 :]
                            )
                            count = (
                                left_count * (left_count + 1) // 2
                                if right_position == left_position
                                else left_count * right_count
                            )
                            moments[channel, axis, left_shape, right_shape] = (
                                checked_bytes(
                                    multiplicity * count,
                                    "triples Fock moment shape multiplicity",
                                )
                            )
                        earlier_pages += left_count
        return vectors, moments

    def _admit_grid(self) -> None:
        limit, fallback = self.triples.tile_shape, 0
        while True:
            vector_counts, moment_counts = self._grid_counts(limit)
            volume = max((math.prod(shape) for _, shape in vector_counts), default=0)
            retained = checked_bytes(
                32 * volume + 32 * (6 * self.prepared.reference.nmo + sum(limit)),
                "triples Fock retained page and binding scratch bytes",
            )
            vectors, moments, bounds = {}, {}, {}
            peak = self.base_bytes
            try:
                for key in vector_counts:
                    tile = build_tile_resolvent_program(self.layout, *key)
                    bound, page_peak = self._bound(tile.program, retained)
                    vectors[key], bounds[tile.program.logical_hash] = tile, bound
                    peak = max(peak, page_peak)
                for key in moment_counts:
                    channel, axis, left_shape, right_shape = key
                    program = build_axis_moment_program(
                        self.layout, channel, left_shape, right_shape, axis
                    )
                    bound, page_peak = self._bound(program, retained)
                    moments[key], bounds[program.logical_hash] = program, bound
                    peak = max(peak, page_peak)
            except MemoryError:
                if max(limit) == 1:
                    raise MemoryError(
                        "minimum triples Fock pages cannot meet numeric capacity"
                    ) from None
                selected = max(range(6), key=lambda position: limit[position])
                limit = tuple(
                    max(1, size // 2) if position == selected else size
                    for position, size in enumerate(limit)
                )
                fallback += 1
                continue
            vector_work = sum(
                count * bounds[vectors[key].program.logical_hash].scalar_work
                for key, count in vector_counts.items()
            )
            moment_work = sum(
                count * bounds[moments[key].logical_hash].scalar_work
                for key, count in moment_counts.items()
            )
            work = checked_bytes(
                self.triples.total_scalar_work + vector_work + moment_work,
                "triples Fock response plus complete paired/recomputed scalar work",
            )
            if work > self.options.max_total_work:
                raise MemoryError("triples Fock complete semantic-work budget exceeded")
            self.tile_shape, self.fallback_steps = limit, fallback
            self.vector_templates, self.moment_templates = (
                MappingProxyType(vectors),
                MappingProxyType(moments),
            )
            self._bounds = MappingProxyType(bounds)
            self.vector_count, self.pair_count = (
                sum(vector_counts.values()),
                sum(moment_counts.values()),
            )
            self.vector_scalar_work, self.moment_scalar_work = vector_work, moment_work
            self.total_scalar_work, self.peak_bytes = work, peak
            return

    def _execute(self, program: Program, feeds: Mapping, counter: dict) -> Mapping:
        self._check_open()
        bound = self._bounds[program.logical_hash]
        if counter["work"] + bound.scalar_work > self.options.max_total_work:
            raise MemoryError("triples Fock execution semantic-work budget exhausted")
        outputs = self.prepared.executor.execute(program, feeds)
        counter["work"] += bound.scalar_work
        counter["native_calls"] += 1
        if outputs.keys() != program.outputs.keys():
            raise ValueError("triples Fock native outputs are incomplete")
        for name, values in outputs.items():
            spec = program.outputs[name].spec
            expected_shape = tuple(index.space.size for index in spec.indices)
            if (
                type(values) is not np.ndarray
                or values.dtype != np.float64
                or values.shape != expected_shape
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    "triples Fock native output violated its FP64 shape contract"
                )
        self._check_open()
        return outputs

    def _vectors(self, channel: str, coordinates: tuple, counter: dict) -> Mapping:
        tile = self.vector_templates[
            channel, tuple(len(values) for values in coordinates)
        ]
        return self._execute(
            tile.program,
            PreparedUCCSDT._bind(self.triples, tile, coordinates),
            counter,
        )

    @staticmethod
    def _scatter(
        target: np.ndarray, errors: np.ndarray, selected: tuple, values: np.ndarray
    ) -> None:
        previous = target[selected]
        correction = values - errors[selected]
        updated = previous + correction
        errors[selected] = (updated - previous) - correction
        target[selected] = updated
        if not np.isfinite(updated).all() or not np.isfinite(errors[selected]).all():
            raise FloatingPointError("nonfinite triples Fock compensated accumulation")

    def response(self) -> UCCSDTriplesFockResponseResult:
        """Replay source bars and every admitted pair, publishing only complete response."""
        self._check_open()
        started = time.perf_counter()
        correction = self.triples.response()
        counter = {
            "work": self.triples.total_scalar_work,
            "native_calls": 0,
            "pairs": 0,
            "vectors": 0,
        }
        moments = {
            name: np.zeros((getattr(self.layout, name[1] + name[-1]),) * 2)
            for name in _FOCK_NAMES
        }
        compensation = {name: np.zeros_like(values) for name, values in moments.items()}
        if correction.converged:
            try:
                with np.errstate(over="raise", invalid="raise"):
                    for channel in self.triples.channels:
                        full_shape = channel_shape(self.layout, channel)
                        ranges = tuple(
                            range(0, size, bound)
                            for size, bound in zip(
                                full_shape, self.tile_shape, strict=True
                            )
                        )
                        for axis in range(6):
                            common = tuple(
                                range(1) if position == axis else options
                                for position, options in enumerate(ranges)
                            )
                            for starts in product(*common):
                                coordinates = tuple(
                                    tuple(range(start, min(start + bound, size)))
                                    for start, bound, size in zip(
                                        starts, self.tile_shape, full_shape, strict=True
                                    )
                                )
                                for left_position, left_start in enumerate(
                                    ranges[axis]
                                ):
                                    left_coordinates = (
                                        coordinates[:axis]
                                        + (
                                            tuple(
                                                range(
                                                    left_start,
                                                    min(
                                                        left_start
                                                        + self.tile_shape[axis],
                                                        full_shape[axis],
                                                    ),
                                                )
                                            ),
                                        )
                                        + coordinates[axis + 1 :]
                                    )
                                    left = self._vectors(
                                        channel, left_coordinates, counter
                                    )
                                    counter["vectors"] += 1
                                    for right_start in ranges[axis][left_position:]:
                                        right_coordinates = (
                                            coordinates[:axis]
                                            + (
                                                tuple(
                                                    range(
                                                        right_start,
                                                        min(
                                                            right_start
                                                            + self.tile_shape[axis],
                                                            full_shape[axis],
                                                        ),
                                                    )
                                                ),
                                            )
                                            + coordinates[axis + 1 :]
                                        )
                                        if right_start == left_start:
                                            right = left
                                        else:
                                            right = self._vectors(
                                                channel, right_coordinates, counter
                                            )
                                            counter["vectors"] += 1
                                        key = (
                                            channel,
                                            axis,
                                            left["x"].shape,
                                            right["x"].shape,
                                        )
                                        program = self.moment_templates[key]
                                        outputs = self._execute(
                                            program,
                                            {
                                                "x_left": left["x"],
                                                "y_left": left["y"],
                                                "x_right": right["x"],
                                                "y_right": right["y"],
                                            },
                                            counter,
                                        )
                                        name, values = next(iter(outputs.items()))
                                        selected = np.ix_(
                                            left_coordinates[axis],
                                            right_coordinates[axis],
                                        )
                                        self._scatter(
                                            moments[name],
                                            compensation[name],
                                            selected,
                                            values,
                                        )
                                        if right_start != left_start:
                                            self._scatter(
                                                moments[name],
                                                compensation[name],
                                                np.ix_(
                                                    right_coordinates[axis],
                                                    left_coordinates[axis],
                                                ),
                                                values.T,
                                            )
                                        counter["pairs"] += 1
                                        del right, outputs
                                    del left
                for name, values in moments.items():
                    expected = correction.cotangents[f"eps_{name[1]}_{name[-1]}"]
                    if (
                        float(np.max(np.abs(np.diag(values) - expected), initial=0))
                        > self.options.diagonal_tolerance
                    ):
                        raise FloatingPointError(
                            "triples full Fock diagonal differs from independently differentiated eps bars"
                        )
                if (counter["vectors"], counter["pairs"], counter["work"]) != (
                    self.vector_count,
                    self.pair_count,
                    self.total_scalar_work,
                ):
                    raise RuntimeError(
                        "triples Fock execution did not cover the admitted complete paired grid"
                    )
            except (
                RuntimeError,
                ValueError,
                FloatingPointError,
                OverflowError,
            ) as error:
                energy = replace(
                    correction.energy,
                    status="failed",
                    reason=str(error),
                    triples_energy=None,
                    channel_energies={},
                    correlation_energy=None,
                    total_energy=None,
                )
                correction = UCCSDTriplesResponseResult(
                    energy, {}, None, None, {"scope": "failed complete Fock response"}
                )
        self._check_open()
        identity = None
        if correction.converged:
            identity = canonical_hash(
                {
                    "binding": self.binding_identity,
                    "correction": correction.response_identity,
                    "moments": {
                        name: sha256(
                            values.astype("<f8", copy=False).tobytes()
                        ).hexdigest()
                        for name, values in moments.items()
                    },
                }
            )
        else:
            moments = {}
        return UCCSDTriplesFockResponseResult(
            correction,
            moments,
            identity,
            {
                "state_identity": self.state_identity,
                "binding_identity": self.binding_identity,
                "tensor_backend": self.prepared.executor.backend,
                "numeric_bound_bytes": self.peak_bytes,
                "tile_shape": self.tile_shape,
                "fallback_steps": self.fallback_steps,
                "planned_pairs": self.pair_count,
                "planned_vectors": self.vector_count,
                "completed_pairs": counter["pairs"],
                "completed_vectors": counter["vectors"],
                "planned_semantic_scalar_work": self.total_scalar_work,
                "executed_semantic_scalar_work": counter["work"],
                "fock_native_calls": counter["native_calls"],
                "options": asdict(self.options),
                "fock_response_endpoint_seconds": self.preparation_seconds
                + time.perf_counter()
                - started,
                "denominator_chain": "complete spin-owned oo/vv resolvent moments replacing eps diagonal",
                "scope": "covariant canonical-reference triples Hamiltonian response, not orbital/nuclear response",
            },
        )

    def close(self) -> None:
        """Close this consumer without closing the borrowed triples/parent/executor."""
        self._closed = True

    def __enter__(self) -> Self:
        self._check_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
