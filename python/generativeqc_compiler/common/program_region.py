"""Measured implementation candidates for contiguous ProgramIR regions.

This module is a compiler planning boundary, not a scientific fusion engine.
ProgramIR keeps opaque provider calls by default. A region may be replaced only
when every covered provider is explicitly proven PURE through the shared effect
contract and an owning subsystem supplies a concrete executable provider
identity.

The shared ScheduleContract remains the sole profitability/promotion vocabulary.
This module only derives whole-region boundaries, binds alternate implementations,
and applies a selected candidate without inventing a second tuner or cache.
"""

from __future__ import annotations

import collections.abc
import typing
from dataclasses import dataclass

from .gpu_profitability import ENDPOINT_NOISE_FRACTION
from .liveness import EffectKind
from .program import PlanCall, ProgramIR
from .provenance import canonical_hash
from .schedule import ScheduleContract, select_measured_schedule_contract

PROGRAM_REGION_SCHEMA = "generativeqc.compiler.program-region.v1"
PROGRAM_REGION_CANDIDATE_SCHEMA = "generativeqc.compiler.program-region-candidate.v1"


def _text(value: typing.Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _digest(value: typing.Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _unique_names(values: typing.Iterable[str], label: str) -> tuple[str, ...]:
    materialized = tuple(_text(value, label) for value in values)
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"duplicate {label}")
    return materialized


def _normalize_effects(
    effects: collections.abc.Mapping[str, EffectKind],
) -> dict[str, EffectKind]:
    if not isinstance(effects, collections.abc.Mapping):
        raise TypeError("ProgramIR region effects must be a provider mapping")
    normalized: dict[str, EffectKind] = {}
    for provider, effect in effects.items():
        provider_name = _text(provider, "effect provider")
        if not isinstance(effect, EffectKind):
            raise TypeError("ProgramIR region effects must use EffectKind")
        normalized[provider_name] = effect
    return normalized


@dataclass(frozen=True, slots=True)
class ProgramRegion:
    """One contiguous, proven-pure ProgramIR subgraph and its external boundary."""

    name: str
    program_identity: str
    call_names: tuple[str, ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    internal_buffers: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.name, "region name")
        _digest(self.program_identity, "program identity")
        object.__setattr__(
            self, "call_names", _unique_names(self.call_names, "region call name")
        )
        object.__setattr__(self, "reads", _unique_names(self.reads, "region read"))
        object.__setattr__(self, "writes", _unique_names(self.writes, "region write"))
        object.__setattr__(
            self,
            "internal_buffers",
            _unique_names(self.internal_buffers, "region internal buffer"),
        )
        if len(self.call_names) < 2:
            raise ValueError("ProgramIR fusion region requires at least two calls")
        if not self.writes:
            raise ValueError("ProgramIR fusion region requires an external result")
        names = set(self.reads)
        if names.intersection(self.writes):
            raise ValueError("ProgramIR region reads and writes must be disjoint")
        if names.intersection(self.internal_buffers):
            raise ValueError("ProgramIR region reads cannot be internal buffers")
        if set(self.writes).intersection(self.internal_buffers):
            raise ValueError("ProgramIR region writes cannot be internal buffers")

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    @property
    def consumer(self) -> str:
        """Stable ScheduleContract consumer key for this exact region."""

        return f"program-region:{self.identity}"

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schema": PROGRAM_REGION_SCHEMA,
            "name": self.name,
            "program_identity": self.program_identity,
            "call_names": list(self.call_names),
            "reads": list(self.reads),
            "writes": list(self.writes),
            "internal_buffers": list(self.internal_buffers),
        }


def _region_boundary(
    program: ProgramIR, start: int, end: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Derive the exact ordered dataflow boundary from the source graph."""
    calls = program.calls[start : end + 1]
    boundary_reads: list[str] = []
    produced_so_far: set[str] = set()
    produced_order: list[str] = []
    for call in calls:
        for read in call.reads:
            if read not in produced_so_far and read not in boundary_reads:
                boundary_reads.append(read)
        for write in call.writes:
            produced_so_far.add(write)
            produced_order.append(write)

    later_reads = {read for call in program.calls[end + 1 :] for read in call.reads}
    externally_needed = later_reads.union(program.outputs)
    boundary_writes = tuple(
        write for write in produced_order if write in externally_needed
    )
    internal_buffers = tuple(
        write for write in produced_order if write not in externally_needed
    )

    return tuple(boundary_reads), boundary_writes, internal_buffers


def derive_program_region(
    program: ProgramIR,
    *,
    name: str,
    start_call: str,
    end_call: str,
    effects: collections.abc.Mapping[str, EffectKind],
) -> ProgramRegion:
    """Derive a fusible contiguous region and fail closed on opaque/effectful calls.

    The region boundary is structural. Reads are values entering from outside the
    region. Writes are region-produced values observed by later calls or exported
    by the ProgramIR. Other produced values are internal materializations that a
    qualified fused implementation may eliminate.
    """

    if not isinstance(program, ProgramIR):
        raise TypeError("ProgramIR region derivation requires ProgramIR")
    name = _text(name, "region name")
    start_call = _text(start_call, "start call")
    end_call = _text(end_call, "end call")
    normalized = _normalize_effects(effects)

    call_names = tuple(call.name for call in program.calls)
    if start_call not in call_names or end_call not in call_names:
        raise ValueError("ProgramIR region references an unknown call")
    start = call_names.index(start_call)
    end = call_names.index(end_call)
    if start > end:
        raise ValueError("ProgramIR region start must not follow its end")
    if start == end:
        raise ValueError("ProgramIR fusion region requires at least two calls")

    calls = program.calls[start : end + 1]
    for call in calls:
        effect = normalized.get(call.provider, EffectKind.OPAQUE)
        if effect is not EffectKind.PURE:
            raise ValueError(
                f"{call.name}: provider {call.provider!r} is not proven pure"
            )

    boundary_reads, boundary_writes, internal_buffers = _region_boundary(
        program, start, end
    )

    return ProgramRegion(
        name=name,
        program_identity=program.identity,
        call_names=tuple(call.name for call in calls),
        reads=tuple(boundary_reads),
        writes=boundary_writes,
        internal_buffers=internal_buffers,
    )


@dataclass(frozen=True, slots=True)
class ProgramRegionCandidate:
    """One original or alternate executable implementation of a ProgramRegion."""

    name: str
    region: ProgramRegion
    schedule: ScheduleContract
    backend: str
    provider: str | None = None
    implementation_identity: str | None = None

    def __post_init__(self) -> None:
        _text(self.name, "region candidate name")
        _text(self.backend, "region candidate backend")
        if not isinstance(self.region, ProgramRegion):
            raise TypeError("region candidate requires ProgramRegion")
        if not isinstance(self.schedule, ScheduleContract):
            raise TypeError("region candidate requires ScheduleContract")
        if self.schedule.consumer != self.region.consumer:
            raise ValueError("region candidate schedule is bound to another region")
        if self.schedule.fallback:
            if self.provider is not None or self.implementation_identity is not None:
                raise ValueError(
                    "fallback region candidate must preserve the original calls"
                )
        else:
            _text(self.provider, "region candidate provider")
            _text(
                self.implementation_identity,
                "region candidate implementation identity",
            )

    @property
    def replacement_identity(self) -> str:
        if self.schedule.fallback:
            raise ValueError("fallback region candidate has no replacement identity")
        return canonical_hash(
            {
                "schema": PROGRAM_REGION_CANDIDATE_SCHEMA,
                "region": self.region.identity,
                "candidate": self.name,
                "backend": self.backend,
                "provider": self.provider,
                "implementation": self.implementation_identity,
                "schedule": self.schedule.identity,
            }
        )

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schema": PROGRAM_REGION_CANDIDATE_SCHEMA,
            "name": self.name,
            "region": self.region.to_payload(),
            "backend": self.backend,
            "provider": self.provider,
            "implementation_identity": self.implementation_identity,
            "schedule": self.schedule.to_payload(),
        }


def select_program_region_candidate(
    candidates: typing.Iterable[ProgramRegionCandidate],
    *,
    minimum_speedup: float,
    endpoint_noise_fraction: float = ENDPOINT_NOISE_FRACTION,
) -> ProgramRegionCandidate:
    """Select one measured region backend using the shared schedule policy."""

    materialized = tuple(candidates)
    if not materialized:
        raise ValueError("ProgramIR region selection requires candidates")
    if any(
        not isinstance(candidate, ProgramRegionCandidate) for candidate in materialized
    ):
        raise TypeError("ProgramIR region selection requires region candidates")
    if len({candidate.name for candidate in materialized}) != len(materialized):
        raise ValueError("duplicate ProgramIR region candidate name")
    if len({candidate.region.identity for candidate in materialized}) != 1:
        raise ValueError("ProgramIR region candidates must target one region")
    if len({candidate.schedule.identity for candidate in materialized}) != len(
        materialized
    ):
        raise ValueError("duplicate ProgramIR region schedule identity")

    selected = select_measured_schedule_contract(
        tuple(candidate.schedule for candidate in materialized),
        minimum_speedup=minimum_speedup,
        endpoint_noise_fraction=endpoint_noise_fraction,
    )
    for candidate in materialized:
        if candidate.schedule is selected:
            return candidate
    raise AssertionError(
        "shared schedule selector returned an unknown region candidate"
    )


def apply_program_region_candidate(
    program: ProgramIR,
    candidate: ProgramRegionCandidate,
) -> ProgramIR:
    """Replace a qualified region with one executable provider call.

    The original fallback is returned unchanged. An alternate implementation
    must be legal and bound to the exact source ProgramIR identity. Internal
    region buffers are removed; boundary buffers retain their original owners,
    layouts, capacities, and public names.
    """

    if not isinstance(program, ProgramIR):
        raise TypeError("ProgramIR region application requires ProgramIR")
    if not isinstance(candidate, ProgramRegionCandidate):
        raise TypeError("ProgramIR region application requires a region candidate")
    region = candidate.region
    if region.program_identity != program.identity:
        raise ValueError("region candidate is stale for this ProgramIR")
    if candidate.schedule.fallback:
        return program
    if not candidate.schedule.legal:
        raise ValueError("cannot apply an illegal ProgramIR region candidate")

    call_names = tuple(call.name for call in program.calls)
    positions = tuple(call_names.index(name) for name in region.call_names)
    expected = tuple(range(positions[0], positions[0] + len(positions)))
    if positions != expected:
        raise ValueError("ProgramIR region calls are no longer contiguous")

    # ProgramRegion is a public value object: a reconstructed/replaced instance
    # can keep the source digest while changing its ABI or elimination list.
    # ProgramIR's SSA check alone cannot detect omitted or reordered inputs.
    boundary = _region_boundary(program, positions[0], positions[-1])
    if boundary != (region.reads, region.writes, region.internal_buffers):
        raise ValueError("ProgramIR region boundary does not match the source graph")

    outside_names = set(call_names).difference(region.call_names)
    if candidate.name in outside_names:
        raise ValueError("region replacement call name collides with another call")

    first = positions[0]
    last = positions[-1]
    replacement = PlanCall(
        candidate.name,
        typing.cast("str", candidate.provider),
        candidate.replacement_identity,
        region.reads,
        region.writes,
    )
    internal = set(region.internal_buffers)
    return ProgramIR(
        program.name,
        tuple(buffer for buffer in program.buffers if buffer.name not in internal),
        program.inputs,
        (*program.calls[:first], replacement, *program.calls[last + 1 :]),
        program.outputs,
        schema_version=program.schema_version,
    )
