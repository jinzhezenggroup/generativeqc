"""Execution facts attached to existing IR operations, never a second tensor IR.

The scientific owner supplies operation identity and admitted precision schedules.
Providers consume these facts to implement that operation; they cannot qualify a
new approximation. This module is pure metadata and imports no device runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import prod
from typing import Literal

from .precision import CastBoundary, ExecutionPrecisionSchedule, PrecisionDirective
from .provenance import canonical_hash
from .resources import checked_bytes
from .schedule import ScheduleTopology

Determinism = Literal["unspecified", "reproducible", "exact-order"]
DETERMINISM = ("unspecified", "reproducible", "exact-order")


def digest(value: str, label: str) -> None:
    """Validate references to existing scientific, schedule and artifact owners."""
    if (
        type(value) is not str
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{label} must be a SHA-256 digest")


def name(value: str, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class OperandLayout:
    """An IR operand's physical view and declared symmetry/alias ownership.

    Modes are consumer-owned normalized axis ordinals, including repeated labels
    for diagonals. Strides are in elements; None denotes a virtual expression
    requiring generated evaluation/materialization. Nonnegative affine strides
    include padded and broadcast input views. Negative-stride adapters must
    explicitly materialize before using this first contract revision.
    """

    operand: str
    modes: tuple[int, ...]
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None
    access: Literal["read", "write", "read-write"] = "read"
    alignment: int = 1
    triangle: Literal["full", "upper", "lower"] = "full"
    alias_group: str | None = None

    def __post_init__(self) -> None:
        name(self.operand, "operand")
        for label in ("modes", "shape", "strides"):
            value = getattr(self, label)
            if value is not None:
                object.__setattr__(self, label, tuple(value))
        if len(self.modes) != len(self.shape) or (
            self.strides is not None and len(self.strides) != len(self.shape)
        ):
            raise ValueError("operand modes, shape and strides must have the same rank")
        for value in (*self.modes, *self.shape, *(self.strides or ())):
            checked_bytes(value, "operand axis/extent/stride")
        checked_bytes(prod(self.shape), "operand elements")
        if self.strides is not None and all(self.shape):
            checked_bytes(
                sum((n - 1) * s for n, s in zip(self.shape, self.strides, strict=True)),
                "operand maximum offset",
            )
        if self.access not in ("read", "write", "read-write") or self.triangle not in (
            "full",
            "upper",
            "lower",
        ):
            raise ValueError("invalid operand access/triangle ownership")
        checked_bytes(self.alignment, "operand alignment")
        if not self.alignment or self.alignment & (self.alignment - 1):
            raise ValueError("operand alignment must be a positive power of two")
        if self.alias_group is not None:
            name(self.alias_group, "alias group")

    def to_payload(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoweringPrecision:
    """One admitted variant, referencing the shared precision schedule directly.

    The region names an existing schedule entry. Casts and audit/refinement are
    obligations of the complete candidate, even if the implementation fuses them.
    Qualification is evidence identity supplied by the scientific owner; this
    record does not independently prove numerical acceptance.
    """

    schedule: ExecutionPrecisionSchedule
    region: str
    input_dtypes: tuple[str, ...]
    publication_dtype: str
    casts: tuple[CastBoundary, ...] = ()
    refinement: str | None = None
    audit: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, ExecutionPrecisionSchedule):
            raise TypeError("lowering precision requires ExecutionPrecisionSchedule")
        if self.region not in dict(self.schedule.regions):
            raise ValueError("lowering precision region is absent from the schedule")
        object.__setattr__(self, "input_dtypes", tuple(self.input_dtypes))
        object.__setattr__(self, "casts", tuple(self.casts))
        if any(
            dtype not in ("float32", "float64", "int64") for dtype in self.input_dtypes
        ) or self.publication_dtype not in ("float32", "float64"):
            raise ValueError("unsupported lowering storage/publication dtype")
        if any(not isinstance(cast, CastBoundary) for cast in self.casts):
            raise TypeError("lowering casts require shared CastBoundary records")
        if len({cast.name for cast in self.casts}) != len(self.casts):
            raise ValueError("lowering cast boundaries must have unique names")
        for label in ("refinement", "audit"):
            if getattr(self, label) is not None:
                name(getattr(self, label), label)

    @property
    def directive(self) -> PrecisionDirective:
        return dict(self.schedule.regions)[self.region]

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict:
        return {
            "schedule": self.schedule.to_payload(),
            "schedule_identity": self.schedule.identity,
            "region": self.region,
            "input_dtypes": self.input_dtypes,
            "publication_dtype": self.publication_dtype,
            "casts": [cast.to_payload() for cast in self.casts],
            "refinement": self.refinement,
            "audit": self.audit,
        }


@dataclass(frozen=True, slots=True)
class LoweringConstraints:
    """Preparation limits for one binding, excluding caller-owned tensor storage.

    Limits are simultaneous additional bytes, not separate allowances that can
    hide a sum over the budget. Unknown limits are explicit None. No flag here
    authorizes changing backend, science or numerical admission on failure.
    """

    workspace_bytes: int | None = None
    provider_bytes: int | None = None
    additional_device_bytes: int | None = None
    host_bytes: int | None = None
    determinism: Determinism = "unspecified"
    capture_required: bool = False
    maximum_candidates: int = 256

    def __post_init__(self) -> None:
        for label in (
            "workspace_bytes",
            "provider_bytes",
            "additional_device_bytes",
            "host_bytes",
            "maximum_candidates",
        ):
            if getattr(self, label) is not None:
                checked_bytes(getattr(self, label), label)
        if not self.maximum_candidates:
            raise ValueError("maximum_candidates must be positive")
        if self.determinism not in DETERMINISM:
            raise ValueError("unknown determinism contract")
        if type(self.capture_required) is not bool:
            raise TypeError("capture_required must be boolean")

    def to_payload(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateExecution:
    """Concrete precision/layout/fusion/algorithm facts for a provider offer.

    Extra temporary and cache bytes exclude LoweringCandidate workspace/provider
    bytes; the sum is charged conservatively as simultaneously live. Host retained
    plans and JIT metadata belong in host_bytes. Providers must supply version and
    toolkit/compiler provenance before an executable binding is cached.
    """

    precision: LoweringPrecision
    algorithm: str
    layouts: tuple[OperandLayout, ...]
    topology: ScheduleTopology = field(default_factory=ScheduleTopology)
    determinism: Determinism = "unspecified"
    capture_safe: bool = False
    temporary_bytes: int = 0
    cache_bytes: int = 0
    host_bytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.precision, LoweringPrecision) or not isinstance(
            self.topology, ScheduleTopology
        ):
            raise TypeError("candidate requires typed precision and schedule topology")
        name(self.algorithm, "algorithm")
        object.__setattr__(self, "layouts", tuple(self.layouts))
        if any(not isinstance(layout, OperandLayout) for layout in self.layouts):
            raise TypeError("candidate layouts require OperandLayout")
        if len({layout.operand for layout in self.layouts}) != len(self.layouts):
            raise ValueError("candidate operand layouts must be unique")
        for label in ("temporary_bytes", "cache_bytes", "host_bytes"):
            checked_bytes(getattr(self, label), label)
        if self.determinism not in DETERMINISM:
            raise ValueError("unknown candidate determinism")
        if type(self.capture_safe) is not bool:
            raise TypeError("capture_safe must be boolean")

    def to_payload(self) -> dict:
        return {
            **asdict(self),
            "precision": self.precision.to_payload(),
            "topology": self.topology.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class LoweringCost:
    """Complete per-operation phase evidence, with unknown distinct from zero.

    Costs are non-overlapping nanoseconds: fused cast/pack work belongs to the
    kernel phase and has zero standalone time, but its traffic is still counted.
    A retained source names measured or estimated evidence; these estimates do
    not promote a method or establish a complete scientific endpoint speedup.
    """

    source: str
    kind: Literal["measured", "estimated"]
    prepare_ns: int | None = None
    kernel_ns: int | None = None
    cast_ns: int | None = None
    pack_ns: int | None = None
    refinement_ns: int | None = None
    audit_ns: int | None = None
    fallback_ns: int | None = None
    cast_bytes: int = 0
    pack_bytes: int = 0
    refinement_bytes: int = 0
    audit_bytes: int = 0
    launches: int = 0

    def __post_init__(self) -> None:
        name(self.source, "cost evidence source")
        if self.kind not in ("measured", "estimated"):
            raise ValueError("unknown lowering cost evidence kind")
        for label, value in asdict(self).items():
            if label not in ("source", "kind") and (
                value is not None or not label.endswith("_ns")
            ):
                checked_bytes(value, label)

    @property
    def replay_ns(self) -> int | None:
        phases = (
            self.kernel_ns,
            self.cast_ns,
            self.pack_ns,
            self.refinement_ns,
            self.audit_ns,
            self.fallback_ns,
        )
        total = 0
        for phase in phases:
            if phase is None:
                return None
            total += phase
        return checked_bytes(total, "complete replay nanoseconds")

    def to_payload(self) -> dict:
        return {**asdict(self), "replay_ns": self.replay_ns}
