"""Backend-neutral lowering-provider request and candidate contracts.

Scientific IR owns operation semantics.  This module only records how an
already-defined operation may be lowered, including composite generated/library
implementations, resource requirements, numerical mode, and negative evidence.
It performs no device probing, compilation, promotion, or method selection.
"""

from __future__ import annotations

import math
import typing
from dataclasses import asdict, dataclass, field, replace
from itertools import islice
from typing import Literal

from .lowering_contract import (
    DETERMINISM,
    CandidateExecution,
    LoweringConstraints,
    LoweringCost,
    LoweringPrecision,
    OperandLayout,
    digest,
)
from .provenance import canonical_hash
from .resources import checked_bytes
from .specialization import TargetCapabilities

LOWERING_REQUEST_SCHEMA = "generativeqc.compiler.lowering-request.v1"
LOWERING_PROVIDER_SCHEMA = "generativeqc.compiler.lowering-provider.v1"
LOWERING_CANDIDATE_SCHEMA = "generativeqc.compiler.lowering-candidate.v1"
LOWERING_DIAGNOSTICS_SCHEMA = "generativeqc.compiler.lowering-diagnostics.v1"

Scalar = bool | int | float | str


def _name(value: typing.Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _scalar(value: typing.Any, label: str) -> Scalar:
    if type(value) not in (bool, int, float, str) or (
        type(value) is float and not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite immutable JSON scalar")
    return value


def _pairs(
    values: typing.Iterable[tuple[str, Scalar]], label: str
) -> tuple[tuple[str, Scalar], ...]:
    result: list[tuple[str, Scalar]] = []
    names: set[str] = set()
    for pair in values:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"{label} must contain name/value pairs")
        name, value = pair
        _name(name, f"{label} name")
        _scalar(value, f"{label} value")
        if name in names:
            raise ValueError(f"duplicate {label} name {name!r}")
        names.add(name)
        result.append((name, value))
    return tuple(sorted(result))


def _nonnegative_bytes(value: typing.Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


class _TypedRecord:
    """Keep equality consistent with type-sensitive canonical JSON identities."""

    __slots__ = ()

    def to_payload(self) -> dict[str, typing.Any]:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        peer = typing.cast("_TypedRecord", other)
        return canonical_hash(self.to_payload()) == canonical_hash(peer.to_payload())

    def __hash__(self) -> int:
        return hash((type(self), canonical_hash(self.to_payload())))


@dataclass(frozen=True, slots=True, eq=False)
class LoweringRequest(_TypedRecord):
    """One semantic operation and its admitted execution choices on a backend.

    Extended requests carry all precision variants together; a provider cannot
    fork the request identity to advertise a different precision or layout.
    ``scientific_identity`` references the existing IR/equation owner, while
    ``semantic_identity`` excludes backend, physical layout and precision offers.
    Legacy diagnostic requests preserve their v1 payload until adapted.
    """

    consumer: str
    operation: str
    backend: str
    dtype: str
    accumulation_dtype: str
    shape: tuple[int, ...]
    semantics: tuple[tuple[str, Scalar], ...] = ()
    scientific_identity: str | None = field(default=None, kw_only=True)
    operands: tuple[OperandLayout, ...] = field(default=(), kw_only=True)
    input_dtypes: tuple[str, ...] = field(default=(), kw_only=True)
    precisions: tuple[LoweringPrecision, ...] = field(default=(), kw_only=True)
    constraints: LoweringConstraints | None = field(default=None, kw_only=True)
    effects: tuple[tuple[str, Scalar], ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        for label in (
            "consumer",
            "operation",
            "backend",
            "dtype",
            "accumulation_dtype",
        ):
            _name(getattr(self, label), label)
        shape = tuple(self.shape)
        if any(type(extent) is not int or extent < 0 for extent in shape):
            raise ValueError("lowering shape extents must be non-negative integers")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "semantics", _pairs(self.semantics, "semantic"))
        object.__setattr__(self, "effects", _pairs(self.effects, "effect"))
        operands = tuple(self.operands)
        precisions = tuple(self.precisions)
        if any(not isinstance(operand, OperandLayout) for operand in operands):
            raise TypeError("request operands require OperandLayout records")
        if len({operand.operand for operand in operands}) != len(operands):
            raise ValueError("request operands must have unique names")
        extents: dict[int, int] = {}
        for operand in operands:
            for mode, extent in zip(operand.modes, operand.shape, strict=True):
                if mode in extents and extents[mode] != extent:
                    raise ValueError("shared operand modes must have identical extents")
                extents[mode] = extent
        if any(
            not isinstance(precision, LoweringPrecision) for precision in precisions
        ):
            raise TypeError("request precision variants require LoweringPrecision")
        if len({precision.identity for precision in precisions}) != len(precisions):
            raise ValueError("request precision variants must be unique")
        if self.scientific_identity is not None:
            digest(self.scientific_identity, "scientific identity")
        if self.constraints is not None and not isinstance(
            self.constraints, LoweringConstraints
        ):
            raise TypeError("request constraints require LoweringConstraints")
        if precisions and self.scientific_identity is None:
            raise ValueError("joint precision selection requires scientific identity")
        input_dtypes = tuple(self.input_dtypes)
        if not input_dtypes and precisions:
            input_dtypes = (self.dtype,) * len(precisions[0].input_dtypes)
        if any(dtype not in ("float32", "float64", "int64") for dtype in input_dtypes):
            raise ValueError("unsupported request input dtype")
        if operands and len(input_dtypes) != sum(
            operand.access != "write" for operand in operands
        ):
            raise ValueError("request input dtypes must describe every read operand")
        object.__setattr__(self, "input_dtypes", input_dtypes)
        for precision in precisions:
            directive = precision.directive
            if precision.publication_dtype != self.dtype:
                raise ValueError(
                    "precision variant changes requested publication dtype"
                )
            if (
                directive.storage_dtype != self.dtype
                or directive.compute_dtype != self.dtype
                or directive.accumulation_dtype != self.accumulation_dtype
                or precision.input_dtypes != input_dtypes
            ) and directive.qualification is None:
                raise ValueError(
                    "alternative precision requires scientific qualification"
                )
            if len(precision.input_dtypes) != len(input_dtypes):
                raise ValueError(
                    "precision input dtypes must describe every read operand"
                )
            if directive.storage_dtype != precision.publication_dtype and not any(
                cast.source_dtype == directive.storage_dtype
                and cast.target_dtype == precision.publication_dtype
                for cast in precision.casts
            ):
                raise ValueError(
                    "publication dtype conversion requires an explicit cast boundary"
                )
        object.__setattr__(self, "operands", operands)
        object.__setattr__(
            self,
            "precisions",
            tuple(sorted(precisions, key=lambda value: value.identity)),
        )

    @property
    def semantic_identity(self) -> str:
        """Provider/backend-independent operation identity under the scientific IR."""
        return canonical_hash(
            {
                "scientific_identity": self.scientific_identity,
                "operation": self.operation,
                "dtype": self.dtype,
                "accumulation_dtype": self.accumulation_dtype,
                "input_dtypes": self.input_dtypes,
                "shape": self.shape,
                "semantics": dict(self.semantics),
                "operands": [
                    {
                        "operand": operand.operand,
                        "modes": operand.modes,
                        "shape": operand.shape,
                        "access": operand.access,
                        "triangle": operand.triangle,
                    }
                    for operand in self.operands
                ],
            }
        )

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, typing.Any]:
        payload = {
            "schema": LOWERING_REQUEST_SCHEMA,
            "consumer": self.consumer,
            "operation": self.operation,
            "backend": self.backend,
            "dtype": self.dtype,
            "accumulation_dtype": self.accumulation_dtype,
            "shape": list(self.shape),
            "semantics": dict(self.semantics),
        }
        if (
            self.scientific_identity is not None
            or self.operands
            or self.precisions
            or self.constraints is not None
            or self.effects
            or self.input_dtypes
        ):
            payload.update(
                {
                    "schema": "generativeqc.compiler.lowering-request.v2",
                    "scientific_identity": self.scientific_identity,
                    "semantic_identity": self.semantic_identity,
                    "operands": [operand.to_payload() for operand in self.operands],
                    "input_dtypes": self.input_dtypes,
                    "precisions": [
                        precision.to_payload() for precision in self.precisions
                    ],
                    "constraints": None
                    if self.constraints is None
                    else self.constraints.to_payload(),
                    "effects": dict(self.effects),
                }
            )
        return payload


@dataclass(frozen=True, slots=True, eq=False)
class ProviderDescriptor(_TypedRecord):
    """Stable provider identity independent of one particular lowering request."""

    name: str
    kind: Literal["generated", "library", "runtime"]
    implementation: str
    version: str | None = None
    required_features: tuple[str, ...] = ()
    provenance: tuple[tuple[str, Scalar], ...] = ()

    def __post_init__(self) -> None:
        _name(self.name, "provider name")
        _name(self.implementation, "provider implementation")
        if self.kind not in ("generated", "library", "runtime"):
            raise ValueError("unknown lowering provider kind")
        if self.version is not None:
            _name(self.version, "provider version")
        features = tuple(self.required_features)
        if any(type(feature) is not str or not feature for feature in features):
            raise ValueError("provider features must be nonempty strings")
        if len(set(features)) != len(features):
            raise ValueError("provider features must be unique")
        object.__setattr__(self, "required_features", tuple(sorted(features)))
        object.__setattr__(
            self, "provenance", _pairs(self.provenance, "provider provenance")
        )

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schema": LOWERING_PROVIDER_SCHEMA,
            "name": self.name,
            "kind": self.kind,
            "implementation": self.implementation,
            "version": self.version,
            "required_features": list(self.required_features),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True, slots=True, eq=False)
class LoweringCandidate(_TypedRecord):
    """One legal or rejected lowering, retaining explicit negative evidence."""

    request: LoweringRequest
    implementation: str
    providers: tuple[ProviderDescriptor, ...]
    status: Literal["ready", "unsupported"]
    numerical_mode: str
    workspace_bytes: int = 0
    provider_bytes: int = 0
    reason: str | None = None
    provenance: tuple[tuple[str, Scalar], ...] = ()
    execution: CandidateExecution | None = field(default=None, kw_only=True)
    cost: LoweringCost | None = field(default=None, kw_only=True)
    target: TargetCapabilities | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.request, LoweringRequest):
            raise TypeError("lowering candidate requires a LoweringRequest")
        _name(self.implementation, "lowering implementation")
        _name(self.numerical_mode, "lowering numerical mode")
        providers = tuple(self.providers)
        if not providers or any(
            not isinstance(provider, ProviderDescriptor) for provider in providers
        ):
            raise ValueError("lowering candidate requires provider descriptors")
        identities = [provider.identity for provider in providers]
        if len(set(identities)) != len(identities):
            raise ValueError("lowering candidate providers must be unique")
        object.__setattr__(self, "providers", providers)
        _nonnegative_bytes(self.workspace_bytes, "lowering workspace bytes")
        _nonnegative_bytes(self.provider_bytes, "lowering provider bytes")
        if self.status not in ("ready", "unsupported"):
            raise ValueError("unknown lowering candidate status")
        if self.status == "ready" and self.reason is not None:
            raise ValueError("ready lowering candidate cannot carry a rejection reason")
        if self.status == "unsupported":
            _name(self.reason, "unsupported lowering rejection reason")
        object.__setattr__(
            self, "provenance", _pairs(self.provenance, "lowering provenance")
        )
        if self.execution is not None:
            if not isinstance(self.execution, CandidateExecution):
                raise TypeError("lowering execution requires CandidateExecution")
            if self.execution.precision not in self.request.precisions:
                raise ValueError(
                    "provider selected a precision outside the admitted request"
                )
            original = self.request.operands
            layouts = self.execution.layouts
            logical = lambda rows: [
                (
                    row.operand,
                    row.modes,
                    row.shape,
                    row.access,
                    row.triangle,
                    row.alias_group,
                )
                for row in rows
            ]
            if logical(original) != logical(layouts):
                raise ValueError(
                    "provider layout changes logical axes, effects or symmetry"
                )
        elif self.request.precisions and self.status == "ready":
            raise ValueError("joint ready candidate requires concrete execution facts")
        if self.cost is not None:
            if not isinstance(self.cost, LoweringCost):
                raise TypeError("lowering cost requires LoweringCost")
            if self.execution is not None and self.cost.cast_bytes < sum(
                cast.read_bytes + cast.write_bytes
                for cast in self.execution.precision.casts
            ):
                raise ValueError("candidate cost omits explicit cast traffic")
        if self.target is not None:
            if not isinstance(self.target, TargetCapabilities):
                raise TypeError("candidate target requires TargetCapabilities")
            if self.target.target.backend != self.request.backend:
                raise ValueError("candidate target changes requested backend")

    @property
    def request_hash(self) -> str:
        return self.request.identity

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, typing.Any]:
        payload = {
            "schema": LOWERING_CANDIDATE_SCHEMA,
            "request": self.request.to_payload(),
            "request_hash": self.request_hash,
            "implementation": self.implementation,
            "providers": [provider.to_payload() for provider in self.providers],
            "status": self.status,
            "numerical_mode": self.numerical_mode,
            "workspace_bytes": self.workspace_bytes,
            "provider_bytes": self.provider_bytes,
            "reason": self.reason,
            "provenance": dict(self.provenance),
        }
        if (
            self.execution is not None
            or self.cost is not None
            or self.target is not None
        ):
            payload.update(
                {
                    "schema": "generativeqc.compiler.lowering-candidate.v2",
                    "execution": None
                    if self.execution is None
                    else self.execution.to_payload(),
                    "cost": None if self.cost is None else self.cost.to_payload(),
                    "target": None if self.target is None else asdict(self.target),
                }
            )
        return payload


class LoweringProvider(typing.Protocol):
    """Side-effect-free candidate producer; selection and promotion live elsewhere."""

    descriptor: ProviderDescriptor

    def candidates(
        self, request: LoweringRequest, target: TargetCapabilities
    ) -> tuple[LoweringCandidate, ...]:
        """Return ready and/or explicit unsupported candidates for one request."""
        ...


def collect_lowering_candidates(
    request: LoweringRequest,
    target: TargetCapabilities,
    providers: typing.Iterable[LoweringProvider],
) -> tuple[LoweringCandidate, ...]:
    """Collect a bounded portfolio, retaining unsupported and admission evidence.

    Provider-specific legality remains with each implementation. Shared resource,
    capture and determinism constraints are enforced here so no provider can
    bypass them. Exhausting the candidate bound fails rather than silently biasing
    selection toward whichever provider happened to register first.
    """

    if not isinstance(request, LoweringRequest):
        raise TypeError("provider collection requires a LoweringRequest")
    if not isinstance(target, TargetCapabilities):
        raise TypeError("provider collection requires TargetCapabilities")
    if request.backend != target.target.backend:
        raise ValueError(
            f"lowering backend {request.backend!r} does not match target "
            f"{target.target.backend!r}"
        )

    collected: list[LoweringCandidate] = []
    identities: set[str] = set()
    constraints = request.constraints or LoweringConstraints()
    for provider in providers:
        descriptor = provider.descriptor
        if not isinstance(descriptor, ProviderDescriptor):
            raise TypeError("lowering provider descriptor has the wrong type")
        offered = tuple(
            islice(
                provider.candidates(request, target),
                constraints.maximum_candidates + 1 - len(collected),
            )
        )
        if len(offered) + len(collected) > constraints.maximum_candidates:
            raise ValueError("lowering candidate bound exceeded")
        if not offered:
            raise ValueError(
                f"provider {descriptor.name!r} must return explicit unsupported evidence"
            )
        for candidate in offered:
            if not isinstance(candidate, LoweringCandidate):
                raise TypeError("lowering provider returned a non-candidate")
            if candidate.request != request:
                raise ValueError(
                    f"provider {descriptor.name!r} returned a candidate for another request"
                )
            if descriptor.identity not in {
                dependency.identity for dependency in candidate.providers
            }:
                raise ValueError(
                    f"candidate from {descriptor.name!r} does not name its provider"
                )
            if candidate.target is not None and candidate.target != target:
                raise ValueError("provider returned a candidate for another target")
            rejection = candidate_constraint_rejection(candidate)
            admitted = (
                replace(candidate, status="unsupported", reason=rejection)
                if candidate.status == "ready" and rejection is not None
                else candidate
            )
            if admitted.target is None and admitted.execution is not None:
                admitted = replace(admitted, target=target)
            if admitted.identity in identities:
                raise ValueError("duplicate lowering candidate identity")
            identities.add(admitted.identity)
            collected.append(admitted)
    return tuple(collected)


def candidate_constraint_rejection(candidate: LoweringCandidate) -> str | None:
    """Apply common limits without asserting provider-specific numerical legality."""
    limits = candidate.request.constraints
    if limits is None:
        return None
    execution = candidate.execution
    if execution is None:
        return "candidate lacks concrete execution facts for constrained selection"
    additional = checked_bytes(
        candidate.workspace_bytes
        + candidate.provider_bytes
        + execution.temporary_bytes
        + execution.cache_bytes,
        "candidate additional device bytes",
    )
    for label, actual in (
        ("workspace_bytes", candidate.workspace_bytes),
        ("provider_bytes", candidate.provider_bytes),
        ("additional_device_bytes", additional),
        ("host_bytes", execution.host_bytes),
    ):
        limit = getattr(limits, label)
        if limit is not None and actual > limit:
            return f"{label} exceeds request limit: {actual} > {limit}"
    if limits.capture_required and not execution.capture_safe:
        return "candidate does not support the requested capture/replay contract"
    if DETERMINISM.index(execution.determinism) < DETERMINISM.index(limits.determinism):
        return "candidate does not meet the requested determinism/reduction order"
    return None


def lowering_diagnostics(
    candidates: typing.Iterable[LoweringCandidate],
) -> dict[str, typing.Any]:
    """Return deterministic provider/candidate provenance without selecting a winner."""

    materialized = tuple(candidates)
    if any(not isinstance(candidate, LoweringCandidate) for candidate in materialized):
        raise TypeError("lowering diagnostics require LoweringCandidate records")
    ordered = sorted(materialized, key=lambda candidate: candidate.identity)
    payloads = [candidate.to_payload() for candidate in ordered]
    providers = sorted(
        {
            provider.name
            for candidate in materialized
            for provider in candidate.providers
        }
    )
    return {
        "schema": LOWERING_DIAGNOSTICS_SCHEMA,
        "identity": canonical_hash(payloads),
        "providers": providers,
        "requests": [
            candidate.request.to_payload()
            for candidate in sorted(
                {
                    candidate.request.identity: candidate for candidate in materialized
                }.values(),
                key=lambda candidate: candidate.request.identity,
            )
        ],
        "candidates": payloads,
    }
