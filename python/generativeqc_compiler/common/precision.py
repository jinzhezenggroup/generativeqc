"""Shared execution-precision contracts for compiler schedules.

Scientific IRs describe equations and requested observables.  This module
describes how an already-admitted execution candidate stores, computes and
accumulates values.  It deliberately does not decide whether lower precision
is scientifically acceptable; method controllers and evidence own that gate.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from .provenance import canonical_hash
from .resources import checked_bytes

DTYPES = frozenset(("float32", "float64"))
STRICT_MATH_MODE = "ieee-rn-no-tf32"
PRECISION_SCHEDULE_SCHEMA = "generativeqc.compiler.execution-precision.v1"


def _dtype(value: typing.Any, label: str) -> str:
    if value not in DTYPES:
        raise ValueError(f"{label} must be float32 or float64")
    return value


def _name(value: typing.Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


@dataclass(frozen=True)
class CastBoundary:
    """One explicit conversion and its logical traffic, shared by IR lowerers.

    These are reads/writes, not a measured memory bandwidth or a claim that
    conversion is free when fused. Allocation liveness belongs to the planner.
    """

    name: str
    source_dtype: str
    target_dtype: str
    elements: int
    read_bytes: int
    write_bytes: int

    def __post_init__(self) -> None:
        _name(self.name, "cast name")
        _dtype(self.source_dtype, "cast source dtype")
        _dtype(self.target_dtype, "cast target dtype")
        for label in ("elements", "read_bytes", "write_bytes"):
            checked_bytes(getattr(self, label), label)
        sizes = {"float32": 4, "float64": 8}
        if self.read_bytes != self.elements * sizes[self.source_dtype] or (
            self.write_bytes != self.elements * sizes[self.target_dtype]
        ):
            raise ValueError("cast traffic must match its element count and dtypes")

    @property
    def simultaneous_bytes(self) -> int:
        return checked_bytes(
            self.read_bytes + self.write_bytes, "cast simultaneous bytes"
        )

    def to_payload(self) -> dict:
        return {
            "name": self.name,
            "source_dtype": self.source_dtype,
            "target_dtype": self.target_dtype,
            "elements": self.elements,
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
            "simultaneous_bytes": self.simultaneous_bytes,
        }


@dataclass(frozen=True, slots=True)
class PrecisionDirective:
    """One region's storage, compute and accumulation precision contract."""

    storage_dtype: str
    compute_dtype: str
    accumulation_dtype: str
    qualification: str | None = None
    math_mode: str = STRICT_MATH_MODE

    def __post_init__(self) -> None:
        for label in ("storage_dtype", "compute_dtype", "accumulation_dtype"):
            _dtype(getattr(self, label), label)
        if self.math_mode != STRICT_MATH_MODE:
            raise ValueError("unsupported compiler arithmetic mode")
        if self.qualification is not None:
            _name(self.qualification, "precision qualification")

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "storage_dtype": self.storage_dtype,
            "compute_dtype": self.compute_dtype,
            "accumulation_dtype": self.accumulation_dtype,
            "qualification": self.qualification,
            "math_mode": self.math_mode,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPrecisionSchedule:
    """Portable precision identity consumed by heterogeneous schedule owners.

    Region names are consumer-owned stable labels.  The contract can therefore
    describe a TensorIR value, a generated ERI recurrence, a Fock accumulation,
    or a grid/XC stage without moving numerical-admission policy into this
    common compiler layer.
    """

    regions: tuple[tuple[str, PrecisionDirective], ...]
    strict_audit_dtype: str = "float64"
    audit_owner: str = "method-controller"
    math_mode: str = STRICT_MATH_MODE

    def __post_init__(self) -> None:
        _dtype(self.strict_audit_dtype, "strict audit dtype")
        _name(self.audit_owner, "precision audit owner")
        if self.math_mode != STRICT_MATH_MODE:
            raise ValueError("unsupported compiler arithmetic mode")
        regions = tuple(self.regions)
        if not regions:
            raise ValueError(
                "execution precision schedule requires at least one region"
            )
        names: set[str] = set()
        normalized: list[tuple[str, PrecisionDirective]] = []
        for row in regions:
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ValueError("precision regions must be name/directive pairs")
            name, directive = row
            _name(name, "precision region")
            if name in names:
                raise ValueError(f"duplicate precision region {name!r}")
            if not isinstance(directive, PrecisionDirective):
                raise TypeError("precision region requires PrecisionDirective")
            if directive.math_mode != self.math_mode:
                raise ValueError("region and schedule arithmetic modes must agree")
            names.add(name)
            normalized.append((name, directive))
        object.__setattr__(self, "regions", tuple(sorted(normalized)))

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    @property
    def is_strict_fp64(self) -> bool:
        return all(
            directive.storage_dtype == "float64"
            and directive.compute_dtype == "float64"
            and directive.accumulation_dtype == "float64"
            for _, directive in self.regions
        )

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schema": PRECISION_SCHEDULE_SCHEMA,
            "strict_audit_dtype": self.strict_audit_dtype,
            "audit_owner": self.audit_owner,
            "math_mode": self.math_mode,
            "regions": [
                {"name": name, **directive.to_payload()}
                for name, directive in self.regions
            ],
        }


def uniform_precision_schedule(
    region: str,
    *,
    storage_dtype: str = "float64",
    compute_dtype: str = "float64",
    accumulation_dtype: str = "float64",
    qualification: str | None = None,
    strict_audit_dtype: str = "float64",
    audit_owner: str = "method-controller",
) -> ExecutionPrecisionSchedule:
    """Construct one explicit uniform region without implying promotion."""

    _name(region, "precision region")
    return ExecutionPrecisionSchedule(
        (
            (
                region,
                PrecisionDirective(
                    storage_dtype=storage_dtype,
                    compute_dtype=compute_dtype,
                    accumulation_dtype=accumulation_dtype,
                    qualification=qualification,
                ),
            ),
        ),
        strict_audit_dtype=strict_audit_dtype,
        audit_owner=audit_owner,
    )
