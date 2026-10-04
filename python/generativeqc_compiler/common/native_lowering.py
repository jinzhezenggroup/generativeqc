"""Emit the shared lowering portfolio for preparation in a native owner.

The input is the existing canonical request/candidate model. This adapter adds
neither science nor provider policy. Runtime-shape emitters may replace explicit
cost/resource expressions before calling the same bounded native selector; the
canonical template identities remain distinct from runtime shape compatibility.
"""

from __future__ import annotations

import json
import re
import typing
from dataclasses import asdict

from .lowering_contract import LoweringConstraints
from .provenance import canonical_hash

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from .lowering_provider import LoweringCandidate, LoweringRequest
    from .specialization import CompilationIdentity, TargetCapabilities

_NATIVE = "generativeqc::runtime::"
_DTYPE = {"float32": "Fp32", "float64": "Fp64"}
_ORDER = {
    "unspecified": "Unspecified",
    "reproducible": "Reproducible",
    "exact-order": "ExactOrder",
}


def _dtype(value: str) -> str:
    if value not in _DTYPE:
        raise ValueError("native lowering currently supports floating operands only")
    return _NATIVE + "PrecisionDtype::" + _DTYPE[value]


def _number(value: int | None) -> str:
    return "std::nullopt" if value is None else f"{value}ULL"


def _boolean(value: bool) -> str:
    return "true" if value else "false"


def native_lowering_portfolio(
    request: LoweringRequest,
    candidates: Sequence[LoweringCandidate],
    target: TargetCapabilities,
    compilation: CompilationIdentity,
    *,
    name: str,
) -> str:
    """Emit immutable C++ metadata consumed by select_native_lowering.

    Negative candidates and unknown phase costs are preserved. Selection is not
    performed here: the native owner supplies the expected replay count and may
    resolve optional provider capabilities before preparing executable entries.
    Returned declarations live in the caller's namespace and require the native
    runtime/lowering_binding.hpp header. All referenced strings have static life.
    """
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError("native portfolio needs a C++ identifier")
    if request.scientific_identity != compilation.scientific_hash:
        raise ValueError("native compilation does not match the scientific request")
    if request.backend != target.target.backend:
        raise ValueError("native target cannot change the requested backend")
    limits = request.constraints or LoweringConstraints()
    if (
        not request.precisions
        or len(request.precisions) > 16
        or len(request.input_dtypes) > 4
    ):
        raise ValueError("native precision/input bound exceeded")
    if len(candidates) > min(256, limits.maximum_candidates):
        raise ValueError("native candidate bound exceeded")
    if any(candidate.request != request for candidate in candidates):
        raise ValueError("native candidates must consume the same canonical request")
    if len({candidate.identity for candidate in candidates}) != len(candidates):
        raise ValueError("duplicate native candidate identity")
    target_identity = canonical_hash(asdict(target))
    compilation_identity = canonical_hash(asdict(compilation))
    lines = [
        f"static const std::string_view {name}_target={json.dumps(target_identity)};",
        f"static const std::string_view {name}_compilation={json.dumps(compilation_identity)};",
        f"static const std::array<{_NATIVE}NativeLoweringPrecision,{len(request.precisions)}> {name}_precisions{{{{",
    ]
    for precision in request.precisions:
        if len(precision.input_dtypes) > 4:
            raise ValueError("native precision input bound exceeded")
        directive = precision.directive
        arithmetic = ",".join(
            (
                _dtype(directive.storage_dtype),
                _dtype(directive.compute_dtype),
                _dtype(directive.accumulation_dtype),
                json.dumps(directive.qualification or ""),
                json.dumps(directive.math_mode),
            )
        )
        lines.append(
            "{"
            + json.dumps(precision.identity)
            + ",{"
            + arithmetic
            + "},{"
            + ",".join(map(_dtype, precision.input_dtypes))
            + "},"
            + str(len(precision.input_dtypes))
            + ","
            + _dtype(precision.publication_dtype)
            + ","
            + ",".join(
                map(
                    json.dumps,
                    (
                        canonical_hash([cast.to_payload() for cast in precision.casts])
                        if precision.casts
                        else "",
                        precision.refinement or "",
                        precision.audit or "",
                    ),
                )
            )
            + "},"
        )
    lines += [
        "}};",
        f"static const {_NATIVE}NativeLoweringRequest {name}_request{{",
        f"{json.dumps(request.scientific_identity)},{json.dumps(request.semantic_identity)},",
        f"{_dtype(request.dtype)},{_dtype(request.accumulation_dtype)},",
        "{"
        + ",".join(map(_dtype, request.input_dtypes))
        + "},"
        + str(len(request.input_dtypes))
        + f",{name}_precisions,{{",
        ",".join(
            _number(getattr(limits, field))
            for field in (
                "workspace_bytes",
                "provider_bytes",
                "additional_device_bytes",
                "host_bytes",
            )
        )
        + ","
        + _NATIVE
        + "LoweringDeterminism::"
        + _ORDER[limits.determinism]
        + f",{_boolean(limits.capture_required)},{limits.maximum_candidates}"
        + "}};",
        f"static const std::array<{_NATIVE}NativeLoweringCandidate,{len(candidates)}> {name}_candidates{{{{",
    ]
    features = dict(target.features)
    for candidate in candidates:
        execution = candidate.execution
        cost = candidate.cost
        reason = candidate.reason or (
            "candidate has no typed execution binding" if execution is None else ""
        )
        fields = [
            json.dumps(candidate.identity),
            json.dumps(request.semantic_identity),
            json.dumps("+".join(provider.name for provider in candidate.providers)),
            json.dumps(
                "+".join(provider.version or "" for provider in candidate.providers)
                if all(provider.version is not None for provider in candidate.providers)
                else ""
            ),
            json.dumps(execution.algorithm if execution else ""),
            json.dumps(
                canonical_hash([layout.to_payload() for layout in execution.layouts])
                if execution
                else ""
            ),
            json.dumps(
                canonical_hash(execution.topology.to_payload()) if execution else ""
            ),
            json.dumps(
                canonical_hash(asdict(candidate.target)) if candidate.target else ""
            ),
            json.dumps(compilation_identity),
            str(request.precisions.index(execution.precision) if execution else 0),
            "{"
            + ",".join(
                [
                    json.dumps(cost.source if cost else ""),
                    _boolean(cost is not None and cost.kind == "measured"),
                    *(
                        _number(getattr(cost, field) if cost else None)
                        for field in (
                            "prepare_ns",
                            "kernel_ns",
                            "cast_ns",
                            "pack_ns",
                            "refinement_ns",
                            "audit_ns",
                            "fallback_ns",
                        )
                    ),
                    *(
                        str(getattr(cost, field) if cost else 0)
                        for field in (
                            "cast_bytes",
                            "pack_bytes",
                            "refinement_bytes",
                            "audit_bytes",
                            "launches",
                        )
                    ),
                ]
            )
            + "}",
            str(candidate.workspace_bytes),
            str(candidate.provider_bytes),
            *(
                str(getattr(execution, field) if execution else 0)
                for field in ("temporary_bytes", "cache_bytes", "host_bytes")
            ),
            _NATIVE
            + "LoweringDeterminism::"
            + _ORDER[execution.determinism if execution else "unspecified"],
            _boolean(execution is not None and execution.capture_safe),
            _boolean(
                all(
                    features.get(feature) is True
                    for provider in candidate.providers
                    for feature in provider.required_features
                )
            ),
            json.dumps(reason),
        ]
        lines.append("{" + ",".join(fields) + "},")
    return "\n".join([*lines, "}};", ""])
