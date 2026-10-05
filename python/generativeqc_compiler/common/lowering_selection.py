"""Bounded prepare-time selection over the shared provider candidate registry.

This is a binding recipe for native/code-generation owners, not an executable
loader or another cache. It never allocates, JITs, searches a vendor heuristic,
or runs scientific work. Runtime owners prepare the selected recipe once and
retain its algorithm/resources with their existing lifetime machinery.
"""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass
from itertools import islice

from .lowering_provider import (
    LoweringCandidate,
    LoweringRequest,
    candidate_constraint_rejection,
    lowering_diagnostics,
)
from .provenance import canonical_hash
from .resources import checked_bytes
from .specialization import CompilationIdentity, TargetCapabilities


def _strict(candidate: LoweringCandidate) -> bool:
    execution = candidate.execution
    if execution is None:
        return False
    precision = execution.precision
    directive = precision.directive
    request = candidate.request
    return (
        directive.storage_dtype == request.dtype
        and directive.compute_dtype == request.dtype
        and directive.accumulation_dtype == request.accumulation_dtype
        and precision.input_dtypes == request.input_dtypes
    )


def _admission(candidate: LoweringCandidate, target: TargetCapabilities) -> str | None:
    if candidate.status != "ready":
        return candidate.reason
    if candidate.execution is None:
        return "candidate has no typed execution binding"
    if candidate.target != target:
        return "candidate legality/cost evidence does not match the binding target"
    rejection = candidate_constraint_rejection(candidate)
    if rejection is not None:
        return rejection
    features = dict(target.features)
    for provider in candidate.providers:
        if provider.version is None:
            return f"provider {provider.name} lacks prepared version/source identity"
        for feature in provider.required_features:
            if features.get(feature) is not True:
                return f"missing required provider capability: {feature}"
    return None


@dataclass(frozen=True, slots=True)
class LoweringBinding:
    """Immutable selected recipe, explicit fallbacks and full negative evidence.

    Fallback may retain the selected precision or restore strict requested
    arithmetic; it cannot silently narrow a strict selection. Preparation/OOM
    failure is handled by the runtime owner using only these admitted recipes,
    before capture/replay. Backend or context changes require a fresh binding.
    """

    selected: LoweringCandidate
    fallbacks: tuple[LoweringCandidate, ...]
    candidates: tuple[LoweringCandidate, ...]
    rejections: tuple[tuple[str, str], ...]
    target: TargetCapabilities
    compilation: CompilationIdentity
    expected_replays: int
    retained_incumbent: bool = False

    @property
    def cache_identity(self) -> str:
        """Execution identity; measurements do not invalidate identical code."""

        def executable(candidate: LoweringCandidate) -> dict:
            payload = candidate.to_payload()
            payload.pop("cost", None)
            return payload

        return canonical_hash(
            {
                "schema": "generativeqc.compiler.lowering-binding.v1",
                "target": asdict(self.target),
                "compilation": asdict(self.compilation),
                "selected": executable(self.selected),
                "fallbacks": [executable(candidate) for candidate in self.fallbacks],
            }
        )

    def validate_context(
        self,
        request: LoweringRequest,
        target: TargetCapabilities,
        compilation: CompilationIdentity,
    ) -> None:
        """Reject stale bindings before executing or looking up native resources."""
        if (
            request != self.selected.request
            or target != self.target
            or compilation != self.compilation
        ):
            raise ValueError(
                "prepared lowering context changed; rebind before execution"
            )

    def to_payload(self) -> dict:
        return {
            "schema": "generativeqc.compiler.lowering-binding.v1",
            "cache_identity": self.cache_identity,
            "scientific_identity": self.compilation.scientific_hash,
            "semantic_identity": self.selected.request.semantic_identity,
            "target": asdict(self.target),
            "compilation": asdict(self.compilation),
            "selected": self.selected.identity,
            "fallbacks": [candidate.identity for candidate in self.fallbacks],
            "rejections": dict(self.rejections),
            "expected_replays": self.expected_replays,
            "retained_incumbent": self.retained_incumbent,
            "diagnostics": lowering_diagnostics(self.candidates),
        }


def select_lowering_binding(
    request: LoweringRequest,
    target: TargetCapabilities,
    compilation: CompilationIdentity,
    candidates: typing.Iterable[LoweringCandidate],
    *,
    expected_replays: int = 1,
    qualified_incumbent: str | None = None,
) -> LoweringBinding:
    """Rank complete prepare + replay costs, preserving a strict fallback.

    Registration order never breaks ties. Incomplete costs are retained as
    negative evidence rather than treated as free work. Explicitly estimated
    costs are allowed for planning; only method endpoint qualification can
    promote a performance default. A provider name is never an input policy.

    During migration, a provider adapter may identify an already-qualified
    incumbent by its complete candidate identity. If its cost is incomplete,
    retain it without making a profitability claim. All legality gates still
    apply, including an executable strict fallback. Missing costs stay visible
    in diagnostics; an unmeasured incumbent cannot be displaced by estimates
    for a competitor. This option cannot qualify a new implementation.
    """
    if (
        not isinstance(request, LoweringRequest)
        or not isinstance(target, TargetCapabilities)
        or not isinstance(compilation, CompilationIdentity)
    ):
        raise TypeError(
            "selection requires typed request, target and compilation identity"
        )
    checked_bytes(expected_replays, "expected replays")
    if not expected_replays:
        raise ValueError("expected replays must be positive")
    if request.scientific_identity != compilation.scientific_hash:
        raise ValueError("binding compilation does not match scientific identity")
    if request.backend != target.target.backend:
        raise ValueError("binding cannot change the requested backend")
    maximum = request.constraints.maximum_candidates if request.constraints else 256
    offered = tuple(islice(candidates, maximum + 1))
    if len(offered) > maximum:
        raise ValueError("lowering candidate bound exceeded")
    if any(not isinstance(candidate, LoweringCandidate) for candidate in offered):
        raise TypeError("selection requires LoweringCandidate records")
    if any(candidate.request != request for candidate in offered):
        raise ValueError("all competing providers must consume the same request")
    if len({candidate.identity for candidate in offered}) != len(offered):
        raise ValueError("duplicate lowering candidate identity")
    offered = tuple(sorted(offered, key=lambda candidate: candidate.identity))
    rejections = []
    ranked = []
    legal = []
    for candidate in offered:
        reason = _admission(candidate, target)
        if reason is None:
            legal.append(candidate)
        cost = candidate.cost
        if reason is None and (
            cost is None or cost.prepare_ns is None or cost.replay_ns is None
        ):
            reason = "complete prepare/cast/pack/kernel/refinement/audit/fallback cost is unavailable"
        if reason is not None:
            rejections.append((candidate.identity, reason))
            continue
        assert (
            cost is not None
            and cost.prepare_ns is not None
            and cost.replay_ns is not None
        )
        ranked.append(
            (
                cost.prepare_ns + expected_replays * cost.replay_ns,
                candidate.identity,
                candidate,
            )
        )
    ranked.sort(key=lambda row: row[:2])
    if qualified_incumbent is not None:
        incumbent = next(
            (
                candidate
                for candidate in legal
                if candidate.identity == qualified_incumbent
            ),
            None,
        )
        if incumbent is None:
            raise ValueError("qualified incumbent is absent or fails binding admission")
        cost = incumbent.cost
        if cost is None or cost.prepare_ns is None or cost.replay_ns is None:
            if not any(_strict(candidate) for candidate in legal):
                raise ValueError(
                    "incumbent retention requires a legal strict candidate"
                )
            assert incumbent.execution is not None
            fallbacks = tuple(
                candidate
                for candidate in legal
                if candidate != incumbent
                and candidate.execution is not None
                and (
                    _strict(candidate)
                    or candidate.execution.precision == incumbent.execution.precision
                )
            )
            return LoweringBinding(
                incumbent,
                fallbacks,
                offered,
                tuple(rejections),
                target,
                compilation,
                expected_replays,
                retained_incumbent=True,
            )
    if not ranked:
        raise ValueError(
            "no admitted lowering candidate with complete cost evidence: "
            + "; ".join(reason for _, reason in rejections)
        )
    if not any(_strict(candidate) for _, _, candidate in ranked):
        raise ValueError(
            "joint selection requires a complete strict requested-precision candidate"
        )
    selected = ranked[0][2]
    assert selected.execution is not None
    fallbacks = tuple(
        candidate
        for _, _, candidate in ranked[1:]
        if candidate.execution is not None
        and (
            _strict(candidate)
            or candidate.execution.precision == selected.execution.precision
        )
    )
    return LoweringBinding(
        selected,
        fallbacks,
        offered,
        tuple(rejections),
        target,
        compilation,
        expected_replays,
    )
