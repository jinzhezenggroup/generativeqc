"""Interpret retained Nsight Compute execution-efficiency evidence.

This module is deliberately diagnostic. It does not promote CUDA candidates or
change production scheduling. It turns measured execution counters into a small
set of auditable mechanism labels so static/PTXAS resource predictions can be
calibrated against real execution without pretending that one counter predicts
endpoint speed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


def _optional_fraction(value: float | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite fraction or None")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be in [0, 1] or None")


def _optional_nonnegative(value: float | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number or None")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number or None")


def _optional_count(value: int | None, name: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{name} must be a non-negative integer or None")


@dataclass(frozen=True, slots=True)
class NcuExecutionEvidence:
    """One source-matched NCU kernel/graph counter record.

    Fractions use [0, 1], never percentages. Local-memory request counters are
    dynamic requests reported by NCU and must not be relabelled as spill bytes.
    Stall fields are cycles per issued instruction from one compatible
    WarpStateStats capture.
    """

    architecture: str
    source_revision: str
    kernel_identity: str
    report_sha256: str
    device: str
    theoretical_occupancy_fraction: float | None = None
    achieved_occupancy_fraction: float | None = None
    executed_threads_per_warp_instruction: float | None = None
    warp_size: int = 32
    issue_active_fraction: float | None = None
    eligible_warps_per_scheduler: float | None = None
    warp_cycles_per_issued_instruction: float | None = None
    barrier_cycles_per_issued_instruction: float | None = None
    wait_cycles_per_issued_instruction: float | None = None
    short_scoreboard_cycles_per_issued_instruction: float | None = None
    long_scoreboard_cycles_per_issued_instruction: float | None = None
    local_load_requests: int | None = None
    local_store_requests: int | None = None
    dram_busy_fraction: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "architecture",
            "source_revision",
            "kernel_identity",
            "report_sha256",
            "device",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not self.architecture.startswith("sm_"):
            raise ValueError("architecture must use sm_XX notation")
        revision = self.source_revision.lower()
        if not 8 <= len(revision) <= 40 or any(
            char not in "0123456789abcdef" for char in revision
        ):
            raise ValueError(
                "source_revision must be an 8-40 digit hexadecimal Git SHA"
            )
        digest = self.report_sha256.lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("report_sha256 must be a 64 digit hexadecimal digest")
        for name in (
            "theoretical_occupancy_fraction",
            "achieved_occupancy_fraction",
            "issue_active_fraction",
            "dram_busy_fraction",
        ):
            _optional_fraction(getattr(self, name), name)
        if type(self.warp_size) is not int or self.warp_size <= 0:
            raise ValueError("warp_size must be a positive integer")
        _optional_nonnegative(
            self.executed_threads_per_warp_instruction,
            "executed_threads_per_warp_instruction",
        )
        if (
            self.executed_threads_per_warp_instruction is not None
            and self.executed_threads_per_warp_instruction > self.warp_size
        ):
            raise ValueError(
                "executed_threads_per_warp_instruction cannot exceed warp_size"
            )
        for name in (
            "eligible_warps_per_scheduler",
            "warp_cycles_per_issued_instruction",
            "barrier_cycles_per_issued_instruction",
            "wait_cycles_per_issued_instruction",
            "short_scoreboard_cycles_per_issued_instruction",
            "long_scoreboard_cycles_per_issued_instruction",
        ):
            _optional_nonnegative(getattr(self, name), name)
        for name in ("local_load_requests", "local_store_requests"):
            _optional_count(getattr(self, name), name)

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> NcuExecutionEvidence:
        """Load the strict v1 retained-evidence schema."""
        if not isinstance(payload, dict):
            raise TypeError("NCU execution evidence must be a JSON object")
        if payload.get("schema") != "generativeqc.compiler.ncu-execution-evidence.v1":
            raise ValueError(
                "NCU execution evidence must use "
                "generativeqc.compiler.ncu-execution-evidence.v1"
            )
        fields = payload.get("evidence")
        if not isinstance(fields, dict):
            raise TypeError("NCU execution evidence requires an evidence object")
        known = set(cls.__dataclass_fields__)
        unknown = set(fields) - known
        if unknown:
            raise ValueError(
                f"unknown NCU execution evidence fields: {sorted(unknown)}"
            )
        return cls(**fields)

    def to_payload(self) -> dict[str, object]:
        """Serialize the measured inputs without deriving profiler claims."""
        return {
            "schema": "generativeqc.compiler.ncu-execution-evidence.v1",
            "scope": (
                "source/report/kernel-bound measured NCU execution counters; "
                "diagnostic only; local-memory requests are not spill bytes"
            ),
            "evidence": asdict(self),
        }


@dataclass(frozen=True, slots=True)
class NcuExecutionAssessment:
    """Mechanism classification derived only from explicit measured counters."""

    confidence: str
    mechanisms: tuple[str, ...]
    guidance: tuple[str, ...]
    derived_metrics: dict[str, float | int | None]
    diagnostics: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "generativeqc.compiler.ncu-execution-assessment.v1",
            "scope": (
                "diagnostic calibration of static/PTXAS predictions; "
                "does not alter production candidate ranking or predict endpoint speed"
            ),
            "confidence": self.confidence,
            "mechanisms": list(self.mechanisms),
            "guidance": list(self.guidance),
            "derived_metrics": self.derived_metrics,
            "diagnostics": list(self.diagnostics),
        }


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def assess_ncu_execution(evidence: NcuExecutionEvidence) -> NcuExecutionAssessment:
    """Classify measured execution mechanisms without fitting opaque weights.

    Thresholds intentionally separate broad, falsifiable regimes seen in retained
    GenerativeQC NCU captures. They are advisory and should be revised only with
    new held-out evidence, never to force a desired candidate ranking.
    """

    if not isinstance(evidence, NcuExecutionEvidence):
        raise TypeError("NCU assessment requires NcuExecutionEvidence")

    active_lane_fraction = (
        None
        if evidence.executed_threads_per_warp_instruction is None
        else evidence.executed_threads_per_warp_instruction / evidence.warp_size
    )
    occupancy_realization = _ratio(
        evidence.achieved_occupancy_fraction,
        evidence.theoretical_occupancy_fraction,
    )
    barrier_fraction = _ratio(
        evidence.barrier_cycles_per_issued_instruction,
        evidence.warp_cycles_per_issued_instruction,
    )
    wait_fraction = _ratio(
        evidence.wait_cycles_per_issued_instruction,
        evidence.warp_cycles_per_issued_instruction,
    )
    long_scoreboard_fraction = _ratio(
        evidence.long_scoreboard_cycles_per_issued_instruction,
        evidence.warp_cycles_per_issued_instruction,
    )
    local_requests = None
    if (
        evidence.local_load_requests is not None
        or evidence.local_store_requests is not None
    ):
        local_requests = (evidence.local_load_requests or 0) + (
            evidence.local_store_requests or 0
        )

    mechanisms: list[str] = []
    guidance: list[str] = []
    diagnostics: list[str] = []

    resource_occupancy_confirmed = (
        evidence.theoretical_occupancy_fraction is not None
        and evidence.theoretical_occupancy_fraction <= 0.25
        and occupancy_realization is not None
        and occupancy_realization >= 0.90
    )
    if resource_occupancy_confirmed:
        mechanisms.append("resource-limited-occupancy-confirmed")

    issue_starved = (
        evidence.issue_active_fraction is not None
        and evidence.issue_active_fraction < 0.15
    ) or (
        evidence.eligible_warps_per_scheduler is not None
        and evidence.eligible_warps_per_scheduler < 0.25
    )
    if issue_starved:
        mechanisms.append("issue-starved")

    lane_underutilized = (
        active_lane_fraction is not None and active_lane_fraction < 0.75
    )
    if lane_underutilized:
        mechanisms.append("lane-underutilized")

    barrier_tail = barrier_fraction is not None and barrier_fraction >= 0.25
    if barrier_tail:
        mechanisms.append("barrier-tail")

    scoreboard_limited = (
        long_scoreboard_fraction is not None and long_scoreboard_fraction >= 0.25
    )
    if scoreboard_limited:
        mechanisms.append("long-scoreboard-latency")

    local_state_traffic = local_requests is not None and local_requests > 0
    if local_state_traffic:
        mechanisms.append("local-state-traffic-present")
        diagnostics.append(
            "dynamic local-memory requests are measured requests, not proven spill bytes"
        )

    if evidence.dram_busy_fraction is not None and evidence.dram_busy_fraction < 0.50:
        mechanisms.append("dram-not-saturated")

    serial_or_underexposed = (
        active_lane_fraction is not None
        and active_lane_fraction <= 0.125
        and barrier_fraction is not None
        and barrier_fraction <= 0.05
        and local_requests == 0
    )
    if serial_or_underexposed:
        mechanisms.append("serial-or-underexposed-reduction")
        guidance.append(
            "prioritize cooperative/tiled reduction or producer fusion before "
            "register-cap tuning"
        )

    barrier_divergence_local = (
        lane_underutilized
        and barrier_tail
        and local_state_traffic
        and evidence.achieved_occupancy_fraction is not None
        and evidence.achieved_occupancy_fraction <= 0.25
    )
    if barrier_divergence_local:
        mechanisms.append("barrier-divergence-local-state")
        guidance.append(
            "prioritize homogeneous class/queue/CTA specialization and shorter "
            "scratch lifetimes"
        )

    scoreboard_local = (
        lane_underutilized
        and not barrier_tail
        and scoreboard_limited
        and local_state_traffic
    )
    if scoreboard_local:
        mechanisms.append("scoreboard-local-state")
        guidance.append(
            "prioritize recurrence/source reuse and local-state lifetime over "
            "barrier tuning"
        )

    if resource_occupancy_confirmed and issue_starved:
        mechanisms.append("occupancy-bound-is-not-root-cause-proof")
        guidance.append(
            "treat static occupancy as a calibrated resource bound, not a standalone "
            "bottleneck ranking"
        )

    if "dram-not-saturated" in mechanisms and issue_starved:
        guidance.append(
            "do not rank DRAM-bandwidth or FP64-throughput saturation first without "
            "matching pipe counters"
        )

    evidence_groups = sum(
        (
            evidence.theoretical_occupancy_fraction is not None
            and evidence.achieved_occupancy_fraction is not None,
            evidence.executed_threads_per_warp_instruction is not None,
            evidence.issue_active_fraction is not None
            or evidence.eligible_warps_per_scheduler is not None,
            evidence.warp_cycles_per_issued_instruction is not None
            and evidence.barrier_cycles_per_issued_instruction is not None,
            evidence.local_load_requests is not None
            and evidence.local_store_requests is not None,
            evidence.dram_busy_fraction is not None,
        )
    )
    confidence = (
        "high" if evidence_groups >= 4 else "medium" if evidence_groups >= 2 else "low"
    )
    if evidence_groups < 2:
        diagnostics.append(
            "too few independent NCU counter groups for a strong mechanism "
            "classification"
        )
    if not mechanisms:
        diagnostics.append(
            "no calibrated mechanism threshold was crossed; retain the raw counters"
        )

    return NcuExecutionAssessment(
        confidence=confidence,
        mechanisms=tuple(mechanisms),
        guidance=tuple(dict.fromkeys(guidance)),
        derived_metrics={
            "occupancy_realization": occupancy_realization,
            "active_lane_fraction": active_lane_fraction,
            "barrier_stall_fraction": barrier_fraction,
            "wait_stall_fraction": wait_fraction,
            "long_scoreboard_stall_fraction": long_scoreboard_fraction,
            "local_memory_requests": local_requests,
        },
        diagnostics=tuple(diagnostics),
    )
