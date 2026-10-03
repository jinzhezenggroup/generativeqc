"""Calibrated CUDA wall-time estimates layered over static cost evidence.

This module deliberately keeps hardware calibration separate from architecture
limits. A CUDA architecture alone is not enough to predict seconds: callers
must provide measured effective rates for the intended device/SKU.

The first model is intentionally small and auditable. It uses a roofline-style
maximum of compute and memory time, explicit launch latency, and a disclosed
parallelism correction derived from the static CUDA cost report.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .cuda_cost_model import StaticCudaCost


def _positive_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return numeric


def _unit_interval(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    numeric = float(value)
    lower = 0.0 if not positive else math.nextafter(0.0, 1.0)
    if not math.isfinite(numeric) or numeric < lower or numeric > 1.0:
        qualifier = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must be in {qualifier}")
    return numeric


@dataclass(frozen=True, slots=True)
class CudaTimingCalibration:
    """Measured device calibration used to turn compiler work into seconds.

    The effective rates are achieved rates from calibration workloads, not
    vendor peak specifications. saturation_occupancy states the whole-device
    occupancy fraction at which those rates are assumed to saturate.

    uncertainty_fraction is a disclosed engineering error band, not a
    statistical confidence interval.
    """

    device: str
    effective_compute_ops_per_second: float
    effective_memory_bytes_per_second: float
    launch_seconds: float
    saturation_occupancy: float = 0.5
    uncertainty_fraction: float = 0.25
    provenance: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        _positive_float(
            self.effective_compute_ops_per_second,
            "effective_compute_ops_per_second",
        )
        _positive_float(
            self.effective_memory_bytes_per_second,
            "effective_memory_bytes_per_second",
        )
        _positive_float(self.launch_seconds, "launch_seconds")
        _unit_interval(self.saturation_occupancy, "saturation_occupancy", positive=True)
        _unit_interval(self.uncertainty_fraction, "uncertainty_fraction")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "generativeqc.compiler.cuda-timing-calibration.v1",
            **asdict(self),
        }


@dataclass(frozen=True, slots=True)
class CudaTimeEstimate:
    """One calibrated wall-time prediction for a CUDA candidate."""

    device: str
    estimated_seconds: float | None
    lower_seconds: float | None
    upper_seconds: float | None
    compute_seconds: float | None
    memory_seconds: float | None
    launch_seconds: float | None
    parallelism_fraction: float | None
    bottleneck: str | None
    diagnostics: tuple[str, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "generativeqc.compiler.cuda-time-estimate.v1",
            "scope": (
                "calibrated engineering estimate; requires validation against "
                "real-device endpoint timings"
            ),
            **asdict(self),
        }


def _parallelism_fraction(cost: StaticCudaCost) -> tuple[float | None, tuple[str, ...]]:
    """Return the strongest available whole-device parallelism evidence."""

    diagnostics: list[str] = []
    if cost.device_occupancy_upper_bound is not None:
        return cost.device_occupancy_upper_bound, ()

    if cost.occupancy_upper_bound is not None:
        diagnostics.append(
            "device SM count or grid size is unavailable; using per-SM occupancy "
            "without a global underfill correction"
        )
        return cost.occupancy_upper_bound, tuple(diagnostics)

    diagnostics.append("parallelism evidence is unavailable")
    return None, tuple(diagnostics)


def estimate_cuda_time(
    cost: StaticCudaCost,
    calibration: CudaTimingCalibration,
) -> CudaTimeEstimate:
    """Estimate CUDA wall time from compiler work and explicit calibration.

    First draft formula:

    max(ops / compute_rate, bytes / memory_rate) / parallel_scale
    + launches * launch_latency

    parallel_scale is min(1, occupancy / saturation_occupancy).

    Missing operation count, semantic traffic, launch count, or usable parallelism
    remains unknown rather than being guessed.
    """

    if not isinstance(cost, StaticCudaCost):
        raise TypeError("CUDA timing estimate requires StaticCudaCost")
    if not isinstance(calibration, CudaTimingCalibration):
        raise TypeError("CUDA timing estimate requires CudaTimingCalibration")

    diagnostics: list[str] = []
    parallelism, parallelism_diagnostics = _parallelism_fraction(cost)
    diagnostics.extend(parallelism_diagnostics)

    required = {
        "arithmetic operation count": cost.arithmetic_operation_count,
        "semantic traffic bytes": cost.semantic_traffic_bytes,
        "launch count": cost.launch_count,
    }
    for label, value in required.items():
        if value is None:
            diagnostics.append(f"{label} is unavailable")

    if parallelism is not None and parallelism <= 0.0:
        diagnostics.append("known launch shape exposes no executable parallelism")

    if any(value is None for value in required.values()):
        return CudaTimeEstimate(
            device=calibration.device,
            estimated_seconds=None,
            lower_seconds=None,
            upper_seconds=None,
            compute_seconds=None,
            memory_seconds=None,
            launch_seconds=None,
            parallelism_fraction=parallelism,
            bottleneck=None,
            diagnostics=tuple(diagnostics),
        )
    if parallelism is None or parallelism <= 0.0:
        return CudaTimeEstimate(
            device=calibration.device,
            estimated_seconds=None,
            lower_seconds=None,
            upper_seconds=None,
            compute_seconds=None,
            memory_seconds=None,
            launch_seconds=None,
            parallelism_fraction=parallelism,
            bottleneck=None,
            diagnostics=tuple(diagnostics),
        )

    assert cost.arithmetic_operation_count is not None
    assert cost.semantic_traffic_bytes is not None
    assert cost.launch_count is not None

    parallel_scale = min(1.0, parallelism / calibration.saturation_occupancy)
    if parallel_scale < 1.0:
        diagnostics.append(
            "effective throughput is linearly reduced below the calibrated "
            "saturation occupancy"
        )

    traffic_bytes = cost.semantic_traffic_bytes
    if cost.spill_bytes is not None:
        traffic_bytes += cost.spill_bytes
        if cost.spill_bytes:
            diagnostics.append("compiled spill bytes are included as memory traffic")
    else:
        diagnostics.append("spill traffic is unknown and omitted from memory work")

    compute_seconds = (
        cost.arithmetic_operation_count
        / calibration.effective_compute_ops_per_second
        / parallel_scale
    )
    memory_seconds = (
        traffic_bytes / calibration.effective_memory_bytes_per_second / parallel_scale
    )
    launch_seconds = cost.launch_count * calibration.launch_seconds
    body_seconds = max(compute_seconds, memory_seconds)
    estimated_seconds = body_seconds + launch_seconds

    if math.isclose(compute_seconds, memory_seconds, rel_tol=1.0e-9, abs_tol=0.0):
        bottleneck = "balanced"
    elif compute_seconds > memory_seconds:
        bottleneck = "compute"
    else:
        bottleneck = "memory"

    uncertainty = calibration.uncertainty_fraction
    return CudaTimeEstimate(
        device=calibration.device,
        estimated_seconds=estimated_seconds,
        lower_seconds=estimated_seconds * (1.0 - uncertainty),
        upper_seconds=estimated_seconds * (1.0 + uncertainty),
        compute_seconds=compute_seconds,
        memory_seconds=memory_seconds,
        launch_seconds=launch_seconds,
        parallelism_fraction=parallelism,
        bottleneck=bottleneck,
        diagnostics=tuple(diagnostics),
    )
