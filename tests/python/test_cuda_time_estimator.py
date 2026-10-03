"""Calibrated CUDA time-estimator behavior."""

import pytest
from generativeqc_compiler.common.cuda_cost_model import static_cuda_cost
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.cuda_time_estimator import (
    CudaTimingCalibration,
    estimate_cuda_time,
)
from generativeqc_compiler.common.gpu_profitability import GpuProfitability


def _calibration(**overrides: object) -> CudaTimingCalibration:
    values: dict[str, object] = {
        "device": "draft-device",
        "effective_compute_ops_per_second": 1.0e12,
        "effective_memory_bytes_per_second": 1.0e11,
        "launch_seconds": 1.0e-5,
        "saturation_occupancy": 0.5,
        "uncertainty_fraction": 0.25,
        "provenance": "unit-test calibration",
    }
    values.update(overrides)
    return CudaTimingCalibration(**values)


def test_estimate_uses_roofline_launch_and_parallelism_terms() -> None:
    cost = static_cuda_cost(
        GpuProfitability(
            semantic_traffic_bytes=100_000_000,
            arithmetic_operation_count=1_000_000_000,
            compiled_registers_per_thread=64,
            spill_store_bytes=0,
            spill_load_bytes=0,
            shared_bytes=0,
            launch_count=2,
        ),
        cuda_target_info("sm_120"),
        128,
        grid_blocks=680,
        sm_count=170,
    )

    estimate = estimate_cuda_time(cost, _calibration())

    assert estimate.parallelism_fraction == pytest.approx(1.0 / 3.0)
    assert estimate.compute_seconds == pytest.approx(0.0015)
    assert estimate.memory_seconds == pytest.approx(0.0015)
    assert estimate.launch_seconds == pytest.approx(0.00002)
    assert estimate.estimated_seconds == pytest.approx(0.00152)
    assert estimate.lower_seconds == pytest.approx(0.00114)
    assert estimate.upper_seconds == pytest.approx(0.00190)
    assert estimate.bottleneck == "balanced"
    assert any("linearly reduced" in item for item in estimate.diagnostics)


def test_estimate_requires_work_counts_instead_of_inventing_them() -> None:
    cost = static_cuda_cost(
        GpuProfitability(
            compiled_registers_per_thread=64,
            spill_store_bytes=0,
            spill_load_bytes=0,
            shared_bytes=0,
        ),
        cuda_target_info("sm_120"),
        128,
        grid_blocks=680,
        sm_count=170,
    )

    estimate = estimate_cuda_time(cost, _calibration())

    assert estimate.estimated_seconds is None
    assert any("arithmetic operation count is unavailable" in d for d in estimate.diagnostics)
    assert any("semantic traffic bytes is unavailable" in d for d in estimate.diagnostics)
    assert any("launch count is unavailable" in d for d in estimate.diagnostics)


def test_per_sm_occupancy_fallback_is_disclosed() -> None:
    cost = static_cuda_cost(
        GpuProfitability(
            semantic_traffic_bytes=1000,
            arithmetic_operation_count=2000,
            estimated_registers_per_thread=64,
            estimated_occupancy_upper_bound=0.5,
            launch_count=1,
        ),
        cuda_target_info("sm_120"),
        128,
    )

    estimate = estimate_cuda_time(cost, _calibration())

    assert estimate.estimated_seconds is not None
    assert estimate.parallelism_fraction == pytest.approx(0.5)
    assert any("without a global underfill correction" in d for d in estimate.diagnostics)


def test_compiled_spills_are_counted_as_memory_traffic() -> None:
    cost = static_cuda_cost(
        GpuProfitability(
            semantic_traffic_bytes=100,
            arithmetic_operation_count=1,
            compiled_registers_per_thread=64,
            spill_store_bytes=100,
            spill_load_bytes=100,
            shared_bytes=0,
            launch_count=1,
        ),
        cuda_target_info("sm_120"),
        128,
        grid_blocks=1360,
        sm_count=170,
    )

    estimate = estimate_cuda_time(
        cost,
        _calibration(
            effective_compute_ops_per_second=1.0e30,
            effective_memory_bytes_per_second=100.0,
            launch_seconds=1.0e-9,
        ),
    )

    assert estimate.memory_seconds == pytest.approx(3.0)
    assert estimate.bottleneck == "memory"
    assert any("spill bytes are included" in d for d in estimate.diagnostics)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("effective_compute_ops_per_second", 0.0),
        ("effective_memory_bytes_per_second", -1.0),
        ("launch_seconds", 0.0),
        ("saturation_occupancy", 0.0),
        ("saturation_occupancy", 1.1),
        ("uncertainty_fraction", -0.1),
        ("uncertainty_fraction", 1.1),
    ],
)
def test_invalid_calibration_fails_closed(field: str, value: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        _calibration(**{field: value})
