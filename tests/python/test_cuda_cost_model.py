"""GPU-free CUDA cost model and global-parallelism screening."""

import pytest
from generativeqc_compiler.common.cuda_cost_model import static_cuda_cost
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.gpu_profitability import GpuProfitability


def test_compiled_resource_screen_uses_register_and_shared_limits() -> None:
    cost = static_cuda_cost(
        GpuProfitability(
            semantic_traffic_bytes=8192,
            arithmetic_operation_count=32768,
            compiled_registers_per_thread=128,
            spill_store_bytes=0,
            spill_load_bytes=0,
            shared_bytes=32768,
            launch_count=1,
        ),
        cuda_target_info("sm_120"),
        128,
    )

    assert cost.evidence_stage == "compiled"
    assert cost.resident_blocks_per_sm_upper_bound == 3
    assert cost.occupancy_upper_bound == pytest.approx(0.25)
    assert cost.limiting_resources == ("shared-memory",)
    assert cost.arithmetic_intensity_ops_per_byte == pytest.approx(4.0)
    assert cost.diagnostics == ()


def test_grid_underfill_is_visible_without_a_gpu_probe() -> None:
    cost = static_cuda_cost(
        GpuProfitability(
            compiled_registers_per_thread=64,
            spill_store_bytes=0,
            spill_load_bytes=0,
            shared_bytes=0,
        ),
        cuda_target_info("sm_120"),
        128,
        grid_blocks=40,
        sm_count=170,
    )

    assert cost.resident_blocks_per_sm_upper_bound == 8
    assert cost.occupancy_upper_bound == pytest.approx(2.0 / 3.0)
    assert cost.grid_saturation_upper_bound == pytest.approx(40 / (8 * 170))
    assert "grid exposes fewer blocks than one resident wave" in cost.diagnostics


def test_unknown_sm_count_stays_unknown_instead_of_inventing_device_topology() -> None:
    cost = static_cuda_cost(
        GpuProfitability(
            estimated_registers_per_thread=64,
            estimated_occupancy_upper_bound=0.5,
        ),
        cuda_target_info("sm_120"),
        128,
        grid_blocks=1,
    )

    assert cost.evidence_stage == "static"
    assert cost.sm_count is None
    assert cost.grid_saturation_upper_bound is None
    assert "grid saturation unavailable without a device SM count" in cost.diagnostics
    assert (
        "resource occupancy uses static estimates; PTXAS evidence is unavailable"
        in cost.diagnostics
    )


def test_screening_priority_penalizes_underfill_before_equal_static_work() -> None:
    target = cuda_target_info("sm_120")
    profitability = GpuProfitability(
        semantic_traffic_bytes=4096,
        arithmetic_operation_count=2048,
        compiled_registers_per_thread=64,
        spill_store_bytes=0,
        spill_load_bytes=0,
        shared_bytes=0,
        launch_count=1,
        source_bytes=1000,
    )
    saturated = static_cuda_cost(
        profitability, target, 128, grid_blocks=1360, sm_count=170
    )
    underfilled = static_cuda_cost(
        profitability, target, 128, grid_blocks=40, sm_count=170
    )

    assert saturated.screening_priority(1) < underfilled.screening_priority(0)


def test_payload_states_that_screening_is_not_a_timing_prediction() -> None:
    payload = static_cuda_cost(
        GpuProfitability(estimated_registers_per_thread=32),
        cuda_target_info("sm_120"),
        128,
    ).to_payload()

    assert payload["schema"] == "generativeqc.compiler.cuda-static-cost.v1"
    assert "not a runtime or speedup prediction" in payload["scope"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"block_threads": 0}, "block_threads"),
        ({"grid_blocks": -1}, "grid_blocks"),
        ({"sm_count": 0}, "sm_count"),
        ({"sm_count": True}, "sm_count"),
    ],
)
def test_invalid_static_cost_inputs_fail_closed(
    kwargs: dict[str, object], match: str
) -> None:
    options = {"block_threads": 128, **kwargs}
    block_threads = options.pop("block_threads")
    with pytest.raises(ValueError, match=match):
        static_cuda_cost(
            GpuProfitability(),
            cuda_target_info("sm_120"),
            block_threads,
            **options,
        )
