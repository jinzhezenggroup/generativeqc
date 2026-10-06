"""Independent arithmetic checks for launch-aware resource admission."""

from dataclasses import replace

import pytest
from generativeqc_compiler.common.cuda_launch import (
    CudaLaunchLimits,
    assess_cuda_launch,
)
from generativeqc_compiler.common.cuda_resources import (
    KernelResources,
    compiled_gpu_profitability,
)
from generativeqc_compiler.common.cuda_target import cuda_target_info


@pytest.fixture
def limits() -> CudaLaunchLimits:
    # Explicit fixture facts; this is not a runtime device observation.
    return CudaLaunchLimits(32, 1024, 1536, 24, 65536, 255, 49152, 101376, 102400, 170)


def test_dynamic_becke_storage_and_small_grid(limits: CudaLaunchLimits) -> None:
    record = assess_cuda_launch(
        limits,
        block_threads=32,
        registers_per_thread=64,
        static_shared_bytes=16,
        dynamic_shared_bytes=23936,
        grid_blocks=606,
    )
    assert record.legal
    assert record.shared_bytes_per_block == 23952
    assert record.resident_blocks_per_sm_upper_bound == 4
    assert record.per_sm_warp_occupancy_upper_bound == pytest.approx(4 / 48)
    assert record.whole_device_warp_occupancy_upper_bound == pytest.approx(
        606 / (170 * 48)
    )


def test_registers_can_defeat_the_proposed_128_thread_gain(
    limits: CudaLaunchLimits,
) -> None:
    record = assess_cuda_launch(
        limits,
        block_threads=128,
        registers_per_thread=255,
        static_shared_bytes=16,
        dynamic_shared_bytes=12096,
        grid_blocks=1024,
    )
    assert record.resident_blocks_per_sm_upper_bound == 2
    assert record.whole_device_warp_occupancy_upper_bound == pytest.approx(8 / 48)


def test_one_block_is_not_whole_device_occupancy(limits: CudaLaunchLimits) -> None:
    record = assess_cuda_launch(
        limits, block_threads=128, registers_per_thread=32, grid_blocks=1
    )
    assert record.per_sm_warp_occupancy_upper_bound == 1
    assert record.whole_device_warp_occupancy_upper_bound == pytest.approx(
        4 / (170 * 48)
    )


@pytest.mark.parametrize("grid_blocks", [None, 0, 1, 606, 1024, 1000000])
def test_grid_bound_never_exceeds_sm_bound(
    limits: CudaLaunchLimits, grid_blocks: int | None
) -> None:
    record = assess_cuda_launch(
        limits, block_threads=128, registers_per_thread=32, grid_blocks=grid_blocks
    )
    if grid_blocks is None:
        assert record.whole_device_warp_occupancy_upper_bound is None
    else:
        assert (
            0
            <= record.whole_device_warp_occupancy_upper_bound
            <= record.per_sm_warp_occupancy_upper_bound
            <= 1
        )


def test_unknown_sm_count_stays_unknown(limits: CudaLaunchLimits) -> None:
    record = assess_cuda_launch(
        replace(limits, sm_count=None),
        block_threads=32,
        registers_per_thread=32,
        grid_blocks=1,
    )
    assert record.whole_device_warp_occupancy_upper_bound is None


def test_dynamic_optin_requires_a_granted_kernel_limit(
    limits: CudaLaunchLimits,
) -> None:
    request = {
        "block_threads": 128,
        "registers_per_thread": 64,
        "static_shared_bytes": 1024,
        "dynamic_shared_bytes": 65536,
    }
    missing = assess_cuda_launch(limits, **request)
    assert not missing.legal
    assert "dynamic-shared-memory-optin-not-granted" in missing.rejections
    granted = assess_cuda_launch(limits, **request, opted_in_dynamic_shared_bytes=65536)
    assert granted.legal and granted.requires_shared_memory_optin
    too_small = assess_cuda_launch(
        limits, **request, opted_in_dynamic_shared_bytes=65535
    )
    assert not too_small.legal


def test_static_storage_is_not_dynamic_optin(limits: CudaLaunchLimits) -> None:
    result = assess_cuda_launch(
        limits,
        block_threads=128,
        registers_per_thread=64,
        static_shared_bytes=65536,
        opted_in_dynamic_shared_bytes=101376,
    )
    assert not result.legal
    assert "static-shared-memory-limit" in result.rejections


def test_driver_reservation_changes_residency(limits: CudaLaunchLimits) -> None:
    args = {
        "block_threads": 32,
        "registers_per_thread": 32,
        "dynamic_shared_bytes": 25600,
    }
    assert assess_cuda_launch(limits, **args).resident_blocks_per_sm_upper_bound == 4
    assert (
        assess_cuda_launch(
            limits, **args, reserved_shared_bytes=1024
        ).resident_blocks_per_sm_upper_bound
        == 3
    )


def test_kernel_thread_cap_and_partial_warp(limits: CudaLaunchLimits) -> None:
    bad = assess_cuda_launch(
        limits,
        block_threads=128,
        registers_per_thread=32,
        kernel_max_threads_per_block=64,
    )
    assert not bad.legal and bad.resident_blocks_per_sm_upper_bound == 0
    partial = assess_cuda_launch(limits, block_threads=33, registers_per_thread=32)
    assert partial.resident_blocks_per_sm_upper_bound == 24
    assert partial.per_sm_warp_occupancy_upper_bound == 1


@pytest.mark.parametrize(
    "field",
    [
        "block_threads",
        "registers_per_thread",
        "static_shared_bytes",
        "dynamic_shared_bytes",
        "reserved_shared_bytes",
        "grid_blocks",
        "opted_in_dynamic_shared_bytes",
        "kernel_max_threads_per_block",
    ],
)
@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_invalid_launch_counts_rejected(
    limits: CudaLaunchLimits, field: str, bad: float
) -> None:
    args = {"block_threads": 128, "registers_per_thread": 32}
    args[field] = bad
    with pytest.raises(ValueError):
        assess_cuda_launch(limits, **args)


@pytest.mark.parametrize(
    "update",
    [
        {"sm_count": 0},
        {"warp_size": 0},
        {"shared_memory_per_block_optin": 100},
        {"maximum_threads_per_sm": 1000},
    ],
)
def test_inconsistent_limits_rejected(
    limits: CudaLaunchLimits, update: dict[str, int]
) -> None:
    with pytest.raises(ValueError):
        replace(limits, **update)


def test_empty_grid_does_not_fabricate_active_work(limits: CudaLaunchLimits) -> None:
    result = assess_cuda_launch(
        limits, block_threads=128, registers_per_thread=32, grid_blocks=0
    )
    assert result.legal
    assert result.whole_device_warp_occupancy_upper_bound == 0


@pytest.fixture
def resource() -> KernelResources:
    return KernelResources("kernel", 32, 0, 0, 0, 16)


def test_profitability_accounts_for_launch_shared_memory_and_grid(
    resource: KernelResources,
) -> None:
    target = cuda_target_info("sm_120").with_runtime_probe(sm_count=170)
    profitability = compiled_gpu_profitability(
        [resource],
        target,
        32,
        dynamic_shared_bytes=23936,
        reserved_shared_bytes=1024,
        grid_blocks=606,
    )
    assert profitability.shared_bytes == 23952
    assert profitability.compiled_occupancy_upper_bound == pytest.approx(
        606 / (170 * 48)
    )
    assert profitability.local_bytes is None


def test_profitability_reservation_lowers_residency_without_inflating_app_bytes(
    resource: KernelResources,
) -> None:
    target = cuda_target_info("sm_120")
    plain = compiled_gpu_profitability(
        [replace(resource, shared_bytes=0)], target, 32, dynamic_shared_bytes=25600
    )
    reserved = compiled_gpu_profitability(
        [replace(resource, shared_bytes=0)],
        target,
        32,
        dynamic_shared_bytes=25600,
        reserved_shared_bytes=1024,
    )
    assert plain.compiled_occupancy_upper_bound == pytest.approx(4 / 48)
    assert reserved.compiled_occupancy_upper_bound == pytest.approx(3 / 48)
    assert plain.shared_bytes == reserved.shared_bytes == 25600


@pytest.mark.parametrize("grid_blocks", [0, 1, 606])
def test_profitability_grid_requires_probed_sm_count(
    resource: KernelResources, grid_blocks: int
) -> None:
    with pytest.raises(ValueError, match="runtime-probed SM count"):
        compiled_gpu_profitability(
            [resource], cuda_target_info("sm_120"), 128, grid_blocks=grid_blocks
        )


@pytest.mark.parametrize("granted", [None, 65535, 65536])
def test_profitability_preserves_dynamic_optin_admission(
    resource: KernelResources, granted: int | None
) -> None:
    profitability = compiled_gpu_profitability(
        [resource],
        cuda_target_info("sm_120"),
        128,
        dynamic_shared_bytes=65536,
        opted_in_dynamic_shared_bytes=granted,
    )
    expected = 4 / 48 if granted == 65536 else 0
    assert profitability.compiled_occupancy_upper_bound == pytest.approx(expected)


@pytest.mark.parametrize("grid_blocks", [0, 1, 10**100])
def test_profitability_rejected_kernel_has_zero_grid_occupancy(
    resource: KernelResources, grid_blocks: int
) -> None:
    profitability = compiled_gpu_profitability(
        [resource],
        cuda_target_info("sm_120").with_runtime_probe(sm_count=170),
        128,
        kernel_max_threads_per_block=64,
        grid_blocks=grid_blocks,
    )
    assert profitability.compiled_occupancy_upper_bound == 0


@pytest.mark.parametrize("excess", [0, 1])
def test_exact_dynamic_shared_device_boundary(
    limits: CudaLaunchLimits, excess: int
) -> None:
    requested = limits.shared_memory_per_block_optin + excess
    result = assess_cuda_launch(
        limits,
        block_threads=128,
        registers_per_thread=32,
        dynamic_shared_bytes=requested,
        opted_in_dynamic_shared_bytes=requested,
    )
    assert result.legal is (excess == 0)
    assert result.resident_blocks_per_sm_upper_bound == (1 if excess == 0 else 0)


def test_large_counts_do_not_wrap_resource_admission(limits: CudaLaunchLimits) -> None:
    huge = 10**100
    result = assess_cuda_launch(
        limits,
        block_threads=huge,
        registers_per_thread=huge,
        dynamic_shared_bytes=huge,
        grid_blocks=huge,
    )
    assert not result.legal
    assert result.resident_blocks_per_sm_upper_bound == 0
    assert result.whole_device_warp_occupancy_upper_bound == 0
