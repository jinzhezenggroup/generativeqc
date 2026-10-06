"""Launch-aware CUDA resource admission; never a measured occupancy or speedup.

Keep static PTXAS storage, dynamic launch storage and the driver's per-block
reservation separate. A per-SM resource bound cannot diagnose a small grid.
No GPU is probed and no scientific or method policy is selected here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cuda_target import CudaTargetInfo


def _count(value: int, name: str, *, positive: bool = False) -> None:
    if type(value) is not int or value < int(positive):
        domain = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {domain} integer")


@dataclass(frozen=True, slots=True)
class CudaLaunchLimits:
    """Device facts, not tuning ceilings or an assumed GPU marketing model."""

    warp_size: int
    maximum_threads_per_block: int
    maximum_threads_per_sm: int
    maximum_blocks_per_sm: int
    registers_per_sm: int
    maximum_registers_per_thread: int
    shared_memory_per_block: int
    shared_memory_per_block_optin: int
    shared_memory_per_sm: int
    sm_count: int | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name != "sm_count":
                _count(getattr(self, name), name, positive=True)
        if self.sm_count is not None:
            _count(self.sm_count, "sm_count", positive=True)
        if not (
            self.warp_size
            <= self.maximum_threads_per_block
            <= self.maximum_threads_per_sm
        ):
            raise ValueError("inconsistent CUDA thread limits")
        if self.maximum_threads_per_sm % self.warp_size:
            raise ValueError("CUDA SM thread limit must contain complete warps")
        if not (
            self.shared_memory_per_block
            <= self.shared_memory_per_block_optin
            <= self.shared_memory_per_sm
        ):
            raise ValueError("inconsistent CUDA shared-memory limits")

    @classmethod
    def from_target(cls, target: CudaTargetInfo) -> CudaLaunchLimits:
        return cls(**{name: getattr(target, name) for name in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class CudaLaunchAssessment:
    """Analytical upper bounds, before allocation granularity and runtime stalls.

    Whole-device occupancy is absent unless both grid extent and the actual SM
    count are supplied. Warp occupancy does not report active lanes, eligible
    warps, time-weighted achieved occupancy or an endpoint performance gain.
    """

    rejections: tuple[str, ...]
    requires_shared_memory_optin: bool
    shared_bytes_per_block: int
    resident_blocks_per_sm_upper_bound: int
    per_sm_warp_occupancy_upper_bound: float
    whole_device_warp_occupancy_upper_bound: float | None

    @property
    def legal(self) -> bool:
        return not self.rejections


def assess_cuda_launch(
    limits: CudaLaunchLimits,
    *,
    block_threads: int,
    registers_per_thread: int,
    static_shared_bytes: int = 0,
    dynamic_shared_bytes: int = 0,
    reserved_shared_bytes: int = 0,
    opted_in_dynamic_shared_bytes: int | None = None,
    kernel_max_threads_per_block: int | None = None,
    grid_blocks: int | None = None,
) -> CudaLaunchAssessment:
    """Admit one launch and bound both per-SM and grid-limited parallelism.

    ``opted_in_dynamic_shared_bytes`` is an explicit *granted kernel limit*,
    never merely the device's opt-in capability. Static shared arrays cannot
    acquire opt-in support by changing a tuning ceiling. The driver reservation
    is charged to SM residency, not to the application's dynamic-byte request.
    Call CUDA's occupancy API on the final kernel for allocation-granular bounds.
    """
    if not isinstance(limits, CudaLaunchLimits):
        raise TypeError("CUDA launch assessment requires CudaLaunchLimits")
    _count(block_threads, "block_threads", positive=True)
    for name, value in (
        ("registers_per_thread", registers_per_thread),
        ("static_shared_bytes", static_shared_bytes),
        ("dynamic_shared_bytes", dynamic_shared_bytes),
        ("reserved_shared_bytes", reserved_shared_bytes),
    ):
        _count(value, name)
    for name, value in (
        ("opted_in_dynamic_shared_bytes", opted_in_dynamic_shared_bytes),
        ("kernel_max_threads_per_block", kernel_max_threads_per_block),
        ("grid_blocks", grid_blocks),
    ):
        if value is not None:
            _count(value, name, positive=name == "kernel_max_threads_per_block")

    rejections: list[str] = []
    if block_threads > limits.maximum_threads_per_block:
        rejections.append("device-thread-limit")
    if (
        kernel_max_threads_per_block is not None
        and block_threads > kernel_max_threads_per_block
    ):
        rejections.append("kernel-thread-limit")
    if registers_per_thread > limits.maximum_registers_per_thread:
        rejections.append("per-thread-register-limit")
    if static_shared_bytes > limits.shared_memory_per_block:
        rejections.append("static-shared-memory-limit")
    application_shared = static_shared_bytes + dynamic_shared_bytes
    requires_optin = application_shared > limits.shared_memory_per_block
    if application_shared > limits.shared_memory_per_block_optin:
        rejections.append("device-optin-shared-memory-limit")
    if requires_optin and opted_in_dynamic_shared_bytes is None:
        rejections.append("dynamic-shared-memory-optin-not-granted")
    if (
        opted_in_dynamic_shared_bytes is not None
        and dynamic_shared_bytes > opted_in_dynamic_shared_bytes
    ):
        rejections.append("kernel-dynamic-shared-memory-limit")

    # A partial warp still consumes a complete warp slot. Register-allocation
    # granularity is intentionally not guessed from a compute-capability name.
    warps = (block_threads + limits.warp_size - 1) // limits.warp_size
    allocated_threads = warps * limits.warp_size
    capacities = [
        limits.maximum_blocks_per_sm,
        limits.maximum_threads_per_sm // allocated_threads,
    ]
    if registers_per_thread:
        capacities.append(
            limits.registers_per_sm // (registers_per_thread * allocated_threads)
        )
    shared = application_shared + reserved_shared_bytes
    if shared:
        capacities.append(limits.shared_memory_per_sm // shared)
    resident = min(capacities)
    if resident == 0:
        rejections.append("no-resident-block")
    if rejections:
        resident = 0
    per_sm = resident * allocated_threads / limits.maximum_threads_per_sm
    device = None
    if grid_blocks is not None and limits.sm_count is not None:
        resident_total = min(grid_blocks, resident * limits.sm_count)
        device = (
            resident_total
            * allocated_threads
            / (limits.sm_count * limits.maximum_threads_per_sm)
        )
    return CudaLaunchAssessment(
        tuple(rejections), requires_optin, shared, resident, per_sm, device
    )
