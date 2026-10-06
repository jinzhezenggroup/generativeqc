"""PTXAS resource records for arbitrary generated kernel symbols."""

import re
import typing
from dataclasses import dataclass

from .cuda_launch import CudaLaunchLimits, assess_cuda_launch
from .cuda_target import CudaTargetInfo
from .gpu_profitability import GpuProfitability


@dataclass(frozen=True, slots=True)
class KernelResources:
    """Static PTXAS resources; stack_bytes bounds both frame and cumulative stack."""

    function: str
    registers: int
    stack_bytes: int
    spill_store_bytes: int
    spill_load_bytes: int
    shared_bytes: int
    local_bytes: int | None = None


def parse_resources(diagnostics: str) -> tuple[KernelResources, ...]:
    """Preserve zero-shared-memory entries omitted by shell-specific parsing."""
    pattern = re.compile(
        r"Function properties for (?P<function>\S+)\n"
        r"\s*(?P<stack>\d+) bytes stack frame, (?P<stores>\d+) bytes spill stores, (?P<loads>\d+) bytes spill loads\n"
        r"ptxas info\s*: Used (?P<registers>\d+) registers(?P<rest>[^\n]*)"
    )
    result = []
    # Windows compiler output uses CRLF even when the caller retains raw text.
    for match in pattern.finditer(diagnostics.replace("\r\n", "\n")):
        shared = re.search(r"(\d+) bytes smem", match["rest"])
        local = re.search(r"(\d+) bytes lmem", match["rest"])
        cumulative = re.search(r"(\d+) bytes cumulative stack size", match["rest"])
        stack = int(match["stack"])
        if cumulative is not None:
            stack = max(stack, int(cumulative[1]))
        result.append(
            KernelResources(
                function=match["function"],
                registers=int(match["registers"]),
                stack_bytes=stack,
                spill_store_bytes=int(match["stores"]),
                spill_load_bytes=int(match["loads"]),
                shared_bytes=int(shared[1]) if shared else 0,
                local_bytes=int(local[1]) if local else None,
            )
        )
    return tuple(result)


def compiled_gpu_profitability(
    resources: typing.Iterable[KernelResources],
    target: CudaTargetInfo,
    block_threads: int,
    *,
    object_bytes: int | None = None,
    compile_seconds: float | None = None,
    dynamic_shared_bytes: int = 0,
    reserved_shared_bytes: int = 0,
    opted_in_dynamic_shared_bytes: int | None = None,
    kernel_max_threads_per_block: int | None = None,
    grid_blocks: int | None = None,
) -> GpuProfitability:
    """Normalize PTXAS rows and optional facts from the same launch contract.

    Storage in PTXAS rows is static: callers must supply dynamic launch storage
    separately. ``opted_in_dynamic_shared_bytes`` means a granted kernel limit,
    not the device capability. Rows with different launches must be assessed
    separately. With no grid extent this reports a per-SM upper bound; a supplied
    grid also requires a runtime-probed SM count and bounds whole-device occupancy.
    Neither form is measured occupancy, active-lane efficiency or a speed claim.
    The lower-level assessment retains separate per-SM/grid bounds and reasons.
    """
    if not isinstance(target, CudaTargetInfo):
        raise TypeError("compiled GPU profitability requires CudaTargetInfo")
    if (
        type(block_threads) is not int
        or not 0 < block_threads <= target.maximum_threads_per_block
    ):
        raise ValueError("compiled GPU profitability requires a valid block size")
    materialized = tuple(resources)
    if not materialized:
        raise ValueError("compiled GPU profitability requires PTXAS resource rows")
    if any(not isinstance(resource, KernelResources) for resource in materialized):
        raise TypeError("compiled GPU profitability requires KernelResources rows")
    if object_bytes is not None and (type(object_bytes) is not int or object_bytes < 0):
        raise ValueError("object_bytes must be non-negative or None")
    if compile_seconds is not None and (
        isinstance(compile_seconds, bool)
        or not isinstance(compile_seconds, (int, float))
        or compile_seconds < 0
    ):
        raise ValueError("compile_seconds must be non-negative or None")
    if grid_blocks is not None and target.sm_count is None:
        raise ValueError("grid-aware profitability requires a runtime-probed SM count")

    limits = CudaLaunchLimits.from_target(target)
    occupancies: list[float] = []
    for resource in materialized:
        assessment = assess_cuda_launch(
            limits,
            block_threads=block_threads,
            registers_per_thread=resource.registers,
            static_shared_bytes=resource.shared_bytes,
            dynamic_shared_bytes=dynamic_shared_bytes,
            reserved_shared_bytes=reserved_shared_bytes,
            opted_in_dynamic_shared_bytes=opted_in_dynamic_shared_bytes,
            kernel_max_threads_per_block=kernel_max_threads_per_block,
            grid_blocks=grid_blocks,
        )
        occupancy = assessment.whole_device_warp_occupancy_upper_bound
        occupancies.append(
            assessment.per_sm_warp_occupancy_upper_bound
            if occupancy is None else occupancy
        )

    local_values = tuple(resource.local_bytes for resource in materialized)
    local_bytes = (
        None
        if any(value is None for value in local_values)
        else max(typing.cast("tuple[int, ...]", local_values), default=0)
    )
    return GpuProfitability(
        compiled_registers_per_thread=max(
            resource.registers for resource in materialized
        ),
        spill_store_bytes=max(resource.spill_store_bytes for resource in materialized),
        spill_load_bytes=max(resource.spill_load_bytes for resource in materialized),
        local_bytes=local_bytes,
        shared_bytes=max(resource.shared_bytes for resource in materialized)
        + dynamic_shared_bytes,
        compiled_occupancy_upper_bound=min(occupancies),
        object_bytes=object_bytes,
        compile_seconds=None if compile_seconds is None else float(compile_seconds),
    )
