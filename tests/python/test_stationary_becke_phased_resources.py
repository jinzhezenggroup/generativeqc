"""Device-free admission gates for the optional shared-owner phase reservation."""

from dataclasses import replace

import pytest
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.method.stationary_resources import (
    plan_stationary_cuda_resources,
)
from generativeqc_compiler.xc.grid_phased import PhasedBeckePlan


@pytest.mark.parametrize("atoms", [33, 48, 96, 128])
@pytest.mark.parametrize("points", [1, 17, 256, 257])
def test_optional_phases_reserve_scratch_seeds_and_indices(
    atoms: int, points: int
) -> None:
    shape = {
        "atoms": atoms,
        "aos": 8 * atoms,
        "primitives": 16 * atoms,
        "points": points,
        "tasks": 256,
        "spins": 1,
        "sources": 8,
        "target": cuda_target_info("sm_120"),
    }
    baseline = plan_stationary_cuda_resources(**shape, budget_bytes=1 << 30)
    planned = plan_stationary_cuda_resources(
        **shape, budget_bytes=1 << 30, phased_becke=True
    )
    phase = PhasedBeckePlan(atoms, points)
    required = phase.scratch_bytes + 8 * points + 8 * phase.pairs
    assert baseline.phased_becke_bytes == 0
    assert planned.phased_becke_bytes == required
    assert planned.allocation_bytes == baseline.allocation_bytes + required
    assert planned.geometry_lanes == baseline.geometry_lanes == points
    for spare, accepted in ((-1, False), (0, True), (1, True)):
        admitted = plan_stationary_cuda_resources(
            **shape,
            budget_bytes=planned.allocation_bytes + spare,
            phased_becke=True,
        )
        assert bool(admitted.phased_becke_bytes) == accepted
        assert admitted.geometry_lanes == points


def test_phases_keep_small_uncached_unqualified_and_multi_point_lanes_bounded() -> None:
    target = cuda_target_info("sm_120")
    shape = {
        "atoms": 96,
        "aos": 768,
        "primitives": 1536,
        "points": 256,
        "tasks": 256,
        "spins": 1,
        "sources": 8,
        "target": target,
    }
    for changes in (
        {"atoms": 24},
        {"points": 1024},
        {"cooperative_becke": False},
        {"target": cuda_target_info("sm_80")},
        {"target": replace(target, maximum_threads_per_block=64)},
    ):
        arguments = {**shape, **changes}
        baseline = plan_stationary_cuda_resources(**arguments, budget_bytes=1 << 30)
        planned = plan_stationary_cuda_resources(
            **arguments, budget_bytes=1 << 30, phased_becke=True
        )
        assert planned == baseline
    baseline = plan_stationary_cuda_resources(**shape, budget_bytes=1 << 30)
    uncached = plan_stationary_cuda_resources(
        **shape,
        budget_bytes=baseline.allocation_bytes - baseline.center_geometry_bytes,
        phased_becke=True,
    )
    assert uncached.center_geometry_bytes == uncached.phased_becke_bytes == 0
    with pytest.raises(ValueError, match="phased Becke selection must be boolean"):
        plan_stationary_cuda_resources(**shape, budget_bytes=1 << 30, phased_becke=1)
