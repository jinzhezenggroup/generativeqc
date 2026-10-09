"""Device-free admission gates for the optional shared-owner phase reservation."""

from dataclasses import replace

import pytest
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.method.stationary_becke_phased import (
    emit_stationary_phased_becke_cuda,
)
from generativeqc_compiler.method.stationary_resources import (
    GEOMETRY_MAX_LANES,
    plan_stationary_cuda_resources,
)
from generativeqc_compiler.xc.becke_partition import (
    recognize_becke_partition_domain_graph,
)
from generativeqc_compiler.xc.grid_partition_ir import grid_partition_domain_program
from generativeqc_compiler.xc.grid_phased import PhasedBeckePlan


@pytest.mark.parametrize(
    "atoms,points",
    [(atoms, points) for atoms in (33, 48, 96, 128) for points in (1, 17, 256, 257)]
    + [(96, 1024)],
)
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
        {"points": GEOMETRY_MAX_LANES + 1},
        {"cooperative_becke": False},
        {"target": cuda_target_info("sm_80")},
        {"target": replace(target, maximum_threads_per_block=64)},
    ):
        arguments = {**shape, **changes}
        baseline = plan_stationary_cuda_resources(**arguments, budget_bytes=1 << 30)
        if "points" in changes:
            assert baseline.geometry_lanes < arguments["points"]
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


@pytest.mark.parametrize("atoms", [24, 33, 48, 96, 128])
def test_primitive_admission_reuses_exact_concurrent_phase_reservation(
    atoms: int,
) -> None:
    shape = {
        "atoms": atoms,
        "aos": 8 * atoms,
        "primitives": 16 * atoms,
        "points": 256,
        "tasks": 256,
        "spins": 1,
        "sources": 8,
        "target": cuda_target_info("sm_120"),
    }
    phased = plan_stationary_cuda_resources(
        **shape, budget_bytes=1 << 30, phased_becke=True
    )
    primitive = plan_stationary_cuda_resources(
        **shape, budget_bytes=1 << 30, becke_primitive=True
    )
    assert replace(primitive, becke_primitive=False) == phased
    assert primitive.becke_primitive == bool(phased.phased_becke_bytes)
    if primitive.becke_primitive:
        bounded = plan_stationary_cuda_resources(
            **shape, budget_bytes=primitive.allocation_bytes - 1, becke_primitive=True
        )
        assert bounded.phased_becke_bytes == 0
        assert bounded.becke_primitive is False
        assert bounded.geometry_lanes == phased.geometry_lanes
    for changes in (
        {"cooperative_becke": False},
        {"target": cuda_target_info("sm_80")},
    ):
        bounded = plan_stationary_cuda_resources(
            **{**shape, **changes}, budget_bytes=1 << 30, becke_primitive=True
        )
        assert bounded.becke_primitive is False
        assert bounded.phased_becke_bytes == 0
    with pytest.raises(ValueError, match="primitive selection must be boolean"):
        plan_stationary_cuda_resources(**shape, budget_bytes=1 << 30, becke_primitive=1)


@pytest.mark.parametrize("iterations", [1, 3, 5])
def test_shared_stationary_emission_binds_actual_dynamic_graph_and_native_bound(
    iterations: int,
) -> None:
    operation = recognize_becke_partition_domain_graph(
        grid_partition_domain_program(4, iterations),
        atom_limit=4,
        iterations=iterations,
    )
    assert operation is not None
    source = emit_stationary_phased_becke_cuda(4, iterations=iterations)
    assert operation.identity in source
    assert "stationary_becke_primitive_max_atoms = 4;" in source
    assert "pair_coefficient_reverse_phase(" in source
    assert "atom_gather_coefficient_phase(" in source
    assert "zero_seed_elision" in source
    assert "input.zero_seed(point)" in source
    assert "atomicAdd(input.zero_seed_points" in source
    assert emit_stationary_phased_becke_cuda(4, iterations=iterations) is source
