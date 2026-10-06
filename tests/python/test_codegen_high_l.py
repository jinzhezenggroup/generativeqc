"""High-L, shell-class, and backend-lowering codegen tests.

Moved behavior-neutrally from the legacy codegen compatibility suite for #489.
"""

from __future__ import annotations

# ruff: noqa: F401  # Keep this move behavior-neutral; imports mirror legacy ownership.
from codegen_test_support import (
    _PRODUCTION_PRELUDE,
    AXES,
    boys_values,
    build_dppp_component_kernel,
    build_dppp_contraction_kernel,
    build_dppp_fused_plan,
    build_fused_shell_plan,
    build_integral_ir,
    build_ppps_rys_force_program,
    build_rys_axis_program,
    build_rys_force_program,
    build_shell_class_component_kernel,
    build_shell_class_contraction_kernel,
    build_weighted_shell_contraction_kernel,
    cartesian_components,
    ContractionSpec,
    cuda_target_info,
    CudaTargetInfo,
    DDDD_SPEC,
    DDPS_SPEC,
    DPDS_SPEC,
    dppp_components,
    DPPP_SPEC,
    emit_rys_force_root_body_cuda,
    emit_shell_class_fused_cuda,
    evaluate_fused_shell_component,
    evaluate_fused_shell_observables,
    evaluate_fused_shell_value,
    evaluate_ppps_rys_component,
    evaluate_rys_component,
    factored_dppp_variables,
    FDDD_SPEC,
    FFPS_SPEC,
    FUSED_SHELL_SPEC_BY_NAME,
    itertools,
    KernelConsumer,
    math,
    OperatorFamily,
    OperatorSpec,
    PairOrientation,
    PairStorage,
    PSSS_SPEC,
    pytest,
    re,
    replace,
    REPOSITORY_ROOT,
    rys_boys_values,
    RysRecurrenceKind,
    RysState,
    sample_variables,
    schedule_candidates,
    ScheduleIR,
    ScheduleKind,
    ShellClassSpec,
    supported_schedule_trials,
    supports_component_lane_rys,
    TEST_CUDA_TARGET,
    TranslationInvariant,
    typing,
)

@pytest.mark.parametrize(
    ("maximum_order", "series_threshold"),
    ((0, 1.0e-8), (1, 0.25), (2, 0.75), (3, 1.25), (4, 2.0)),
)
def test_generated_low_order_boys_thresholds_preserve_upward_recurrence(
    maximum_order: int, series_threshold: float
) -> None:
    """Keep the fast low-order branch accurate at its least stable point."""

    threshold_literal = "1.0e-8" if maximum_order == 0 else str(series_threshold)
    assert f"MaximumOrder == {maximum_order} ? {threshold_literal}" in (
        _PRODUCTION_PRELUDE
    )
    for argument in (
        series_threshold,
        series_threshold + 1.0e-6,
        0.5 * (series_threshold + 6.0),
        6.0,
    ):
        values = [0.5 * math.sqrt(math.pi / argument) * math.erf(math.sqrt(argument))]
        exponential = math.exp(-argument)
        for order in range(1, maximum_order + 1):
            values.append(
                ((2.0 * order - 1.0) * values[-1] - exponential) / (2.0 * argument)
            )
        reference = rys_boys_values(argument, maximum_order + 1)[maximum_order]
        assert values[maximum_order] == pytest.approx(
            reference, rel=5.0e-14, abs=1.0e-15
        )


def test_generic_cuda_emitter_uses_backend_lowering_not_dppp_compatibility() -> None:
    """Keep generic compilation independent of historical shell adapters."""

    emitter = (
        REPOSITORY_ROOT
        / "python"
        / "generativeqc_compiler"
        / "integral"
        / "cuda_emitter.py"
    ).read_text(encoding="utf-8")
    production = (
        REPOSITORY_ROOT
        / "python"
        / "generativeqc_compiler"
        / "integral"
        / "production.py"
    ).read_text(encoding="utf-8")
    benchmark = (
        REPOSITORY_ROOT
        / "python"
        / "generativeqc_compiler"
        / "integral"
        / "benchmark.py"
    ).read_text(encoding="utf-8")
    assert "from . import cuda_lowering as _implementation" in emitter
    assert "dppp_dispatch" not in emitter
    assert not (
        REPOSITORY_ROOT
        / "python"
        / "generativeqc_compiler"
        / "integral"
        / "dppp_dispatch.py"
    ).exists()
    assert "from .dppp_dispatch import" not in production
    assert "from .dppp_dispatch import" not in benchmark


@pytest.mark.parametrize("architecture", ("sm_80", "sm_86", "sm_89", "sm_90", "sm_120"))
def test_cuda_target_catalog_covers_the_compile_matrix(architecture: str) -> None:
    """Expose target-derived scheduling and resource limits for supported SMs."""

    target = cuda_target_info(architecture)
    assert isinstance(target, CudaTargetInfo)
    assert target.architecture == architecture
    assert target.warp_size == 32
    assert target.maximum_threads_per_block == 1024
    assert target.tuning_maximum_shared_bytes <= target.shared_memory_per_block


@pytest.mark.parametrize("name", ("fsss", "fsps"))
def test_scalar_thread_force_lowering_is_structural_for_f_shells(
    name: str,
) -> None:
    """Generate scalar subset/Wick force code without a shell-name allowlist.

    These classes intentionally are not production promotions.  Emitting them
    here proves that the compiler-owned fallback can cover a new f-shell
    derivative class from its component metadata and derivative IR alone.
    """

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=1,
    )
    source = emit_shell_class_fused_cuda(
        spec,
        build_fused_shell_plan(
            spec, schedule=schedule, recurrence="subset_wick", target=TEST_CUDA_TARGET
        ),
    )

    class_name = name[0].upper() + name[1:]
    assert f"generated_{name}_scalar_thread_force_task" in source
    assert f"generated_{name}_scalar_thread_accumulate_components_" in source
    assert f"Generated{class_name}ScalarThreadStorage" in source
    assert "scalar thread-task force lowering is currently specialized" not in source


def test_ppps_scalar_thread_lowering_uses_explicit_derivative_center_slots() -> None:
    """Route scalar-thread force atomics through non-final IR recovery."""

    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    operator = OperatorSpec(
        family=OperatorFamily.FOUR_CENTER_ERI,
        centers=(0, 1, 2, 3),
        invariants=(TranslationInvariant(dependent_center=1),),
    )
    force = ContractionSpec(
        consumer="direct_force",
        density="rhf|uhf",
        output="atomic_force",
    )
    integral = build_integral_ir(
        spec,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(force,),
    )
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=8,
    )
    source = emit_shell_class_fused_cuda(
        spec,
        build_fused_shell_plan(
            spec, integral=integral, schedule=schedule, target=TEST_CUDA_TARGET
        ),
    )

    assert "double decay_gradients[4][3];" in source
    assert "storage.primitive.decay_gradients[3][2]" in source
    recovery_begin = source.index(
        "const double fourth_force",
        source.index("generated_ppps_scalar_thread_force_task"),
    )
    recovery = source[recovery_begin : recovery_begin + 600]
    assert "static_cast<std::size_t>(task.atom[1])" in recovery
    assert "static_cast<std::size_t>(task.atom[3])" not in recovery


def test_ppps_rys_program_is_a_compact_unique_state_recurrence() -> None:
    """Keep the independent backend at recurrence-state granularity."""

    program = build_ppps_rys_force_program()
    assert program.spec == FUSED_SHELL_SPEC_BY_NAME["ppps"]
    assert program.nroots == 3
    assert program.independent_derivative_centers == (0, 1, 2)
    assert program.recovered_derivative_centers == (3,)
    assert program.independent_force_centers == (0, 1, 2)
    assert program.component_order == program.spec.components
    assert len(program.axis_program.requested_states) == 20
    assert len(program.axis_program.instructions) == 23
    states = [instruction.state for instruction in program.axis_program.instructions]
    assert len(states) == len(set(states))
    emitted: set[RysState] = set()
    for instruction in program.axis_program.instructions:
        assert set(instruction.dependencies) <= emitted
        emitted.add(instruction.state)
    assert {instruction.kind for instruction in program.axis_program.instructions} == {
        RysRecurrenceKind.SEED,
        RysRecurrenceKind.TRR_BRA,
        RysRecurrenceKind.TRR_KET,
        RysRecurrenceKind.HRR_BRA,
    }
    minimal = build_rys_axis_program((RysState(1, 1, 1, 0),))
    assert minimal.instructions[-1].state == RysState(1, 1, 1, 0)

    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=program.spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=12,
    )
    plan = build_fused_shell_plan(
        program.spec, schedule=schedule, recurrence="rys3", target=TEST_CUDA_TARGET
    )
    assert plan.kernel.integral.recurrence == "rys3"
    with pytest.raises(ValueError, match="requires rys4"):
        build_fused_shell_plan(DPPP_SPEC, recurrence="rys3", target=TEST_CUDA_TARGET)


def test_dddd_rys_program_exposes_five_root_backend_requirements() -> None:
    """Quantify the high-order state surface without emitting scalar algebra."""

    program = build_rys_force_program(DDDD_SPEC)
    assert program.nroots == 5
    assert len(program.component_order) == 1296
    assert len(program.axis_program.requested_states) == 162
    assert len(program.axis_program.instructions) == 216


def test_dddp_rys5_recurrence_matches_every_symbolic_component() -> None:
    """Lock the first promoted five-root class against symbolic lowering."""

    spec = FUSED_SHELL_SPEC_BY_NAME["dddp"]
    values = factored_dppp_variables(sample_variables())
    for component in spec.components:
        actual = evaluate_rys_component(spec, component, values)
        expected = evaluate_fused_shell_observables(spec, component, values)
        assert actual.value == pytest.approx(expected.value, rel=8.0e-13, abs=8.0e-13)
        for center in range(4):
            for axis in range(3):
                assert actual.gradients[center][axis] == pytest.approx(
                    expected.gradients[center][axis],
                    rel=2.0e-12,
                    abs=2.0e-12,
                )


def test_dddd_rys5_recurrence_matches_representative_symbolic_components() -> None:
    """Cover every Cartesian axis pattern without a 1296-case duplicate gate."""

    spec = DDDD_SPEC
    values = factored_dppp_variables(sample_variables())
    for component_index in (0, 1, 17, 215, 647, 648, 1024, 1295):
        component = spec.components[component_index]
        actual = evaluate_rys_component(spec, component, values)
        expected = evaluate_fused_shell_observables(spec, component, values)
        assert actual.value == pytest.approx(expected.value, rel=8.0e-13, abs=8.0e-13)
        for center in range(4):
            for axis in range(3):
                assert actual.gradients[center][axis] == pytest.approx(
                    expected.gradients[center][axis],
                    rel=2.0e-12,
                    abs=2.0e-12,
                )


def test_dppp_rys_program_bounds_four_root_state_groups() -> None:
    """Expose the exact DPPP Rys4 surface before production integration."""

    program = build_rys_force_program(DPPP_SPEC)
    assert program.nroots == 4
    assert len(program.component_order) == 162
    assert len(program.axis_program.requested_states) == 56
    assert len(program.axis_program.instructions) == 67
    body = emit_rys_force_root_body_cuda(DPPP_SPEC, component_group=3)
    assert body.count("const double component_density_weight") == 162
    assert body.count("double rys_state_") == 1375
    assert "boys_" not in body
    assert "component_gradient" not in body


@pytest.mark.parametrize("name", ("psss", "psps", "ppss", "dsss"))
def test_low_order_shells_share_scalar_rys2_force_backend(name: str) -> None:
    """Emit each two-root shell with one complete quartet per CUDA lane."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=8,
    )
    plan = build_fused_shell_plan(
        spec, schedule=schedule, recurrence="rys2", target=TEST_CUDA_TARGET
    )
    source = emit_shell_class_fused_cuda(spec, plan)
    assert f"generated_{name}_rys2_force_task" in source
    assert f"generated_{name}_rys2_roots" in source
    assert f"component_weights[kGenerated{name.title()}ComponentCount][32]" in source
    assert "root_index < 2U" in source


def test_ppps_rys_recurrence_matches_every_symbolic_component() -> None:
    """Lock component order, force signs, and translation recovery."""

    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    values = factored_dppp_variables(sample_variables())
    for component in spec.components:
        actual = evaluate_ppps_rys_component(component, values)
        expected = evaluate_fused_shell_observables(spec, component, values)
        assert actual.value == pytest.approx(expected.value, rel=3.0e-13, abs=3.0e-13)
        for center in range(4):
            for axis in range(3):
                assert actual.gradients[center][axis] == pytest.approx(
                    expected.gradients[center][axis],
                    rel=8.0e-13,
                    abs=8.0e-13,
                )


@pytest.mark.parametrize("name", ("dppp", "dpdp", "dpds", "ddpp", "ddps", "ddds"))
def test_cooperative_rys4_recurrence_matches_every_symbolic_component(
    name: str,
) -> None:
    """Lock each promoted four-root recurrence against symbolic lowering."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    values = factored_dppp_variables(sample_variables())
    for component in spec.components:
        actual = evaluate_rys_component(spec, component, values)
        expected = evaluate_fused_shell_observables(spec, component, values)
        assert actual.value == pytest.approx(expected.value, rel=6.0e-13, abs=6.0e-13)
        for center in range(4):
            for axis in range(3):
                assert actual.gradients[center][axis] == pytest.approx(
                    expected.gradients[center][axis],
                    rel=1.5e-12,
                    abs=1.5e-12,
                )


@pytest.mark.parametrize(
    "name", ("dpps", "dpss", "dsps", "dspp", "dsds", "ddss", "pppp")
)
def test_cooperative_rys3_recurrence_matches_every_symbolic_component(
    name: str,
) -> None:
    """Lock each promoted three-root recurrence against symbolic lowering."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    values = factored_dppp_variables(sample_variables())
    for component in spec.components:
        actual = evaluate_rys_component(spec, component, values)
        expected = evaluate_fused_shell_observables(spec, component, values)
        assert actual.value == pytest.approx(expected.value, rel=8.0e-13, abs=8.0e-13)
        for center in range(4):
            for axis in range(3):
                assert actual.gradients[center][axis] == pytest.approx(
                    expected.gradients[center][axis],
                    rel=2.0e-12,
                    abs=2.0e-12,
                )


def test_dppp_cooperative_rys4_uses_uniform_runtime_indexed_axis_recurrence() -> None:
    """Prevent regression to a divergent 162-way component dispatcher."""

    spec = FUSED_SHELL_SPEC_BY_NAME["dppp"]
    schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=192,
        component_tile=spec.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.MATERIALIZED,
        unroll_pair_terms=True,
        minimum_blocks_per_sm=2,
    )
    plan = build_fused_shell_plan(
        spec, schedule=schedule, recurrence="rys4", target=TEST_CUDA_TARGET
    )
    source = emit_shell_class_fused_cuda(spec, plan)
    assert "generated_dppp_rys4_component_lane_task" in source
    assert "volatile double trr[5][4]" in source
    assert "generated_dppp_rys4_axis" in source
    assert "switch (component)" not in source
    assert "generated_dppp_rys4_fill_weights" not in source
    assert "component_weights[kGeneratedDpppComponentCount][32]" not in source
    assert "GeneratedDpppPrimitiveGeometry primitive" not in source
    assert "generated_dppp_rys4_roots" in source


def test_dppp_rys4_uniform_warps_advance_32_quartets_per_block() -> None:
    """Keep the 2111-style task and component coordinates explicit."""

    schedule = ScheduleIR(
        kind=ScheduleKind.SUBGROUP_TASKS,
        block_threads=256,
        component_tile=DPPP_SPEC.component_count,
        tasks_per_warp=4,
        shared_coulomb=True,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.MATERIALIZED,
        unroll_pair_terms=True,
        minimum_blocks_per_sm=1,
    )
    plan = build_fused_shell_plan(
        DPPP_SPEC, schedule=schedule, recurrence="rys4", target=TEST_CUDA_TARGET
    )
    source = emit_shell_class_fused_cuda(DPPP_SPEC, plan)
    assert schedule.tasks_per_block == 32
    assert schedule.subgroup_lanes == 8
    assert "kGeneratedDpppRys4TaskCount = 32U" in source
    assert "kGeneratedDpppRys4ComponentLanes = 8U" in source
    assert "const unsigned sq = thread & 31U" in source
    assert "const unsigned component_lane = thread >> 5U" in source
    assert "atomicAdd(task_head, kGeneratedDpppRys4TaskCount)" in source
    assert "generated_dppp_rys4_uniform_warp_roots" in source
    assert "switch (component_lane)" in source
    assert "generated_dppp_subgroup_force_task" not in source
    assert "generated_dppp_rys4_component_lane_task" not in source

    mixed_plan = build_fused_shell_plan(
        DPPP_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
        recurrence="rys4",
        target=TEST_CUDA_TARGET,
    )
    mixed_source = emit_shell_class_fused_cuda(DPPP_SPEC, mixed_plan)
    assert "kGeneratedDpppBlockThreads = 256U" in mixed_source
    assert "kGeneratedDpppFockBlockThreads = 192U" in mixed_source
    assert "GeneratedDpppSubgroupFockStorage" not in mixed_source


@pytest.mark.parametrize("name", ("dddp", "dddd"))
def test_high_order_rys5_uniform_warps_advance_32_quartets_per_block(
    name: str,
) -> None:
    """Keep each five-root task/component mapping explicit."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    schedule = ScheduleIR(
        kind=ScheduleKind.SUBGROUP_TASKS,
        block_threads=256,
        component_tile=spec.component_count,
        tasks_per_warp=4,
        shared_coulomb=True,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.MATERIALIZED,
        unroll_pair_terms=True,
        minimum_blocks_per_sm=1,
    )
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FORCE,),
        schedule=schedule,
        recurrence="rys5",
        target=TEST_CUDA_TARGET,
    )
    source = emit_shell_class_fused_cuda(spec, plan)
    class_name = name[0].upper() + name[1:]
    assert f"kGenerated{class_name}Rys5TaskCount = 32U" in source
    assert f"kGenerated{class_name}Rys5ComponentLanes = 8U" in source
    assert f"generated_{name}_rys5_uniform_warp_roots" in source
    assert "root_index < 5U" in source
    assert f"generated_{name}_subgroup_force_task" not in source


@pytest.mark.parametrize(
    ("name", "fock_block_threads"),
    (("dpps", 64), ("dspp", 64), ("pppp", 96)),
)
def test_rys3_uniform_warps_split_components_without_scalar_spills(
    name: str, fock_block_threads: int
) -> None:
    """Reuse the 32-task geometry when one Rys3 thread owns too much state."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    schedule = ScheduleIR(
        kind=ScheduleKind.SUBGROUP_TASKS,
        block_threads=256,
        component_tile=spec.component_count,
        tasks_per_warp=4,
        shared_coulomb=True,
        minimum_blocks_per_sm=1,
    )
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
        recurrence="rys3",
        target=TEST_CUDA_TARGET,
    )
    source = emit_shell_class_fused_cuda(spec, plan)
    class_name = name[0].upper() + name[1:]
    assert f"kGenerated{class_name}Rys3TaskCount = 32U" in source
    assert f"kGenerated{class_name}Rys3ComponentLanes = 8U" in source
    assert f"generated_{name}_rys3_uniform_warp_roots" in source
    assert "root_index < 3U" in source
    assert f"kGenerated{class_name}FockBlockThreads = {fock_block_threads}U" in source
    assert f"generated_{name}_rys3_force_task" not in source
    assert f"generated_{name}_subgroup_force_task" not in source


@pytest.mark.parametrize(
    ("name", "block_threads", "trr_shape"),
    (
        ("dpps", 64, "volatile double trr[5][3]"),
        ("dsps", 32, "volatile double trr[4][3]"),
        ("dsds", 64, "volatile double trr[4][4]"),
        ("ddss", 64, "volatile double trr[6][2]"),
        ("pppp", 96, "volatile double trr[4][4]"),
    ),
)
def test_cooperative_rys3_hot_classes_use_uniform_component_lanes(
    name: str,
    block_threads: int,
    trr_shape: str,
) -> None:
    """Promote measured Rys3 hotspots without changing their direct Fock."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=block_threads,
        component_tile=spec.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
    )
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
        recurrence="rys3",
        target=TEST_CUDA_TARGET,
    )
    source = emit_shell_class_fused_cuda(spec, plan)
    assert f"generated_{name}_rys3_component_lane_task" in source
    assert f"generated_{name}_rys3_roots" in source
    assert trr_shape in source
    assert "switch (component)" not in source
    assert f"generated_{name}_shell_class_fock_rhf_kernel" in source


@pytest.mark.parametrize(
    ("name", "recurrence", "block_threads"),
    (("dpss", "rys3", 32), ("ddss", "rys3", 64)),
)
def test_component_lane_rys_fock_lowering_uses_structural_capabilities(
    name: str, recurrence: str, block_threads: int
) -> None:
    """Use the fixed-root Fock worker for legal classes beyond the old list."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=block_threads,
        component_tile=spec.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
        minimum_blocks_per_sm=1,
    )
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
        recurrence=recurrence,
        target=TEST_CUDA_TARGET,
    )
    assert supports_component_lane_rys(spec, schedule)
    source = emit_shell_class_fused_cuda(spec, plan)

    assert f"generated_{name}_rys3_value_axis" in source
    assert f"generated_{name}_shell_class_fock_rhf_kernel" in source
    fock_marker = f"generated_{name}_shell_class_fock_task("
    fock_source = source[source.index(fock_marker) :]
    assert f"generated_{name}_component_value" not in fock_source


def test_weighted_psss_graph_cse_matches_component_oracle() -> None:
    """Combine density-weighted components before CUDA primitive traversal."""

    weights = (0.7, -0.2, 1.1)
    variables = factored_dppp_variables(sample_variables())
    variables.update(
        {
            f"component_weight_{component}": weight
            for component, weight in enumerate(weights)
        }
    )
    weighted = build_weighted_shell_contraction_kernel(PSSS_SPEC)
    individual_node_count = sum(
        len(build_shell_class_contraction_kernel(PSSS_SPEC, component).graph.nodes)
        for component in PSSS_SPEC.components
    )
    assert len(weighted.graph.nodes) < individual_node_count
    expected_value = sum(
        weight * evaluate_fused_shell_value(PSSS_SPEC, component, variables)
        for weight, component in zip(weights, PSSS_SPEC.components, strict=True)
    )
    assert weighted.graph.evaluate(weighted.value, variables) == pytest.approx(
        expected_value,
        rel=2.0e-12,
        abs=2.0e-12,
    )
    for center in range(4):
        for axis in range(3):
            expected = sum(
                weight
                * evaluate_fused_shell_component(
                    PSSS_SPEC,
                    component,
                    variables,
                )[center][axis]
                for weight, component in zip(
                    weights,
                    PSSS_SPEC.components,
                    strict=True,
                )
            )
            assert weighted.graph.evaluate(
                weighted.gradients[center][axis],
                variables,
            ) == pytest.approx(expected, rel=3.0e-12, abs=3.0e-12)


def assert_rtx5090_resources(
    ptxas_output: str,
    limits: dict[str, tuple[int, int, int]],
) -> None:
    """Reject CUDA 12.9 resource regressions before production integration."""

    for function, (register_limit, stack_limit, shared_limit) in limits.items():
        match = re.search(
            rf"Function properties for {function}\n"
            r"\s+(\d+) bytes stack frame, (\d+) bytes spill stores, "
            r"(\d+) bytes spill loads\n"
            r"ptxas info\s+: Used (\d+) registers([^\n]*)",
            ptxas_output,
        )
        assert match is not None, f"missing ptxas resources for {function}"
        stack, spill_stores, spill_loads, registers = map(int, match.groups()[:4])
        shared_match = re.search(r"(\d+) bytes smem", match.group(5))
        shared = int(shared_match.group(1)) if shared_match is not None else 0
        assert registers <= register_limit
        assert stack <= stack_limit
        assert spill_stores == 0
        assert spill_loads == 0
        assert shared <= shared_limit


def test_shell_spec_generates_cca_components_and_compile_time_bounds() -> None:
    """Derive shell schedules without handwritten component tables."""

    assert cartesian_components(0) == ("",)
    assert cartesian_components(1) == AXES
    assert cartesian_components(2) == ("xx", "xy", "xz", "yy", "yz", "zz")
    assert cartesian_components(3) == (
        "xxx",
        "xxy",
        "xxz",
        "xyy",
        "xyz",
        "xzz",
        "yyy",
        "yyz",
        "yzz",
        "zzz",
    )
    assert DPPP_SPEC.pair_orders == (3, 2)
    assert DPDS_SPEC.pair_orders == (3, 2)
    assert DDPS_SPEC.pair_orders == (4, 1)
    assert DPPP_SPEC.maximum_force_coulomb_order == 6
    assert DPPP_SPEC.component_count == 162
    assert DPPP_SPEC.component_strides == (27, 9, 3, 1)


def test_large_dddd_class_defaults_to_tiled_lowering() -> None:
    """Keep AO products beyond CUDA's block limit in the generated catalog."""

    assert DDDD_SPEC.component_count == 1296
    assert DDDD_SPEC.pair_orders == (4, 4)
    integral = build_integral_ir(DDDD_SPEC)
    candidates = schedule_candidates(integral, target=TEST_CUDA_TARGET)
    # Search now exposes all target-legal mappings; the production default
    # below must still use the qualified 64-component tile.
    tiled = [item for item in candidates if item.kind == ScheduleKind.TILED_COMPONENTS]
    limit = min(
        TEST_CUDA_TARGET.maximum_threads_per_block,
        TEST_CUDA_TARGET.maximum_threads_per_sm,
    )
    expected_tiles = [
        TEST_CUDA_TARGET.warp_size * 2**power
        for power in range(1, limit.bit_length())
        if TEST_CUDA_TARGET.warp_size * 2**power <= limit
        and TEST_CUDA_TARGET.warp_size * 2**power < DDDD_SPEC.component_count
    ]
    assert [item.component_tile for item in tiled] == expected_tiles
    assert {item.kind for item in candidates} == {
        ScheduleKind.PACKED_TASKS,
        ScheduleKind.SHELL_TASK,
        ScheduleKind.SUBGROUP_TASKS,
        ScheduleKind.TILED_COMPONENTS,
    }
    plan = build_fused_shell_plan(DDDD_SPEC, target=TEST_CUDA_TARGET)
    assert plan.schedule.kind == ScheduleKind.TILED_COMPONENTS
    assert plan.block_threads == 64
    assert len(plan.coulomb_states) == 220
    source = emit_shell_class_fused_cuda(DDDD_SPEC, plan)
    assert "kGeneratedDdddComponentCount = 1296U" in source
    assert "__constant__ short generated_dddd_coulomb_indices[1000]" in source
    assert "state >> 4U" in source
    assert "state >> 8U" in source
    assert "component_tile_begin += 64U" in source

    trials = supported_schedule_trials(DDDD_SPEC, target=TEST_CUDA_TARGET)
    assert len({trial.schedule_id for trial in trials}) == len(trials)
    tiled_trials = [
        trial
        for trial in trials
        if trial.schedule.kind == ScheduleKind.TILED_COMPONENTS
    ]
    assert (
        len(tiled_trials)
        == len(expected_tiles) * len(PairStorage) * len(PairOrientation) * 2
    )
    assert {
        (
            trial.schedule.component_tile,
            trial.schedule.pair_storage,
            trial.schedule.pair_orientation,
            trial.schedule.unroll_pair_terms,
        )
        for trial in tiled_trials
    } == {
        (tile, storage, orientation, unrolled)
        for tile in expected_tiles
        for storage in PairStorage
        for orientation in PairOrientation
        for unrolled in (True, False)
    }


def test_f_shell_cuda_lowering_emits_axes_triple_matchings_and_tiles() -> None:
    """Cover pair order six and a component product above the block limit."""

    ffps_source = emit_shell_class_fused_cuda(FFPS_SPEC, target=TEST_CUDA_TARGET)
    assert "generated_ffps_f_axes[10][3]" in ffps_source
    assert "if constexpr (PairOrder >= 6U)" in ffps_source
    assert "first_removed | second_removed | third_removed, 3U" in ffps_source
    assert "__constant__ short generated_ffps_coulomb_indices[729]" in ffps_source

    fddd_plan = build_fused_shell_plan(FDDD_SPEC, target=TEST_CUDA_TARGET)
    assert fddd_plan.schedule.kind == ScheduleKind.TILED_COMPONENTS
    assert fddd_plan.block_threads == 64
    assert FDDD_SPEC.component_count == 2160
    fddd_source = emit_shell_class_fused_cuda(FDDD_SPEC, fddd_plan)
    assert "generated_fddd_f_axes[10][3]" in fddd_source
    assert "component_tile_begin += 64U" in fddd_source


def test_shell_spec_rejects_invalid_metadata_and_components() -> None:
    with pytest.raises(ValueError):
        ShellClassSpec("bad", (2, 1, 1))
    with pytest.raises(ValueError):
        ShellClassSpec("Bad", (2, 1, 1, 1))
    with pytest.raises(ValueError):
        DPPP_SPEC.validate_component(("xx", "x", "y", "xx"))
    with pytest.raises(IndexError):
        DPPP_SPEC.component_from_index(DPPP_SPEC.component_count)


@pytest.mark.parametrize(
    ("spec", "component"),
    (
        (DPDS_SPEC, ("xy", "z", "xz", "")),
        (DDPS_SPEC, ("xy", "xz", "z", "")),
    ),
)
def test_generic_shell_ad_matches_factored_lowering(
    spec: typing.Any, component: typing.Any
) -> None:
    """Exercise pair orders 3+2 and 4+1 without handwritten builders."""

    full = build_shell_class_component_kernel(spec, component)
    factored = build_shell_class_contraction_kernel(spec, component)
    full_values = sample_variables()
    argument = full.graph.evaluate(full.boys_argument, full_values)
    for order, value in enumerate(
        boys_values(argument, spec.maximum_force_coulomb_order + 1)
    ):
        full_values[f"boys_{order}"] = value
    factored_values = factored_dppp_variables(full_values)

    full_value = full.graph.evaluate(full.value, full_values)
    factored_value = factored_values["prefactor"] * factored.graph.evaluate(
        factored.value, factored_values
    )
    assert factored_value == pytest.approx(full_value, rel=5.0e-13, abs=5.0e-13)
    for center in range(4):
        for axis in range(3):
            actual = factored.graph.evaluate(
                factored.gradients[center][axis], factored_values
            )
            expected = full.graph.evaluate(full.gradients[center][axis], full_values)
            assert actual == pytest.approx(expected, rel=3.0e-11, abs=3.0e-11)


@pytest.mark.parametrize(
    ("d_component", "p_components"),
    (("xx", "xxx"), ("xy", "xyz"), ("zz", "zyx")),
)
def test_factored_dppp_lowering_matches_full_symbolic_kernel(
    d_component: str, p_components: str
) -> None:
    full = build_dppp_component_kernel(d_component, tuple(p_components))
    factored = build_dppp_contraction_kernel(d_component, tuple(p_components))
    full_values = sample_variables()
    argument = full.graph.evaluate(full.boys_argument, full_values)
    for order, value in enumerate(boys_values(argument, 7)):
        full_values[f"boys_{order}"] = value
    factored_values = factored_dppp_variables(full_values)

    full_value = full.graph.evaluate(full.value, full_values)
    factored_value = factored_values["prefactor"] * factored.graph.evaluate(
        factored.value, factored_values
    )
    assert factored_value == pytest.approx(full_value, rel=3.0e-13, abs=3.0e-13)
    for center in range(4):
        for axis in range(3):
            assert factored.graph.evaluate(
                factored.gradients[center][axis], factored_values
            ) == pytest.approx(
                full.graph.evaluate(full.gradients[center][axis], full_values),
                rel=2.0e-11,
                abs=2.0e-11,
            )


def test_dppp_fused_plan_covers_components_and_shared_coulomb_states() -> None:
    plan = build_dppp_fused_plan()
    components = dppp_components()
    assert plan.components == components
    assert len(components) == 162
    assert len(plan.coulomb_states) == 84
    assert len(plan.coulomb_indices) == 7**3
    assert plan.block_threads == 192
    assert plan.warp_count == 6
    for index, (x_order, y_order, z_order) in enumerate(plan.coulomb_states):
        dense_index = (x_order * 7 + y_order) * 7 + z_order
        assert plan.coulomb_indices[dense_index] == index
        assert x_order + y_order + z_order <= 6


@pytest.mark.parametrize("unrestricted", (False, True))
def test_closed_density_orbit_matches_unique_permutations(
    unrestricted: bool,
) -> None:
    """Prove the closed force coefficient for every AO equality pattern."""

    order = 4
    alpha = [
        [
            float((min(row, column) + 1) * 7 + max(row, column))
            for column in range(order)
        ]
        for row in range(order)
    ]
    beta = [
        [
            float((min(row, column) + 2) * 11 - max(row, column))
            for column in range(order)
        ]
        for row in range(order)
    ]

    for i, j, k, l in itertools.product(range(order), repeat=4):
        permutations = (
            (i, j, k, l),
            (j, i, k, l),
            (i, j, l, k),
            (j, i, l, k),
            (k, l, i, j),
            (l, k, i, j),
            (k, l, j, i),
            (l, k, j, i),
        )
        old = 0.0
        seen: set[tuple[int, int, int, int]] = set()
        for a, b, c, d in permutations:
            if (a, b, c, d) in seen:
                continue
            seen.add((a, b, c, d))
            if unrestricted:
                old += 0.5 * (alpha[a][b] + beta[a][b]) * (alpha[c][d] + beta[c][d])
                old -= 0.5 * (alpha[a][c] * alpha[b][d] + beta[a][c] * beta[b][d])
            else:
                old += (
                    0.5 * alpha[a][b] * alpha[c][d] - 0.25 * alpha[a][c] * alpha[b][d]
                )

        orbit_scale = 0.5 if i == j else 1.0
        if k == l:
            orbit_scale *= 0.5
        if (i == k and j == l) or (i == l and j == k):
            orbit_scale *= 0.5
        if unrestricted:
            closed = orbit_scale * (
                4.0 * (alpha[i][j] + beta[i][j]) * (alpha[k][l] + beta[k][l])
                - 2.0
                * (
                    alpha[i][k] * alpha[j][l]
                    + alpha[i][l] * alpha[j][k]
                    + beta[i][k] * beta[j][l]
                    + beta[i][l] * beta[j][k]
                )
            )
        else:
            closed = orbit_scale * (
                4.0 * alpha[i][j] * alpha[k][l]
                - alpha[i][k] * alpha[j][l]
                - alpha[i][l] * alpha[j][k]
            )
        assert closed == pytest.approx(old, abs=1.0e-12)


def test_equal_shell_pair_component_domain_matches_active_tile_triangle() -> None:
    """Avoid double-counting (ij|kl) and (kl|ij) in shell-wide workers."""

    source = emit_shell_class_fused_cuda(
        FUSED_SHELL_SPEC_BY_NAME["pppp"], target=TEST_CUDA_TARGET
    )
    assert (
        "shared.task.shell_pair[0] != shared.task.shell_pair[1] || "
        "(first_p * 3U + second_p) >= (third_p * 3U + fourth_p)" in source
    )


def test_fused_cuda_can_emit_fock_values_and_force_gradients_together() -> None:
    """Generate both consumers from one integral and component schedule IR."""

    plan = build_fused_shell_plan(
        DPDS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        target=TEST_CUDA_TARGET,
    )
    source = emit_shell_class_fused_cuda(DPDS_SPEC, plan)
    assert "generated_dpds_component_value" in source
    assert "generated_dpds_component_gradient" in source
    assert "generated_dpds_shell_class_fock_rhf_kernel" in source
    assert "generated_dpds_shell_class_fock_uhf_persistent_kernel" in source
    assert "generated_dpds_shell_class_force_rhf_kernel" in source
    force_only = emit_shell_class_fused_cuda(DPDS_SPEC, target=TEST_CUDA_TARGET)
    assert "shell_class_fock" not in force_only
    assert "coordinate_gradient" not in source
    assert "Dual3" not in source


def test_dpds_fused_cuda_is_generated_from_shell_spec() -> None:
    source = emit_shell_class_fused_cuda(DPDS_SPEC, target=TEST_CUDA_TARGET)
    assert "kGeneratedDpdsComponentCount = 108U" in source
    assert "kGeneratedDpdsBlockThreads = 128U" in source
    assert "const unsigned third_d = component % 6U" in source
    assert "const unsigned fourth_s = 0U" in source
    assert "generated_dpds_d_axes[third_d][1]" in source
    assert "GeneratedDpdsPairTerm second_terms[4]" in source
    assert "generated_dpds_shell_class_force_rhf_kernel" in source
    assert "generated_dpds_shell_class_force_uhf_persistent_kernel" in source
    assert "generated_dppp" not in source
    assert "__noinline__" not in source
    assert "Dual3" not in source


def test_pair_orientation_changes_the_materialized_contraction_pair() -> None:
    """Make pair orientation a measured CUDA code shape, not manifest metadata."""

    base = build_fused_shell_plan(
        DPDS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        target=TEST_CUDA_TARGET,
    ).schedule
    canonical = emit_shell_class_fused_cuda(
        DPDS_SPEC,
        build_fused_shell_plan(
            DPDS_SPEC,
            consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
            schedule=replace(base, pair_orientation=PairOrientation.CANONICAL),
            target=TEST_CUDA_TARGET,
        ),
    )
    swapped = emit_shell_class_fused_cuda(
        DPDS_SPEC,
        build_fused_shell_plan(
            DPDS_SPEC,
            consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
            schedule=replace(base, pair_orientation=PairOrientation.SWAPPED),
            target=TEST_CUDA_TARGET,
        ),
    )
    assert "GeneratedDpdsPairTerm second_terms[4]" in canonical
    assert "GeneratedDpdsValueTerm second_terms[4]" in canonical
    assert "GeneratedDpdsPairTerm first_terms[8]" in swapped
    assert "GeneratedDpdsValueTerm first_terms[8]" in swapped
    assert "GeneratedDpdsPairTerm first_terms[8]" not in canonical
    assert "GeneratedDpdsValueTerm first_terms[8]" not in canonical


def test_ddps_fused_cuda_generates_order_four_double_matchings() -> None:
    source = emit_shell_class_fused_cuda(DDPS_SPEC, target=TEST_CUDA_TARGET)
    assert "kGeneratedDdpsComponentCount = 108U" in source
    assert "kGeneratedDdpsBlockThreads = 128U" in source
    assert "PairOrder == 1U || PairOrder == 4U" in source
    assert "first_removed | second_removed, 2U" in source
    assert "first_d >= second_d" in source
    assert "GeneratedDdpsPairTerm second_terms[2]" in source
    assert "generated_ddps_shell_class_force_rhf_kernel" in source
    assert "generated_dppp" not in source
    assert "__noinline__" not in source
    assert "Dual3" not in source


def test_rys4_component_lanes_raise_a_second_center_d_shell() -> None:
    """Generate the exact b=3 HRR state needed by a d-center derivative."""

    schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=128,
        component_tile=DDPS_SPEC.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
        minimum_blocks_per_sm=1,
    )
    plan = build_fused_shell_plan(
        DDPS_SPEC, schedule=schedule, recurrence="rys4", target=TEST_CUDA_TARGET
    )
    source = emit_shell_class_fused_cuda(DDPS_SPEC, plan)
    assert "if (b == 2U)" in source
    assert "trr, a + 3U, c, d, cd" in source
    assert "3.0 * ab * raised_twice" in source
    assert "__noinline__ void\ngenerated_ddps_rys4_component_lane_task" in source
    assert "generated_ddps_shell_class_force_rhf_kernel" in source
