"""Production selection, runtime ownership, and generated-registry codegen tests.

Moved behavior-neutrally from the legacy codegen compatibility suite for #489.
"""

from __future__ import annotations

# ruff: noqa: F401  # Keep this move behavior-neutral; imports mirror legacy ownership.
from codegen_test_support import (
    CAPABILITY_MIXED_FOCK,
    CAPABILITY_STREAMING_FOCK,
    DPPP_SPEC,
    FUSED_SHELL_SPEC_BY_NAME,
    PSPS_SPEC,
    REPOSITORY_ROOT,
    TEST_CUDA_TARGET,
    AlgebraPlacement,
    ContractionSpec,
    KernelConsumer,
    OperatorFamily,
    OperatorSpec,
    PairOrientation,
    PairStorage,
    Path,
    ScheduleIR,
    ScheduleKind,
    TranslationInvariant,
    _direct_cuda_source,
    _partition_production_selections,
    boys_values,
    build_capability_report,
    build_fused_shell_plan,
    build_integral_ir,
    emit_low_order_weighted_header,
    emit_registry_header,
    emit_registry_source,
    emit_shell_class_fused_cuda,
    json,
    load_production_fock_manifest,
    load_production_kernel_selections,
    load_production_manifest,
    math,
    pytest,
    re,
    replace,
    schedule_candidates,
    subprocess,
    typing,
    write_production_bundle,
)


@pytest.mark.parametrize("name", ("psps", "ppss"))
def test_low_order_production_force_is_generated_by_common_rys2_pipeline(
    name: str,
) -> None:
    """Keep low-order production ownership in the shared IR and CUDA emitter."""

    selection = next(
        item
        for item in load_production_kernel_selections(
            REPOSITORY_ROOT
            / "python"
            / "generativeqc_compiler"
            / "integral"
            / "production_shell_classes.json",
            "sm_120",
        )
        if item.spec.name == name
    )
    plan = build_fused_shell_plan(
        selection.spec,
        consumers=selection.consumers,
        schedule=selection.schedule,
        recurrence=selection.recurrence,
        target=TEST_CUDA_TARGET,
    )
    source = emit_shell_class_fused_cuda(
        selection.spec,
        plan,
        fock_schedule=selection.fock_schedule,
    )
    assert selection.recurrence == "rys2"
    assert selection.schedule.kind == ScheduleKind.THREAD_TASKS
    assert selection.schedule.block_threads == 32
    assert f"generated_{name}_rys2_force_task" in source
    assert f"generated_{name}_shell_class_force_rhf_persistent_kernel" in source
    assert f"generated_{name}_shell_class_force_uhf_persistent_kernel" in source
    assert f"generated_{name}_shell_class_fock_rhf_persistent_kernel" in source
    assert "atomicAdd(task_head, 32U)" in source
    assert "GENERATIVEQC_LOW_ORDER_TASK_BEGIN" not in source
    assert f"generated_{name}_contract_weighted_coulomb" not in source
    assert "Dual3" not in source


def test_packed_force_geometry_omits_component_coulomb_tables() -> None:
    """Keep packed-force shared storage limited to fields its CSE consumes."""

    source = emit_shell_class_fused_cuda(
        PSPS_SPEC,
        build_fused_shell_plan(
            PSPS_SPEC,
            consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
            schedule=ScheduleIR(
                kind=ScheduleKind.PACKED_TASKS,
                block_threads=32,
                component_tile=PSPS_SPEC.component_count,
                tasks_per_warp=32,
                shared_coulomb=False,
            ),
            target=TEST_CUDA_TARGET,
        ),
    )
    force_geometry = source.split(
        "struct GeneratedPspsPackedForceGeometry", maxsplit=1
    )[1].split("};", maxsplit=1)[0]
    assert "coordinate_powers" not in force_geometry
    assert "negative_two_rho_powers" not in force_geometry
    assert "pair_shifts[3][3]" in force_geometry
    assert (
        "pair_shifts[3][axis]"
        not in source.split("generated_psps_make_packed_force_geometry", maxsplit=1)[
            1
        ].split("/** Density-weighted shell gradient", maxsplit=1)[0]
    )
    assert "GeneratedPspsPackedForceLaneStorage" in source
    assert "GeneratedPspsPackedFockLaneStorage" in source


@pytest.mark.parametrize(
    ("spec", "pair_shift_rows"),
    ((PSPS_SPEC, 3), (DPPP_SPEC, 4)),
)
def test_packed_force_geometry_cuda_is_lowered_from_backend_neutral_algebra(
    spec: typing.Any, pair_shift_rows: typing.Any
) -> None:
    """Keep packed geometry setup derived from the shared scalar IR."""

    source = emit_shell_class_fused_cuda(
        spec,
        build_fused_shell_plan(
            spec,
            consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
            schedule=ScheduleIR(
                kind=ScheduleKind.PACKED_TASKS,
                block_threads=32,
                component_tile=spec.component_count,
                tasks_per_warp=32,
                shared_coulomb=False,
            ),
            target=TEST_CUDA_TARGET,
        ),
    )
    setup = source.split(
        f"generated_{spec.name}_make_packed_force_geometry", maxsplit=1
    )[1].split("/** Density-weighted shell gradient", maxsplit=1)[0]
    assert f"pair_shifts[{pair_shift_rows}][3]" in source
    assert "generated_dppp_axis(" not in setup
    assert "argument_squared_distance +=" not in setup
    assert "geometry.pair_shifts[0][0] =" in setup
    assert "geometry.decay_gradients[2][2] =" in setup
    assert "geometry.primitive_coefficient =" in setup
    assert f"boys_values<{spec.maximum_force_coulomb_order}>" in setup
    assert "sqrt(" in setup


def test_packed_force_lowering_uses_explicit_derivative_center_slots() -> None:
    """Route packed force atomics through non-final IR recovery metadata."""

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
        PSPS_SPEC,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(force,),
    )
    plan = build_fused_shell_plan(
        PSPS_SPEC,
        integral=integral,
        schedule=ScheduleIR(
            kind=ScheduleKind.PACKED_TASKS,
            block_threads=32,
            component_tile=PSPS_SPEC.component_count,
            tasks_per_warp=32,
            shared_coulomb=False,
        ),
        target=TEST_CUDA_TARGET,
    )
    source = emit_shell_class_fused_cuda(PSPS_SPEC, plan)

    # Independent slots are A/C/D, while the recovered force is accumulated
    # into B.  Differentiating center D also requires retaining its decay row.
    assert "decay_gradients[4][3]" in source
    assert "geometry.decay_gradients[3][2]" in source
    assert "0U, 2U, 3U};" in source
    recovery_begin = source.index(
        "const double fourth_force",
        source.index("generated_psps_packed_force_lane"),
    )
    recovery = source[recovery_begin : recovery_begin + 600]
    assert "static_cast<std::size_t>(task.atom[1])" in recovery
    assert "static_cast<std::size_t>(task.atom[3])" not in recovery


def test_explicit_component_lane_fock_width_reaches_streaming_wrapper() -> None:
    """Keep a wider tuned ppps Fock CTA consistent across generated entry points."""

    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    force_schedule = ScheduleIR(
        kind=ScheduleKind.SUBGROUP_TASKS,
        block_threads=256,
        component_tile=spec.component_count,
        tasks_per_warp=4,
        shared_coulomb=True,
    )
    fock_schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=64,
        component_tile=spec.component_count,
        shared_coulomb=True,
    )
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=force_schedule,
        recurrence="rys3",
        target=TEST_CUDA_TARGET,
    )
    source = emit_shell_class_fused_cuda(
        spec,
        plan,
        fock_schedule=fock_schedule,
        capabilities=("streaming_fock",),
    )
    assert "kGeneratedPppsFockBlockThreads = 64U" in source


def test_high_impact_fock_classes_emit_generated_mixed_capability() -> None:
    """Keep the profiled FP32 AOT set explicit and independently routed."""

    sources = {}
    for name in ("ppps", "dpps", "ddds", "dspp"):
        spec = FUSED_SHELL_SPEC_BY_NAME[name]
        plan = build_fused_shell_plan(
            spec,
            consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
            target=TEST_CUDA_TARGET,
        )
        sources[name] = emit_shell_class_fused_cuda(
            spec,
            plan,
            capabilities=(CAPABILITY_MIXED_FOCK,) if name != "dspp" else (),
        )

    dpps = sources["dpps"]
    assert "generated_dpps_shell_class_mixed_fock_rhf_persistent_kernel" in dpps
    assert "generated_dpps_shell_class_mixed_fock_uhf_persistent_kernel" in dpps
    assert "kGeneratedDppsMixedFockBlockThreads" in dpps
    assert "struct GeneratedDppsMixedValueTerm" in dpps
    assert "  float component_integral = 0.0F;" in dpps
    assert "const double* density" in dpps
    assert "double* fock" in dpps
    assert "generated_ppps_shell_class_mixed_fock" in sources["ppps"]
    assert "generated_ddds_shell_class_mixed_fock" in sources["ddds"]
    assert "generated_dspp_shell_class_mixed_fock" not in sources["dspp"]


def test_mixed_fock_recomputed_coulomb_scratch_uses_fp32() -> None:
    """Compile the mixed path when a Fock schedule recomputes Coulomb state."""

    spec = FUSED_SHELL_SPEC_BY_NAME["ddds"]
    force_schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=224,
        component_tile=spec.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.RECOMPUTED,
        unroll_pair_terms=False,
        minimum_blocks_per_sm=1,
    )
    fock_schedule = replace(
        force_schedule,
        shared_coulomb=False,
        pair_storage=PairStorage.MATERIALIZED,
        unroll_pair_terms=True,
        minimum_blocks_per_sm=0,
    )
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=force_schedule,
        recurrence="rys4",
        target=TEST_CUDA_TARGET,
    )
    source = emit_shell_class_fused_cuda(
        spec,
        plan,
        fock_schedule=fock_schedule,
        capabilities=(CAPABILITY_MIXED_FOCK,),
    )
    mixed = source.split("struct GeneratedDddsMixedPrimitiveGeometry", maxsplit=1)[1]
    assert "float coulomb[1];" in mixed
    assert "double coulomb[1];" not in mixed


def test_simple_registry_dispatches_profiled_mixed_fock_classes() -> None:
    """Expose a selectable capability mask for the profiled mixed workers."""

    manifest = (
        REPOSITORY_ROOT
        / "python"
        / "generativeqc_compiler"
        / "integral"
        / "production_shell_classes.json"
    )
    selections = load_production_kernel_selections(manifest, "sm_120")
    header = emit_registry_header(selections)
    source = emit_registry_source(selections)

    assert "kMixedFockShellKernels" in header
    mixed_rows = header.split("kMixedFockShellKernels", maxsplit=1)[1].split(
        "}};", maxsplit=1
    )[0]
    expected = {
        "ppps",
        "dpps",
        "dsps",
        "dsds",
        "ddss",
        "ddps",
        "ddds",
        "pppp",
    }
    for name in expected:
        assert f'"{name}"' in mixed_rows
        assert f"generativeqc_launch_generated_{name}_mixed_fock" in source
    assert '"dspp"' not in mixed_rows
    assert "enabled_mixed_fock_shell_class_mask" in header
    assert "launch_shell_class_mixed_fock" in header
    assert "GENERATIVEQC_AOT_MIXED_FOCK_SHELL_CLASSES" in source
    assert "generativeqc_launch_generated_dspp_mixed_fock" not in source


def test_direct_tile_validation_is_opt_in_and_reports_descriptor_context() -> None:
    """Keep the large-AO queue validator diagnostic-only and actionable."""

    source = _direct_cuda_source()
    policy = (REPOSITORY_ROOT / "src" / "scf" / "cuda" / "rhf_policy.cpp").read_text(
        encoding="utf-8"
    )
    assert '"GENERATIVEQC_DIRECT_TILE_VALIDATION"' in policy
    assert "validate_direct_tile_descriptors_kernel" in source
    assert "DirectTileValidationRecord" in source
    assert "direct-tile-validation error=" in source
    # The validator must stop before a consumer can turn a bad descriptor into
    # a secondary illegal access; normal runs never enter this branch.
    assert "if (direct_tile_validation &&" in source
    assert "return cudaSuccess;" in source


def test_graph_native_eigensolver_override_covers_all_solver_calls() -> None:
    """Keep the large-matrix escape hatch off cuSOLVER in every phase."""

    source = _direct_cuda_source()
    override_begin = source.index("if (requested_graph_native_eigensolver_override) {")
    override_end = source.index(
        "    }\n  }\n  const CudaEigensolverFamily", override_begin
    )
    override = source[override_begin:override_end]
    assert "plan.eigensolver_diagnostic.family =" in override
    assert "plan.eigensolver_diagnostic.ordinary_family =" in override
    assert "CudaEigensolverFamily::graph_native" in override
    probe_call = source.index(
        "probe_xsyev_batched_device_launch_graph(", override_begin
    )
    assert probe_call > override_begin
    assert (
        "Do not probe that provider first"
        in source[override_begin - 300 : override_begin]
    )
    # Finalization and split ordinary-stream iterations use ordinary_family;
    # an override that changes only family silently reintroduces XsyevBatched.
    assert (
        "ordinary_eigensolver_family == CudaEigensolverFamily::xsyev_batched" in source
    )


def test_large_matrix_stream_fallback_matches_gpu4pyscf_solver_contract() -> None:
    """Keep Graph-rejected large matrices on the standard Xsyevd provider."""

    source = _direct_cuda_source()
    assert "CudaEigensolverFamily::xsyevd" in source
    assert "cusolverDnXsyevd_bufferSize" in source
    eigensolver = (REPOSITORY_ROOT / "src/scf/cuda/eigensolver.cpp").read_text()
    assert "cusolverDnXsyevd(" in eigensolver
    probe_end = source.index(
        "const XsyevBatchedDispatch dispatch =",
        source.index("probe_xsyev_batched_device_launch_graph("),
    )
    assert "cudaGetLastError" in source[probe_end - 320 : probe_end]
    assert "dispatch.device_launch_graph_provider" in source
    assert "plan.eigensolver_diagnostic.ordinary_family =" in source
    assert "CudaEigensolverFamily::xsyevd" in source
    assert "Unlike XsyevBatched" in eigensolver


def test_bounded_force_registry_gaps_use_exact_runtime_fallback() -> None:
    """Prevent large-AO force runs from regressing to a hard CUDA error."""

    source = _direct_cuda_source()
    fallback = source.index("const auto launch_bounded_generic_force")
    dispatch = source.index("const auto launch_bounded_force")
    dispatch_boundary = re.search(
        r"if \((?:options\.compute_forces &&\s+)?quartet_direct &&\s+plan\.shell_quartet_tile_capacities",
        source[dispatch:],
    )
    assert dispatch_boundary is not None
    dispatch_end = dispatch + dispatch_boundary.start()
    assert fallback < dispatch < dispatch_end
    assert "bounded_direct_shell_quartet_kernel" in source[fallback:dispatch]
    assert "uncovered_force_shell_class_mask == 0U" in source[fallback:dispatch]
    assert "launch_bounded_generic_force" in source[dispatch:dispatch_end]
    # Registry incompleteness must not be converted into the old hard failure.
    assert "return cudaErrorNotSupported;" not in source[dispatch:dispatch_end]


def test_bounded_fock_registry_gaps_use_exact_runtime_fallback() -> None:
    """Keep high-l bounded Fock correct without an unbounded descriptor arena."""

    source = _direct_cuda_source()
    fallback = source.index("const auto launch_bounded_generic_fock")
    dispatch = source.index("const auto launch_bounded_generated_fock", fallback)
    dispatch_end = source.index(
        "// The exact provider is resolved/validated by run_hf_cuda_bucket_cached.",
        dispatch,
    )
    assert fallback < dispatch < dispatch_end
    assert "host_uncovered_fock_shell_class_mask == 0U" in source[fallback:dispatch]
    assert (
        "launch_bounded_direct_fock_shell_quartet_kernel" in source[fallback:dispatch]
    )
    assert "host_generated_fock_shell_class_mask" in source[fallback:dispatch]
    assert "launch_bounded_generic_fock" in source[dispatch:dispatch_end]
    assert "return cudaErrorNotSupported;" not in source[dispatch:dispatch_end]

    fallback_source = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_bounded_fallback.cu"
    ).read_text()
    fock_wrapper = fallback_source.index(
        "void launch_bounded_direct_fock_shell_quartet_kernel("
    )
    assert (
        "bounded_direct_shell_quartet_kernel<true, DirectScreeningPurpose::Fock, false>"
        in fallback_source[fock_wrapper:]
    )
    assert (
        "bounded_direct_shell_quartet_kernel<false, DirectScreeningPurpose::Fock, false>"
        in fallback_source[fock_wrapper:]
    )
    # The method-neutral force fallback may use Fock screening while still writing forces.
    # Do not conflate screening purpose with the scientific consumer again.
    force_wrapper = fallback_source.index(
        "void launch_bounded_direct_shell_quartet_kernel_scaled("
    )
    assert (
        "bounded_direct_shell_quartet_kernel<true, DirectScreeningPurpose::Fock, true>"
        in fallback_source[force_wrapper:fock_wrapper]
    )


def test_production_manifest_drives_generated_registry_and_shards(
    tmp_path: Path,
) -> None:
    """Keep machine CUDA out of Git while retaining deterministic builds."""

    manifest = (
        REPOSITORY_ROOT
        / "python"
        / "generativeqc_compiler"
        / "integral"
        / "production_shell_classes.json"
    )
    specifications = load_production_manifest(manifest)
    fock_specifications = load_production_fock_manifest(manifest)
    assert tuple(spec.name for spec in specifications) == (
        "ssss",
        "dppp",
        "dpdp",
        "dddp",
        "dddd",
        "dpss",
        "dsds",
        "ddss",
        "ddpp",
        "ddds",
        "dpds",
        "ddps",
        "fpps",
        "ppps",
        "dpps",
        "dsps",
        "dspp",
        "pppp",
        "psps",
        "ppss",
        "dsss",
    )
    assert tuple(spec.name for spec in fock_specifications) == (
        "ssss",
        "psss",
        "dppp",
        "dpdp",
        "dddp",
        "dddd",
        "dpss",
        "dsds",
        "ddss",
        "ddpp",
        "ddds",
        "dpds",
        "ddps",
        "ppps",
        "dpps",
        "dsps",
        "dspp",
        "pppp",
        "psps",
        "ppss",
        "dsss",
    )
    selections = load_production_kernel_selections(manifest, "sm_120")
    assert tuple(
        selection.spec.name
        for selection in selections
        if KernelConsumer.FORCE in selection.consumers
    ) == tuple(spec.name for spec in specifications)
    assert all(selection.architecture == "sm_120" for selection in selections)
    assert all(
        selection.schedule.algebra_placement == AlgebraPlacement.MATERIALIZED_CSE
        for selection in selections
    )
    canonical_spd = {
        "ssss",
        "psss",
        "psps",
        "ppss",
        "ppps",
        "pppp",
        "dsss",
        "dsps",
        "dspp",
        "dsds",
        "dpss",
        "dpps",
        "dppp",
        "dpds",
        "dpdp",
        "ddss",
        "ddps",
        "ddpp",
        "ddds",
        "dddp",
        "dddd",
    }
    generated_force = {
        selection.spec.name
        for selection in selections
        if KernelConsumer.FORCE in selection.consumers
    }
    # psss force reuses the exact bounded scheduler; every other canonical
    # s/p/d class has a production-selected generated force consumer.
    assert (generated_force | {"psss"}) & canonical_spd == canonical_spd
    direct_source = _direct_cuda_source()
    assert "unexpected_tuned_spd_fallback_mask" in direct_source
    assert "kCanonicalSpdShellClassMask" in direct_source
    assert "aot_shell_class_selection_override_requested()" in direct_source
    shards = _partition_production_selections(selections, shard_count=8)
    shard_by_name = {
        selection.spec.name: shard_index
        for shard_index, shard in enumerate(shards)
        for selection in shard
    }
    # Removing a manifest entry must not invalidate unrelated source shards.
    without_dppp = _partition_production_selections(
        tuple(selection for selection in selections if selection.spec.name != "dppp"),
        shard_count=8,
    )
    assert {
        selection.spec.name: shard_index
        for shard_index, shard in enumerate(without_dppp)
        for selection in shard
    } == {
        name: shard_index
        for name, shard_index in shard_by_name.items()
        if name != "dppp"
    }
    assert {
        selection.spec.name: selection.schedule.pair_storage for selection in selections
    } == {
        spec.name: (
            PairStorage.RECOMPUTED
            if spec.name in ("dddp", "dddd", "ddds")
            else PairStorage.MATERIALIZED
        )
        for spec in (selection.spec for selection in selections)
    }
    assert tuple(selection.consumers for selection in selections) == tuple(
        (KernelConsumer.FOCK,)
        if selection.spec.name == "psss"
        else (
            (KernelConsumer.FORCE,)
            if selection.spec.name == "fpps"
            else (KernelConsumer.FOCK, KernelConsumer.FORCE)
        )
        for selection in selections
    )
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first = write_production_bundle(manifest, first_directory, shard_count=4)
    second = write_production_bundle(manifest, second_directory, shard_count=4)
    assert [path.name for path in first] == [path.name for path in second]
    for first_path, second_path in zip(first, second, strict=True):
        assert first_path.read_bytes() == second_path.read_bytes()
        assert b"\0" not in first_path.read_bytes()
    header = emit_registry_header(selections)
    assert '{"dppp", 12U, 5U, 128U, 3U, 162U}' in header
    assert '{"dpds", 13U, 5U, 256U, 3U, 108U}' in header
    assert '{"ddps", 16U, 5U, 256U, 3U, 108U}' in header
    assert '{"ppps", 4U, 3U, 256U, 3U, 27U}' in header
    assert '{"dsps", 7U, 3U, 32U, 3U, 18U}' in header
    assert '{"dpdp", 14U, 6U, 256U, 3U, 324U}' in header
    assert '{"dddp", 19U, 7U, 256U, 3U, 648U}' in header
    assert '{"dddd", 20U, 8U, 256U, 3U, 1296U}' in header
    assert '{"dpss", 10U, 3U, 32U, 3U, 18U}' in header
    assert '{"dsds", 9U, 4U, 64U, 3U, 36U}' in header
    assert '{"ddss", 15U, 4U, 64U, 3U, 36U}' in header
    assert '{"ddpp", 17U, 6U, 256U, 3U, 324U}' in header
    assert '{"ddds", 18U, 6U, 224U, 3U, 216U}' in header
    assert '{"dspp", 8U, 4U, 128U, 3U, 54U}' in header
    assert '{"dpps", 11U, 4U, 128U, 3U, 54U}' in header
    assert '{"pppp", 5U, 4U, 128U, 3U, 81U}' in header
    assert '{"psps", 2U, 2U, 32U, 3U, 9U}' in header
    assert '{"ppss", 3U, 2U, 32U, 3U, 9U}' in header
    assert '{"dsss", 6U, 2U, 32U, 3U, 6U}' in header
    assert "GENERATIVEQC_AOT_SHELL_CLASSES" in header
    assert "GENERATIVEQC_AOT_FOCK_SHELL_CLASSES" in header
    shards = "\n".join(
        path.read_text(encoding="utf-8") for path in first if "shard" in path.name
    )
    assert "offsetof(GeneratedDpppShellTask, shell_pair)" in shards
    assert "offsetof(GeneratedDpppPrimitivePairData, product_center)" in shards
    assert "const std::uint32_t* task_offset" in header
    generated_sources = [path.read_text(encoding="utf-8") for path in first]
    assert any("*task_offset + task_index" in source for source in generated_sources)
    assert any(
        "worker_blocks, tasks, task_offset" in source for source in generated_sources
    )


def test_runtime_buckets_all_generated_classes_before_dispatch() -> None:
    """Prevent production promotion from restoring one scan per exact class."""

    source = _direct_cuda_source()
    assert "classify_generated_shell_tasks_kernel" in source
    assert "prefix_generated_shell_task_counts_kernel" in source
    assert "materialize_generated_shell_tasks_kernel" in source
    assert "compact_generated_shell_tasks_kernel" not in source
    assert source.count("classify_generated_shell_tasks_kernel<<<") == 1


def test_one_electron_force_uses_only_compiler_owned_derivatives() -> None:
    """Retire the native derivative owner without removing schedule diagnostics."""

    source = _direct_cuda_source()
    policy = (REPOSITORY_ROOT / "src/scf/cuda/rhf_policy.cpp").read_text(
        encoding="utf-8"
    )
    rhf = (REPOSITORY_ROOT / "src/scf/cuda_rhf.cpp").read_text(encoding="utf-8")
    assert "GENERATIVEQC_ONE_ELECTRON_DERIVATIVES" not in policy
    assert "generated_one_electron_derivatives_requested" not in policy
    assert "launch_generated_one_electron_gradient(" in rhf
    assert "launch_one_electron_force_cooperative_kernel" not in rhf
    assert "OneElectronDerivativeHermiteCoefficients" not in rhf
    for retired in (
        "one_electron_force_reference.cu",
        "one_electron_force_reference.hpp",
        "one_electron_force_workspace.hpp",
        "one_electron_native_attraction_gradient.cuh",
        "one_electron_native_force.cuh",
    ):
        assert not (REPOSITORY_ROOT / "src/scf/cuda" / retired).exists()
    begin = policy.index("unsigned one_electron_derivative_mapping_requested()")
    end = policy.index("bool resident_psss_bra_requested()", begin)
    mapping = policy[begin:end]
    assert 'std::getenv("GENERATIVEQC_ONE_ELECTRON_DERIVATIVE_MAPPING")' in mapping
    assert (
        "if (selection == nullptr) return NucleusCooperativeSchedule::schedule_code;"
        in mapping
    )
    assert 'std::strcmp(selection, "nucleus_cooperative") == 0' in mapping
    assert "return NucleusCooperativeSchedule::schedule_code;" in mapping
    assert "one_electron_force_cooperative_kernel" not in source


def test_batched_finalization_reuses_each_converged_raw_fock() -> None:
    """Reuse converged peers; both spins delegate baseline admission to HF policy."""

    source = _direct_cuda_source()
    assert "template <bool RetainConvergedDensity>" in source
    assert 'std::getenv("GENERATIVEQC_FINAL_FOCK_REBUILD")' in source
    assert "select_final_fock_rebuild_kernel" in source
    assert "kTightConvergedFockReuseDensityRms = 1.0e-12" in source
    assert "kExpandedConvergedFockReuseDensityTolerance = 1.0e-9" in source
    assert "kExpandedConvergedFockReuseDensityRms = 2.0e-9" in source
    assert "converged_fock_reuse_density_rms(options.density_tolerance)" in source
    assert "copy_selected_matrices_kernel" in source
    assert "launch_direct_quartet_metadata(density, false, false)" in source
    # A resident dm0 is already normalized for its cached overlap matrix, so a
    # geometry change must re-run the warm-density normalization path.
    assert "plan.resident_warm_positions == host.positions" in source
    assert "plan.resident_warm_density == host.warm_density" in source
    assert source.count("hf_iteration_converged(") == 2
    assert "update_convergence_kernel<true>" in source
    assert "update_uhf_convergence_kernel<true>" in source


def test_ppps_queue_buckets_orientation_and_primitive_signature_on_device() -> None:
    """Keep Phase-3 bucketing on the compact production queue and A/B-able."""

    source = _direct_cuda_source()
    assert "kPppsSignatureBucketCount" in source
    assert "resident_ppps_signature_bucket" in source
    assert "prefix_ppps_resident_signature_buckets_kernel" in source
    assert 'std::getenv("GENERATIVEQC_PPPS_SIGNATURE_BUCKETING")' in source
    assert 'std::getenv("GENERATIVEQC_PPPS_BLOCK_THREADS")' in source
    assert "ppps_resident_block_threads_requested" in source
    assert "resident_signature_offsets[bucket_index]" in source
    assert "atomicAdd(resident_signature_write_counts + bucket_index" in source
    assert "std::uint32_t* generated_ppps_resident_signatures =" in source
    assert re.search(
        r"shell_class_profiling\s*\?\s*arena_pointer<std::uint32_t>",
        source,
    )
    assert "kBoundedForceSignatureShellClassMask" in source
    assert "bounded_force_signature_bucket" in source
    assert "scan_bounded_force_signature_counts_kernel" in source
    assert "prefix_bounded_force_signature_blocks_kernel" in source
    assert "bounded_paged_force_shell_class_mask" in source
    assert "bounded_force_signature_offsets, true" in source
    # Force-only classes (for example fpps) are not part of the Fock registry;
    # bounded force paging must enumerate the force registry itself.
    assert "generated::selected_shell_kernels(bounded_force_kernel_count)" in source
    assert "bounded_page_density_tails" in source


def test_bounded_force_signature_mask_tracks_warp_uniform_schedules() -> None:
    """Keep page sorting aligned with every production lockstep task worker."""

    manifest = (
        REPOSITORY_ROOT
        / "python"
        / "generativeqc_compiler"
        / "integral"
        / "production_shell_classes.json"
    )
    selections = load_production_kernel_selections(manifest, "sm_120")
    lockstep_kinds = {
        ScheduleKind.PACKED_TASKS,
        ScheduleKind.THREAD_TASKS,
        ScheduleKind.SUBGROUP_TASKS,
    }
    expected_constants = {
        f"k{selection.spec.name.capitalize()}ShellClass"
        for selection in selections
        if selection.schedule.kind in lockstep_kinds
    }
    source = _direct_cuda_source()
    mask_begin = source.index(
        "constexpr std::uint64_t kBoundedForceSignatureShellClassMask"
    )
    mask_end = source.index(";", mask_begin)
    configured_constants = set(
        re.findall(r"<< (k[A-Za-z0-9]+ShellClass)", source[mask_begin:mask_end])
    )
    assert configured_constants == expected_constants


def test_bounded_dppp_force_uses_nonterminating_paged_screening() -> None:
    """Keep DPPP paged without assuming Schwarz-sorted ket segments."""

    source = _direct_cuda_source()
    mask_begin = source.index(
        "const std::uint64_t bounded_force_legacy_queue_shell_class_mask"
    )
    mask_end = source.index(
        "const std::uint64_t covered_force_shell_class_mask", mask_begin
    )
    mask_source = source[mask_begin:mask_end]
    assert "std::uint64_t{1} << kDpppShellClass" not in mask_source

    queue_source = (REPOSITORY_ROOT / "src/scf/cuda/queue_plan.cpp").read_text()
    order_begin = queue_source.index("bool make_bounded_stream_shell_pair_order")
    order_end = queue_source.index(
        "std::uint64_t bounded_lower_triangle_row", order_begin
    )
    assert "std::sort" not in queue_source[order_begin:order_end]

    # Compaction now has its own owner. Bound each assertion by the next
    # function in that owner rather than by a comment in a different file.
    compact_source = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_bounded_pages.cu"
    ).read_text()
    compact_begin = compact_source.index(
        "__global__ void compact_bounded_exact_class_force_wave_kernel"
    )
    compact_end = compact_source.index(
        "void launch_compact_bounded_exact_class_force_wave_kernel", compact_begin
    )
    low_order_begin = source.index(
        "__global__ void contract_bounded_exact_low_order_force_page_kernel"
    )
    low_order_end = source.index(
        "__global__ __launch_bounds__(kBoundedDirectThreads, 1) void bounded_direct_shell_quartet_kernel",
        low_order_begin,
    )
    for page_source in (
        compact_source[compact_begin:compact_end],
        source[low_order_begin:low_order_end],
    ):
        force_gate = re.search(
            r"page_density_tails\.force < force_tolerance\) \{\s*(\w+);",
            page_source,
        )
        fock_gate = re.search(
            r"page_density_tails\.fock < screening_tolerance\) \{\s*(\w+);",
            page_source,
        )
        assert force_gate is not None
        assert force_gate.group(1) == "continue"
        assert fock_gate is not None
        assert fock_gate.group(1) == "continue"

    page_begin = source.index("const auto launch_bounded_overflow_force")
    page_end = source.index("const auto launch_bounded_native_force", page_begin)
    page_source = source[page_begin:page_end]
    assert "bounded_force_legacy_queue_shell_class_mask" in page_source
    assert page_source.index("bounded_force_legacy_queue_shell_class_mask") < (
        page_source.index("bounded_generated_page_range")
    )


def test_bounded_page_range_tracks_end_across_systems() -> None:
    """Do not stop a page at the first system that it intersects."""

    source = _direct_cuda_source()
    source = (REPOSITORY_ROOT / "src/scf/cuda/queue_plan.cpp").read_text()
    range_begin = source.index("BoundedGeneratedPageRange bounded_generated_page_range")
    range_source = source[range_begin:]
    assert "bool found_end = false;" in range_source
    assert "found_begin && !found_end && page_end <= system_end" in range_source
    assert "found_end = true;" in range_source
    assert "bra_end == 0U" not in range_source


def test_bounded_fock_pages_do_not_duplicate_streaming_consumers() -> None:
    """Run full generated pages before overflow-only streaming workers."""

    source = _direct_cuda_source()
    begin = source.index("const auto launch_bounded_paged_generated_fock")
    end = source.index("const auto launch_bounded_generated_fock", begin)
    page_source = source[begin:end]
    # Generated streaming consumers are overflow-only: they return immediately
    # when the per-class overflow flag is clear.  The paged exact consumer must
    # therefore not skip those classes, or normal tasks disappear from Fock.
    assert "host_generated_streaming_fock_shell_class_mask" not in page_source
    assert "host_native_streaming_fock_shell_class_mask" in page_source


def test_fixed_generated_task_arena_has_a_memory_admission_limit() -> None:
    """Route large grid-addressable buckets before a multi-GiB allocation."""

    source = _direct_cuda_source()
    assert "direct_schedule.fixed_topology.arena_maximum_bytes" in source
    assert "direct_jk_bounded_streaming_task_capacity_limit" in source
    assert "resolve_direct_jk_schedule_policy" in source
    assert "direct_task_layout.exact_tile_count >" in source
    assert "sizeof(GeneratedShellTask)" in source
    assert "requested_bounded_direct_streaming = true" in source


def test_direct_task_resource_domains_remain_separate() -> None:
    """Do not reuse fixed-topology storage to size bounded streaming pages."""

    source = (REPOSITORY_ROOT / "src/scf/cuda/rhf_policy.cpp").read_text()
    begin = source.index("direct_jk_bounded_streaming_task_capacity_limit")
    end = source.index("bool reuse_converged_fock_requested", begin)
    capacity_source = source[begin:end]
    assert "policy.fixed_topology" not in capacity_source
    assert "policy.bounded_streaming.task_capacity_ceiling" in capacity_source
    assert "policy.bounded_streaming.arena_maximum_bytes" in capacity_source


def test_bounded_force_keeps_fock_only_classes_out_of_force_dispatch() -> None:
    """Do not call the force registry for the remaining Fock-only psss entry."""

    source = _direct_cuda_source()
    begin = source.index(
        "const std::uint64_t explicit_generated_force_shell_class_mask"
    )
    end = source.index("const auto launch_bounded_generated_force", begin)
    mask_source = source[begin:end]
    assert "selected_fock_shell_kernels" not in mask_source
    assert "selected_shell_kernels(bounded_force_kernel_count)" in mask_source
    assert "~kBoundedNativePagedForceShellClassMask" in mask_source
    assert "cudaErrorNotSupported" in mask_source


def test_psss_force_codegen_emits_only_independent_gradient_roots() -> None:
    """Keep the Direct-HF psss candidate free of unused value/center-four roots."""

    source = emit_low_order_weighted_header(inline_single_use=True)
    begin = source.index("IndependentGradient psss_force(")
    end = source.index("IndependentGradient ssss_force(", begin)
    psss_force = source[begin:end]
    assert "Gradient psss(" in source
    assert "result.value" not in psss_force
    assert "result.center[3]" not in psss_force
    for center in range(3):
        for axis in range(3):
            assert f"result.center[{center}][{axis}]" in psss_force


def test_psss_force_math_is_unconditionally_generated() -> None:
    """Keep retired handwritten psss force math and its route selector absent."""

    native_source = (REPOSITORY_ROOT / "src/scf/cuda/direct_native_psss.cuh").read_text(
        encoding="utf-8"
    )
    low_order_source = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_force_low_order.cuh"
    ).read_text(encoding="utf-8")
    policy_source = (REPOSITORY_ROOT / "src/scf/cuda/rhf_policy.cpp").read_text(
        encoding="utf-8"
    )
    assert "GeneratedMath" not in native_source
    assert "if constexpr (GeneratedMath)" not in native_source
    assert "generated_weighted_eri::Geometry geometry;" in native_source
    assert "generated_weighted_eri::Geometry geometry{};" not in native_source
    assert "generated_weighted_eri::psss_force" in native_source
    assert "generated_psss_weighted" not in low_order_source
    assert "GENERATIVEQC_PSSS_WEIGHTED" not in policy_source
    assert (
        "contracted_eri_cartesian_source_psss_weighted_gradient<ResidentBra>"
        in low_order_source
    )


def test_ssss_force_codegen_emits_only_independent_gradient_roots() -> None:
    """Keep the native-adapter ssss helper free of unused value/center-four work."""

    source = emit_low_order_weighted_header(inline_single_use=True)
    begin = source.index("IndependentGradient ssss_force(")
    end = source.index(
        "}  // namespace generativeqc::scf::generated_weighted_eri", begin
    )
    ssss_force = source[begin:end]
    assert "result.value" not in ssss_force
    assert "result.center[3]" not in ssss_force
    assert "geometry.product_scales[3]" not in ssss_force
    assert "geometry.decay[3]" not in ssss_force
    for center in range(3):
        for axis in range(3):
            assert f"result.center[{center}][{axis}]" in ssss_force


def test_ssss_force_retires_handwritten_math_and_selector() -> None:
    """Keep ssss force science compiler-owned on the qualified native scheduler."""

    manifest = load_production_kernel_selections(
        REPOSITORY_ROOT
        / "python"
        / "generativeqc_compiler"
        / "integral"
        / "production_shell_classes.json",
        "sm_120",
    )
    ssss = next(selection for selection in manifest if selection.spec.name == "ssss")
    assert KernelConsumer.FORCE in ssss.consumers

    types_source = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_gradient_types.cuh"
    ).read_text(encoding="utf-8")
    low_order_source = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_force_low_order.cuh"
    ).read_text(encoding="utf-8")
    assert "SsssWeightedGradient" not in types_source
    assert "contract_two_electron_force_ssss_task" in low_order_source
    assert "generated_weighted_eri::ssss_force" in low_order_source
    assert "direct_native_order01_gradient.cuh" not in low_order_source
    assert "generated_math" not in low_order_source
    assert "geometry.product_scales[3]" not in low_order_source
    assert "geometry.decay[3][axis]" not in low_order_source

    policy = (REPOSITORY_ROOT / "src/scf/cuda/rhf_policy.cpp").read_text(
        encoding="utf-8"
    )
    resources = (REPOSITORY_ROOT / "python/generativeqc/resources_hf.py").read_text(
        encoding="utf-8"
    )
    driver = _direct_cuda_source()
    assert "GENERATIVEQC_SSSS_FORCE" not in policy
    assert "GENERATIVEQC_SSSS_FORCE" not in resources
    assert "generated_ssss_force" not in driver
    assert "const std::uint64_t ssss_shell_class_mask" in driver
    assert "~ssss_shell_class_mask" in driver
    assert "~explicit_generated_force_shell_class_mask" in driver


def test_order01_force_retires_handwritten_generic_fallback() -> None:
    """Keep total-order-zero/one Direct-HF force mathematics compiler-owned."""

    assert not (
        REPOSITORY_ROOT / "src/scf/cuda/direct_native_order01_gradient.cuh"
    ).exists()

    quartet = (REPOSITORY_ROOT / "src/scf/cuda/direct_force_quartet.cuh").read_text(
        encoding="utf-8"
    )
    assert "direct_native_order01_gradient.cuh" not in quartet
    assert "contracted_eri_cartesian_source_order01_gradient" not in quartet
    assert "static_assert(AngularOrder >= 2U" in quartet

    bounded = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_bounded_contraction.cuh"
    ).read_text(encoding="utf-8")
    assert "GENERATIVEQC_BOUNDED_FORCE_CASE(0)" not in bounded
    assert "GENERATIVEQC_BOUNDED_FORCE_CASE(1)" not in bounded


def test_order2_force_codegen_emits_only_independent_gradient_roots() -> None:
    """Keep PSPS/PPSS/DSSS native schedulers backed by force-only compiler roots."""

    source = emit_low_order_weighted_header(inline_single_use=True)
    names = ("psps_force", "ppss_force", "dsss_force")
    for index, name in enumerate(names):
        begin = source.index(f"IndependentGradient {name}(")
        if index + 1 < len(names):
            end = source.index(f"IndependentGradient {names[index + 1]}(", begin)
        else:
            end = source.index(
                "}  // namespace generativeqc::scf::generated_weighted_eri", begin
            )
        function = source[begin:end]
        assert "result.value" not in function
        assert "result.center[3]" not in function
        for center in range(3):
            for axis in range(3):
                assert f"result.center[{center}][{axis}]" in function


def test_order2_force_retires_handwritten_gradient_bodies() -> None:
    """Keep exact order-two Direct-HF force mathematics compiler-owned."""

    for name in ("dsss", "ppss", "psps"):
        assert not (
            REPOSITORY_ROOT / f"src/scf/cuda/direct_native_{name}_gradient.cuh"
        ).exists()

    source = (REPOSITORY_ROOT / "src/scf/cuda/direct_force_order2.cuh").read_text(
        encoding="utf-8"
    )
    assert (
        "contracted_eri_cartesian_source_order2_generated_weighted_gradient" in source
    )
    for name in ("psps", "ppss", "dsss"):
        assert f"generated_weighted_eri::{name}_force" in source
        assert f"direct_native_{name}_gradient.cuh" not in source
        assert f"contracted_eri_cartesian_source_{name}_weighted_gradient" not in source
    assert not (
        REPOSITORY_ROOT / "src/scf/cuda/direct_native_order2_gradient.cuh"
    ).exists()
    assert "contracted_eri_cartesian_source_order2_generated_gradient" in source
    quartet_source = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_force_quartet.cuh"
    ).read_text(encoding="utf-8")
    assert "direct_native_order2_gradient.cuh" not in quartet_source
    assert "contracted_eri_cartesian_source_order2_generated_gradient" in quartet_source
    assert "generated_weighted_eri::Geometry geometry;" in source
    assert "generated_weighted_eri::Geometry geometry{};" not in source


def test_order3_force_retires_handwritten_gradient_bodies() -> None:
    """Keep all total-order-three Direct-HF force mathematics compiler-owned."""

    generated = emit_low_order_weighted_header(inline_single_use=True)
    for name in ("ppps", "dsps", "dpss", "fsss"):
        assert f"IndependentGradient {name}_force(" in generated

    assert not (
        REPOSITORY_ROOT / "src/scf/cuda/direct_native_order3_gradient.cuh"
    ).exists()
    assert not (
        REPOSITORY_ROOT / "src/scf/cuda/direct_native_pair_order3_gradient.cuh"
    ).exists()

    source = (REPOSITORY_ROOT / "src/scf/cuda/direct_force_order3.cuh").read_text(
        encoding="utf-8"
    )
    assert (
        "contracted_eri_cartesian_source_order3_generated_weighted_gradient" in source
    )
    for name in ("ppps", "dsps", "dpss", "fsss"):
        assert f"generated_weighted_eri::{name}_force" in source

    generic = (REPOSITORY_ROOT / "src/scf/cuda/direct_force_quartet.cuh").read_text(
        encoding="utf-8"
    )
    assert "contracted_eri_cartesian_source_order3_gradient" not in generic
    assert "direct_native_order3_gradient.cuh" not in generic


def test_bounded_order3_force_uses_generated_shell_task_math() -> None:
    """Keep bounded streaming disjoint from the retired order-three AO formula."""

    bounded = (REPOSITORY_ROOT / "src/scf/cuda/direct_bounded_fallback.cu").read_text(
        encoding="utf-8"
    )
    assert "contract_two_electron_force_order3_task_scaled<Unrestricted>(" in bounded
    dispatcher = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_bounded_contraction.cuh"
    ).read_text(encoding="utf-8")
    assert "GENERATIVEQC_BOUNDED_FORCE_CASE(3)" not in dispatcher


def test_bounded_psss_resident_path_is_allocated_and_disjoint_from_page_fallback() -> (
    None
):
    """Use the validated resident-bra consumer before paging psss force work."""

    source = _direct_cuda_source()
    assert re.search(
        r"requested_quartet_direct\s*\?\s*host\.psss_resident_tasks\.size\(\)",
        source,
    )
    assert "requested_quartet_direct && !requested_bounded_direct_streaming" in source
    assert "launch_bounded_resident_psss_force" in source
    assert "~(bounded_resident_psss_force_enabled" in source


def test_warm_density_validation_parallelizes_each_system_matrix() -> None:
    """Keep fixed-dm0 setup from regressing to one serial N^2 worker."""

    source = _direct_cuda_source()
    assert "constexpr unsigned kWarmDensityThreads = 256" in source
    assert "warm_density_block_sum<kWarmDensityThreads>" in source
    for kernel in ("apply_warm_density_kernel", "apply_uhf_warm_density_kernel"):
        launch = rf"launch_{kernel}\(\s*static_cast<unsigned>\(batch_size\),\s*"
        assert re.search(launch + r"kWarmDensityThreads", source)
        assert f"{kernel}<<<grid, block, shared_bytes, stream>>>" in source


def test_force_density_product_screening_is_force_only_and_conservative() -> None:
    """Keep the force queue optional without weakening the SCF Fock gate."""

    source = _direct_cuda_source()
    assert "enum class DirectScreeningPurpose" in source
    assert "DirectScreeningPurpose::Fock" in source
    assert "DirectScreeningPurpose::Force" in source
    assert "kForceDensityProductScreeningTolerance = 1.0e-14" in source
    assert "fmin(screening_tolerance, kForceDensityProductScreeningTolerance)" in source
    assert 'std::getenv("GENERATIVEQC_FORCE_DENSITY_PRODUCT_SCREENING")' in source
    assert "launch_direct_force_compaction();" in source


def test_cached_direct_plan_reuses_immutable_task_layout() -> None:
    """Keep quadratic shell-pair topology enumeration out of warm replay."""

    source = _direct_cuda_source()
    layout_begin = source.index("detail::DirectQuartetTaskLayout direct_task_layout")
    layout_end = source.index(
        "// Direct consumers expand each compact logical tile", layout_begin
    )
    layout_setup = source[layout_begin:layout_end]
    assert "requested_quartet_direct && first_setup" in layout_setup
    assert "plan.total_shell_quartet_tiles" in layout_setup
    assert source.count("detail::make_direct_quartet_task_layout(") == 1
    bucket_source = (
        REPOSITORY_ROOT / "src" / "scf" / "cuda" / "rhf_bucket.cpp"
    ).read_text(encoding="utf-8")
    assert "**plan, candidate, options" in bucket_source


def test_mixed_precision_is_budgeted_per_item_on_the_prepared_census() -> None:
    """Keep the mixed route budgeted on the prepared census and refined in FP64.

    The public ``auto`` policy resolves the FP32 cutoff from the accumulated-error
    budget and the mixed-capable tile census of this reference, so a route that
    cannot supply a census (bounded streaming) keeps the FP64 operator instead of
    accumulating an unbounded rounding error. The cutoff and the admission are
    resolved per item, so one batch keeps a cold item on the exact FP64 operator
    while a warm item runs the mixed route. The legacy diagnostic switch stays a
    separate, deliberately unbudgeted override.
    """

    source = _direct_cuda_source()
    threshold_begin = source.index(
        "const MixedPrecisionFockPolicy requested_precision_policy"
    )
    threshold_end = source.index(
        "const bool requested_mixed_precision_fock", threshold_begin
    )
    resolution = source[threshold_begin:threshold_end]
    assert re.search(
        r"requested_quartet_direct\s*\?\s*resolve_mixed_precision_fock_policy\(",
        resolution,
    )
    assert "mixed_precision_eligible_tile_count" in resolution
    # The per-item census is kept per system, uploaded per execution, and read
    # both by the tile gate and by the per-item refinement entry.
    assert "system_mixed_capable_tile_counts" in source
    assert "mixed_precision_system_census" in source
    assert "mixed_fock_item_cutoff(mixed_precision_cutoff_ceiling" in source
    assert "host_mixed_item_census.data()" in source
    assert "enter_target_refinement_kernel<<<" in source
    policy = (REPOSITORY_ROOT / "src" / "scf" / "cuda" / "rhf_policy.cpp").read_text(
        encoding="utf-8"
    )
    assert "admit_auto_mixed_precision_fock" in policy
    assert "kMixedPrecisionFloat32UnitRoundoff * eligible_tiles" in policy
    assert "resolve_mixed_precision_item" in policy
    assert "allow_mixed_precision && mixed_precision_fock" in source
    # The finalization path must explicitly disable the iterative mixed route.
    assert "launch_fock_builder(density, false, false)" in source


def test_bounded_streaming_uses_monotonic_system_density_tail() -> None:
    """Prune large class segments with a conservative density coarse bound."""

    topology = (REPOSITORY_ROOT / "src" / "scf" / "generated_shell_task.hpp").read_text(
        encoding="utf-8"
    )
    generator = (
        REPOSITORY_ROOT
        / "python"
        / "generativeqc_compiler"
        / "integral"
        / "production_emission.py"
    ).read_text(encoding="utf-8")
    assert "const double* system_density_bounds" in topology
    assert "const double* system_pair_density_bounds" in topology
    assert "const std::uint32_t* generated_overflow" in topology
    assert "topology.system_pair_density_bounds[" in generator
    assert "system_density_bound < screening_tolerance" in generator
    assert "topology.generated_overflow[{shell_class}U]" in generator


def test_bounded_streaming_profiles_executed_precision_per_shell_class() -> None:
    """Count actual retained quartets without changing normal kernel work."""

    generator = (
        REPOSITORY_ROOT
        / "python"
        / "generativeqc_compiler"
        / "integral"
        / "production_emission.py"
    ).read_text(encoding="utf-8")
    source = _direct_cuda_source()
    assert "record_fock_precision" in generator
    assert "fp64_work_count, fp32_work_count" in generator
    assert "bounded_fock_fp64_work_counts + shell_class" in source
    assert "bounded_fock_fp32_work_counts + shell_class" in source
    assert "fp64_quartets=%llu fp32_quartets=%llu" in source


def test_generated_order2_fock_masks_handwritten_fallback() -> None:
    """Prevent generated order-two Fock quartets from being scattered twice."""

    source = _direct_cuda_source()
    task_begin = source.index("contract_fock_direct_order2_task(")
    task_end = source.index(
        "/** Fixed-capacity wrapper retained for high-register angular orders. */",
        task_begin,
    )
    task_source = source[task_begin:task_end]
    assert "generated_fock_shell_class_mask" in task_source
    assert "std::uint64_t{1} << shell_class" in task_source

    worker_begin = source.index("void build_fock_direct_order2_persistent_kernel(")
    worker_end = source.index(
        "/** Consume only the active compacted Fock domain from a device queue. */",
        worker_begin,
    )
    worker_source = source[worker_begin:worker_end]
    assert "generated_fock_shell_class_mask" in worker_source
    assert "contract_fock_direct_order2_task<Unrestricted>" in worker_source


def test_production_codegen_cmake_tracks_transitive_generator_inputs(
    tmp_path: typing.Any,
) -> None:
    """Regenerate production CUDA whenever shared compiler stages change."""

    # Dynamic dependencies exist only after the generator has run. Exercise the
    # real CPU-configurable pilot, then inspect its emitted depfile rather than
    # requiring the retired all-compiler glob in Ninja's pre-build graph.
    subprocess.run(
        [
            "cmake",
            "-S",
            str(REPOSITORY_ROOT),
            "-B",
            str(tmp_path),
            "-G",
            "Ninja",
            "-DGENERATIVEQC_ENABLE_CUDA=OFF",
            "-DGENERATIVEQC_BUILD_TESTS=OFF",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = "generated/shell_kernels/eri_psss_x_gradient.cuh"
    subprocess.run(
        ["cmake", "--build", str(tmp_path), "--target", output],
        check=True,
        capture_output=True,
        text=True,
    )
    dependencies = (tmp_path / f"{output}.d").read_text(encoding="utf-8")
    for dependency in (
        "python/generativeqc_compiler/integral/blocks.py",
        "python/generativeqc_compiler/integral/cache.py",
        "python/generativeqc_compiler/integral/cuda.py",
        "python/generativeqc_compiler/integral/capabilities.py",
        "python/generativeqc_compiler/integral/cuda_lowering.py",
        "python/generativeqc_compiler/integral/expr.py",
        "python/generativeqc_compiler/integral/fused_schedule.py",
        "python/generativeqc_compiler/integral/ir.py",
        "python/generativeqc_compiler/integral/ir_serialization.py",
        "python/generativeqc_compiler/integral/production.py",
        "python/generativeqc_compiler/integral/rys.py",
        "python/generativeqc_compiler/integral/rys3_data.py",
        "python/generativeqc_compiler/integral/rys5_data.py",
        "python/generativeqc_compiler/integral/shell_class.py",
        "python/generativeqc_compiler/integral/shell_signature.py",
        "python/generativeqc_compiler/integral/shell_spec.py",
    ):
        assert dependency in dependencies
    # Production AOT uses the same depfile-enabled registration while retaining
    # its explicit non-Python manifest dependency. Unrelated compiler stages must
    # not be reintroduced as unconditional dependencies.
    assert "python/generativeqc_compiler/tensor/layout.py" not in dependencies
    cuda = (REPOSITORY_ROOT / "cmake/GenerativeQCCuda.cmake").read_text(
        encoding="utf-8"
    )
    production = cuda.split("generativeqc_register_generated_sources(", 1)[1]
    production_dependencies = production.split("DEPENDS", 1)[1].split("ARGS", 1)[0]
    assert "${GENERATIVEQC_AOT_SHELL_MANIFEST}" in production_dependencies
    assert "${GENERATIVEQC_SCIENTIFIC_COMPILER_INPUTS}" not in production_dependencies


def test_cuda_target_request_is_resolved_before_language_enablement() -> None:
    """Do not let CMake/NVCC invent a compiler-default CUDA target."""

    cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    cuda_block = cmake.split("if(GENERATIVEQC_ENABLE_CUDA)", 1)[1].split(
        "# Scoped GenerativeQC-owned GFN2 CPU runtime", 1
    )[0]
    target_error = cuda_block.index("CUDA target architecture is required")
    target_assignment = cuda_block.index(
        "set(CMAKE_CUDA_ARCHITECTURES ${_generativeqc_cuda_requested_architectures})"
    )
    language_enable = cuda_block.index("enable_language(CUDA)")
    assert target_error < target_assignment < language_enable
    assert "GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES" in cuda_block[:language_enable]
    assert "GENERATIVEQC_CUDA_ARCHITECTURES" in cuda_block[:language_enable]
    assert "DEFINED CMAKE_CUDA_ARCHITECTURES" in cuda_block[:language_enable]
    assert "DEFINED ENV{CUDAARCHS}" in cuda_block[:language_enable]
    assert "set(CMAKE_CUDA_ARCHITECTURES 120)" not in cuda_block


def test_codegen_capability_report_covers_catalog_and_manifest() -> None:
    """Report structural backend reasons for all 55 canonical shell classes."""

    manifest = (
        REPOSITORY_ROOT
        / "python"
        / "generativeqc_compiler"
        / "integral"
        / "production_shell_classes.json"
    )
    report = build_capability_report(
        architecture="sm_120",
        manifest=manifest,
    )
    assert report["total_shell_classes"] == 55
    assert report["generic_fused_supported"] == 55
    assert report["backend"] == {
        "name": "cuda",
        "architecture": "sm_120",
        "compute_capability": "12.0",
        "generator_abi": 1,
        "schedule_source": "schedule_candidates",
        "emitter_validation": "emit_shell_class_fused_cuda",
    }
    assert report["recurrence_supported"] == {
        "subset_wick": 55,
        "rys2": 4,
        "rys3": 11,
        "rys4": 16,
        "rys5": 14,
    }
    assert report["force_derivative_supported"] == {"1": 55, "2": 0}
    rows = {row["shell_class"]: row for row in report["shell_classes"]}
    assert rows["psss"]["recurrences"]["rys2"]["supported"] is True
    assert rows["dppp"]["recurrences"]["rys4"]["supported"] is True
    assert rows["dppp"]["recurrences"]["rys3"]["supported"] is False
    assert rows["dppp"]["force_derivative_orders"]["1"]["supported"] is True
    second_force = rows["dppp"]["force_derivative_orders"]["2"]
    assert second_force["supported"] is False
    assert "order-one derivatives" in second_force["reasons"][0]
    assert rows["fsps"]["production"]["force"] is False
    assert rows["fsps"]["production"]["status"] == "manifest_gap"
    assert (
        rows["fsps"]["production"]["promotion_gate"]
        == "real_molecular_endpoint_and_resource_gates"
    )
    assert rows["dpps"]["production"]["force"] is True
    assert rows["dpps"]["production"]["status"] == "manifest_selected"
    assert CAPABILITY_STREAMING_FOCK in rows["dpps"]["production"]["capabilities"]
