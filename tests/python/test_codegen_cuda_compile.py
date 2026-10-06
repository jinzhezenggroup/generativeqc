"""NVCC compile, resource, and generated-kernel benchmark codegen tests.

Moved behavior-neutrally from the legacy codegen compatibility suite for #489.
"""

from __future__ import annotations

# ruff: noqa: F401  # Keep this move behavior-neutral; imports mirror legacy ownership.
from codegen_test_support import (
    _PRODUCTION_PRELUDE,
    boys_values,
    build_fused_shell_plan,
    build_integral_ir,
    DDDD_SPEC,
    DDPS_SPEC,
    DPDS_SPEC,
    DPPP_SPEC,
    emit_shell_class_benchmark_cuda,
    emit_shell_class_fused_cuda,
    FDDD_SPEC,
    FFPS_SPEC,
    FUSED_SHELL_SPEC_BY_NAME,
    json,
    KernelConsumer,
    load_production_kernel_selections,
    os,
    PairOrientation,
    PairStorage,
    Path,
    PSSS_SPEC,
    pytest,
    re,
    replace,
    REPOSITORY_ROOT,
    RTX5090_DDPS_RESOURCE_LIMITS,
    RTX5090_DPDS_RESOURCE_LIMITS,
    RTX5090_DPPP_RESOURCE_LIMITS,
    RTX5090_DPPP_UNIFORM_RYS4_RESOURCE_LIMITS,
    RTX5090_DPSS_SCALAR_RYS3_RESOURCE_LIMITS,
    RTX5090_PPPS_SCALAR_THREAD_RESOURCE_LIMITS,
    RTX5090_PPSS_RESOURCE_LIMITS,
    RTX5090_PSPS_RESOURCE_LIMITS,
    schedule_candidates,
    ScheduleIR,
    ScheduleKind,
    subprocess,
    supported_schedule_trials,
    TEST_CUDA_TARGET,
    time,
    typing,
)

@pytest.mark.parametrize(
    ("name", "recurrence", "resource_limits"),
    (
        ("ppps", "rys3", None),
        ("dpss", "rys3", RTX5090_DPSS_SCALAR_RYS3_RESOURCE_LIMITS),
        ("psss", "rys2", None),
        ("psps", "rys2", None),
        ("ppss", "rys2", None),
        ("dsss", "rys2", None),
    ),
)
def test_scalar_rys_cuda_compiles_with_bounded_call_save_when_nvcc_is_configured(
    tmp_path: Path,
    name: str,
    recurrence: str,
    resource_limits: dict[str, tuple[int, int, int]] | None,
) -> None:
    """Bound scalar fixed-root resources before production promotion."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
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
        spec, schedule=schedule, recurrence=recurrence, target=TEST_CUDA_TARGET
    )
    source = tmp_path / f"generated_{name}_{recurrence}.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(spec, plan),
        encoding="utf-8",
    )
    cubin = tmp_path / f"generated_{name}_{recurrence}.cubin"
    compile_started = time.perf_counter()
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(cubin),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    compile_seconds = time.perf_counter() - compile_started
    if os.environ.get("GENERATIVEQC_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    print(
        json.dumps(
            {
                "compile_seconds": compile_seconds,
                "cubin_bytes": cubin.stat().st_size,
            },
            sort_keys=True,
        )
    )
    if cuda_architecture == "sm_120":
        output = result.stdout + result.stderr
        assert f"generated_{name}_shell_class_force_rhf_persistent_kernel" in output
        if resource_limits is not None:
            assert_rtx5090_resources(output, resource_limits)
        resource_records = re.findall(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
            r"(\d+) bytes spill loads",
            output,
        )
        assert resource_records
        numeric_records = tuple(tuple(map(int, record)) for record in resource_records)
        if name == "ppps":
            assert max(record[0] for record in numeric_records) <= 56
            assert max(record[1] for record in numeric_records) <= 64
            assert max(record[2] for record in numeric_records) <= 64
        else:
            assert all(record == (0, 0, 0) for record in numeric_records)


def test_dppp_cooperative_rys4_compiles_without_spills_when_nvcc_is_configured(
    tmp_path: Path,
) -> None:
    """Apply the sm_120 resource gate before any production promotion."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=192,
        component_tile=DPPP_SPEC.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.MATERIALIZED,
        unroll_pair_terms=True,
        minimum_blocks_per_sm=2,
    )
    plan = build_fused_shell_plan(
        DPPP_SPEC, schedule=schedule, recurrence="rys4", target=TEST_CUDA_TARGET
    )
    source = tmp_path / "generated_dppp_cooperative_rys4.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(DPPP_SPEC, plan),
        encoding="utf-8",
    )
    cubin = tmp_path / "generated_dppp_cooperative_rys4.cubin"
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-O3",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(cubin),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if os.environ.get("GENERATIVEQC_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    if cuda_architecture == "sm_120":
        assert_rtx5090_resources(
            result.stdout + result.stderr,
            {
                "generated_dppp_shell_class_force_rhf_kernel": (168, 160, 1024),
                "generated_dppp_shell_class_force_uhf_kernel": (168, 160, 1024),
                "generated_dppp_shell_class_force_rhf_persistent_kernel": (
                    168,
                    160,
                    1024,
                ),
                "generated_dppp_shell_class_force_uhf_persistent_kernel": (
                    168,
                    160,
                    1024,
                ),
            },
        )


@pytest.mark.parametrize(
    ("name", "schedule", "ordinary_limit", "persistent_limit"),
    (
        (
            "dpdp",
            ScheduleIR(
                kind=ScheduleKind.COMPONENT_LANES,
                block_threads=352,
                component_tile=324,
                tasks_per_warp=1,
                shared_coulomb=True,
                pair_orientation=PairOrientation.SWAPPED,
                pair_storage=PairStorage.RECOMPUTED,
                unroll_pair_terms=False,
                minimum_blocks_per_sm=1,
            ),
            (168, 200, 1312),
            (168, 200, 1320),
        ),
        (
            "dpds",
            ScheduleIR(
                kind=ScheduleKind.SUBGROUP_TASKS,
                block_threads=256,
                component_tile=108,
                tasks_per_warp=4,
                shared_coulomb=True,
                pair_orientation=PairOrientation.CANONICAL,
                pair_storage=PairStorage.MATERIALIZED,
                unroll_pair_terms=True,
                minimum_blocks_per_sm=1,
            ),
            (254, 112, 36872),
            (254, 112, 36872),
        ),
        (
            "ddpp",
            ScheduleIR(
                kind=ScheduleKind.COMPONENT_LANES,
                block_threads=352,
                component_tile=324,
                tasks_per_warp=1,
                shared_coulomb=True,
                pair_orientation=PairOrientation.SWAPPED,
                pair_storage=PairStorage.RECOMPUTED,
                unroll_pair_terms=False,
                minimum_blocks_per_sm=1,
            ),
            (168, 192, 1312),
            (167, 192, 1320),
        ),
        (
            "ddps",
            ScheduleIR(
                kind=ScheduleKind.SUBGROUP_TASKS,
                block_threads=256,
                component_tile=108,
                tasks_per_warp=4,
                shared_coulomb=True,
                pair_orientation=PairOrientation.CANONICAL,
                pair_storage=PairStorage.MATERIALIZED,
                unroll_pair_terms=True,
                minimum_blocks_per_sm=1,
            ),
            (254, 112, 36872),
            (254, 112, 36872),
        ),
        (
            "ddds",
            ScheduleIR(
                kind=ScheduleKind.COMPONENT_LANES,
                block_threads=224,
                component_tile=216,
                tasks_per_warp=1,
                shared_coulomb=True,
                pair_orientation=PairOrientation.SWAPPED,
                pair_storage=PairStorage.RECOMPUTED,
                unroll_pair_terms=False,
                minimum_blocks_per_sm=1,
            ),
            (254, 192, 1024),
            (254, 192, 1032),
        ),
    ),
)
def test_batched_rys4_hot_classes_compile_without_spills_when_nvcc_is_configured(
    tmp_path: Path,
    name: str,
    schedule: ScheduleIR,
    ordinary_limit: tuple[int, int, int],
    persistent_limit: tuple[int, int, int],
) -> None:
    """Lock the sm_120 resource envelope for batched Rys4 promotions."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    plan = build_fused_shell_plan(
        spec, schedule=schedule, recurrence="rys4", target=TEST_CUDA_TARGET
    )
    source = tmp_path / f"generated_{name}_promoted_rys4.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(spec, plan),
        encoding="utf-8",
    )
    cubin = tmp_path / f"generated_{name}_promoted_rys4.cubin"
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-O3",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(cubin),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    output = result.stdout + result.stderr
    if os.environ.get("GENERATIVEQC_NVCC_VERBOSE"):
        print(output)
    assert result.returncode == 0, output
    assert cubin.exists()
    if cuda_architecture == "sm_120":
        function_prefix = f"generated_{name}_shell_class_force"
        assert_rtx5090_resources(
            output,
            {
                f"{function_prefix}_rhf_kernel": ordinary_limit,
                f"{function_prefix}_uhf_kernel": ordinary_limit,
                f"{function_prefix}_rhf_persistent_kernel": (persistent_limit),
                f"{function_prefix}_uhf_persistent_kernel": (persistent_limit),
            },
        )
        helper_records = re.findall(
            r"\d+ bytes stack frame, (\d+) bytes spill stores, "
            r"(\d+) bytes spill loads",
            output,
        )
        assert helper_records
        assert all(
            int(spill_stores) == 0 and int(spill_loads) == 0
            for spill_stores, spill_loads in helper_records
        )


def test_dppp_rys4_uniform_warps_compile_when_nvcc_is_configured(
    tmp_path: Path,
) -> None:
    """Compile the 32-task/eight-warp force worker before endpoint testing."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
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
    source = tmp_path / "generated_dppp_uniform_warp_rys4.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(DPPP_SPEC, plan),
        encoding="utf-8",
    )
    cubin = tmp_path / "generated_dppp_uniform_warp_rys4.cubin"
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-O3",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(cubin),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if os.environ.get("GENERATIVEQC_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert cubin.exists()
    if cuda_architecture == "sm_120":
        assert_rtx5090_resources(
            result.stdout + result.stderr,
            RTX5090_DPPP_UNIFORM_RYS4_RESOURCE_LIMITS,
        )


@pytest.mark.parametrize(
    ("name", "ordinary_limit", "persistent_limit"),
    (
        ("dpps", (216, 56, 36360), (218, 56, 36360)),
        ("dspp", (214, 56, 36360), (216, 56, 36360)),
        ("pppp", (230, 88, 36360), (232, 88, 36360)),
    ),
)
def test_rys3_uniform_warps_compile_without_spills_when_nvcc_is_configured(
    tmp_path: Path,
    name: str,
    ordinary_limit: tuple[int, int, int],
    persistent_limit: tuple[int, int, int],
) -> None:
    """Reject uniform Rys3 mappings that exceed their resource envelope."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
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
        spec, schedule=schedule, recurrence="rys3", target=TEST_CUDA_TARGET
    )
    source = tmp_path / f"generated_{name}_uniform_warp_rys3.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(spec, plan),
        encoding="utf-8",
    )
    cubin = tmp_path / f"generated_{name}_uniform_warp_rys3.cubin"
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-O3",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(cubin),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if os.environ.get("GENERATIVEQC_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert cubin.exists()
    if cuda_architecture == "sm_120":
        function_prefix = f"generated_{name}_shell_class_force"
        assert_rtx5090_resources(
            result.stdout + result.stderr,
            {
                f"{function_prefix}_rhf_kernel": ordinary_limit,
                f"{function_prefix}_uhf_kernel": ordinary_limit,
                f"{function_prefix}_rhf_persistent_kernel": persistent_limit,
                f"{function_prefix}_uhf_persistent_kernel": persistent_limit,
            },
        )


@pytest.mark.parametrize(
    ("name", "block_threads", "resource_limit"),
    (
        ("dpps", 64, (168, 120, 656)),
        ("dsps", 32, (166, 96, 584)),
        ("pppp", 96, (168, 128, 728)),
    ),
)
def test_cooperative_rys3_hot_classes_compile_without_spills_when_nvcc_is_configured(
    tmp_path: Path,
    name: str,
    block_threads: int,
    resource_limit: tuple[int, int, int],
) -> None:
    """Apply a zero-spill sm_120 gate to every promoted Rys3 force class."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
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
    source = tmp_path / f"generated_{name}_cooperative_rys3.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(spec, plan),
        encoding="utf-8",
    )
    cubin = tmp_path / f"generated_{name}_cooperative_rys3.cubin"
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-O3",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(cubin),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    output = result.stdout + result.stderr
    if os.environ.get("GENERATIVEQC_NVCC_VERBOSE"):
        print(output)
    assert result.returncode == 0, output
    if cuda_architecture == "sm_120":
        class_name = name[0].upper() + name[1:]
        assert_rtx5090_resources(
            output,
            {
                f"generated_{name}_shell_class_force_rhf_kernel": resource_limit,
                f"generated_{name}_shell_class_force_uhf_kernel": resource_limit,
                f"generated_{name}_shell_class_force_rhf_persistent_kernel": (
                    resource_limit
                ),
                f"generated_{name}_shell_class_force_uhf_persistent_kernel": (
                    resource_limit
                ),
            },
        )
        assert f"Generated{class_name}Rys3Primitive" in source.read_text(
            encoding="utf-8"
        )


def test_ppps_rys3_benchmark_runs_against_component_lanes_when_nvcc_is_configured(
    tmp_path: Path,
) -> None:
    """Compare direct Rys recurrence with the current ppps task topology."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA benchmark gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    direct_schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=8,
    )
    direct_plan = build_fused_shell_plan(
        spec, schedule=direct_schedule, recurrence="rys3", target=TEST_CUDA_TARGET
    )
    production_plan = next(
        build_fused_shell_plan(
            spec,
            consumers=selection.consumers,
            schedule=selection.schedule,
            target=TEST_CUDA_TARGET,
        )
        for selection in load_production_kernel_selections(
            REPOSITORY_ROOT
            / "python"
            / "generativeqc_compiler"
            / "integral"
            / "production_shell_classes.json",
            "sm_120",
        )
        if selection.spec == spec
    )
    environment = dict(os.environ)

    def compile_and_run(label: str, plan: typing.Any) -> dict[str, object]:
        source = tmp_path / f"generated_ppps_{label}_benchmark.cu"
        source.write_text(
            emit_shell_class_benchmark_cuda(
                spec,
                task_count=512,
                primitive_count=2,
                warmups=1,
                iterations=3,
                samples=3,
                plan=plan,
                benchmark_kernel_only=True,
                persistent_kernel=True,
            ),
            encoding="utf-8",
        )
        executable = tmp_path / f"generated_ppps_{label}_benchmark"
        compile_result = subprocess.run(
            [
                nvcc,
                "-std=c++17",
                f"-arch={cuda_architecture}",
                "-O3",
                str(source),
                "-o",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert compile_result.returncode == 0, (
            compile_result.stdout + compile_result.stderr
        )
        run_result = subprocess.run(
            [
                "srun",
                "--partition=main",
                "--gres=gpu:5090:1",
                "--nodes=1",
                "--ntasks=1",
                "--time=00:03:00",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=210,
            env=environment,
        )
        assert run_result.returncode == 0, run_result.stdout + run_result.stderr
        payload = json.loads(run_result.stdout.strip().splitlines()[-1])
        assert payload["maximum_force_error"] <= (
            2.0e-10 * max(1.0, payload["maximum_force"])
        )
        return payload

    direct = compile_and_run("rys3", direct_plan)
    production = compile_and_run("component_lanes", production_plan)
    print(
        json.dumps(
            {
                "rys3": direct,
                "component_lanes": production,
                "speedup_vs_component_lanes": (
                    production["fused_ms"] / direct["fused_ms"]
                ),
            },
            sort_keys=True,
        )
    )


def test_dppp_cooperative_rys4_benchmark_runs_against_component_lanes_when_nvcc_is_configured(
    tmp_path: Path,
) -> None:
    """Measure 192-lane cooperative Rys4 against its subset-Wick predecessor."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA benchmark gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    rys4_schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=192,
        component_tile=DPPP_SPEC.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.MATERIALIZED,
        unroll_pair_terms=True,
        minimum_blocks_per_sm=2,
    )
    rys4_plan = build_fused_shell_plan(
        DPPP_SPEC, schedule=rys4_schedule, recurrence="rys4", target=TEST_CUDA_TARGET
    )
    baseline_plan = build_fused_shell_plan(
        DPPP_SPEC,
        schedule=rys4_schedule,
        recurrence="subset_wick",
        target=TEST_CUDA_TARGET,
    )
    environment = dict(os.environ)

    def compile_and_run(label: str, plan: typing.Any) -> dict[str, object]:
        source = tmp_path / f"generated_dppp_{label}_benchmark.cu"
        source.write_text(
            emit_shell_class_benchmark_cuda(
                DPPP_SPEC,
                task_count=8192,
                primitive_count=3,
                warmups=1,
                iterations=3,
                samples=3,
                plan=plan,
                benchmark_kernel_only=True,
                persistent_kernel=True,
            ),
            encoding="utf-8",
        )
        executable = tmp_path / f"generated_dppp_{label}_benchmark"
        compiled = subprocess.run(
            [
                nvcc,
                "-std=c++17",
                f"-arch={cuda_architecture}",
                "-O3",
                str(source),
                "-o",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert compiled.returncode == 0, compiled.stdout + compiled.stderr
        run = subprocess.run(
            [
                "srun",
                "--partition=main",
                "--gres=gpu:5090:1",
                "--nodes=1",
                "--ntasks=1",
                "--time=00:05:00",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=330,
            env=environment,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        payload = json.loads(run.stdout.strip().splitlines()[-1])
        assert payload["maximum_force_error"] <= (
            2.0e-10 * max(1.0, payload["maximum_force"])
        )
        return payload

    rys4 = compile_and_run("cooperative_rys4", rys4_plan)
    baseline = compile_and_run("component_lanes", baseline_plan)
    result = {
        "cooperative_rys4": rys4,
        "component_lanes": baseline,
        "speedup_vs_component_lanes": (baseline["fused_ms"] / rys4["fused_ms"]),
    }
    print(json.dumps(result, sort_keys=True))
    assert result["speedup_vs_component_lanes"] > 1.0


def test_dppp_uniform_warp_rys4_benchmark_runs_against_component_lanes_when_nvcc_is_configured(
    tmp_path: Path,
) -> None:
    """Compare the 32-task mapping with the previously accepted force path."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA benchmark gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    uniform_schedule = ScheduleIR(
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
    uniform_plan = build_fused_shell_plan(
        DPPP_SPEC, schedule=uniform_schedule, recurrence="rys4", target=TEST_CUDA_TARGET
    )
    # Keep the comparison independent of the mutable production manifest.  If
    # the candidate is promoted, loading the manifest here would silently
    # benchmark the new kernel against itself and erase the rejection signal.
    component_lane_schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=192,
        component_tile=DPPP_SPEC.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.MATERIALIZED,
        unroll_pair_terms=True,
        minimum_blocks_per_sm=2,
    )
    component_lane_plan = build_fused_shell_plan(
        DPPP_SPEC,
        schedule=component_lane_schedule,
        recurrence="rys4",
        target=TEST_CUDA_TARGET,
    )
    environment = dict(os.environ)

    def compile_and_run(label: str, plan: typing.Any) -> dict[str, object]:
        source = tmp_path / f"generated_dppp_{label}_benchmark.cu"
        source.write_text(
            emit_shell_class_benchmark_cuda(
                DPPP_SPEC,
                task_count=8192,
                primitive_count=3,
                warmups=1,
                iterations=3,
                samples=3,
                plan=plan,
                benchmark_kernel_only=True,
                persistent_kernel=True,
            ),
            encoding="utf-8",
        )
        executable = tmp_path / f"generated_dppp_{label}_benchmark"
        compiled = subprocess.run(
            [
                nvcc,
                "-std=c++17",
                f"-arch={cuda_architecture}",
                "-O3",
                str(source),
                "-o",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert compiled.returncode == 0, compiled.stdout + compiled.stderr
        run = subprocess.run(
            [
                "srun",
                "--partition=main",
                "--gres=gpu:5090:1",
                "--nodes=1",
                "--ntasks=1",
                "--time=00:05:00",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=330,
            env=environment,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        payload = json.loads(run.stdout.strip().splitlines()[-1])
        assert payload["maximum_force_error"] <= (
            2.0e-10 * max(1.0, payload["maximum_force"])
        )
        return payload

    uniform = compile_and_run("uniform_warp_rys4", uniform_plan)
    component_lanes = compile_and_run("component_lane_rys4", component_lane_plan)
    result = {
        "uniform_warp_rys4": uniform,
        "component_lane_rys4": component_lanes,
        "speedup_vs_component_lanes": (
            component_lanes["fused_ms"] / uniform["fused_ms"]
        ),
    }
    print(json.dumps(result, sort_keys=True))
    assert result["speedup_vs_component_lanes"] > 1.0


@pytest.mark.parametrize("name", ("dpps", "dpss", "dsps", "dspp", "pppp"))
def test_cooperative_rys3_benchmark_runs_against_component_lanes_when_nvcc_is_configured(
    tmp_path: Path,
    name: str,
) -> None:
    """Gate each promoted Rys3 class against its accepted force recurrence."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA benchmark gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    selection = next(
        selection
        for selection in load_production_kernel_selections(
            REPOSITORY_ROOT
            / "python"
            / "generativeqc_compiler"
            / "integral"
            / "production_shell_classes.json",
            "sm_120",
        )
        if selection.spec == spec
    )
    rys3_plan = build_fused_shell_plan(
        spec, schedule=selection.schedule, recurrence="rys3", target=TEST_CUDA_TARGET
    )
    baseline_plan = build_fused_shell_plan(
        spec,
        schedule=selection.schedule,
        recurrence="subset_wick",
        target=TEST_CUDA_TARGET,
    )
    environment = dict(os.environ)

    def compile_and_run(label: str, plan: typing.Any) -> dict[str, object]:
        source = tmp_path / f"generated_{name}_{label}_benchmark.cu"
        source.write_text(
            emit_shell_class_benchmark_cuda(
                spec,
                task_count=8192,
                primitive_count=3,
                warmups=1,
                iterations=3,
                samples=3,
                plan=plan,
                benchmark_kernel_only=True,
                persistent_kernel=True,
            ),
            encoding="utf-8",
        )
        executable = tmp_path / f"generated_{name}_{label}_benchmark"
        compiled = subprocess.run(
            [
                nvcc,
                "-std=c++17",
                f"-arch={cuda_architecture}",
                "-O3",
                str(source),
                "-o",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert compiled.returncode == 0, compiled.stdout + compiled.stderr
        run = subprocess.run(
            [
                "srun",
                "--partition=main",
                "--gres=gpu:5090:1",
                "--nodes=1",
                "--ntasks=1",
                "--time=00:05:00",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=330,
            env=environment,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        payload = json.loads(run.stdout.strip().splitlines()[-1])
        assert payload["maximum_force_error"] <= (
            2.0e-10 * max(1.0, payload["maximum_force"])
        )
        return payload

    rys3 = compile_and_run("cooperative_rys3", rys3_plan)
    baseline = compile_and_run("component_lanes", baseline_plan)
    result = {
        "shell_class": name,
        "cooperative_rys3": rys3,
        "component_lanes": baseline,
        "speedup_vs_component_lanes": baseline["fused_ms"] / rys3["fused_ms"],
    }
    print(json.dumps(result, sort_keys=True))
    assert result["speedup_vs_component_lanes"] > 1.0


@pytest.mark.parametrize(
    ("spec", "resource_limits"),
    (
        (DPPP_SPEC, RTX5090_DPPP_RESOURCE_LIMITS),
        (DPDS_SPEC, RTX5090_DPDS_RESOURCE_LIMITS),
        (DDPS_SPEC, RTX5090_DDPS_RESOURCE_LIMITS),
    ),
)
def test_fused_cuda_compiles_when_nvcc_is_configured(
    tmp_path: Path, spec: typing.Any, resource_limits: typing.Any
) -> None:
    """Compile every generated shell class for explicit resource probes."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    source = tmp_path / f"generated_{spec.name}_fused.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(spec, target=TEST_CUDA_TARGET),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(tmp_path / f"generated_{spec.name}_fused.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if os.environ.get("GENERATIVEQC_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    if cuda_architecture == "sm_120" and resource_limits is not None:
        assert_rtx5090_resources(result.stdout + result.stderr, resource_limits)


def test_ppps_scalar_thread_cuda_compiles_without_spills_when_nvcc_is_configured(
    tmp_path: Path,
) -> None:
    """Gate the scalar ppps prototype before any production routing."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=8,
    )
    source = tmp_path / "generated_ppps_scalar_thread.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(
            spec,
            build_fused_shell_plan(spec, schedule=schedule, target=TEST_CUDA_TARGET),
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(tmp_path / "generated_ppps_scalar_thread.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if os.environ.get("GENERATIVEQC_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    if cuda_architecture == "sm_120":
        assert_rtx5090_resources(
            result.stdout + result.stderr,
            RTX5090_PPPS_SCALAR_THREAD_RESOURCE_LIMITS,
        )
        resource_records = re.findall(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
            r"(\d+) bytes spill loads",
            result.stdout + result.stderr,
        )
        assert resource_records
        assert all(tuple(map(int, record)) == (0, 0, 0) for record in resource_records)


def test_ppps_scalar_thread_benchmark_runs_when_nvcc_is_configured(
    tmp_path: Path,
) -> None:
    """Execute the scalar persistent worker against the component oracle."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA benchmark gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=8,
    )
    environment = dict(os.environ)
    if environment.get("CUDA_VISIBLE_DEVICES") == "":
        environment.pop("CUDA_VISIBLE_DEVICES")

    def compile_and_run(
        label: str,
        selected_schedule: ScheduleIR,
    ) -> dict[str, object]:
        source = tmp_path / f"generated_ppps_{label}_benchmark.cu"
        source.write_text(
            emit_shell_class_benchmark_cuda(
                spec,
                task_count=512,
                primitive_count=2,
                warmups=1,
                iterations=3,
                samples=3,
                schedule=selected_schedule,
                benchmark_kernel_only=True,
                persistent_kernel=True,
                target=TEST_CUDA_TARGET,
            ),
            encoding="utf-8",
        )
        executable = tmp_path / f"generated_ppps_{label}_benchmark"
        compile_result = subprocess.run(
            [
                nvcc,
                "-std=c++17",
                f"-arch={cuda_architecture}",
                "-O3",
                str(source),
                "-o",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert compile_result.returncode == 0, (
            compile_result.stdout + compile_result.stderr
        )
        run_result = subprocess.run(
            [str(executable)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
        assert run_result.returncode == 0, run_result.stdout + run_result.stderr
        payload = json.loads(run_result.stdout.strip().splitlines()[-1])
        assert payload["consumer"] == "force"
        assert payload["topology"] == "persistent_shared"
        assert payload["maximum_force_error"] <= (
            2.0e-10 * max(1.0, payload["maximum_force"])
        )
        return payload

    production_schedule = next(
        selection.schedule
        for selection in load_production_kernel_selections(
            REPOSITORY_ROOT
            / "python"
            / "generativeqc_compiler"
            / "integral"
            / "production_shell_classes.json",
            "sm_120",
        )
        if selection.spec == spec
    )
    scalar_payload = compile_and_run("scalar_thread", schedule)
    production_payload = compile_and_run(
        "component_lanes",
        production_schedule,
    )
    print(
        json.dumps(
            {
                "scalar_thread_ms": scalar_payload["fused_ms"],
                "component_lanes_ms": production_payload["fused_ms"],
                "speedup_vs_component_lanes": (
                    production_payload["fused_ms"] / scalar_payload["fused_ms"]
                ),
            },
            sort_keys=True,
        )
    )


def test_joint_fock_force_cuda_compiles_when_nvcc_is_configured(
    tmp_path: Path,
) -> None:
    """Compile the dual-consumer pilot through the real CUDA frontend."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    plan = build_fused_shell_plan(
        DPDS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        target=TEST_CUDA_TARGET,
    )
    source = tmp_path / "generated_dpds_fock_force.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(DPDS_SPEC, plan),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            str(source),
            "-o",
            str(tmp_path / "generated_dpds_fock_force.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_tiled_joint_fock_force_cuda_compiles_when_nvcc_is_configured(
    tmp_path: Path,
) -> None:
    """Compile the tiled dual-consumer lowering through the real frontend."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    integral = build_integral_ir(
        DPPP_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    schedule = next(
        item
        for item in schedule_candidates(integral, target=TEST_CUDA_TARGET)
        if item.kind == ScheduleKind.TILED_COMPONENTS and item.component_tile == 64
    )
    plan = build_fused_shell_plan(
        DPPP_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
        target=TEST_CUDA_TARGET,
    )
    source = tmp_path / "generated_dppp_tiled_fock_force.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(DPPP_SPEC, plan),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            str(source),
            "-o",
            str(tmp_path / "generated_dppp_tiled_fock_force.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dddd_tiled_cuda_compiles_when_nvcc_is_configured(tmp_path: Path) -> None:
    """Compile a 1296-component class that cannot use one lane per quartet."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    schedule = replace(
        build_fused_shell_plan(DDDD_SPEC, target=TEST_CUDA_TARGET).schedule,
        block_threads=128,
        component_tile=128,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.RECOMPUTED,
        unroll_pair_terms=True,
    )
    plan = build_fused_shell_plan(DDDD_SPEC, schedule=schedule, target=TEST_CUDA_TARGET)
    source = tmp_path / "generated_dddd_tiled.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(DDDD_SPEC, plan),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(tmp_path / "generated_dddd_tiled.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("spec", "consumers"),
    (
        (FFPS_SPEC, (KernelConsumer.FOCK, KernelConsumer.FORCE)),
        (FDDD_SPEC, (KernelConsumer.FORCE,)),
    ),
)
def test_f_shell_cuda_compiles_when_nvcc_is_configured(
    tmp_path: Path, spec: typing.Any, consumers: typing.Any
) -> None:
    """Compile pair-order-six and tiled f-shell gradients with CUDA 12.9."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    schedule = replace(
        build_fused_shell_plan(spec, target=TEST_CUDA_TARGET).schedule,
        unroll_pair_terms=False,
    )
    if spec == FDDD_SPEC:
        schedule = replace(
            schedule,
            block_threads=128,
            component_tile=128,
            pair_storage=PairStorage.RECOMPUTED,
        )
    plan = build_fused_shell_plan(
        spec, consumers=consumers, schedule=schedule, target=TEST_CUDA_TARGET
    )
    source = tmp_path / f"generated_{spec.name}.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(spec, plan),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(tmp_path / f"generated_{spec.name}.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("name", "recurrence", "schedule"),
    (
        (
            "fsss",
            "rys3",
            ScheduleIR(
                kind=ScheduleKind.THREAD_TASKS,
                block_threads=32,
                component_tile=10,
                tasks_per_warp=32,
                shared_coulomb=False,
                minimum_blocks_per_sm=1,
            ),
        ),
        (
            "fpps",
            "rys4",
            ScheduleIR(
                kind=ScheduleKind.SUBGROUP_TASKS,
                block_threads=256,
                component_tile=90,
                tasks_per_warp=4,
                shared_coulomb=True,
                minimum_blocks_per_sm=1,
            ),
        ),
    ),
)
def test_structural_rys_capability_examples_compile_when_nvcc_is_configured(
    tmp_path: Path,
    name: str,
    recurrence: str,
    schedule: ScheduleIR,
) -> None:
    """Compile f-shell candidates admitted without shell-name allowlists."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    plan = build_fused_shell_plan(
        spec, schedule=schedule, recurrence=recurrence, target=TEST_CUDA_TARGET
    )
    source = tmp_path / f"generated_{name}_{recurrence}_capability.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(spec, plan),
        encoding="utf-8",
    )
    cubin = tmp_path / f"generated_{name}_{recurrence}_capability.cubin"
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-O3",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(cubin),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if os.environ.get("GENERATIVEQC_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert cubin.exists()
    if cuda_architecture == "sm_120":
        resource_records = re.findall(
            r"\d+ bytes stack frame, (\d+) bytes spill stores, "
            r"(\d+) bytes spill loads",
            result.stdout + result.stderr,
        )
        assert resource_records
        assert all(tuple(map(int, record)) == (0, 0) for record in resource_records)


def test_psss_shell_task_cuda_compiles_when_nvcc_is_configured(
    tmp_path: Path,
) -> None:
    """Compile a zero-order ket pair through generated Fock/force lowering."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    integral = build_integral_ir(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    schedule = next(
        item
        for item in schedule_candidates(integral, target=TEST_CUDA_TARGET)
        if item.kind == ScheduleKind.SHELL_TASK
    )
    plan = build_fused_shell_plan(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
        target=TEST_CUDA_TARGET,
    )
    source = tmp_path / "generated_psss_shell_task.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(PSSS_SPEC, plan),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            str(source),
            "-o",
            str(tmp_path / "generated_psss_shell_task.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_psss_packed_cuda_compiles_when_nvcc_is_configured(
    tmp_path: Path,
) -> None:
    """Compile 32 independent low-order tasks per warp for Fock and force."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    integral = build_integral_ir(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    schedule = next(
        item
        for item in schedule_candidates(integral, target=TEST_CUDA_TARGET)
        if item.kind == ScheduleKind.PACKED_TASKS
    )
    plan = build_fused_shell_plan(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
        target=TEST_CUDA_TARGET,
    )
    source = tmp_path / "generated_psss_packed.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(PSSS_SPEC, plan),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(tmp_path / "generated_psss_packed.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("name", "resource_limits"),
    (
        ("psps", RTX5090_PSPS_RESOURCE_LIMITS),
        ("ppss", RTX5090_PPSS_RESOURCE_LIMITS),
    ),
)
def test_low_order_production_rys2_cuda_compiles_when_nvcc_is_configured(
    tmp_path: Path,
    name: str,
    resource_limits: dict[str, tuple[int, int, int]],
) -> None:
    """Compile the common production Rys2 source and reject spills."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
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
    source = tmp_path / f"generated_{name}_production_rys2.cu"
    source.write_text(
        _PRODUCTION_PRELUDE.replace('#include "scf/generated_shell_task.hpp"\n', "")
        + "#include <cstddef>\n#include <cstdint>\n"
        + emit_shell_class_fused_cuda(
            selection.spec,
            plan,
            fock_schedule=selection.fock_schedule,
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(tmp_path / f"generated_{name}_production_rys2.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if os.environ.get("GENERATIVEQC_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    if cuda_architecture == "sm_120":
        assert_rtx5090_resources(result.stdout + result.stderr, resource_limits)


def test_high_component_fock_oracle_block_covers_every_component() -> None:
    """Do not truncate the independent oracle for subgroup candidates."""

    trial = next(
        trial
        for trial in supported_schedule_trials(
            FUSED_SHELL_SPEC_BY_NAME["dppp"],
            KernelConsumer.FOCK,
            target=TEST_CUDA_TARGET,
        )
        if trial.schedule.kind == ScheduleKind.SUBGROUP_TASKS
        and trial.schedule.block_threads == 128
        and trial.schedule.tasks_per_warp == 4
        and trial.schedule.pair_orientation == PairOrientation.SWAPPED
        and trial.schedule.pair_storage == PairStorage.MATERIALIZED
        and trial.schedule.unroll_pair_terms
        and trial.schedule.minimum_blocks_per_sm == 0
    )
    source = emit_shell_class_benchmark_cuda(
        FUSED_SHELL_SPEC_BY_NAME["dppp"],
        task_count=1,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
        consumer=KernelConsumer.FOCK,
        schedule=trial.schedule,
        target=TEST_CUDA_TARGET,
    )
    baseline = source.split("/** Per-component Fock baseline", maxsplit=1)[1]
    assert "__launch_bounds__(192)" in baseline
    assert "<<<kTaskCount,\n        192>>>" in baseline


def test_fock_benchmark_runs_when_nvcc_is_configured(tmp_path: Path) -> None:
    """Execute the swapped value benchmark and its independent oracle."""

    nvcc = os.environ.get("GENERATIVEQC_NVCC")
    if nvcc is None:
        pytest.skip("set GENERATIVEQC_NVCC to run the generated CUDA benchmark gate")
    cuda_architecture = os.environ.get("GENERATIVEQC_CUDA_ARCH", "sm_90")
    schedule = replace(
        build_fused_shell_plan(DPDS_SPEC, target=TEST_CUDA_TARGET).schedule,
        pair_orientation=PairOrientation.SWAPPED,
    )
    source = tmp_path / "generated_dpds_fock_benchmark.cu"
    source.write_text(
        emit_shell_class_benchmark_cuda(
            DPDS_SPEC,
            task_count=2,
            primitive_count=1,
            warmups=0,
            iterations=1,
            samples=1,
            consumer=KernelConsumer.FOCK,
            schedule=schedule,
            target=TEST_CUDA_TARGET,
        ),
        encoding="utf-8",
    )
    executable = tmp_path / "generated_dpds_fock_benchmark"
    compile_result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-O3",
            str(source),
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    environment = dict(os.environ)
    if environment.get("CUDA_VISIBLE_DEVICES") == "":
        environment.pop("CUDA_VISIBLE_DEVICES")
    run_result = subprocess.run(
        [str(executable)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    payload = json.loads(run_result.stdout.strip().splitlines()[-1])
    assert payload["consumer"] == "fock"
    assert payload["maximum_fock_error"] <= (
        2.0e-10 * max(1.0, payload["maximum_fock"])
    )
