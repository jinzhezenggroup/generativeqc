"""Keep all CUDA jobs mandatory and resource sources owned by the CMake graph."""

from __future__ import annotations

import glob
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_ci_prose_workflow import assert_full_coverage_step

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/ci.yml").read_text()


def _job(name: str) -> str:
    section = WORKFLOW.split(f"\n  {name}:\n", 1)[1]
    return re.split(r"\n  [a-zA-Z0-9_-]+:\n", section, maxsplit=1)[0]


def _step(job: str, name: str) -> str:
    section = _job(job).split(f"      - name: {name}\n", 1)[1]
    return section.split("      - name: ", 1)[0]


def _host_commands(step: str) -> list[list[str]]:
    script = step.split("        run: |\n", 1)[1].replace("\\\n", " ")
    commands = []
    for line in script.splitlines():
        lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        commands.append(list(lexer))
    return commands


def _option_values(argv: list[str], names: tuple[str, ...]) -> list[str]:
    values = []
    for index, arg in enumerate(argv):
        for name in names:
            if arg == name:
                assert index + 1 < len(argv)
                values.append(argv[index + 1])
            elif arg.startswith(name + "="):
                values.append(arg[len(name) + 1 :])
            elif len(name) == 2 and arg.startswith(name):
                values.append(arg[2:])
    return values


def _assert_host_preflight_commands(step: str) -> None:
    required = {
        "generativeqc_fock_build_tests",
        "generativeqc_scf_proposal_tests",
        "generativeqc_ordered_gram_host_admission_tests",
    }
    commands = _host_commands(step)
    builds = [argv for argv in commands if argv[:2] == ["cmake", "--build"]]
    runs = [argv for argv in commands if argv[:1] == ["ctest"]]
    assert len(builds) == len(runs) == 1
    assert not {";", "&", "&&", "|", "||"}.intersection(builds[0] + runs[0])
    targets = builds[0][builds[0].index("--target") + 1 :]
    end = next(
        (i for i, arg in enumerate(targets) if arg.startswith("-")), len(targets)
    )
    assert required <= set(targets[:end])
    run = runs[0]
    assert _option_values(run, ("--test-dir",)) == ["build/cuda-release-sm120"]
    assert "--output-on-failure" in run
    selection = _option_values(run, ("-R", "--tests-regex"))
    assert len(selection) == 1
    assert all(re.search(selection[0], test) for test in required)
    exclusions = _option_values(run, ("-E", "--exclude-regex"))
    assert not any(
        re.search(pattern, test) for pattern in exclusions for test in required
    )
    # Other narrowing modes need actual label/index/file state, and dry runs do
    # not execute the mandatory host checks. Neither belongs to this gate.
    forbidden = (
        "-I",
        "--tests-information",
        "-L",
        "--label-regex",
        "-LE",
        "--label-exclude",
        "--tests-from-file",
        "--exclude-from-file",
        "--rerun-failed",
        "-N",
        "--show-only",
    )
    assert not any(
        arg == flag
        or arg.startswith(flag + "=")
        or (len(flag) == 2 and arg.startswith(flag))
        for arg in run
        for flag in forbidden
    )


def test_all_cuda_jobs_remain_required_with_the_same_source_and_finite_budgets() -> (
    None
):
    for name in ("cuda-compile", "cuda-resources", "cuda-direct-aot"):
        job = _job(name)
        header = job.split("    steps:\n", 1)[0]
        assert "needs: merge_queue_liveness" in header
        assert "if: needs.merge_queue_liveness.outputs.active != 'false'" in header
        expected_minutes = 50 if name == "cuda-compile" else 25
        assert f"timeout-minutes: {expected_minutes}" in header
        assert "continue-on-error:" not in header
        checkout = job.split("      - uses: actions/checkout@", 1)[1]
        checkout = checkout.split("      - name: ", 1)[0]
        assert "ref: ${{ github.sha }}" in checkout
        assert "persist-credentials: false" in checkout
        assert "CUDACXX: /usr/local/cuda-12.9/bin/nvcc" in header
    aggregate = _job("pass")
    needs = re.search(r"needs: \[([^\]]+)\]", aggregate)
    assert needs is not None
    assert {
        "merge_queue_liveness",
        "cpu",
        "cuda-compile",
        "cuda-resources",
        "cuda-direct-aot",
        "python",
    } <= {name.strip() for name in needs.group(1).split(",")}
    assert "jobs: ${{ toJSON(needs) }}" in aggregate
    assert "allowed-failures:" not in aggregate
    # The exact event-dependent skip set is exercised by test_native_cli_workflow:
    # only the scheduled artifact job is exempt, never a CUDA gate.


@pytest.mark.parametrize(
    "job,name,command",
    [
        (
            "cuda-compile",
            "Compile production CUDA and runtime CTest",
            "--target generativeqc_cuda_runtime_tests --parallel 3",
        ),
        (
            "cuda-compile",
            "Compile opted-in AO radial kernels with release resources",
            "test_cuda_ao_radial_reuse.py::test_emitted_ao_cuda_compiles_with_specialized_resources",
        ),
        (
            "cuda-compile",
            "Check Fock and proposal preflight without a GPU",
            None,
        ),
        (
            "cuda-resources",
            "Compile five representative f classes with release resources",
            "tools/validate_f_shells.py --tier compile --smoke",
        ),
        (
            "cuda-resources",
            "Compile generated XC derivative consumers",
            "tools/validate_xc.py --tier cuda-compile",
        ),
        (
            "cuda-resources",
            "Compile native grid/XC region resources",
            "tools/validate_native_xc_resources.py",
        ),
        (
            "cuda-resources",
            "Compile generated r2SCAN stationary CUDA consumer",
            "tools/validate_stationary_cuda_compile.py",
        ),
        (
            "cuda-resources",
            "Compile generated TensorIR static-data artifacts",
            "tools/validate_tensor_cuda_compile.py",
        ),
        (
            "cuda-direct-aot",
            "Qualify direct J/K AOT static resources (shard 6)",
            "tools/validate_direct_jk_aot_resources.py",
        ),
    ],
)
def test_existing_gates_are_mandatory_for_full_scope_and_owned_once(
    job: str, name: str, command: str | None
) -> None:
    step = _step(job, name)
    if command is None:
        _assert_host_preflight_commands(step)
    else:
        assert command in step
    assert_full_coverage_step(step)
    assert "continue-on-error:" not in step
    assert WORKFLOW.count(f"      - name: {name}\n") == 1


def _host_step_with_commands(step: str, commands: list[list[str]]) -> str:
    prefix = step.split("        run: |\n", 1)[0]
    return (
        prefix
        + "        run: |\n"
        + "".join("          " + shlex.join(argv) + "\n" for argv in commands if argv)
    )


@pytest.mark.parametrize("surface", ("build", "selection"))
@pytest.mark.parametrize(
    "missing",
    (
        "generativeqc_fock_build_tests",
        "generativeqc_scf_proposal_tests",
        "generativeqc_ordered_gram_host_admission_tests",
    ),
)
def test_host_preflight_rejects_missing_builds_or_selections(
    surface: str, missing: str
) -> None:
    step = _step("cuda-compile", "Check Fock and proposal preflight without a GPU")
    commands = _host_commands(step)
    if surface == "build":
        build = next(argv for argv in commands if argv[:2] == ["cmake", "--build"])
        build.remove(missing)
    else:
        remaining = {
            "generativeqc_fock_build_tests",
            "generativeqc_scf_proposal_tests",
            "generativeqc_ordered_gram_host_admission_tests",
        } - {missing}
        run = next(argv for argv in commands if argv[:1] == ["ctest"])
        run[run.index("-R") + 1] = "^(" + "|".join(sorted(remaining)) + ")$"
    with pytest.raises(AssertionError):
        _assert_host_preflight_commands(_host_step_with_commands(step, commands))


def test_host_preflight_accepts_equivalent_regex_order() -> None:
    step = _step("cuda-compile", "Check Fock and proposal preflight without a GPU")
    commands = _host_commands(step)
    run = next(argv for argv in commands if argv[:1] == ["ctest"])
    run[run.index("-R") + 1] = (
        "^(generativeqc_ordered_gram_host_admission_tests|"
        "generativeqc_scf_proposal_tests|generativeqc_fock_build_tests)$"
    )
    _assert_host_preflight_commands(_host_step_with_commands(step, commands))


@pytest.mark.parametrize(
    "extra", (("-E", "ordered_gram_host_admission"), ("-N",), ("||", "true"))
)
def test_host_preflight_rejects_exclusion_dry_run_or_softening(
    extra: tuple[str, ...],
) -> None:
    step = _step("cuda-compile", "Check Fock and proposal preflight without a GPU")
    commands = _host_commands(step)
    run = next(argv for argv in commands if argv[:1] == ["ctest"])
    run.extend(extra)
    # Preserve shell operators as operators, rather than quoting them as data.
    broken = _host_step_with_commands(step, commands).replace("'||'", "||")
    with pytest.raises(AssertionError):
        _assert_host_preflight_commands(broken)


def test_resources_generate_the_native_cmake_source_without_rebuilding_library() -> (
    None
):
    production = _step("cuda-compile", "Configure production CUDA compile check")
    resources = _step("cuda-resources", "Configure native grid source generation")
    assert (
        production.split("        run: >-\n", 1)[1].strip()
        == resources.split("        run: >-\n", 1)[1].strip()
    )
    generation = _step(
        "cuda-resources", "Generate exact native grid source and prerequisite headers"
    )
    assert "cmake --build --preset cuda-release-sm120" in generation
    assert "--target generated/generated_grid_policy.cu --parallel 2" in generation
    assert "generativeqc_cuda_runtime_tests" not in _job("cuda-resources")
    validation = _step("cuda-resources", "Compile native grid/XC region resources")
    assert "--generated-dir build/cuda-release-sm120/generated" in validation
    assert (
        "--cmake-source build/cuda-release-sm120/generated/generated_grid_policy.cu"
        in validation
    )
    assert "--functional PBE --npoint 4096 --tile-points 256 --nao 96 --spins 2" in (
        validation
    )
    reports = _step("cuda-resources", "Preserve generated XC resource reports")
    assert "if: always()" in reports
    assert "build/xc-compile" in reports
    assert "build/native-xc-resource" in reports


def test_direct_aot_has_an_independent_bounded_cold_compile_and_receipts() -> None:
    job = _job("cuda-direct-aot")
    header = job.split("    steps:\n", 1)[0]
    assert "github.event_name != 'push'" in header
    compile_step = _step(
        "cuda-direct-aot", "Qualify direct J/K AOT static resources (shard 6)"
    )
    job_minutes = int(re.search(r"timeout-minutes: (\d+)", header)[1])
    step_minutes = int(re.search(r"timeout-minutes: (\d+)", compile_step)[1])
    compiler_seconds = int(re.search(r"--compile-timeout (\d+)", compile_step)[1])
    # Reserve time for compiler-group shutdown/metadata, setup and publication.
    assert 0 < compiler_seconds <= (step_minutes - 2) * 60
    assert step_minutes <= job_minutes - 3
    assert "--record-output" not in compile_step
    assert "build/cuda-release-sm120/CMakeCache.txt" in compile_step
    assert "build/cuda-release-sm120/generated/production_shell_kernels" in compile_step
    assert (
        job.index("Install ccache 4.14")
        < job.index("Configure direct J/K AOT release context")
        < job.index("Qualify direct J/K AOT static resources")
    )
    # Sibling validators do not consume the direct shard's wall-clock budget.
    for tool in (
        "validate_f_shells.py",
        "validate_xc.py",
        "validate_native_xc_resources.py",
        "validate_stationary_cuda_compile.py",
        "validate_tensor_cuda_compile.py",
    ):
        assert tool not in job
    restore = _step("cuda-direct-aot", "Restore direct J/K AOT resource cache")
    save = _step("cuda-direct-aot", "Save direct J/K AOT resource cache")
    assert "path: build/direct-jk-aot-cache" in restore
    assert "direct-jk-aot-12.9-sm120-" in restore
    assert "github.event_name != 'merge_group'" in save
    assert "steps.direct_jk_aot_cache.outputs.cache-hit != 'true'" in save
    assert "always()" not in save and "continue-on-error:" not in save
    reports = _step(
        "cuda-direct-aot", "Preserve direct J/K AOT resource qualification report"
    )
    assert "if: always()" in reports
    for receipt in (
        "report.json",
        "cuobjdump.txt",
        "compile.json",
        "compiler-cache.json",
        "nvcc.log",
    ):
        assert receipt in reports


@pytest.mark.parametrize("active", ["true", "false", ""])
@pytest.mark.parametrize(
    "event,ref,expected",
    [
        ("pull_request", "refs/pull/1/merge", True),
        ("merge_group", "refs/heads/gh-readonly-queue/master/pr-1", True),
        ("schedule", "refs/heads/master", True),
        ("workflow_dispatch", "refs/heads/master", True),
        ("workflow_dispatch", "refs/heads/feature", True),
        ("push", "refs/heads/master", True),
        ("push", "refs/heads/feature", False),
        ("push", "refs/tags/v1.2.3", False),
    ],
)
def test_direct_aot_preserves_required_events_and_only_seeds_master_pushes(
    active: str, event: str, ref: str, expected: bool
) -> None:
    header = _job("cuda-direct-aot").split("    steps:\n", 1)[0]
    expression = re.search(r"^    if: (.+)$", header, re.MULTILINE)[1]
    expression = (
        expression.replace("needs.merge_queue_liveness.outputs.active", '"$ACTIVE"')
        .replace("github.event_name", '"$EVENT"')
        .replace("github.ref", '"$REF"')
    )
    result = subprocess.run(
        ["bash", "-c", f"if [[ {expression} ]]; then printf run; else printf skip; fi"],
        env={**os.environ, "ACTIVE": active, "EVENT": event, "REF": ref},
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout == ("run" if expected and active != "false" else "skip")


def _cache_patterns(prefix: str) -> list[str]:
    line = next(line for line in WORKFLOW.splitlines() if f"key: {prefix}" in line)
    return re.findall(r"'([^']+)'", line.split("hashFiles(", 1)[1])


def _selected_cache_inputs(root: Path, patterns: list[str]) -> list[Path]:
    selected: set[Path] = set()
    for pattern in patterns:
        exclude = pattern.startswith("!")
        matches = {
            root / value
            for value in glob.glob(
                pattern.removeprefix("!"), root_dir=root, recursive=True
            )
            if (root / value).is_file()
        }
        if exclude:
            selected -= matches
        else:
            selected |= matches
    return sorted(selected)


@pytest.mark.parametrize(
    "prefix",
    [
        "f-shell-cuda-v2-",
        "xc-cuda-",
        "stationary-cuda-",
        "tensor-static-cuda-",
        "direct-jk-aot-",
    ],
)
def test_resource_cache_keys_ignore_bytecode_but_track_source(
    tmp_path: Path, prefix: str
) -> None:
    patterns = _cache_patterns(prefix)
    source = tmp_path / "python/generativeqc_compiler/common/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n")

    def fingerprint() -> str:
        return hashlib.sha256(
            b"".join(
                path.read_bytes() for path in _selected_cache_inputs(tmp_path, patterns)
            )
        ).hexdigest()

    original = fingerprint()
    bytecode = source.parent / "__pycache__/example.cpython-311.pyc"
    bytecode.parent.mkdir()
    for generated in (
        bytecode,
        bytecode.with_suffix(".receipt"),
        source.with_suffix(".pyc"),
        source.with_suffix(".pyo"),
    ):
        generated.write_bytes(b"checkout-specific compiled bytecode")
    assert _selected_cache_inputs(tmp_path, patterns) == [source]
    assert fingerprint() == original
    bytecode.write_bytes(b"changed import timestamp")
    assert fingerprint() == original
    source.write_text("value = 2\n")
    assert fingerprint() != original


def test_direct_aot_transport_key_covers_inner_implementation_inputs(
    tmp_path: Path,
) -> None:
    from tools.validate_direct_jk_aot_resources import COMPILE_IMPLEMENTATIONS

    expected = []
    for name in COMPILE_IMPLEMENTATIONS:
        source = tmp_path / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(name)
        expected.append(source)
    assert _selected_cache_inputs(
        tmp_path, _cache_patterns("direct-jk-aot-")
    ) == sorted(expected)


def _assert_direct_aot_is_required(aggregate: str) -> None:
    dependencies = re.search(r"needs: \[([^\]]+)\]", aggregate)
    assert dependencies is not None
    assert "cuda-direct-aot" in {name.strip() for name in dependencies[1].split(",")}
    assert "jobs: ${{ toJSON(needs) }}" in aggregate
    assert "allowed-failures:" not in aggregate
    assert "continue-on-error:" not in aggregate
    allowed = re.search(r"allowed-skips: (.+)", aggregate)
    assert allowed is not None and "cuda-direct-aot" not in allowed[1]


@pytest.mark.parametrize("fault", ["missing", "skippable", "non_gating"])
def test_direct_aot_aggregate_rejects_weakened_gate_controls(fault: str) -> None:
    aggregate = _job("pass")
    _assert_direct_aot_is_required(aggregate)
    if fault == "missing":
        aggregate = aggregate.replace("cuda-direct-aot, ", "")
    elif fault == "skippable":
        aggregate = aggregate.replace(
            "'native-cli-artifact'", "'native-cli-artifact,cuda-direct-aot'"
        )
    else:
        aggregate += "\n    continue-on-error: true\n"
    with pytest.raises(AssertionError):
        _assert_direct_aot_is_required(aggregate)


@pytest.mark.skipif(
    not all(shutil.which(command) for command in ("cmake", "ninja", "ccache")),
    reason="requires the CI build tools",
)
def test_native_grid_file_target_generates_its_exact_header_closure(
    tmp_path: Path,
) -> None:
    """Exercise the real production declarations without enabling a CUDA compiler."""
    from generativeqc_compiler.dft.ao_cuda import emit_grid_source

    from tools.generate_libxc_semilocal_cuda_registry import emit_header, emit_registry

    subprocess.run(["ccache", "--version"], check=True, capture_output=True, timeout=10)
    for name in ("tools", "python", "src", "include", "manifests", "upstream", "data"):
        (tmp_path / name).symlink_to(ROOT / name, target_is_directory=True)
    (tmp_path / "dummy.cpp").write_text("int fixture;\n")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(CudaGridCodegen LANGUAGES CXX)\n"
        f'include("{ROOT / "cmake/GenerativeQCGenerated.cmake"}")\n'
        f'include("{ROOT / "cmake/GenerativeQCGeneratedSources.cmake"}")\n'
        f'include("{ROOT / "cmake/GenerativeQCCuda.cmake"}")\n'
        f'set(Python3_EXECUTABLE "{sys.executable}")\n'
        "set(GENERATIVEQC_ENABLE_CUDA ON)\n"
        "set(GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT OFF)\n"
        "set(CMAKE_CUDA_ARCHITECTURES 120)\n"
        "add_library(CUDA::cudart INTERFACE IMPORTED)\n"
        "add_library(generativeqc STATIC dummy.cpp)\n"
        "generativeqc_register_host_generated_sources(generativeqc)\n"
        "generativeqc_register_cuda_generated_sources(generativeqc)\n"
    )
    build = tmp_path / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "generated/generated_grid_policy.cu",
            "--parallel",
            "2",
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    generated = build / "generated"
    assert {
        path.relative_to(generated).as_posix()
        for path in generated.rglob("*")
        if path.is_file() and path.suffix != ".d"
    } == {
        "generated_grid_policy.cu",
        "generated_b3lyp_device.cuh",
        "generated_r2scan_device.cuh",
        "generated_wb97mv_device.cuh",
        "generated_split_hybrid_registry.cuh",
        "libxc_semilocal_cuda/generated_libxc_semilocal_cuda_registry.hpp",
        "libxc_semilocal_cuda/generated_libxc_semilocal_cuda_registry.cpp",
    }
    registry = generated / "libxc_semilocal_cuda"
    assert (registry / "generated_libxc_semilocal_cuda_registry.hpp").read_text() == (
        emit_header()
    )
    assert (registry / "generated_libxc_semilocal_cuda_registry.cpp").read_text() == (
        emit_registry(enabled=False)
    )
    assert (generated / "generated_grid_policy.cu").read_text() == emit_grid_source(
        native_ks=True
    )[0]
    assert not list(build.rglob("*.o"))
    assert not list(build.glob("*.a"))
    assert not list(build.glob("*.so"))
