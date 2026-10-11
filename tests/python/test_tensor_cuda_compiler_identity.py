"""CPU-only dependency and executable identity regression tests."""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    cuda_execute,
    input_tensor,
    reduce_sum,
)
from generativeqc_compiler.tensor import cuda_compiler_identity as identity
from generativeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda

from tools.tensor_reduction_qualification import qualification_reduction_plan


@pytest.fixture
def toolkit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Fake NVCC delegates only preprocessing to the available C++ preprocessor."""
    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("CPU-only dependency tests require g++ preprocessing")
    root = tmp_path / "toolkit"
    native = tmp_path / "native" / "src" / "tensor"
    native.mkdir(parents=True)
    (native / "cuda_graph_context.cuh").write_text("#include <cuda_runtime.h>\n")
    for relative, text in {
        "include/cuda_runtime.h": "#pragma once\n",
        "include/cub/block/block_reduce.cuh": "#include <cub/block/specializations/block_reduce_warp_reductions.cuh>\n#include <cub/detail.cuh>\n",
        "include/cub/block/specializations/block_reduce_warp_reductions.cuh": "#define WARP_REDUCTION 1\n",
        "include/cub/detail.cuh": "#define DETAIL_REDUCTION 2\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "bin").mkdir()
    driver = f"""#!{sys.executable}
import os, shlex, sys
from pathlib import Path
if sys.argv[1:] == ["--version"]:
    print("fixture compiler version 1")
    raise SystemExit(0)
if "-M" not in sys.argv:
    raise SystemExit("the fake compiler permits dependency preprocessing only")
flags = shlex.split(os.environ.get("NVCC_PREPEND_FLAGS", "")) + sys.argv[1:] + shlex.split(os.environ.get("NVCC_APPEND_FLAGS", ""))
command = [{cxx!r}, "-x", "c++", "-std=c++17", "-isystem", str(Path(__file__).resolve().parents[1] / "include")]
i = 0
while i < len(flags):
    flag = flags[i]
    if flag in ("-M",):
        command.append(flag)
    elif flag in ("-MF", "-MT", "-I", "-isystem"):
        command.extend(flags[i:i+2]); i += 1
    elif flag.startswith(("-I", "-D")) or flag.endswith(".cu"):
        command.append(flag)
    i += 1
os.execv({cxx!r}, command)
"""
    for name in ("nvcc", "ptxas", "gcc"):
        path = root / "bin" / name
        path.write_text(driver)
        path.chmod(0o700)
    for name in identity._ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "PATH", str(root / "bin") + os.pathsep + os.environ.get("PATH", "")
    )
    monkeypatch.setattr(identity, "asset_path", lambda _: native)
    monkeypatch.setattr(
        cuda_execute, "tensor_source_identity", lambda: "fixture-source"
    )
    identity._version.cache_clear()
    calls = []
    run = identity.run_compiler

    def record(command: list[str], timeout: float, **kwargs: Any) -> Any:
        calls.append((command, timeout))
        return run(command, timeout, **kwargs)

    monkeypatch.setattr(identity, "run_compiler", record)
    axis = Index("i", IndexSpace("axis", "batch", 129))
    values = input_tensor("values", TensorSpec((axis,), role="input"))
    plan = plan_cuda(
        Program({"result": reduce_sum(values, (0,))}),
        cuda_target_info("sm_80"),
        schedule=TensorSchedule(stream_reductions=True),
    )
    return (
        root,
        native,
        plan,
        CudaCompilerAdapter(root / "bin/nvcc", plan.target),
        calls,
    )


def test_transitive_cub_dependency_and_filename_association(toolkit: Any) -> None:
    root, _, plan, compiler, _ = toolkit
    before = cuda_execute.tensor_compiler_identity(plan, compiler)
    first = root / "include/cub/block/specializations/block_reduce_warp_reductions.cuh"
    second = root / "include/cub/detail.cuh"
    first.write_text("#define WARP_REDUCTION 3\n")
    changed = cuda_execute.tensor_compiler_identity(plan, compiler)
    assert changed != before
    a, b = first.read_text(), second.read_text()
    first.write_text(b)
    second.write_text(a)
    assert cuda_execute.tensor_compiler_identity(plan, compiler) != changed


@pytest.mark.parametrize(
    "environment", ["CPATH", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"]
)
def test_resolved_include_override_and_new_shadowing_header(
    toolkit: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, environment: str
) -> None:
    _, _, plan, compiler, _ = toolkit
    override = tmp_path / "override"
    override.mkdir()
    monkeypatch.setenv(
        environment, str(override) if environment == "CPATH" else f"-I{override}"
    )
    before = cuda_execute.tensor_compiler_identity(plan, compiler)
    shadow = override / "cub/block/specializations/block_reduce_warp_reductions.cuh"
    shadow.parent.mkdir(parents=True)
    shadow.write_text("#define OVERRIDE_REDUCTION 1\n")
    after = cuda_execute.tensor_compiler_identity(plan, compiler)
    assert after != before
    shadow.write_text("#define OVERRIDE_REDUCTION 2\n")
    assert cuda_execute.tensor_compiler_identity(plan, compiler) != after


@pytest.mark.parametrize("name", ["nvcc", "ptxas", "gcc"])
def test_same_banner_executable_replacement_invalidates_identity(
    toolkit: Any, name: str
) -> None:
    root, _, plan, compiler, _ = toolkit
    before = cuda_execute.tensor_compiler_identity(plan, compiler)
    executable = root / "bin" / name
    executable.write_text(executable.read_text() + "\n# replaced executable bytes\n")
    assert cuda_execute.tensor_compiler_identity(plan, compiler) != before


def test_versions_are_cached_but_header_resolution_is_refreshed(toolkit: Any) -> None:
    _, _, plan, compiler, calls = toolkit
    before = cuda_execute.tensor_compiler_identity(plan, compiler)
    assert cuda_execute.tensor_compiler_identity(plan, compiler) == before
    assert len([command for command, _ in calls if "--version" in command]) == 3
    dependency_commands = [command for command, _ in calls if "-M" in command]
    assert len(dependency_commands) == 4
    assert all(timeout == 30 for _, timeout in calls)
    assert all(
        "-arch=sm_80" in command and "--fmad=false" in command
        for command in dependency_commands
    )
    assert all("-MD" not in command and "-c" not in command for command, _ in calls)


def test_unused_header_does_not_invalidate_and_toolkit_relocation_is_portable(
    toolkit: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, _, plan, compiler, _ = toolkit
    before = cuda_execute.tensor_compiler_identity(plan, compiler)
    (root / "include/unused.h").write_text("unused header")
    assert cuda_execute.tensor_compiler_identity(plan, compiler) == before
    moved = tmp_path / "relocated"
    shutil.copytree(root, moved)
    monkeypatch.setenv("PATH", str(moved / "bin") + os.pathsep + os.environ["PATH"])
    relocated = replace(compiler, nvcc=moved / "bin/nvcc")
    assert cuda_execute.tensor_compiler_identity(plan, relocated) == before
    facts = cuda_execute._tensor_compiler_facts(plan, relocated)
    assert str(tmp_path) not in json.dumps(facts)


def test_bound_and_unbound_contexts_match_and_missing_cub_is_optional(
    toolkit: Any, tmp_path: Path
) -> None:
    root, _, plan, compiler, _ = toolkit
    before = cuda_execute.tensor_compiler_identity(plan, compiler)
    bound = qualification_reduction_plan(plan)
    assert cuda_execute.tensor_compiler_identity(bound, compiler) == before
    (root / "include/cub/block/block_reduce.cuh").unlink()
    absent = cuda_execute.tensor_compiler_identity(plan, compiler)
    assert absent != before
    assert cuda_execute.tensor_compiler_identity(bound, compiler) == absent
    assert (
        cuda_execute._tensor_compiler_facts(plan, compiler)["dependencies"]["cub"]
        is None
    )
    index, binding = bound.lowering_bindings[0]
    bound = replace(
        bound,
        lowering_bindings=(
            (
                index,
                replace(
                    binding,
                    compilation=replace(binding.compilation, compiler_hash=absent),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="no verified dependency closure"):
        cuda_execute.compile_cuda(bound, compiler, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(
    "override",
    [
        "-MD",
        "-run",
        "--output-file=evil",
        "-oevil",
        "-MFredirect",
        "--options-file=hidden",
        "-Xcompiler=-c",
        "-Xcompiler=@hidden",
    ],
)
def test_unsafe_dependency_overrides_fail_before_processes(
    toolkit: Any, monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    _, _, plan, compiler, calls = toolkit
    monkeypatch.setenv("NVCC_APPEND_FLAGS", override)
    with pytest.raises(ValueError, match="unsafe NVCC identity-probe override"):
        cuda_execute.tensor_compiler_identity(plan, compiler)
    assert not calls


def test_make_dependencies_support_escaped_paths_and_reject_ambiguity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.cu"
    header = tmp_path / "space #$.h"
    escaped = str(header).replace("$", "$$").replace("#", "\\#").replace(" ", "\\ ")
    manifest = f"tensor_identity: {source} \\\n {escaped}\n"
    assert identity._dependency_paths(manifest, source) == (header,)
    with pytest.raises(ValueError, match="omits its source"):
        identity._dependency_paths(f"tensor_identity: {escaped}\n", source)
    with pytest.raises(ValueError, match="unrecognized"):
        identity._dependency_paths(f"tensor_identity: {source}\nother: value\n", source)


def test_dependency_count_and_byte_budgets_fail_closed(
    toolkit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, plan, compiler, _ = toolkit
    monkeypatch.setattr(identity, "_MAX_DEPENDENCIES", 1)
    with pytest.raises(ValueError, match="count exceeds budget"):
        cuda_execute.tensor_compiler_identity(plan, compiler)
    monkeypatch.setattr(identity, "_MAX_DEPENDENCIES", 16384)
    monkeypatch.setattr(identity, "_MAX_DEPENDENCY_BYTES", 1)
    with pytest.raises(ValueError, match="file-byte budget"):
        cuda_execute.tensor_compiler_identity(plan, compiler)


@pytest.mark.parametrize("selection", ["NVCC_CCBIN", "-ccbin", "--compiler-bindir"])
def test_selected_host_override_is_identified(
    toolkit: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    root, _, plan, compiler, _ = toolkit
    directory = tmp_path / "other-host" / "bin"
    directory.mkdir(parents=True)
    executable = directory / "gcc"
    shutil.copyfile(root / "bin/gcc", executable)
    executable.chmod(0o700)
    if selection == "NVCC_CCBIN":
        monkeypatch.setenv(selection, str(directory))
    else:
        monkeypatch.setenv("NVCC_APPEND_FLAGS", f"{selection}={directory}")
    before = cuda_execute.tensor_compiler_identity(plan, compiler)
    executable.write_text(executable.read_text() + "\n# replaced selected host\n")
    assert cuda_execute.tensor_compiler_identity(plan, compiler) != before


@pytest.mark.parametrize("timeout", [0.0, float("inf"), float("nan")])
def test_dependency_timeout_must_be_finite_positive(
    toolkit: Any, timeout: float
) -> None:
    _, _, plan, compiler, _ = toolkit
    with pytest.raises(ValueError, match="finite positive timeout"):
        cuda_execute.tensor_compiler_identity(
            plan, replace(compiler, compile_timeout=timeout)
        )


def test_runtime_dependency_failure_refuses_identity(
    toolkit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from generativeqc_compiler.common.compiler_process import CompileResult

    _, _, plan, compiler, _ = toolkit
    original = identity.run_compiler
    timeouts = []

    def fail(command: list[str], timeout: float, **kwargs: Any) -> Any:
        if "-M" in command:
            timeouts.append(timeout)
            return CompileResult(124, True, timeout, "", "fixture timeout")
        return original(command, timeout, **kwargs)

    monkeypatch.setattr(identity, "run_compiler", fail)
    with pytest.raises(ValueError, match="runtime dependency probe failed.*timeout"):
        cuda_execute.tensor_compiler_identity(
            plan, replace(compiler, compile_timeout=0.125)
        )
    assert timeouts == [0.125]


def test_unknown_dependency_location_refuses_identity(
    toolkit: Any, tmp_path: Path
) -> None:
    _, native, plan, compiler, _ = toolkit
    external = tmp_path / "unknown-root" / "external.h"
    external.parent.mkdir()
    external.write_text("#pragma once\n")
    (native / "cuda_graph_context.cuh").write_text(f'#include "{external}"\n')
    with pytest.raises(ValueError, match="no unique portable identity"):
        cuda_execute.tensor_compiler_identity(plan, compiler)


@pytest.mark.parametrize(
    "failure", ["missing", "non-utf8", "malformed", "vanished-header"]
)
@pytest.mark.parametrize("cub_only", [False, True])
def test_unavailable_dependency_manifests_fail_closed(
    toolkit: Any, monkeypatch: pytest.MonkeyPatch, failure: str, cub_only: bool
) -> None:
    _, _, plan, compiler, _ = toolkit
    original = identity.run_compiler

    def corrupt(command: list[str], timeout: float, **kwargs: Any) -> Any:
        result = original(command, timeout, **kwargs)
        if "-M" in command:
            source = Path(command[-1])
            cub = "cub/block/block_reduce.cuh" in source.read_text()
            if not cub_only or cub:
                output = Path(command[command.index("-MF") + 1])
                if failure == "missing":
                    output.unlink()
                elif failure == "non-utf8":
                    output.write_bytes(b"\xff")
                elif failure == "malformed":
                    output.write_text("invalid dependency rule")
                else:
                    output.write_text(
                        f"tensor_identity: {source} {compiler.nvcc.parent.parent}/include/vanished.h\n"
                    )
        return result

    monkeypatch.setattr(identity, "run_compiler", corrupt)
    if cub_only:
        facts = cuda_execute._tensor_compiler_facts(plan, compiler)
        assert facts["dependencies"]["cub"] is None
        assert cuda_execute.tensor_compiler_identity(
            plan, compiler
        ) == cuda_execute.tensor_compiler_identity(plan, compiler)
    else:
        with pytest.raises(ValueError, match="runtime dependency probe failed"):
            cuda_execute.tensor_compiler_identity(plan, compiler)
