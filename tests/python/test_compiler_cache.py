"""Runtime C++/CUDA compilers must use one verified compiler cache."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from generativeqc_compiler.common import compiler_cache, cpp_adapter, cuda_adapter
from generativeqc_compiler.common.compiler_process import CompileResult
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info


@pytest.fixture(autouse=True)
def _clear_cache_resolution() -> Any:
    compiler_cache._resolve_compiler_cache.cache_clear()
    yield
    compiler_cache._resolve_compiler_cache.cache_clear()


def test_compiler_cache_prefers_verified_sccache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes: list[list[str]] = []

    def which(name: str, *, path: str) -> str | None:
        assert path == "/test/bin"
        return f"/test/bin/{name}"

    def run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        probes.append(command)
        return subprocess.CompletedProcess(command, 0, "sccache 0.16.0\n", "")

    monkeypatch.setenv("PATH", "/test/bin")
    monkeypatch.setattr(compiler_cache.shutil, "which", which)
    monkeypatch.setattr(compiler_cache.subprocess, "run", run)

    launcher = compiler_cache.resolve_compiler_cache()
    assert launcher.name == "sccache"
    assert launcher.path == Path("/test/bin/sccache")
    assert launcher.version == "sccache 0.16.0"
    assert launcher.wrap(["nvcc", "-c", "x.cu"]) == [
        "/test/bin/sccache",
        "nvcc",
        "-c",
        "x.cu",
    ]
    assert probes == [["/test/bin/sccache", "--version"]]


def test_compiler_cache_falls_back_to_verified_ccache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes: list[list[str]] = []

    def which(name: str, *, path: str) -> str | None:
        assert path == "/test/bin"
        return f"/test/bin/{name}"

    def run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        probes.append(command)
        if command[0].endswith("/sccache"):
            return subprocess.CompletedProcess(command, 1, "", "broken sccache")
        return subprocess.CompletedProcess(command, 0, "ccache version 4.12\n", "")

    monkeypatch.setenv("PATH", "/test/bin")
    monkeypatch.setattr(compiler_cache.shutil, "which", which)
    monkeypatch.setattr(compiler_cache.subprocess, "run", run)

    launcher = compiler_cache.resolve_compiler_cache()
    assert launcher.name == "ccache"
    assert launcher.path == Path("/test/bin/ccache")
    assert probes == [
        ["/test/bin/sccache", "--version"],
        ["/test/bin/ccache", "--version"],
    ]


def test_compiler_cache_rejects_uncached_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/test/bin")
    monkeypatch.setattr(compiler_cache.shutil, "which", lambda _name, *, path: None)
    with pytest.raises(RuntimeError, match="python -m pip install sccache"):
        compiler_cache.cached_compiler_command(["c++", "-c", "x.cpp"])


def test_cpu_and_cuda_adapters_wrap_cache_miss_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wrapped: list[list[str]] = []
    launched: list[list[str]] = []

    def wrap(command: list[str]) -> list[str]:
        wrapped.append(command)
        return ["verified-cache", *command]

    def compile_run(command: list[str], timeout: float, *, label: str) -> CompileResult:
        assert timeout > 0 and label in {"C++", "NVCC"}
        launched.append(command)
        return CompileResult(0, False, 0.01, "", "")

    monkeypatch.setattr(cpp_adapter, "cached_compiler_command", wrap)
    monkeypatch.setattr(cpp_adapter, "run_compiler", compile_run)
    cpu = CppCompilerAdapter(Path(sys.executable))
    cpu.compile_shared(tmp_path / "x.cpp", tmp_path / "x.so")
    assert launched[-1][0] == "verified-cache"
    assert wrapped[-1][0] == str(cpu.cxx)

    monkeypatch.setattr(cuda_adapter, "cached_compiler_command", wrap)
    monkeypatch.setattr(cuda_adapter, "run_compiler", compile_run)
    cuda = CudaCompilerAdapter(Path("/opt/cuda/bin/nvcc"), cuda_target_info("sm_120"))
    cuda.compile(tmp_path / "x.cu", tmp_path / "x.o")
    assert launched[-1][0] == "verified-cache"
    assert wrapped[-1][0] == "/opt/cuda/bin/nvcc"

    link_commands: list[list[str]] = []

    def link_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        link_commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cuda_adapter.subprocess, "run", link_run)
    cuda.link(tmp_path / "driver.cu", [], tmp_path / "driver")
    assert link_commands[-1][0] == "verified-cache"
    assert "/opt/cuda/bin/nvcc" in link_commands[-1]
