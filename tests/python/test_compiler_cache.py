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
    launched: list[list[str]] = []

    def compile_run(command: list[str], timeout: float, *, label: str) -> CompileResult:
        assert timeout > 0 and label in {"C++", "NVCC"}
        launched.append(command)
        return CompileResult(0, False, 0.01, "", "")

    monkeypatch.setattr(cpp_adapter, "run_cached_compiler", compile_run)
    cpu = CppCompilerAdapter(Path(sys.executable))
    cpu.compile_shared(tmp_path / "x.cpp", tmp_path / "x.so")
    assert launched[-1][0] == str(cpu.cxx)

    monkeypatch.setattr(cuda_adapter, "run_cached_compiler", compile_run)
    cuda = CudaCompilerAdapter(Path("/opt/cuda/bin/nvcc"), cuda_target_info("sm_120"))
    cuda.compile(tmp_path / "x.cu", tmp_path / "x.o")
    assert launched[-1][0] == "/opt/cuda/bin/nvcc"
    assert "-c" in launched[-1]
    cuda.compile_shared(tmp_path / "x.cu", tmp_path / "x.so")
    assert "--shared" in launched[-1]
    cuda.link_shared_objects([tmp_path / "x.o"], tmp_path / "x.so")
    assert "--shared" in launched[-1]
    result = cuda.link(tmp_path / "driver.cu", [], tmp_path / "driver")
    assert launched[-1][0] == "/opt/cuda/bin/nvcc"
    assert result.returncode == 0


def test_older_sccache_falls_back_before_starting_an_unowned_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compiler_cache.shutil, "which", lambda name, *, path: f"/bin/{name}"
    )
    monkeypatch.setattr(
        compiler_cache.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            "sccache 0.10.0" if "sccache" in command[0] else "ccache 4.14.1",
            "",
        ),
    )
    assert compiler_cache.resolve_compiler_cache().name == "ccache"


def test_discovery_timeout_consumes_compilation_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    monkeypatch.setattr(
        compiler_cache.shutil, "which", lambda name, *, path: f"/bin/{name}"
    )

    def stalled(command: list[str], **kwargs: Any) -> Any:
        time.sleep(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(compiler_cache.subprocess, "run", stalled)
    result = compiler_cache.run_cached_compiler(
        ["c++", "-c", "x.cpp"], 0.05, label="probe"
    )
    assert result.timed_out and result.returncode == 124
    assert result.duration_seconds < 0.5
