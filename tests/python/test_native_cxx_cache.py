"""Native probe caching must reuse equivalent inputs without hiding regressions."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import NativeCxx


@pytest.mark.parametrize("missing", ("CXX", "CCACHE"))
@pytest.mark.parametrize("required", (False, True))
def test_missing_tools_preserve_optional_and_required_gates(
    tmp_path: Path, missing: str, required: bool
) -> None:
    # Run real fixture setup in a separate pytest process, before any test body.
    (tmp_path / "conftest.py").write_text(
        Path(__file__).with_name("conftest.py").read_text()
    )
    fixture = "required_native_cxx" if required else "native_cxx"
    (tmp_path / "test_probe.py").write_text(
        f"def test_probe({fixture}):\n    raise AssertionError('body must not run')\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            missing: str(tmp_path / "missing-tool"),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == (1 if required else 0), result.stdout + result.stderr
    assert ("1 error" if required else "1 skipped") in result.stdout


def test_cache_reuses_sessions_and_invalidates_changed_inputs(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    # A private test cache makes hit/miss evidence independent of other workers.
    env = {"CCACHE_DIR": str(tmp_path / "cache")}

    def hits() -> int:
        result = subprocess.run(
            [native_cxx.cache, "--print-stats"],
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        stats = dict(line.split() for line in result.stdout.splitlines())
        return int(stats["direct_cache_hit"]) + int(stats["preprocessed_cache_hit"])

    def build(session: str, header: int = 41, extra: int = 0, offset: int = 0) -> int:
        base = tmp_path / session
        directory = base / "test_probe0"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "value.hpp").write_text(f"#define VALUE {header}\n")
        source = directory / "probe.cpp"
        source.write_text(
            f'#include "value.hpp"\nint main() {{ return VALUE + OFFSET + {extra}; }}\n'
        )
        output = directory / "probe"
        replace(native_cxx, base_dir=base).build_executable(
            [source], output, compile_args=("-std=c++17", f"-DOFFSET={offset}"), env=env
        )
        return subprocess.run([str(output)], check=False, timeout=10).returncode

    assert build("pytest-1/popen-gw0") == 41
    before = hits()
    assert build("pytest-2/popen-gw3") == 41
    assert hits() > before
    assert build("pytest-2/popen-gw3", header=42) == 42
    assert build("pytest-2/popen-gw3", header=42, extra=1) == 43
    assert build("pytest-2/popen-gw3", header=42, extra=1, offset=2) == 45


def test_relative_compiler_and_cache_paths_survive_compile_cwd(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    toolchain = tmp_path / "toolchain"
    toolchain.mkdir()
    (toolchain / "c++").symlink_to(native_cxx.compiler)
    (toolchain / "ccache").symlink_to(native_cxx.cache)
    (tmp_path / "conftest.py").write_text(
        Path(__file__).with_name("conftest.py").read_text()
    )
    (tmp_path / "test_probe.py").write_text(
        "import subprocess\n"
        "def test_probe(tmp_path, required_native_cxx):\n"
        "    source, output = tmp_path / 'probe.cpp', tmp_path / 'probe'\n"
        "    source.write_text('#include <string>\\nint main(){ return std::string(\"x\").size() != 1; }\\n')\n"
        "    required_native_cxx.build_executable([source], output)\n"
        "    subprocess.run([str(output)], check=True)\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "CXX": "./toolchain/c++",
            "CCACHE": "./toolchain/ccache",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


@pytest.mark.parametrize("include_args", (("-Iinclude",), ("-I", "include")))
def test_relative_probe_inputs_keep_caller_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_cxx: NativeCxx,
    include_args: tuple[str, ...],
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("include").mkdir()
    Path("include/value.hpp").write_text("#define VALUE 17\n")
    Path("probe.cpp").write_text("#include <value.hpp>\nint main(){ return VALUE; }\n")
    native_cxx.build_executable(
        [Path("probe.cpp")], Path("probe"), compile_args=include_args
    )
    assert (
        subprocess.run([str(tmp_path / "probe")], check=False, timeout=10).returncode
        == 17
    )


def test_compile_and_link_errors_never_run_stale_output(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    source, output = tmp_path / "probe.cpp", tmp_path / "probe"
    source.write_text("int main(){ return 0; }\n")
    native_cxx.build_executable([source], output)
    source.write_text("invalid C++ source\n")
    with pytest.raises(subprocess.CalledProcessError):
        native_cxx.build_executable([source], output)
    source.write_text("int function_without_main(){ return 0; }\n")
    with pytest.raises(subprocess.CalledProcessError):
        native_cxx.build_executable([source], output)


def test_symlink_source_keeps_quoted_header_directory(
    tmp_path: Path, native_cxx: NativeCxx
) -> None:
    actual, alias = tmp_path / "actual", tmp_path / "alias"
    actual.mkdir()
    alias.mkdir()
    (actual / "probe.cpp").write_text(
        '#include "value.hpp"\nint main(){ return VALUE; }\n'
    )
    (actual / "value.hpp").write_text("#define VALUE 41\n")
    (alias / "value.hpp").write_text("#define VALUE 17\n")
    (alias / "probe.cpp").symlink_to(actual / "probe.cpp")
    output = tmp_path / "probe"
    native_cxx.build_executable([alias / "probe.cpp"], output)
    assert subprocess.run([str(output)], check=False, timeout=10).returncode == 17
