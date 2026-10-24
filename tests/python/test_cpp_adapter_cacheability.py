"""CPU JIT objects are cacheable without weakening flags or process ownership."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from generativeqc_compiler.common import cpp_adapter
from generativeqc_compiler.common.compiler_cache import (
    CompilerCacheLauncher,
    verify_compiler_cache,
)
from generativeqc_compiler.common.cpp_adapter import CppCompilerAdapter


@pytest.fixture
def real_compiler(monkeypatch: pytest.MonkeyPatch) -> CppCompilerAdapter:
    cxx, ccache = shutil.which("c++"), shutil.which("ccache")
    if cxx is None or ccache is None:
        pytest.skip("real C++ compiler and ccache required")
    launcher = verify_compiler_cache(Path(ccache))
    monkeypatch.setattr(cpp_adapter, "resolve_compiler_cache", lambda _: launcher)
    return CppCompilerAdapter(Path(cxx))


def _answer(library: Path) -> int:
    function = ctypes.CDLL(str(library)).answer
    function.restype = ctypes.c_int
    return function()


def test_repeated_objects_hit_cache_and_execute_changed_headers_and_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_compiler: CppCompilerAdapter
) -> None:
    include = tmp_path / "include"
    include.mkdir()
    header = include / "value.hpp"
    header.write_text("constexpr int value = 7;\n")
    first = tmp_path / "first.cpp"
    second = tmp_path / "second.cpp"
    first.write_text('#include "value.hpp"\nint part() { return value * SCALE; }\n')
    second.write_text('int part(); extern "C" int answer() { return part() + 1; }\n')

    def build(name: str, scale: int) -> tuple[Path, str]:
        output, log = tmp_path / f"{name}.so", tmp_path / f"{name}.log"
        monkeypatch.setenv("CCACHE_LOGFILE", str(log))
        result = real_compiler.compile_shared_many(
            iter((first, second)),
            output,
            includes=iter((include,)),
            options=iter(("-ffp-contract=off", "-D", f"SCALE={scale}")),
        )
        assert result.returncode == 0, result.stderr
        assert not list(tmp_path.glob(".gqc-cpp-*"))
        return output, log.read_text()

    initial, _ = build("initial", 2)
    repeat, log = build("repeat", 2)
    assert _answer(initial) == _answer(repeat) == 15
    # Read only this invocation's log; never clear/zero a shared compiler cache.
    assert (
        log.count("Result: direct_cache_hit")
        + log.count("Result: preprocessed_cache_hit")
        >= 2
    )
    header.write_text("constexpr int value = 11;\n")
    changed, _ = build("changed_header", 2)
    assert _answer(changed) == 23
    header_repeat, log = build("header_repeat", 2)
    assert _answer(header_repeat) == 23
    assert (
        log.count("Result: direct_cache_hit")
        + log.count("Result: preprocessed_cache_hit")
        >= 2
    )
    flagged, _ = build("changed_flags", 3)
    assert _answer(flagged) == 34
    repeated, log = build("changed_repeat", 3)
    assert _answer(repeated) == 34
    assert (
        log.count("Result: direct_cache_hit")
        + log.count("Result: preprocessed_cache_hit")
        >= 2
    )


def test_ordered_static_libraries_link_after_all_translation_units(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_compiler: CppCompilerAdapter,
    native_cxx: Any,
) -> None:
    if shutil.which("ar") is None:
        pytest.skip("static archive tool required")
    for name, code in (
        (
            "first",
            'extern "C" int second(); extern "C" int first() { return second() + 2; }',
        ),
        ("second", 'extern "C" int second() { return 40; }'),
    ):
        source, obj = tmp_path / f"{name}.cpp", tmp_path / f"{name}.o"
        source.write_text(code)
        native_cxx.compile_object(source, obj, args=("-fPIC",))
        # Archive creation is fixture setup, not the adapter's deadline contract.
        # Use the same finite budget as native fixture linking on busy runners.
        subprocess.run(
            ["ar", "rcs", str(tmp_path / f"lib{name}.a"), str(obj)],
            check=True,
            timeout=60,
        )
    monkeypatch.setenv("LIBRARY_PATH", str(tmp_path))
    source = tmp_path / "caller.cpp"
    source.write_text(
        'extern "C" int first(); extern "C" int answer() { return first(); }'
    )
    output = tmp_path / "ordered.so"
    result = real_compiler.compile_shared(
        source, output, libraries=iter(("first", "second"))
    )
    assert result.returncode == 0, result.stderr
    assert _answer(output) == 42


@pytest.mark.parametrize("stage", ["first_source", "later_source", "link"])
@pytest.mark.parametrize("preexisting", [False, True])
def test_failure_never_publishes_partial_output_or_removes_prior_library(
    tmp_path: Path, real_compiler: CppCompilerAdapter, stage: str, preexisting: bool
) -> None:
    first, second = tmp_path / "first.cpp", tmp_path / "second.cpp"
    first.write_text(
        "this is not C++"
        if stage == "first_source"
        else 'extern "C" int first() { return 1; }'
    )
    second.write_text(
        "this is not C++"
        if stage == "later_source"
        else 'extern "C" int second() { return 2; }'
    )
    output = tmp_path / "result.so"
    if preexisting:
        output.write_bytes(b"previous caller-owned library")
    result = real_compiler.compile_shared_many(
        (first, second),
        output,
        libraries=("gqc_missing_library_for_test",) if stage == "link" else (),
    )
    assert result.returncode != 0 and not result.timed_out
    if preexisting:
        assert output.read_bytes() == b"previous caller-owned library"
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".gqc-cpp-*"))


def test_opaque_driver_options_preserve_combined_command_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_compiler: CppCompilerAdapter
) -> None:
    # The .txt input needs -x; applying that flag to later .o operands would
    # incorrectly parse binary objects as C++. A response file is also opaque.
    source = tmp_path / "source.txt"
    source.write_text('extern "C" int answer() { return SCALE; }')
    response = tmp_path / "options.rsp"
    response.write_text("-DSCALE=29")
    log = tmp_path / "fallback.log"
    monkeypatch.setenv("CCACHE_LOGFILE", str(log))
    output = tmp_path / "fallback.so"
    result = real_compiler.compile_shared(
        source, output, options=("-x", "c++", f"@{response}", "-Wl,-z,defs")
    )
    assert result.returncode == 0, result.stderr
    assert _answer(output) == 29
    assert "called_for_link" in log.read_text()
    assert "cache_hit" not in log.read_text()


def _running(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


@pytest.mark.parametrize("slow_phase", ["compile", "link"])
def test_total_deadline_owns_children_and_preserves_unrelated_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slow_phase: str
) -> None:
    records = tmp_path / "children.jsonl"
    driver = tmp_path / "driver"
    driver.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, subprocess, sys, time\n"
        "phase = 'compile' if '-c' in sys.argv else 'link'\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"with open({str(records)!r}, 'a') as f: f.write(json.dumps(child.pid) + '\\n')\n"
        "print(phase, flush=True)\n"
        f"time.sleep({0.5 if slow_phase == 'compile' else 0.8} if phase == {slow_phase!r} else 0.08)\n"
        "pathlib.Path(sys.argv[sys.argv.index('-o') + 1]).write_text('owned output')\n"
    )
    driver.chmod(0o755)
    cache = tmp_path / "ccache"
    cache.write_text(
        f"#!{sys.executable}\nimport os, sys\nos.execv(sys.argv[1], sys.argv[1:])\n"
    )
    cache.chmod(0o755)
    launcher = CompilerCacheLauncher(cache, "ccache", "test launcher")

    def discover(timeout: float) -> CompilerCacheLauncher:
        assert timeout > 0
        time.sleep(0.1)
        return launcher

    monkeypatch.setattr(cpp_adapter, "resolve_compiler_cache", discover)
    output = tmp_path / "result.so"
    output.write_bytes(b"prior library")
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        result = CppCompilerAdapter(driver, compile_timeout=0.8).compile_shared_many(
            (tmp_path / "a.cpp", tmp_path / "b.cpp"), output
        )
        assert result.returncode == 124 and result.timed_out
        assert 0.75 <= result.duration_seconds < 1.6
        assert "total budget" in result.stderr
        assert "compile" in result.stdout
        if slow_phase == "link":
            assert "link" in result.stdout
        assert output.read_bytes() == b"prior library"
        assert sentinel.poll() is None
        assert not list(tmp_path.glob(".gqc-cpp-*"))
        children = [json.loads(line) for line in records.read_text().splitlines()]
        deadline = time.monotonic() + 2
        while any(_running(pid) for pid in children):
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        sentinel.kill()
        sentinel.wait()


def test_missing_output_parent_returns_failed_outcome(
    tmp_path: Path, real_compiler: CppCompilerAdapter
) -> None:
    source = tmp_path / "source.cpp"
    source.write_text('extern "C" int answer() { return 42; }')
    output = tmp_path / "missing" / "output.so"
    result = real_compiler.compile_shared(source, output)
    assert result.returncode != 0 and not result.timed_out
    assert result.stderr and not output.parent.exists()


def test_unwritable_output_parent_returns_failed_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_compiler: CppCompilerAdapter
) -> None:
    def denied(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("output parent is not writable")

    monkeypatch.setattr(cpp_adapter.tempfile, "TemporaryDirectory", denied)
    result = real_compiler.compile_shared(
        tmp_path / "source.cpp", tmp_path / "output.so"
    )
    assert result.returncode != 0 and not result.timed_out
    assert "not writable" in result.stderr


def test_opaque_side_output_stays_beside_requested_library(
    tmp_path: Path, real_compiler: CppCompilerAdapter
) -> None:
    source = tmp_path / "source.cpp"
    source.write_text('extern "C" int answer() { return 42; }')
    output = tmp_path / "caller.so"
    result = real_compiler.compile_shared(source, output, options=("-save-temps=obj",))
    assert result.returncode == 0, result.stderr
    assert _answer(output) == 42
    assert list(tmp_path.glob("*.ii")) and list(tmp_path.glob("*.s"))


def test_publication_failure_preserves_existing_destination(
    tmp_path: Path, real_compiler: CppCompilerAdapter
) -> None:
    source = tmp_path / "source.cpp"
    source.write_text('extern "C" int answer() { return 42; }')
    output = tmp_path / "existing-directory"
    output.mkdir()
    sentinel = output / "unrelated"
    sentinel.write_bytes(b"caller data")
    result = real_compiler.compile_shared(source, output)
    assert result.returncode != 0 and not result.timed_out
    assert sentinel.read_bytes() == b"caller data"
    assert not list(tmp_path.glob(".gqc-cpp-*"))


def test_discovery_timeout_does_not_publish_or_leave_intermediates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def expired(timeout: float) -> Any:
        raise subprocess.TimeoutExpired("compiler cache discovery", timeout)

    monkeypatch.setattr(cpp_adapter, "resolve_compiler_cache", expired)
    output = tmp_path / "prior.so"
    output.write_bytes(b"prior library")
    result = CppCompilerAdapter(Path(sys.executable)).compile_shared(
        tmp_path / "source.cpp", output
    )
    assert result.returncode == 124 and result.timed_out
    assert output.read_bytes() == b"prior library"
    assert not list(tmp_path.glob(".gqc-cpp-*"))


def test_path_object_operand_uses_opaque_fallback(
    tmp_path: Path, real_compiler: CppCompilerAdapter, native_cxx: Any
) -> None:
    provider, obj = tmp_path / "provider.cpp", tmp_path / "provider.o"
    provider.write_text("int provider() { return 42; }")
    native_cxx.compile_object(provider, obj, args=("-fPIC",))
    source = tmp_path / "caller.cpp"
    source.write_text('int provider(); extern "C" int answer() { return provider(); }')
    output = tmp_path / "with-object.so"
    result = real_compiler.compile_shared(source, output, options=(obj,))
    assert result.returncode == 0, result.stderr
    assert _answer(output) == 42


def test_source_and_header_path_macros_match_combined_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, real_compiler: CppCompilerAdapter
) -> None:
    header, source = tmp_path / "where.hpp", tmp_path / "source.cpp"
    header.write_text("inline const char* header_path() { return __FILE__; }")
    source.write_text(
        '#include "where.hpp"\n'
        'extern "C" const char* source_path() { return __FILE__; }\n'
        'extern "C" const char* included_path() { return header_path(); }\n'
    )
    monkeypatch.setenv("CCACHE_BASEDIR", str(tmp_path))
    results = []
    for name, options in (
        ("combined", ("-Wl,-z,defs",)),
        ("split", ()),
        ("repeated", ()),
    ):
        output, log = tmp_path / f"{name}.so", tmp_path / f"{name}.log"
        monkeypatch.setenv("CCACHE_LOGFILE", str(log))
        result = real_compiler.compile_shared(source, output, options=options)
        assert result.returncode == 0, result.stderr
        library = ctypes.CDLL(str(output))
        paths = []
        for symbol in ("source_path", "included_path"):
            function = getattr(library, symbol)
            function.restype = ctypes.c_char_p
            paths.append(function().decode())
        results.append(paths)
        assert os.environ["CCACHE_BASEDIR"] == str(tmp_path)
        if name == "repeated":
            assert "cache_hit" in log.read_text()
    assert results == [[str(source), str(header)]] * 3
