"""Explicit GCC C++ protocol controls and mandatory native Linux qualification."""

from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.common.compiler_cache import resolve_compiler_cache
from generativeqc_compiler.common.compiler_process import CompileResult, run_compiler
from generativeqc_compiler.common.provenance import file_hash
from generativeqc_compiler.xc import bulk_aot_closure as c
from generativeqc_compiler.xc import bulk_aot_store as store
from test_bulk_aot_closure import compile_controlled, dependency, real_gcc, variant

__all__ = ["real_gcc"]

if TYPE_CHECKING:
    from typing import Any

FLAGS = ("-std=c++20", "-O1", "-fPIC")


def gcc_quoted(value: str) -> str:
    """Emit the GCC driver's diagnostic quoting, independently of decoding."""
    return (
        '"' + "".join("\\" + char if char in '\\"$' else char for char in value) + '"'
    )


@pytest.mark.parametrize(
    "value",
    [
        "ordinary.cpp",
        "dollar$unit.cpp",
        r"literal\$unit.cpp",
        r"two\\$unit.cpp",
        'quote"$.cpp',
        "apos' and `tick`.cpp",
        "",
    ],
)
def test_gcc_diagnostic_argument_identity(value: str) -> None:
    diagnostic = " /cc1plus " + gcc_quoted(value) + ' -o "/tmp/assembly.s"'
    assert c._gcc_job_arguments(diagnostic) == [
        "/cc1plus",
        value,
        "-o",
        "/tmp/assembly.s",
    ]


@pytest.mark.parametrize(
    "diagnostic",
    [' /cc1plus "unfinished', ' /cc1plus "trailing\\', r' /cc1plus "unknown\q"'],
)
def test_unsupported_gcc_diagnostic_quoting(diagnostic: str) -> None:
    with pytest.raises(ValueError, match="unsupported-gxx-job-model"):
        c._gcc_job_arguments(diagnostic)


@pytest.fixture
def fake_gxx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    driver = tmp_path / "g++"
    frontend = tmp_path / "cc1plus"
    assembler = tmp_path / "as"
    for tool in (driver, frontend, assembler):
        tool.write_bytes(b"\x7fELFfake")
    source = tmp_path / "point.cpp"
    source.write_bytes(b"int point() { return 1; }\n")
    monkeypatch.setattr(c.sys, "platform", "linux")
    monkeypatch.setattr(c.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(c, "_system_gxx", lambda: driver)
    monkeypatch.setattr(
        c.shutil,
        "which",
        lambda name, **kw: (
            str(driver) if name == "g++" else (str(assembler) if name == "as" else name)
        ),
    )

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        assert kwargs["environment"] == c.CPU_ENVIRONMENT and timeout > 0
        diagnostic = ""
        if "--version" in command:
            text = "g++ Free Software Foundation 12.3.0"
        elif "-dumpfullversion" in command:
            text = "12.3.0"
        elif "-dumpmachine" in command:
            text = "x86_64-linux-gnu"
        elif "-print-prog-name=cc1plus" in command:
            text = str(frontend)
        elif "-print-prog-name=as" in command:
            text = str(assembler)
        elif "-dumpspecs" in command:
            text = "native specs"
        elif "-###" in command:
            text = ""
            assembly = tmp_path / "probe.s"
            diagnostic = (
                f" {gcc_quoted(str(frontend))} {gcc_quoted(str(source))} "
                f'"-std=c++20" -o {gcc_quoted(str(assembly))}\n'
                f" as -o /dev/null {gcc_quoted(str(assembly))}"
            )
        elif "-dM" in command:
            text = "#define __cplusplus 202002L\n#define __GXX_ABI_VERSION 1017"
        elif "-v" in command:
            text = ""
            diagnostic = '#include "..." search starts here:\n#include <...> search starts here:\n /usr/include/c++/12\nEnd of search list.'
        elif "-E" in command:
            text = "preprocessed C++"
        else:
            text = "closure: " + str(source).replace(" ", "\\ ")
        return CompileResult(0, False, 0.001, text, diagnostic)

    monkeypatch.setattr(c, "run_compiler", run)
    manifest = c.CpuToolchainManifest(
        tuple(dependency(p) for p in (driver, frontend, assembler)), True
    )
    return source, manifest


def collect(
    source: Path, manifest: c.CpuToolchainManifest, **kwargs: Any
) -> c.CpuClosureResult:
    return c.collect_cpu_closure(
        variant(source.read_text()),
        source,
        compiler=kwargs.pop("compiler", "g++"),
        language="c++",
        flags=kwargs.pop("flags", FLAGS),
        toolchain=manifest,
        **kwargs,
    )


def test_explicit_language_and_manifest(fake_gxx: tuple) -> None:
    source, manifest = fake_gxx
    result = collect(source, manifest)
    assert not result.reasons and result.closure is not None and result.closure.reusable
    assert result.recipe is not None and result.recipe["language"] == "c++20"
    assert result.recipe["compiler"]["executable"].endswith("g++")
    assert c.collect_cpu_closure(variant(source.read_text()), source).reasons
    for incomplete in (
        replace(manifest, complete=False),
        replace(manifest, files=manifest.files[:1]),
    ):
        failed = collect(source, incomplete)
        assert (
            failed.reasons
            and failed.closure is not None
            and not failed.closure.reusable
        )


@pytest.mark.parametrize(
    "flags",
    [
        (),
        ("-std=c99",),
        ("-std=c++17",),
        (*FLAGS, "-flto"),
        (*FLAGS, "@args"),
        (*FLAGS, "-fmodules-ts"),
        (*FLAGS, "-D_GLIBCXX_USE_CXX11_ABI=0"),
        (*FLAGS, "--sysroot=/tmp"),
    ],
)
def test_unsupported_cxx_flags(fake_gxx: tuple, flags: tuple) -> None:
    source, manifest = fake_gxx
    assert collect(source, manifest, flags=flags).reasons == ("unsupported-cpu-flags",)


def test_cxx_abi_and_pch_identity(
    fake_gxx: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, manifest = fake_gxx
    before = collect(source, manifest)
    assert not before.reasons and before.closure is not None
    original = c.run_compiler

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        result = original(command, timeout, **kwargs)
        return (
            replace(result, stdout=result.stdout.replace("1017", "1018"))
            if "-dM" in command
            else result
        )

    monkeypatch.setattr(c, "run_compiler", run)
    after = collect(source, manifest)
    assert not after.reasons and after.closure is not None
    assert before.closure.cache_key != after.closure.cache_key
    Path(str(source) + ".gch").write_bytes(b"unsupported precompiled header")
    rejected = collect(source, manifest)
    assert (
        rejected.reasons == ("unsupported-gxx-pch",)
        and rejected.closure is not None
        and not rejected.closure.reusable
    )


@pytest.mark.parametrize(
    "fault",
    [
        "frontend",
        "assembler",
        "extra-job",
        "handoff",
        "abi",
        "selection",
        "timeout",
        "late-abi",
        "nonregular",
    ],
)
def test_cxx_observations_fail_closed(
    fake_gxx: tuple, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    source, manifest = fake_gxx
    original = c.run_compiler
    macros = 0

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        nonlocal macros
        result = original(command, timeout, **kwargs)
        if fault == "timeout":
            return replace(result, timed_out=True)
        if "-###" in command:
            text = result.stderr
            if fault in {"frontend", "assembler"}:
                text = text.replace(
                    "cc1plus" if fault == "frontend" else "\n as ",
                    "other",
                )
                if fault == "assembler":
                    text += '\n "relative-as" -o /dev/null /tmp/probe.s'
            if fault == "extra-job":
                text += f'\n "{source.parent / "ld"}" -o /dev/null'
            if fault == "handoff":
                text = text.replace("probe.s", "different.s", 1)
            result = replace(result, stderr=text)
        if "-dM" in command:
            macros += 1
            if fault == "abi" or (fault == "late-abi" and macros > 1):
                result = replace(
                    result, stdout=result.stdout.replace("1017", "invalid")
                )
        if fault == "selection":
            monkeypatch.setattr(c.shutil, "which", lambda *a, **kw: str(source))
        return result

    monkeypatch.setattr(c, "run_compiler", run)
    if fault == "nonregular":
        Path(manifest.files[0].identity).unlink()
        Path(manifest.files[0].identity).mkdir()
    result = collect(source, manifest)
    assert result.reasons and (
        result.closure is None or result.closure.cache_key is None
    )


@pytest.fixture
def real_gxx(tmp_path: Path) -> tuple[str, c.CpuToolchainManifest]:
    if sys.platform != "linux":
        pytest.skip("native Linux G++ qualification runs separately")
    selected = shutil.which("g++", path=c.CPU_ENVIRONMENT["PATH"])
    assert selected is not None, "Linux qualification requires native /usr/bin/g++"
    compiler = str(Path(selected).resolve())
    files = {Path(compiler)}
    for name in ("cc1plus", "as"):
        query = run_compiler(
            [compiler, "-print-prog-name=" + name],
            30,
            label="G++ fixture tools",
            environment=c.CPU_ENVIRONMENT,
        )
        assert query.returncode == 0 and not query.timed_out
        tool = shutil.which(query.stdout.strip(), path=c.CPU_ENVIRONMENT["PATH"])
        assert tool is not None
        files.add(Path(tool).resolve())
    for tool in tuple(files):
        loaded = run_compiler(
            ["/usr/bin/ldd", str(tool)],
            30,
            label="G++ fixture loader",
            environment=c.CPU_ENVIRONMENT,
        )
        assert loaded.returncode == 0
        files.update(
            Path(p).resolve() for p in loaded.stdout.split() if p.startswith("/")
        )
    config = tmp_path / "asserted snapshot"
    config.write_text("controlled native GCC C++20 fixture/v1\n")
    files.add(config)
    # Producer assertion: the controlled image, include namespaces (including
    # absent/PCH candidates), loaded libraries/config and opaque inputs stay
    # immutable in each collect/compile/collect bracket. ldd is not this proof.
    return compiler, c.CpuToolchainManifest(
        tuple(dependency(p) for p in sorted(files)), True
    )


def test_real_cxx_store_execution_and_invalidation(
    tmp_path: Path, real_gxx: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler, manifest = real_gxx
    source = tmp_path / "point.cpp"
    header = tmp_path / "value.hpp"
    header.write_text("#define VALUE 1\n")
    text = '#include <array>\n#include <numeric>\n#include "value.hpp"\nextern "C" double point(double r) { constexpr std::array<int,3> a{1,2,3}; return r*r*r + r*r/2 + std::accumulate(a.begin(),a.end(),0) + VALUE; }\n'
    source.write_text(text)
    before = collect(source, manifest, compiler=compiler)
    assert (
        not before.reasons
        and before.closure is not None
        and before.closure.reusable
        and before.recipe is not None
    )
    assert any(
        "c++" in p.identity and "c++config.h" in p.identity for p in before.headers
    )
    obj, dep = tmp_path / "point.o", tmp_path / "point.d"
    command = [
        compiler,
        *before.recipe["flags"],
        "-MD",
        "-MF",
        str(dep),
        "-MT",
        "closure",
        "-c",
        str(source),
        "-o",
        str(obj),
    ]
    built = compile_controlled(command, monkeypatch)
    assert built.returncode == 0 and not built.timed_out, built.stderr
    assert set(c.parse_gcc_dependencies(dep.read_text())) == {
        str(source),
        *(p.identity for p in before.headers),
    }
    after = collect(source, manifest, compiler=compiler)
    assert (
        not after.reasons
        and after.closure is not None
        and after.closure.cache_key == before.closure.cache_key
    )
    raw_hash = file_hash(obj)
    stored = store.store_artifact(tmp_path / "store", after.closure, obj)
    assert stored.object_sha256 == raw_hash
    obj.unlink()
    replay = collect(source, manifest, compiler=compiler)
    assert not replay.reasons and replay.closure is not None
    hit = store.lookup_artifact(tmp_path / "store", replay.closure)
    assert (
        hit == stored
        and hit.artifact_path is not None
        and file_hash(hit.artifact_path) == raw_hash
    )
    shared = tmp_path / "point.so"
    linked = compile_controlled(
        [compiler, "-shared", str(hit.artifact_path), "-o", str(shared)], monkeypatch
    )
    assert linked.returncode == 0 and not linked.timed_out, linked.stderr
    library = ctypes.CDLL(str(shared))
    library.point.argtypes, library.point.restype = [ctypes.c_double], ctypes.c_double
    rows = []
    for value in ("0", ".125", "1", "2.75"):
        r = Decimal(value)
        expected = r**3 + r * r / 2 + 7
        actual = library.point(float(r))
        assert actual == pytest.approx(float(expected), rel=1e-14)
        rows.append([value, actual, str(expected)])
    keys = {before.closure.cache_key}
    for changed_flags in (
        ("-std=c++20", "-O2", "-fPIC"),
        ("-fPIC", "-O1", "-std=c++20"),
    ):
        changed = collect(source, manifest, compiler=compiler, flags=changed_flags)
        assert not changed.reasons and changed.closure is not None
        keys.add(changed.closure.cache_key)
    header.write_text("#define VALUE 2\n")
    changed = collect(source, manifest, compiler=compiler)
    assert not changed.reasons and changed.closure is not None
    keys.add(changed.closure.cache_key)
    assert store.lookup_artifact(tmp_path / "store", changed.closure).status == "miss"
    source.write_text(text + "// source identity change\n")
    changed = collect(source, manifest, compiler=compiler)
    assert not changed.reasons and changed.closure is not None
    keys.add(changed.closure.cache_key)
    assert len(keys) == 5
    receipt = {
        "object_sha256": raw_hash,
        "stored_object": str(hit.artifact_path),
        "executed_library": str(shared),
        "library_sha256": file_hash(shared),
        "commands": [
            command,
            [compiler, "-shared", str(hit.artifact_path), "-o", str(shared)],
        ],
        "closure": before.closure.to_payload(),
        "rows": rows,
    }
    (tmp_path / "qualification.json").write_text(json.dumps(receipt, indent=2))


def test_real_same_emission_c_and_cxx_are_distinct(
    tmp_path: Path, real_gxx: tuple, real_gcc: tuple
) -> None:
    cxx_driver, cxx_manifest = real_gxx
    c_driver, c_manifest = real_gcc
    text = "int point(void) { return 7; }\n"
    source = tmp_path / "point.cpp"
    source.write_text(text)
    c_source = tmp_path / "point.c"
    c_source.write_text(text)
    cxx_result = collect(source, cxx_manifest, compiler=cxx_driver)
    c_result = c.collect_cpu_closure(
        variant(text), c_source, compiler=c_driver, toolchain=c_manifest
    )
    assert not cxx_result.reasons and not c_result.reasons
    assert cxx_result.closure is not None and c_result.closure is not None
    assert cxx_result.closure.emission_identity == c_result.closure.emission_identity
    assert cxx_result.closure.cache_key != c_result.closure.cache_key


@pytest.mark.parametrize(
    ("directory", "name"),
    [
        ("inputs", "ordinary.cpp"),
        ("inputs", "dollar$unit.cpp"),
        ("inputs", r"literal\$unit.cpp"),
        ("dollar$directory", "ordinary.cpp"),
        ("inputs", "back`unit.cpp"),
    ],
)
def test_real_cxx_diagnostic_source_paths(
    tmp_path: Path,
    real_gxx: tuple,
    monkeypatch: pytest.MonkeyPatch,
    directory: str,
    name: str,
) -> None:
    compiler, manifest = real_gxx
    folder = tmp_path / directory
    folder.mkdir()
    source = folder / name
    source.write_bytes(b'extern "C" int point() { return 7; }\n')
    before = collect(source, manifest, compiler=compiler)
    assert not before.reasons and before.closure is not None and before.closure.reusable
    assert before.recipe is not None
    obj = tmp_path / "point.o"
    dep = tmp_path / "point.d"
    command = [
        compiler,
        *before.recipe["flags"],
        "-MD",
        "-MF",
        str(dep),
        "-MT",
        "closure",
        "-c",
        str(source),
        "-o",
        str(obj),
    ]
    built = compile_controlled(command, monkeypatch)
    assert built.returncode == 0 and not built.timed_out, built.stderr
    assert str(source) in c.parse_gcc_dependencies(dep.read_text())
    after = collect(source, manifest, compiler=compiler)
    assert not after.reasons and after.closure is not None
    assert before.closure.cache_key == after.closure.cache_key
    diagnostic = run_compiler(
        [
            compiler,
            *before.recipe["flags"],
            "-###",
            "-c",
            str(source),
            "-o",
            "/dev/null",
        ],
        30,
        label="independent G++ path diagnostic",
        environment=c.CPU_ENVIRONMENT,
    )
    assert diagnostic.returncode == 0 and not diagnostic.timed_out
    (tmp_path / "path-qualification.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": file_hash(source),
                "object_sha256": file_hash(obj),
                "key": after.closure.cache_key,
                "command": command,
                "driver_diagnostic": diagnostic.stderr,
            },
            indent=2,
        )
    )


def test_real_cxx_opaque_cache_and_identity(
    tmp_path: Path, real_gxx: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler, manifest = real_gxx
    monkeypatch.chdir(tmp_path)
    payload = tmp_path / "payload.bin"
    source = tmp_path / "point.cpp"
    source.write_text(
        '__asm__(".pushsection .rodata\\n.incbin \\"payload.bin\\"\\n.popsection");\nextern "C" int point() { return 1; }\n'
    )
    keys, objects = [], []
    launcher = resolve_compiler_cache(60)
    # Retain only cache location/address settings. Compiler and loader inputs
    # must match collection even when the developer's shell is contaminated.
    cache_environment = {
        **c.CPU_ENVIRONMENT,
        **{
            key: os.environ[key]
            for key in (
                "HOME",
                "SCCACHE_DIR",
                "SCCACHE_SERVER_PORT",
                "SCCACHE_SERVER_UDS",
                "CCACHE_DIR",
            )
            if key in os.environ
        },
    }
    for key in (
        "GCC_EXEC_PREFIX",
        "COMPILER_PATH",
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "LIBRARY_PATH",
        "LD_PRELOAD",
    ):
        monkeypatch.setenv(key, str(tmp_path / "inherited-compiler-poison"))
    receipts = []
    for content in (b"CXX-AAAA", b"CXX-BBBB"):
        payload.write_bytes(content)
        snapshot = replace(manifest, files=(*manifest.files, dependency(payload)))
        before = collect(source, snapshot, compiler=compiler)
        assert (
            not before.reasons
            and before.closure is not None
            and before.recipe is not None
        )
        obj = tmp_path / "point.o"
        command = [compiler, *before.recipe["flags"], "-c", str(source), "-o", str(obj)]
        stats = run_compiler(
            [str(launcher.path), "--show-stats"],
            30,
            label="cache stats",
            environment=cache_environment,
        )
        built = compile_controlled(command, monkeypatch)
        assert built.returncode == 0 and not built.timed_out, built.stderr
        after = collect(source, snapshot, compiler=compiler)
        assert (
            not after.reasons
            and after.closure is not None
            and after.closure.cache_key == before.closure.cache_key
        )
        assert content in obj.read_bytes()
        keys.append(after.closure.cache_key)
        objects.append(file_hash(obj))
        receipts.append(
            {
                "command": launcher.wrap(command),
                "stats_before": stats.stdout,
                "object_sha256": objects[-1],
            }
        )
    assert keys[0] != keys[1] and objects[0] != objects[1]
    # Ordinary C++ is cacheable. Exact replay must hit; refreshing a manifest
    # input unseen by the preprocessor must miss through the returned seed.
    # CI restores persistent caches, including earlier runs of this fixture.
    # Change real opaque bytes in each run instead of assuming fixed payloads
    # are cold, clearing the cache, or overriding a recorded input identity.
    namespace = os.urandom(32)
    stats_option = "--show-stats"
    if launcher.name == "ccache":
        cache_environment["CCACHE_STATSLOG"] = str(tmp_path / "ordinary-stats.log")
        stats_option = "--show-log-stats"
    source.write_text('extern "C" int point() { return 1; }\n')
    commands: list[dict[str, Any]] = []
    for content in (b"salt-one", b"salt-one", b"salt-two"):
        payload.write_bytes(namespace + content)
        snapshot = replace(manifest, files=(*manifest.files, dependency(payload)))
        observed = collect(source, snapshot, compiler=compiler)
        assert not observed.reasons and observed.recipe is not None
        command = [
            compiler,
            *observed.recipe["flags"],
            "-c",
            str(source),
            "-o",
            str(tmp_path / "ordinary.o"),
        ]
        # Use this verified launcher's ordinary foreground client for counter
        # evidence; it does not stop/reconfigure the existing shared cache.
        built = run_compiler(
            launcher.wrap(command),
            60,
            label="ordinary G++ cache replay",
            environment=cache_environment,
        )
        assert built.returncode == 0 and not built.timed_out, built.stderr
        stats = run_compiler(
            [str(launcher.path), stats_option],
            30,
            label="cache counters",
            environment=cache_environment,
        )
        assert stats.returncode == 0 and not stats.timed_out, stats.stderr
        commands.append({"command": launcher.wrap(command), "stats": stats.stdout})
    assert commands[0]["command"] == commands[1]["command"]
    assert commands[1]["command"] != commands[2]["command"]
    counter = "Cache hits" if launcher.name == "sccache" else "Hits:"

    def count(text: str, label: str) -> int:
        match = re.search(r"^\s*" + re.escape(label) + r"\s+(\d+)", text, re.MULTILINE)
        assert match is not None, text
        return int(match[1])

    assert count(commands[1]["stats"], counter) > count(commands[0]["stats"], counter)
    counter = "Cache misses" if launcher.name == "sccache" else "Misses:"
    assert count(commands[2]["stats"], counter) > count(commands[1]["stats"], counter)
    (tmp_path / "cache-evidence.json").write_text(
        json.dumps(
            {
                "launcher": launcher.version,
                "counter_source": stats_option,
                "opaque_namespace": namespace.hex(),
                "opaque": receipts,
                "ordinary": commands,
            },
            indent=2,
        )
    )


def test_real_cxx_selection_time_probe_and_timeout(
    tmp_path: Path, real_gxx: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler, manifest = real_gxx
    source = tmp_path / "point.cpp"
    source.write_text(
        '#if __has_include("optional.hpp")\n#define V 2\n#else\n#define V 1\n#endif\nint point() { return V; }\n'
    )
    first = collect(source, manifest, compiler=compiler)
    assert not first.reasons and first.closure is not None
    (tmp_path / "optional.hpp").write_text("// successful non-including probe\n")
    second = collect(source, manifest, compiler=compiler)
    assert (
        not second.reasons
        and second.closure is not None
        and first.closure.cache_key != second.closure.cache_key
    )
    entry = tmp_path / "gxx-entry"
    entry.symlink_to(compiler)
    original = c.run_compiler

    def retarget(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        result = original(command, timeout, **kwargs)
        entry.unlink()
        entry.symlink_to(source)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(c, "run_compiler", retarget)
        assert collect(source, manifest, compiler=str(entry)).reasons == (
            "compiler-selection-changed",
        )
    source.write_text("const char* value = __TIME__;\n")
    assert collect(source, manifest, compiler=compiler).reasons == (
        "unsupported-time-macro",
    )
    os.mkfifo(tmp_path / "blocked.hpp")
    source.write_text('#include "blocked.hpp"\n')
    result = collect(source, manifest, compiler=compiler, timeout=2)
    assert result.reasons == ("collection-timed-out",) and (
        result.closure is None or not result.closure.reusable
    )
