"""Explicit Clang C++20 protocol controls and mandatory native Linux qualification."""

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
from test_bulk_aot_clang_closure import fake_clang, real_clang
from test_bulk_aot_closure import compile_controlled, dependency, variant

__all__ = ["fake_clang", "real_clang"]

if TYPE_CHECKING:
    from typing import Any

FLAGS = ("-std=c++20", "-O1", "-fPIC")


def llvm_quoted(word: str) -> str:
    """Independent LLVM diagnostic printer for protocol fixtures."""
    return '"' + "".join("\\" + ch if ch in '\\"$' else ch for ch in word) + '"'


@pytest.fixture
def fake_clangxx(fake_clang: tuple, monkeypatch: pytest.MonkeyPatch) -> tuple:
    _, old_source, driver, manifest = fake_clang
    source = old_source.with_suffix(".cpp")
    source.write_bytes(old_source.read_bytes())
    gcc = source.parent / "gcc-installation"
    gcc.mkdir()
    monkeypatch.setattr(c, "_system_clangxx", lambda: driver)
    original = c.run_compiler

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        assert "--driver-mode=g++" in command
        result = original(command, timeout, **kwargs)
        if "-###" in command:
            tokens = [
                str(driver),
                "-cc1",
                "-triple",
                "x86_64-linux-gnu",
                "-emit-obj",
                "-resource-dir",
                str(source.parent / "resource"),
                "-x",
                "c++",
                "-std=c++20",
                str(source),
            ]
            return replace(
                result, stderr=" ".join(llvm_quoted(token) for token in tokens)
            )
        if "-dump-raw-tokens" in command:
            projected = os.read(kwargs["pass_fds"][1], 1024 * 1024).decode()
            records = []

            def emit(kind: str, value: str, start: bool = False) -> None:
                records.append(
                    kind
                    + " '"
                    + value
                    + "'\t"
                    + (" [StartOfLine]" if start else "")
                    + "\tLoc=<fixture:1:1>"
                )

            for line in projected.splitlines():
                if line.startswith("#define "):
                    fields = line.split(maxsplit=2)
                    name = fields[1]
                    emit("hash", "#", True)
                    emit("unknown", " ")
                    emit("raw_identifier", "define")
                    emit("unknown", " ")
                    emit("raw_identifier", name.removesuffix("()"))
                    if name.endswith("()"):
                        emit("l_paren", "(")
                        emit("r_paren", ")")
                    if len(fields) > 2:
                        emit("unknown", " ")
                        emit(
                            "numeric_constant"
                            if fields[2][0].isdigit()
                            else "raw_identifier",
                            fields[2],
                        )
                elif line:
                    emit("raw_identifier", line, True)
                emit("unknown", "\n")
            os.write(kwargs["pass_fds"][0], "\n".join(records).encode())
            return replace(result, stdout="", stderr="")
        if "-dump-tokens" in command:
            data = (
                b"identifier 'point'\t\tLoc=<fixture:1:1>\neof ''\t\tLoc=<fixture:1:8>"
            )
            os.write(kwargs["pass_fds"][0], data)
            return replace(result, stdout="", stderr="")
        if "-dM" in command:
            return replace(
                result,
                stdout="#define __cplusplus 202002L\n#define __GXX_ABI_VERSION 1002",
            )
        if "-v" in command:
            return replace(
                result,
                stderr=f'Selected GCC installation: {gcc}\nSelected multilib: .;@m64\n#include "..." search starts here:\n#include <...> search starts here:\n /usr/include/c++/12\nEnd of search list.',
            )
        result = replace(
            result, stdout=result.stdout.replace(str(old_source), str(source))
        )
        if kwargs.get("pass_fds"):
            os.write(kwargs["pass_fds"][0], (result.stdout + result.stderr).encode())
            return replace(result, stdout="", stderr="")
        return result

    monkeypatch.setattr(c, "run_compiler", run)
    return source, manifest


@pytest.fixture
def real_clangxx(tmp_path: Path) -> tuple:
    if sys.platform != "linux":
        pytest.skip("native Linux Clang++ qualification runs separately")
    selected = shutil.which("clang++", path=c.CPU_ENVIRONMENT["PATH"])
    assert selected is not None, "Linux qualification requires /usr/bin/clang++"
    compiler = str(Path(selected).resolve())
    resource = run_compiler(
        [compiler, "--driver-mode=g++", "--no-default-config", "-print-resource-dir"],
        30,
        label="Clang++ fixture resource",
        environment=c.CPU_ENVIRONMENT,
    )
    assert resource.returncode == 0 and not resource.timed_out, resource.stderr
    files = {
        Path(compiler),
        *(p for p in Path(resource.stdout.strip()).rglob("*") if p.is_file()),
    }
    loaded = run_compiler(
        ["/usr/bin/ldd", compiler],
        30,
        label="Clang++ fixture loader",
        environment=c.CPU_ENVIRONMENT,
    )
    assert loaded.returncode == 0 and not loaded.timed_out, loaded.stderr
    files.update(Path(p).resolve() for p in loaded.stdout.split() if p.startswith("/"))
    config = tmp_path / "asserted C++ snapshot contract"
    config.write_text("controlled native Clang integrated C++20 fixture/v1\n")
    files.add(config)
    # Producer assertion: executable/loader/resource/C++ header and search
    # namespaces (including absent/PCH candidates) stay immutable per bracket.
    # Enumeration is not proof of completeness. The C entry is independent.
    return compiler, c.CpuToolchainManifest(
        tuple(dependency(p) for p in sorted(files)), complete=True
    )


def collect(
    source: Path, manifest: c.CpuToolchainManifest, **kwargs: Any
) -> c.CpuClosureResult:
    return c.collect_cpu_closure(
        variant(source.read_text()),
        source,
        compiler=kwargs.pop("compiler", "clang++"),
        compiler_family="clang",
        language="c++",
        flags=kwargs.pop("flags", FLAGS),
        toolchain=manifest,
        **kwargs,
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
def test_unsupported_cxx_flags(fake_clangxx: tuple, flags: tuple) -> None:
    source, manifest = fake_clangxx
    assert collect(source, manifest, flags=flags).reasons == ("unsupported-cpu-flags",)


def test_cxx_abi_and_pch_identity(
    fake_clangxx: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, manifest = fake_clangxx
    before = collect(source, manifest)
    assert not before.reasons and before.closure is not None
    original = c.run_compiler

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        result = original(command, timeout, **kwargs)
        return (
            replace(result, stdout=result.stdout.replace("1002", "1003"))
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


def test_real_cxx_store_execution_and_invalidation(
    tmp_path: Path, real_clangxx: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler, manifest = real_clangxx
    source = tmp_path / "point.cpp"
    header = tmp_path / "value.hpp"
    header.write_text("#define VALUE 1\n")
    text = '#include <array>\n#include <numeric>\n#include <span>\n#include <concepts>\n#include "value.hpp"\ntemplate<std::floating_point T> T cube(T r) { return r*r*r; }\nextern "C" double point(double r) { constexpr std::array<int,3> a{1,2,3}; auto view = std::span{a}; return cube(r) + r*r/2 + std::accumulate(view.begin(),view.end(),0) + VALUE; }\n'
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
    observations = {}
    for label, options in (
        ("driver_plan", ["-###", "-c", str(source), "-o", "/dev/null"]),
        ("macro_abi", ["-dM", "-E", str(source)]),
        ("macro_history", ["-dD", "-E", str(source)]),
        ("include_selection", ["-E", "-v", str(source)]),
    ):
        observed = run_compiler(
            [compiler, *before.recipe["flags"], *options],
            30,
            label="Clang++ qualification observations",
            environment=c.CPU_ENVIRONMENT,
        )
        assert observed.returncode == 0 and not observed.timed_out, observed.stderr
        observations[label] = (
            observed.stdout
            if label in {"macro_abi", "macro_history"}
            else observed.stderr
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
        [
            compiler,
            "--driver-mode=g++",
            "--no-default-config",
            "-shared",
            str(hit.artifact_path),
            "-o",
            str(shared),
        ],
        monkeypatch,
    )
    assert linked.returncode == 0 and not linked.timed_out, linked.stderr
    library = ctypes.CDLL(str(shared))
    library.point.argtypes, library.point.restype = [ctypes.c_double], ctypes.c_double
    rows = []
    for value in ("-2.25", "-.5", "0", ".125", "1", "2.75", "7.125"):
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
            [
                compiler,
                "--driver-mode=g++",
                "--no-default-config",
                "-shared",
                str(hit.artifact_path),
                "-o",
                str(shared),
            ],
        ],
        "closure": before.closure.to_payload(),
        "recipe": before.recipe,
        "observations": observations,
        "environment": c.CPU_ENVIRONMENT,
        "source_text": text,
        "source_sha256": variant(text).source_sha256,
        "compiler_sha256": file_hash(Path(compiler)),
        "rows": rows,
    }
    (tmp_path / "qualification.json").write_text(json.dumps(receipt, indent=2))


def test_real_cxx_opaque_cache_and_identity(
    tmp_path: Path, real_clangxx: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler, manifest = real_clangxx
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
    if launcher.name == "sccache":
        # Attribute ordinary-client statistics to this test alone while reusing
        # the persistent disk cache. Never stop the shared server.
        cache_environment.pop("SCCACHE_SERVER_PORT", None)
        cache_environment["SCCACHE_SERVER_UDS"] = (
            "/tmp/gqc-clangxx-" + os.urandom(12).hex() + ".sock"
        )
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
    previous_snapshot = None
    for content in (b"CXX-AAAA", b"CXX-BBBB"):
        payload.write_bytes(content)
        if previous_snapshot is not None:
            rejected = collect(source, previous_snapshot, compiler=compiler)
            assert rejected.reasons == ("dependency-changed",)
            assert rejected.closure is None or rejected.closure.cache_key is None
        snapshot = replace(manifest, files=(*manifest.files, dependency(payload)))
        previous_snapshot = snapshot
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
            label="ordinary Clang++ cache replay",
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
                "cache_environment": cache_environment,
            },
            indent=2,
        )
    )


def test_explicit_clangxx_domain(fake_clangxx: tuple) -> None:
    source, manifest = fake_clangxx
    result = collect(source, manifest)
    assert not result.reasons and result.closure is not None and result.closure.reusable
    assert result.recipe is not None
    assert result.recipe["language"] == "c++20"
    assert result.recipe["flags"][:3] == [
        "--driver-mode=g++",
        "--no-default-config",
        "-fintegrated-as",
    ]
    for incomplete in (replace(manifest, complete=False), replace(manifest, files=())):
        rejected = collect(source, incomplete)
        assert rejected.reasons and rejected.closure is not None
        assert rejected.closure.cache_key is None


@pytest.mark.parametrize(
    "fault",
    [
        "language",
        "external",
        "standard",
        "abi",
        "search",
        "selection",
        "late-selection",
        "timeout",
        "nonregular",
    ],
)
def test_clangxx_observations_fail_closed(
    fake_clangxx: tuple, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    source, manifest = fake_clangxx
    original = c.run_compiler
    searches = 0

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        nonlocal searches
        result = original(command, timeout, **kwargs)
        if fault == "timeout":
            return replace(result, timed_out=True)
        if "-###" in command:
            diagnostic = result.stderr
            if fault == "language":
                diagnostic = diagnostic.replace('"c++"', '"c"')
            if fault == "external":
                diagnostic += '\n "/usr/bin/as" "input.s"'
            if fault == "standard":
                diagnostic = diagnostic.replace("c++20", "c++17")
            result = replace(result, stderr=diagnostic)
        if "-dM" in command and fault == "abi":
            result = replace(result, stdout=result.stdout.replace("202002L", "201703L"))
        if "-v" in command:
            searches += 1
            if fault == "search":
                result = replace(result, stderr="unknown include search")
            if fault == "late-selection" and searches > 1:
                result = replace(result, stderr=result.stderr + "\nchanged multilib")
        if fault == "selection":
            monkeypatch.setattr(c.shutil, "which", lambda *a, **kw: str(source))
        return result

    monkeypatch.setattr(c, "run_compiler", run)
    if fault == "nonregular":
        driver = Path(manifest.files[0].identity)
        driver.unlink()
        driver.mkdir()
    result = collect(source, manifest)
    assert result.reasons and (
        result.closure is None or result.closure.cache_key is None
    )


@pytest.mark.parametrize(
    "name",
    [
        "ordinary.cpp",
        "space unit.cpp",
        "dollar$unit.cpp",
        "tick`unit.cpp",
        r"back\$unit.cpp",
    ],
)
def test_real_clangxx_diagnostic_paths(
    tmp_path: Path, real_clangxx: tuple, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    compiler, manifest = real_clangxx
    source = tmp_path / name
    source.write_text(
        '#include <array>\nextern "C" int point() { constexpr std::array a{2,3,5}; return a[0]*a[1]+a[2]; }\n'
    )
    result = collect(source, manifest, compiler=compiler)
    raw = run_compiler(
        [
            compiler,
            "--driver-mode=g++",
            "--no-default-config",
            *FLAGS,
            "-###",
            "-c",
            str(source),
        ],
        30,
        label="raw Clang++ plan",
        environment=c.CPU_ENVIRONMENT,
    )
    (tmp_path / "driver-plan.txt").write_text(raw.stderr)
    if "\\" in name:
        compiled = compile_controlled(
            [
                compiler,
                "--driver-mode=g++",
                "--no-default-config",
                *FLAGS,
                "-c",
                str(source),
                "-o",
                str(tmp_path / "rejected-path.o"),
            ],
            monkeypatch,
        )
        assert compiled.returncode == 0 and not compiled.timed_out, compiled.stderr
        assert result.reasons and (
            result.closure is None or not result.closure.reusable
        )
        return
    assert (
        not result.reasons and result.closure is not None and result.recipe is not None
    )
    obj = tmp_path / "point.o"
    dep = tmp_path / "point.d"
    built = compile_controlled(
        [
            compiler,
            *result.recipe["flags"],
            "-MD",
            "-MF",
            str(dep),
            "-MT",
            "closure",
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        monkeypatch,
    )
    assert built.returncode == 0 and not built.timed_out, built.stderr
    assert set(c.parse_gcc_dependencies(dep.read_text())) == {
        str(source),
        *(p.identity for p in result.headers),
    }
    after = collect(source, manifest, compiler=compiler)
    assert not after.reasons and after.closure is not None
    assert after.closure.cache_key == result.closure.cache_key


def test_real_clangxx_unopened_probe_and_timeout(
    tmp_path: Path, real_clangxx: tuple
) -> None:
    compiler, manifest = real_clangxx
    source = tmp_path / "point.cpp"
    source.write_text(
        '#if __has_include("optional.hpp")\n#define V 2\n#else\n#define V 1\n#endif\nint point() { return V; }\n'
    )
    before = collect(source, manifest, compiler=compiler)
    assert not before.reasons
    (tmp_path / "optional.hpp").write_text("// unopened probe\n")
    assert collect(source, manifest, compiler=compiler).reasons == (
        "clang-header-trace-mismatch",
    )
    source.write_text("const char* value = __TIME__;\n")
    assert collect(source, manifest, compiler=compiler).reasons == (
        "unsupported-time-macro",
    )
    os.mkfifo(tmp_path / "blocked.hpp")
    source.write_text('#include "blocked.hpp"\n')
    assert collect(source, manifest, compiler=compiler, timeout=2).reasons == (
        "collection-timed-out",
    )


def test_real_clang_c_and_cxx_identity(
    tmp_path: Path, real_clangxx: tuple, real_clang: tuple
) -> None:
    compiler, manifest = real_clangxx
    c_compiler, c_manifest = real_clang
    union = {p.identity: p for p in (*manifest.files, *c_manifest.files)}
    manifest = c.CpuToolchainManifest(
        tuple(union[k] for k in sorted(union)), complete=True
    )
    c_manifest = manifest
    text = "int point(void) { return 7; }\n"
    cpp, cc = tmp_path / "same.cpp", tmp_path / "same.c"
    cpp.write_text(text)
    cc.write_text(text)
    cxx_result = collect(cpp, manifest, compiler=compiler)
    c_result = c.collect_cpu_closure(
        variant(text),
        cc,
        compiler=c_compiler,
        compiler_family="clang",
        toolchain=c_manifest,
    )
    assert not cxx_result.reasons and not c_result.reasons
    assert cxx_result.closure is not None and c_result.closure is not None
    assert cxx_result.closure.emission_identity == c_result.closure.emission_identity
    assert cxx_result.closure.cache_key != c_result.closure.cache_key


@pytest.mark.parametrize(
    "macro",
    [
        "_GLIBCXX_USE_CXX11_ABI 0",
        "_GLIBCXX_USE_CXX11_ABI() 1",
        "_GLIBCXX_DEBUG 1",
        "_GLIBCXX_PARALLEL 1",
        "_GLIBCXX_PROFILE 1",
        "_LIBCPP_VERSION 180000",
    ],
)
def test_alternate_clangxx_abi_rejected(
    fake_clangxx: tuple, monkeypatch: pytest.MonkeyPatch, macro: str
) -> None:
    source, manifest = fake_clangxx
    original = c.run_compiler

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        result = original(command, timeout, **kwargs)
        if "-dD" in command:
            os.write(kwargs["pass_fds"][0], ("\n#define " + macro + "\n").encode())
        return result

    monkeypatch.setattr(c, "run_compiler", run)
    rejected = collect(source, manifest)
    assert rejected.reasons == ("unsupported-clangxx-abi",)
    assert rejected.closure is not None and rejected.closure.cache_key is None


@pytest.mark.parametrize(
    "word",
    [
        "ordinary.cpp",
        "space unit.cpp",
        "dollar$unit.cpp",
        r"back\$unit.cpp",
        r"two\\$unit.cpp",
        'quote"$.cpp',
        "apos' and `tick`.cpp",
        "",
    ],
)
def test_llvm_diagnostic_identity(word: str) -> None:
    # LLVM 18 sys::printArg always quotes -### arguments, escaping exactly
    # quote, backslash and dollar. This is a printer protocol, not shell input.
    printed = llvm_quoted(word)
    assert c._clang_cxx_job_arguments(' "/clang" "-cc1" ' + printed) == [
        "/clang",
        "-cc1",
        word,
    ]


@pytest.mark.parametrize(
    "text", ['"unfinished', '"trailing\\', r'"unsupported\q"', '"one""two"', "unquoted"]
)
def test_llvm_diagnostic_malformed(text: str) -> None:
    with pytest.raises(ValueError, match="unsupported-clang-job-model"):
        c._clang_cxx_job_arguments(text)


@pytest.mark.parametrize("definition", [" 0", "() 1"])
@pytest.mark.parametrize("restore", ["", "#define _GLIBCXX_USE_CXX11_ABI 1\n"])
@pytest.mark.parametrize("wrapper", [False, True])
def test_real_transient_alternate_abi_fails_closed(
    tmp_path: Path,
    real_clangxx: tuple,
    monkeypatch: pytest.MonkeyPatch,
    restore: str,
    wrapper: bool,
    definition: str,
) -> None:
    compiler, manifest = real_clangxx
    source = tmp_path / "transient.cpp"
    definitions = (
        "#define _GLIBCXX_USE_CXX11_ABI"
        + definition
        + "\n#include <string>\n#undef _GLIBCXX_USE_CXX11_ABI\n"
        + restore
    )
    if wrapper:
        (tmp_path / "wrapper.hpp").write_text(definitions)
        definitions = '#include "wrapper.hpp"\n'
    source.write_text(
        definitions + 'std::string fixture_value() { return "old ABI"; }\n'
    )
    # Independent symbol observation proves that the temporary definition really
    # selected the old ABI, even though the final macro table hides/restores it.
    obj = tmp_path / "old-abi.o"
    built = compile_controlled(
        [
            compiler,
            "--driver-mode=g++",
            "--no-default-config",
            *FLAGS,
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        monkeypatch,
    )
    assert built.returncode == 0 and not built.timed_out, built.stderr
    symbols = run_compiler(
        ["/usr/bin/nm", "-C", str(obj)],
        30,
        label="independent ABI symbols",
        environment=c.CPU_ENVIRONMENT,
    )
    assert symbols.returncode == 0 and not symbols.timed_out, symbols.stderr
    assert " T fixture_value()" in symbols.stdout
    assert "fixture_value[abi:cxx11]" not in symbols.stdout
    macro_table = run_compiler(
        [
            compiler,
            "--driver-mode=g++",
            "--no-default-config",
            *FLAGS,
            "-E",
            "-dM",
            str(source),
        ],
        30,
        label="final macro state",
        environment=c.CPU_ENVIRONMENT,
    )
    assert macro_table.returncode == 0 and not macro_table.timed_out
    assert "#define _GLIBCXX_USE_CXX11_ABI 0" not in macro_table.stdout
    rejected = collect(source, manifest, compiler=compiler)
    (tmp_path / "transient-abi.json").write_text(
        json.dumps(
            {
                "symbols": symbols.stdout,
                "final_macros": macro_table.stdout,
                "reasons": rejected.reasons,
                "object_sha256": file_hash(obj),
            },
            indent=2,
        )
    )
    assert rejected.reasons == ("unsupported-clangxx-abi",)
    assert rejected.closure is not None and rejected.closure.cache_key is None


def test_real_inactive_abi_definition_keeps_native_abi(
    tmp_path: Path,
    real_clangxx: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler, manifest = real_clangxx
    source = tmp_path / "native.cpp"
    source.write_text(
        '#if 0\n#define _GLIBCXX_USE_CXX11_ABI 0\n#endif\n#include <string>\nstd::string fixture_value() { return "native ABI"; }\n'
    )
    result = collect(source, manifest, compiler=compiler)
    assert (
        not result.reasons
        and result.closure is not None
        and result.closure.reusable
        and result.recipe is not None
    )
    obj = tmp_path / "native.o"
    built = compile_controlled(
        [compiler, *result.recipe["flags"], "-c", str(source), "-o", str(obj)],
        monkeypatch,
    )
    assert built.returncode == 0 and not built.timed_out, built.stderr
    symbols = run_compiler(
        ["/usr/bin/nm", "-C", str(obj)],
        30,
        label="independent native ABI symbols",
        environment=c.CPU_ENVIRONMENT,
    )
    assert symbols.returncode == 0 and not symbols.timed_out
    assert "fixture_value[abi:cxx11]()" in symbols.stdout
    after = collect(source, manifest, compiler=compiler)
    assert (
        not after.reasons
        and after.closure is not None
        and after.closure.cache_key == result.closure.cache_key
    )


@pytest.mark.parametrize(
    "declaration",
    [
        "export module fixture;\n",
        "module;\nexport module fixture;\n",
        "#define DECLARE module\nexport DECLARE fixture;\n",
        "import fixture;\n",
    ],
)
def test_real_cxx_modules_fail_closed(
    tmp_path: Path, real_clangxx: tuple, declaration: str
) -> None:
    compiler, manifest = real_clangxx
    source = tmp_path / "unit.cpp"
    source.write_text(declaration + "int point() { return 7; }\n")
    result = collect(source, manifest, compiler=compiler)
    assert result.reasons == ("unsupported-clangxx-module-token",)
    assert result.closure is not None and result.closure.cache_key is None


def test_real_cxx_module_words_in_literals_and_inactive_code(
    tmp_path: Path,
    real_clangxx: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler, manifest = real_clangxx
    source = tmp_path / "unit.cpp"
    source.write_text(
        '#if 0\nexport module fixture;\nimport fixture;\n#endif\n/* module import */\nextern "C" const char* point() { return "module import" R"tag(module import)tag"; }\n'
    )
    before = collect(source, manifest, compiler=compiler)
    assert (
        not before.reasons
        and before.closure is not None
        and before.closure.reusable
        and before.recipe is not None
    )
    obj = tmp_path / "unit.o"
    built = compile_controlled(
        [compiler, *before.recipe["flags"], "-c", str(source), "-o", str(obj)],
        monkeypatch,
    )
    assert built.returncode == 0 and not built.timed_out, built.stderr
    after = collect(source, manifest, compiler=compiler)
    assert (
        not after.reasons
        and after.closure is not None
        and before.closure.cache_key == after.closure.cache_key
    )


@pytest.mark.parametrize("name", ["module", "import"])
def test_real_ordinary_contextual_identifier_is_conservatively_rejected(
    tmp_path: Path,
    real_clangxx: tuple,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    compiler, manifest = real_clangxx
    source = tmp_path / "unit.cpp"
    source.write_text("int point() { int " + name + " = 7; return " + name + "; }\n")
    built = compile_controlled(
        [
            compiler,
            "--driver-mode=g++",
            "--no-default-config",
            *FLAGS,
            "-c",
            str(source),
            "-o",
            str(tmp_path / "unit.o"),
        ],
        monkeypatch,
    )
    assert built.returncode == 0 and not built.timed_out, built.stderr
    result = collect(source, manifest, compiler=compiler)
    assert result.reasons == ("unsupported-clangxx-module-token",)
    assert result.closure is not None and result.closure.cache_key is None


@pytest.mark.parametrize(
    "declaration",
    [
        'import "header.hpp";\n',
        'export import "header.hpp";\n',
        '#define IMPORT import\nIMPORT "header.hpp";\n',
    ],
)
def test_real_header_unit_forms_fail_closed(
    tmp_path: Path, real_clangxx: tuple, declaration: str
) -> None:
    compiler, manifest = real_clangxx
    (tmp_path / "header.hpp").write_text("int header_value();\n")
    source = tmp_path / "unit.cpp"
    source.write_text(declaration + "int point() { return 7; }\n")
    result = collect(source, manifest, compiler=compiler)
    assert result.reasons and (
        result.closure is None or result.closure.cache_key is None
    )


@pytest.mark.parametrize(
    "text",
    [
        "identifier 'point'\t\tLoc=<fixture:1:1>",
        "eof ''\t\tLoc=<fixture:1:1>\nidentifier 'module'\t\tLoc=<fixture:1:2>",
        "eof ''\t\tLoc=<fixture:1:1>\neof ''\t\tLoc=<fixture:1:2>",
        "warning: identifier 'module'\n" + "eof ''\t\tLoc=<fixture:1:2>",
    ],
)
def test_incomplete_or_ambiguous_token_output_fails_closed(
    fake_clangxx: tuple, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    source, manifest = fake_clangxx
    original = c.run_compiler

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        result = original(command, timeout, **kwargs)
        if "-dump-tokens" in command:
            fd = kwargs["pass_fds"][0]
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, text.encode())
        return result

    monkeypatch.setattr(c, "run_compiler", run)
    result = collect(source, manifest)
    assert result.reasons == ("unsupported-clangxx-token-stream",)
    assert result.closure is not None and result.closure.cache_key is None


def test_real_token_capture_budget_fails_closed(
    tmp_path: Path, real_clangxx: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler, manifest = real_clangxx
    source = tmp_path / "unit.cpp"
    source.write_text("int point() { return 7; }\n")
    monkeypatch.setattr(c, "_CLANGXX_TOKEN_BYTES", 128)
    result = collect(source, manifest, compiler=compiler)
    assert result.reasons == ("clangxx-token-output-limit",)
    assert result.closure is not None and result.closure.cache_key is None


def test_real_multiline_raw_strings_do_not_define_abi_modes(
    tmp_path: Path, real_clangxx: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler, manifest = real_clangxx
    source = tmp_path / "unit.cpp"
    source.write_text(
        'const char* value=R"tag(module\nimport\n#define _GLIBCXX_DEBUG 1\n#define _GLIBCXX_USE_CXX11_ABI 0\n#define _GLIBCXX_PARALLEL 1\n)tag";\n'
    )
    result = collect(source, manifest, compiler=compiler)
    assert (
        not result.reasons
        and result.closure is not None
        and result.closure.reusable
        and result.recipe is not None
    )
    built = compile_controlled(
        [
            compiler,
            *result.recipe["flags"],
            "-c",
            str(source),
            "-o",
            str(tmp_path / "unit.o"),
        ],
        monkeypatch,
    )
    assert built.returncode == 0 and not built.timed_out, built.stderr
    after = collect(source, manifest, compiler=compiler)
    assert (
        not after.reasons
        and after.closure is not None
        and after.closure.cache_key == result.closure.cache_key
    )


@pytest.mark.parametrize("transient", [False, True])
def test_real_parallel_mode_objects_are_not_qualified(
    tmp_path: Path,
    real_clangxx: tuple,
    monkeypatch: pytest.MonkeyPatch,
    transient: bool,
) -> None:
    compiler, manifest = real_clangxx
    query = run_compiler(
        [compiler, "--driver-mode=g++", "--no-default-config", "-print-resource-dir"],
        30,
        label="matching OpenMP resource",
        environment=c.CPU_ENVIRONMENT,
    )
    assert query.returncode == 0 and not query.timed_out
    include = Path(query.stdout.strip()) / "include"
    if not (include / "omp.h").is_file():
        pytest.skip(
            "matching Clang OpenMP header unavailable; parallel object proof remains unqualified"
        )
    source = tmp_path / "parallel.cpp"
    source.write_text(
        "#define _GLIBCXX_PARALLEL 1\n#include <algorithm>\n"
        + ("#undef _GLIBCXX_PARALLEL\n" if transient else "")
        + "void fixture_sort(int* a,int* b) { std::sort(a,b); }\n"
    )
    flags = (*FLAGS, "-I" + str(include))
    obj = tmp_path / "parallel.o"
    built = compile_controlled(
        [
            compiler,
            "--driver-mode=g++",
            "--no-default-config",
            *flags,
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        monkeypatch,
    )
    assert built.returncode == 0 and not built.timed_out, built.stderr
    nm = run_compiler(
        ["/usr/bin/nm", "-C", str(obj)],
        30,
        label="independent parallel symbols",
        environment=c.CPU_ENVIRONMENT,
    )
    assert nm.returncode == 0 and not nm.timed_out
    assert re.search(
        r"\bU omp_(?:get_max_threads|get_num_threads|get_thread_num)\b", nm.stdout
    )
    rejected = collect(source, manifest, compiler=compiler, flags=flags)
    assert rejected.reasons == ("unsupported-clangxx-abi",)
    assert rejected.closure is not None and rejected.closure.cache_key is None
    (tmp_path / "parallel-rejection.json").write_text(
        json.dumps(
            {
                "header": str(include / "omp.h"),
                "symbols": nm.stdout,
                "object_sha256": file_hash(obj),
                "reasons": rejected.reasons,
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("unused_macro", [False, True])
def test_real_raw_abi_text_is_not_a_directive(
    tmp_path: Path,
    real_clangxx: tuple,
    monkeypatch: pytest.MonkeyPatch,
    unused_macro: bool,
) -> None:
    compiler, manifest = real_clangxx
    source = tmp_path / "unit.cpp"
    literal = 'R"tag(\n#define _GLIBCXX_DEBUG 1\n#define _GLIBCXX_PARALLEL 1\n#define _GLIBCXX_USE_CXX11_ABI 0\n)tag"'
    text = (
        ("#define UNUSED " + literal + "\n")
        if unused_macro
        else ("const char* data=" + literal + ";\n")
    )
    source.write_text(text + "int point(){return 7;}\n")
    before = collect(source, manifest, compiler=compiler)
    assert (
        not before.reasons
        and before.closure is not None
        and before.closure.reusable
        and before.recipe is not None
    )
    built = compile_controlled(
        [
            compiler,
            *before.recipe["flags"],
            "-c",
            str(source),
            "-o",
            str(tmp_path / "unit.o"),
        ],
        monkeypatch,
    )
    assert built.returncode == 0 and not built.timed_out, built.stderr
    after = collect(source, manifest, compiler=compiler)
    assert (
        not after.reasons
        and after.closure is not None
        and before.closure.cache_key == after.closure.cache_key
    )


@pytest.mark.parametrize("fault", ["missing-sentinel", "trailing-token", "diagnostic"])
def test_macro_projection_protocol_is_complete_before_policy(
    fake_clangxx: tuple, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    source, manifest = fake_clangxx
    original = c.run_compiler

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        result = original(command, timeout, **kwargs)
        if "-dump-raw-tokens" in command:
            fd = kwargs["pass_fds"][0]
            os.lseek(fd, 0, os.SEEK_SET)
            data = os.read(fd, 1024 * 1024).decode()
            if fault == "missing-sentinel":
                data = data.replace("generativeqc_macro_end_", "missing_marker_")
            elif fault == "trailing-token":
                data += "\nraw_identifier 'extra'\t\tLoc=<fixture:2:1>"
            else:
                data = "warning: macro text\n" + data
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, data.encode())
        return result

    monkeypatch.setattr(c, "run_compiler", run)
    result = collect(source, manifest)
    assert result.reasons == ("unsupported-clangxx-macro-stream",)
    assert result.closure is not None and result.closure.cache_key is None
