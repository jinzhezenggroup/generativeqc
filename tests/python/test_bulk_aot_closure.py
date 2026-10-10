"""Dependency-parser controls and mandatory Linux real-GCC/store integration."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.common import compiler_cache, compiler_process
from generativeqc_compiler.common.compiler_cache import run_cached_compiler
from generativeqc_compiler.common.compiler_process import CompileResult, run_compiler
from generativeqc_compiler.common.provenance import file_hash
from generativeqc_compiler.xc import bulk_aot, bulk_aot_cache, bulk_aot_store
from generativeqc_compiler.xc import bulk_aot_closure as collector

if TYPE_CHECKING:
    from typing import Any


def variant(source: str) -> bulk_aot.SourceVariant:
    return bulk_aot.SourceVariant(
        name="A",
        family="lda",
        spin="unpolarized",
        import_identity="A",
        domain="test/v1",
        features=("rho",),
        derivative_order=0,
        backend="cpu",
        source=source,
        energy_nodes=1,
        ssa={},
    )


def dependency(path: Path) -> bulk_aot_cache.CacheDependency:
    return bulk_aot_cache.CacheDependency("toolchain-file", str(path), file_hash(path))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("closure: /a.c /usr/include/x.h\n", ("/a.c", "/usr/include/x.h")),
        ("closure: /a\\ b.c \\\n /h\\#1.h /d$$x.h\n", ("/a b.c", "/d$x.h", "/h#1.h")),
        ("closure: /a\\\\b.c /a\\\\b.c\n", ("/a\\\\b.c",)),
        ("closure: /a\\q.c /a\\\\\\ b.h\n", ("/a\\ b.h", "/a\\q.c")),
        ("closure: /a.c \\\r\n\t/b.h\r\n", ("/a.c", "/b.h")),
        ("closure: /a.c /tail\\ \n", ("/a.c", "/tail ")),
        ("closure: /a.c /tail\\\t\n", ("/a.c", "/tail\t")),
    ],
)
def test_parser(text: str, expected: tuple[str, ...]) -> None:
    assert collector.parse_gcc_dependencies(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "wrong: /a.c",
        "closure:",
        "closure: /a\\",
        "closure: /a$(x)",
        "closure: /a #comment",
        "closure: /a\nother: /b",
        "closure: /a\\\\ /b.h",
        "closure: /a\\ /b.h",
        "closure: /a\\  /b.h",
        "closure: /a\\\n /b.h\n",
        "closure: /a\\\n",
    ],
)
def test_unsupported_make_syntax(text: str) -> None:
    with pytest.raises(ValueError):
        collector.parse_gcc_dependencies(text)


def test_import_does_not_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("import must not execute a compiler")

    # Patch the imported owner, since reloading replaces a collector-local mock.
    with monkeypatch.context() as patch:
        patch.setattr(compiler_process, "run_compiler", forbidden)
        importlib.reload(collector)
    importlib.reload(collector)


@pytest.fixture
def fake_gcc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[bulk_aot.SourceVariant, Path, collector.CpuToolchainManifest]:
    compiler = tmp_path / "gcc"
    compiler.write_bytes(b"\x7fELFfake-driver")
    source = tmp_path / "point.c"
    source.write_bytes(b"int point(void) { return 1; }\n")
    header = tmp_path / "nested.h"
    header.write_text("#define VALUE 1\n", encoding="utf-8")
    monkeypatch.setattr(collector.sys, "platform", "linux")
    monkeypatch.setattr(collector.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(collector, "_system_gcc", lambda: compiler)
    monkeypatch.setattr(
        collector.shutil, "which", lambda *args, **kwargs: str(compiler)
    )

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        assert kwargs["environment"] == collector.CPU_ENVIRONMENT
        assert timeout > 0
        if "--version" in command:
            text = "gcc Free Software Foundation"
        elif "-dumpfullversion" in command:
            text = "12.2.0"
        elif "-dumpmachine" in command:
            text = "x86_64-linux-gnu"
        elif any(arg.startswith("-print-prog-name=") for arg in command):
            text = str(compiler)
        elif "-dumpspecs" in command:
            text = "specs"
        elif "-E" in command:
            text = "preprocessed"
        else:

            def escape(path: Path) -> str:
                return str(path).replace(" ", "\\ ")

            text = "closure: " + escape(source) + " " + escape(header)
        return CompileResult(0, False, 0.001, text, "")

    monkeypatch.setattr(collector, "run_compiler", run)
    manifest = collector.CpuToolchainManifest((dependency(compiler),), complete=True)
    return variant(source.read_text(encoding="utf-8")), source, manifest


def test_manifest_is_never_inferred(fake_gcc: tuple) -> None:
    item, source, manifest = fake_gcc
    result = collector.collect_cpu_closure(item, source)
    assert result.closure is not None
    assert result.closure.cache_key is None
    assert result.reasons == (
        "toolchain-manifest-incomplete",
        "downstream-toolchain-not-covered",
    )
    result = collector.collect_cpu_closure(
        item, source, toolchain=replace(manifest, complete=False)
    )
    assert result.closure is not None and not result.closure.complete
    assert result.reasons == ("toolchain-manifest-incomplete",)
    result = collector.collect_cpu_closure(
        item, source, toolchain=collector.CpuToolchainManifest(complete=True)
    )
    assert result.closure is not None and not result.closure.complete
    assert result.reasons == ("downstream-toolchain-not-covered",)


def test_dependency_query_preserves_terminal_space(
    fake_gcc: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    item, source, manifest = fake_gcc
    header = source.parent / "tail.h "
    header.write_text("/* trailing-space dependency */\n", encoding="utf-8")
    original = collector.run_compiler

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        result = original(command, timeout, **kwargs)
        if "-M" in command:
            names = [str(source), str(header)]
            text = "closure: " + " ".join(n.replace(" ", "\\ ") for n in names) + "\n"
            return replace(result, stdout=text)
        return result

    monkeypatch.setattr(collector, "run_compiler", run)
    result = collector.collect_cpu_closure(item, source, toolchain=manifest)
    assert not result.reasons and result.closure is not None
    assert result.closure.reusable
    assert [record.identity for record in result.headers] == [str(header)]


@pytest.mark.parametrize(
    "flag",
    [
        "@args",
        "-fplugin=x",
        "-B/tmp",
        "--sysroot=/x",
        "-march=native",
        "-include/x",
        "-Irelative",
        "-x",
        "-DANY=1",
    ],
)
def test_unsupported_flags_fail_closed(fake_gcc: tuple, flag: str) -> None:
    item, source, manifest = fake_gcc
    result = collector.collect_cpu_closure(
        item, source, flags=(flag,), toolchain=manifest
    )
    assert result.closure is None
    assert result.reasons == ("unsupported-cpu-flags",)


@pytest.mark.parametrize(
    "fault",
    [
        "timeout",
        "failed",
        "missing",
        "mutation",
        "set-change",
        "source",
        "compiler",
        "wrapper",
        "version",
        "specs",
        "target",
        "relative",
        "preprocessed",
        "late-mutation",
    ],
)
def test_collection_negative_controls(
    fake_gcc: tuple,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    item, source, manifest = fake_gcc
    original = collector.run_compiler
    scans = 0
    versions = 0
    specs = 0
    preprocesses = 0

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        nonlocal scans, versions, specs, preprocesses
        result = original(command, timeout, **kwargs)
        if "--version" in command:
            versions += 1
            if fault == "version" and versions > 1:
                return replace(result, stdout="changed version")
        if "-dumpspecs" in command:
            specs += 1
            if fault == "specs" and specs > 1:
                return replace(result, stdout="changed specs")
        if "-dumpmachine" in command and fault == "target":
            return replace(result, stdout="aarch64-none-elf")
        if "-E" in command:
            preprocesses += 1
            if preprocesses == 2:
                if fault == "preprocessed":
                    return replace(result, stdout="changed preprocessor output")
                if fault == "late-mutation":
                    source.with_name("nested.h").write_text(
                        "late change", encoding="utf-8"
                    )
        if "-M" in command:
            scans += 1
            if fault == "timeout":
                return replace(result, timed_out=True, returncode=124)
            if fault == "failed":
                return replace(result, returncode=1)
            if fault == "relative":
                return replace(result, stdout="closure: relative.c")
            header = source.with_name("nested.h")
            if fault == "missing":
                header.unlink(missing_ok=True)
            if scans == 2:
                if fault == "mutation":
                    header.write_text("changed", encoding="utf-8")
                elif fault == "set-change":
                    return replace(result, stdout=result.stdout + " /other.h")
                elif fault == "source":
                    source.write_text("changed", encoding="utf-8")
                elif fault == "compiler":
                    Path(manifest.files[0].identity).write_bytes(b"changed")
        return result

    if fault == "wrapper":
        Path(manifest.files[0].identity).write_bytes(b"#!/bin/sh")
    monkeypatch.setattr(collector, "run_compiler", run)
    result = collector.collect_cpu_closure(item, source, toolchain=manifest)
    assert result.reasons
    assert result.closure is None or (
        not result.closure.complete and result.closure.cache_key is None
    )


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_deadline(fake_gcc: tuple, timeout: float) -> None:
    item, source, _ = fake_gcc
    with pytest.raises(ValueError, match="positive and finite"):
        collector.collect_cpu_closure(item, source, timeout=timeout)


def test_source_mismatch(fake_gcc: tuple) -> None:
    item, source, manifest = fake_gcc
    source.write_text("changed", encoding="utf-8")
    result = collector.collect_cpu_closure(item, source, toolchain=manifest)
    assert result.reasons == ("source-mismatch",)
    assert result.closure is None


def test_ordered_flags_and_context_identity(fake_gcc: tuple) -> None:
    item, source, manifest = fake_gcc
    first = collector.collect_cpu_closure(item, source, toolchain=manifest)
    changed = collector.collect_cpu_closure(
        item, source, toolchain=manifest, flags=("-fPIC", "-O1", "-std=c99")
    )
    assert first.closure is not None and changed.closure is not None
    assert first.closure.reusable and changed.closure.reusable
    assert first.closure.cache_key != changed.closure.cache_key
    assert "compiler-flags" in bulk_aot_cache.invalidation_reasons(
        first.closure, changed.closure
    )


@pytest.fixture
def real_gcc(tmp_path: Path) -> tuple[str, collector.CpuToolchainManifest]:
    if sys.platform != "linux":
        pytest.skip("Linux GCC integration runs in mandatory repository CI")
    compiler = shutil.which("gcc", path=collector.CPU_ENVIRONMENT["PATH"])
    assert compiler is not None, "Linux integration requires real GCC"
    compiler = str(Path(compiler).resolve())

    def query(args: list[str]) -> str:
        result = run_compiler(
            [compiler, *args],
            30,
            label="integration manifest",
            environment=collector.CPU_ENVIRONMENT,
        )
        assert result.returncode == 0 and not result.timed_out
        return result.stdout.strip()

    files = {Path(compiler)}
    for name in ("cc1", "as"):
        found = shutil.which(
            query(["-print-prog-name=" + name]), path=collector.CPU_ENVIRONMENT["PATH"]
        )
        assert found is not None
        files.add(Path(found).resolve())
    # Explicit test producer assertion, not a collector inference: the CI image
    # and include namespaces remain unchanged within each bracketed compilation.
    # Include actual loaded libraries in the fixture's asserted toolchain files.
    for tool in tuple(files):
        result = run_compiler(
            ["/usr/bin/ldd", str(tool)],
            30,
            label="integration libraries",
            environment=collector.CPU_ENVIRONMENT,
        )
        assert result.returncode == 0
        for token in result.stdout.split():
            if token.startswith("/"):
                files.add(Path(token).resolve())
    config = tmp_path / "explicit toolchain config"
    config.write_text("controlled native GCC C fixture/v1\n", encoding="utf-8")
    files.add(config)
    return compiler, collector.CpuToolchainManifest(
        tuple(dependency(path) for path in sorted(files)), complete=True
    )


def compile_controlled(
    command: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> CompileResult:
    # Verify the incoming absolute launcher before restricting compiler PATH.
    # Cache execution provenance is not part of scientific/build identity.
    launcher = compiler_cache.resolve_compiler_cache(60)
    with monkeypatch.context() as clean:
        clean.setattr(
            compiler_cache, "resolve_compiler_cache", lambda timeout: launcher
        )
        for key in tuple(os.environ):
            clean.delenv(key)
        for key, value in collector.CPU_ENVIRONMENT.items():
            clean.setenv(key, value)
        return run_cached_compiler(command, 60, label="real CPU closure integration")


@pytest.mark.parametrize("position", ["followed", "wrapped", "final"])
def test_real_gcc_terminal_backslash_decoys_fail_closed(
    real_gcc: tuple, position: str
) -> None:
    compiler, manifest = real_gcc
    # A short path keeps both user headers on one GCC -M line. Cover a decoy
    # for merged tokens, a wrapped separator and the final rule terminator, so
    # existence is no oracle for any of the ambiguous spellings.
    with tempfile.TemporaryDirectory(prefix="closure-", dir="/tmp") as temporary:
        root = Path(temporary)
        if position == "wrapped":
            root /= "long-" + "n" * 90
            root.mkdir()
        header = root / "foo\\"
        other = root / "z.h"
        header.write_text("/* real first header */\n", encoding="utf-8")
        other.write_text("/* real second header */\n", encoding="utf-8")
        if position == "followed":
            merged = Path(str(header)[:-1] + " " + str(other))
            merged.parent.mkdir(parents=True)
            merged.write_text("/* merged decoy */\n", encoding="utf-8")
        elif position == "wrapped":
            (root / "foo ").write_text("/* wrapped decoy */\n", encoding="utf-8")
        else:
            (root / "foo").write_text("/* final-continuation decoy */\n", encoding="utf-8")
        source = root / "point.c"
        includes = ['#include "foo\\"\n', '#include "z.h"\n']
        if position == "final":
            includes.reverse()
        text = "".join(includes) + "int point(void) { return 0; }\n"
        source.write_text(text, encoding="utf-8")
        emitted = run_compiler(
            [compiler, "-M", "-MT", "closure", str(source)],
            30,
            label="terminal-backslash GCC evidence",
            environment=collector.CPU_ENVIRONMENT,
        )
        assert emitted.returncode == 0 and not emitted.timed_out, emitted.stderr
        if position == "followed":
            assert str(header) + " " + str(other) in emitted.stdout, repr(
                emitted.stdout
            )
        elif position == "wrapped":
            assert str(header) + " \\\n" in emitted.stdout, repr(emitted.stdout)
        else:
            assert emitted.stdout.endswith(str(header) + "\n"), repr(emitted.stdout)
        for comment in ("first", "changed"):
            header.write_text(f"/* {comment} real header */\n", encoding="utf-8")
            result = collector.collect_cpu_closure(
                variant(text), source, compiler=compiler, toolchain=manifest
            )
            assert result.reasons
            assert result.closure is not None and result.closure.cache_key is None


@pytest.mark.parametrize("suffix", [" ", "\t"])
def test_real_gcc_terminal_whitespace_dependencies(
    tmp_path: Path, real_gcc: tuple, suffix: str
) -> None:
    compiler, manifest = real_gcc
    header = tmp_path / ("tail.h" + suffix)
    header.write_text("/* real trailing-whitespace header */\n", encoding="utf-8")
    source = tmp_path / "point.c"
    text = f'#include "{header.name}"\nint point(void) {{ return 0; }}\n'
    source.write_text(text, encoding="utf-8")
    emitted = run_compiler(
        [compiler, "-M", "-MT", "closure", str(source)],
        30,
        label="terminal-whitespace GCC evidence",
        environment=collector.CPU_ENVIRONMENT,
    )
    assert emitted.returncode == 0 and not emitted.timed_out, emitted.stderr
    assert emitted.stdout.endswith("\\" + suffix + "\n"), repr(emitted.stdout)
    keys = []
    for comment in ("first", "changed"):
        header.write_text(f"/* {comment} header comment */\n", encoding="utf-8")
        result = collector.collect_cpu_closure(
            variant(text), source, compiler=compiler, toolchain=manifest
        )
        assert not result.reasons and result.closure is not None
        assert result.closure.reusable
        observed = {entry.identity: entry.content_sha256 for entry in result.headers}
        assert observed[str(header)] == file_hash(header)
        keys.append(result.closure.cache_key)
    assert keys[0] != keys[1]


def test_real_gcc_identity_headers_and_store(
    tmp_path: Path,
    real_gcc: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler, manifest = real_gcc
    root = tmp_path / "headers with spaces"
    root.mkdir()
    nested = root / "nested \\ #$ header.h"
    nested.write_text("#define VALUE 1\n", encoding="utf-8")
    outer = root / "outer.h"
    outer.write_text(
        f'#include "{nested.name}"\n#include <stddef.h>\n#include <stdint.h>\n',
        encoding="utf-8",
    )
    source = root / "same source.c"
    text = '#include "outer.h"\nint point(void) { return VALUE + sizeof(size_t); }\n'
    source.write_text(text, encoding="utf-8")
    item = variant(text)

    def collect(
        current: bulk_aot.SourceVariant = item, **kwargs: Any
    ) -> collector.CpuClosureResult:
        result = collector.collect_cpu_closure(
            current,
            source,
            compiler=compiler,
            toolchain=kwargs.pop("toolchain", manifest),
            **kwargs,
        )
        assert not result.reasons, result.reasons
        assert result.closure is not None and result.closure.reusable
        return result

    first = collect()
    assert first.recipe is not None and first.closure is not None
    names = {record.identity for record in first.headers}
    assert str(nested) in names and str(outer) in names
    assert any(name.startswith("/usr/") and name.endswith("stddef.h") for name in names)
    for header in first.headers:
        assert header.content_sha256 == file_hash(Path(header.identity))
    assert first.recipe["compiler"]["executable_sha256"] == file_hash(Path(compiler))
    assert first.recipe["translation_unit_sha256"] == file_hash(source)
    # Independently observe GCC's raw emitted data, including real continuations.
    dep = run_compiler(
        [compiler, *first.recipe["flags"], "-M", "-MT", "closure", str(source)],
        30,
        label="independent GCC dependencies",
        environment=collector.CPU_ENVIRONMENT,
    )
    assert dep.returncode == 0 and "\\\n" in dep.stdout
    assert set(collector.parse_gcc_dependencies(dep.stdout)) == names | {str(source)}
    obj = root / "point.o"
    compiled = compile_controlled(
        [compiler, *first.recipe["flags"], "-c", str(source), "-o", str(obj)],
        monkeypatch,
    )
    assert compiled.returncode == 0 and not compiled.timed_out, compiled.stderr
    assert obj.is_file()
    after = collect()
    assert (
        after.closure is not None and first.closure.cache_key == after.closure.cache_key
    )
    cache = tmp_path / "cache"
    stored = bulk_aot_store.store_artifact(cache, after.closure, obj)
    assert stored.status == "hit" and stored.object_sha256 == file_hash(obj)
    assert bulk_aot_store.lookup_artifact(cache, collect().closure) == stored
    alias = collect(replace(item, name="B", import_identity="B"))
    assert (
        alias.closure is not None and alias.closure.cache_key == first.closure.cache_key
    )
    keys = {first.closure.cache_key}
    nested.write_text("#define VALUE 2\n", encoding="utf-8")
    changed = collect()
    assert changed.closure is not None
    keys.add(changed.closure.cache_key)
    assert bulk_aot_store.lookup_artifact(cache, changed.closure).status == "miss"
    source.write_text(text + "int extra(void) { return 2; }\n", encoding="utf-8")
    changed_source = variant(source.read_text(encoding="utf-8"))
    result = collect(changed_source)
    assert result.closure is not None
    keys.add(result.closure.cache_key)
    result = collect(changed_source, flags=("-std=c99", "-O2", "-fPIC"))
    assert result.closure is not None
    keys.add(result.closure.cache_key)
    config = tmp_path / "explicit toolchain config"
    config.write_text("changed configuration", encoding="utf-8")
    stale = collector.collect_cpu_closure(
        changed_source, source, compiler=compiler, toolchain=manifest
    )
    assert stale.reasons == ("dependency-changed",)
    assert stale.closure is not None and stale.closure.cache_key is None
    refreshed = replace(
        manifest,
        files=tuple(dependency(Path(record.identity)) for record in manifest.files),
    )
    result = collect(changed_source, toolchain=refreshed)
    assert result.closure is not None
    keys.add(result.closure.cache_key)
    assert len(keys) == 5
    nested.unlink()
    missing = collector.collect_cpu_closure(
        changed_source, source, compiler=compiler, toolchain=refreshed
    )
    assert missing.closure is not None and missing.closure.cache_key is None
    assert missing.reasons == ("compiler-command-failed",)


def test_real_compiler_timeout(tmp_path: Path, real_gcc: tuple) -> None:
    compiler, manifest = real_gcc
    source = tmp_path / "point.c"
    source.write_text("int point(void) { return 0; }\n", encoding="utf-8")
    result = collector.collect_cpu_closure(
        variant(source.read_text(encoding="utf-8")),
        source,
        compiler=compiler,
        toolchain=manifest,
        timeout=0.000001,
    )
    assert result.reasons == ("collection-timed-out",)
    assert result.closure is None or result.closure.cache_key is None


def test_real_gcc_blocked_include_times_out(
    tmp_path: Path,
    real_gcc: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler, manifest = real_gcc
    # GCC opens this include while preprocessing; no writer ever opens it.
    # The existing process-group owner must kill the actual blocked compiler.
    os.mkfifo(tmp_path / "blocked.h")
    text = '#include "blocked.h"\nint point(void) { return 0; }\n'
    source = tmp_path / "point.c"
    source.write_text(text, encoding="utf-8")
    original = collector.run_compiler
    timed_out_commands = []

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        result = original(command, timeout, **kwargs)
        if result.timed_out:
            timed_out_commands.append(command)
        return result

    monkeypatch.setattr(collector, "run_compiler", run)
    result = collector.collect_cpu_closure(
        variant(text),
        source,
        compiler=compiler,
        toolchain=manifest,
        timeout=10,
    )
    assert any("-E" in command for command in timed_out_commands)
    assert result.reasons == ("collection-timed-out",)
    assert result.closure is not None and not result.closure.complete
    assert result.closure.cache_key is None


def test_real_header_mutation_during_collection(
    tmp_path: Path,
    real_gcc: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler, manifest = real_gcc
    header = tmp_path / "value.h"
    header.write_text("#define VALUE 1\n", encoding="utf-8")
    text = '#include "value.h"\nint point(void) { return VALUE; }\n'
    source = tmp_path / "point.c"
    source.write_text(text, encoding="utf-8")
    scans = 0
    original = collector.run_compiler

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        nonlocal scans
        result = original(command, timeout, **kwargs)
        if "-M" in command:
            scans += 1
            if scans == 2:
                header.write_text("#define VALUE 2\n", encoding="utf-8")
        return result

    monkeypatch.setattr(collector, "run_compiler", run)
    result = collector.collect_cpu_closure(
        variant(text), source, compiler=compiler, toolchain=manifest
    )
    assert scans == 2 and result.reasons == ("dependency-changed",)
    assert result.closure is not None and not result.closure.complete
    assert result.closure.cache_key is None


def test_real_has_include_without_inclusion_changes_identity(
    tmp_path: Path,
    real_gcc: tuple,
) -> None:
    compiler, manifest = real_gcc
    text = '#if __has_include("optional.h")\n#define VALUE 2\n#else\n#define VALUE 1\n#endif\nint point(void) { return VALUE; }\n'
    source = tmp_path / "point.c"
    source.write_text(text, encoding="utf-8")
    before = collector.collect_cpu_closure(
        variant(text), source, compiler=compiler, toolchain=manifest
    )
    (tmp_path / "optional.h").write_text(
        "/* intentionally never included */\n", encoding="utf-8"
    )
    after = collector.collect_cpu_closure(
        variant(text), source, compiler=compiler, toolchain=manifest
    )
    assert not before.reasons and not after.reasons
    assert before.headers == after.headers
    assert before.closure is not None and after.closure is not None
    assert before.closure.cache_key != after.closure.cache_key


def test_time_macros_are_unsupported(fake_gcc: tuple) -> None:
    item, source, manifest = fake_gcc
    text = "const char *time = __TIME__;\n"
    source.write_bytes(text.encode("utf-8"))
    result = collector.collect_cpu_closure(
        replace(item, source=text), source, toolchain=manifest
    )
    assert result.reasons == ("unsupported-time-macro",)
    assert result.closure is None


def test_working_directory_mutation_fails_closed(
    fake_gcc: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item, source, manifest = fake_gcc
    original = collector.run_compiler

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        result = original(command, timeout, **kwargs)
        if "-E" in command:
            monkeypatch.chdir(source.parent)
        return result

    monkeypatch.setattr(collector, "run_compiler", run)
    result = collector.collect_cpu_closure(item, source, toolchain=manifest)
    assert result.reasons == ("working-directory-changed",)
    assert result.closure is not None and result.closure.cache_key is None


def test_real_working_directory_changes_object_and_key(
    tmp_path: Path,
    real_gcc: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler, manifest = real_gcc
    text = '__asm__(".pushsection .rodata\\n.incbin \\"payload.bin\\"\\n.popsection");\nint point(void) { return 0; }\n'
    source = tmp_path / "point.c"
    source.write_text(text, encoding="utf-8")
    payloads = []
    for name, value in (("left", b"AAAA"), ("right", b"BBBB")):
        folder = tmp_path / name
        folder.mkdir()
        payload = folder / "payload.bin"
        payload.write_bytes(value)
        payloads.append(payload)
    # The same complete manifest covers both possible immutable assembler inputs.
    # Only cwd chooses the one consumed; file-set hashing alone cannot bind it.
    manifest = replace(
        manifest, files=(*manifest.files, *(dependency(p) for p in payloads))
    )
    keys = []
    objects = []
    store = tmp_path / "store"
    for payload in payloads:
        monkeypatch.chdir(payload.parent)
        before = collector.collect_cpu_closure(
            variant(text), source, compiler=compiler, toolchain=manifest
        )
        assert (
            not before.reasons
            and before.closure is not None
            and before.recipe is not None
        )
        assert before.closure.reusable
        assert before.recipe["working_directory"] == str(payload.parent)
        obj = payload.parent / "point.o"
        built = compile_controlled(
            [compiler, *before.recipe["flags"], "-c", str(source), "-o", str(obj)],
            monkeypatch,
        )
        assert built.returncode == 0 and not built.timed_out, built.stderr
        after = collector.collect_cpu_closure(
            variant(text), source, compiler=compiler, toolchain=manifest
        )
        assert not after.reasons and after.closure is not None
        assert before.closure.cache_key == after.closure.cache_key
        keys.append(after.closure.cache_key)
        objects.append(file_hash(obj))
        assert bulk_aot_store.lookup_artifact(store, after.closure).status == "miss"
        stored = bulk_aot_store.store_artifact(store, after.closure, obj)
        assert stored.status == "hit" and stored.object_sha256 == objects[-1]
    assert keys[0] != keys[1]
    assert objects[0] != objects[1]
    # Change an opaque assembler input in the SAME working directory. GCC -M/-E
    # cannot see its contents; the returned closure salt must invalidate the
    # compiler cache too, and must deliver new object bytes, not merely a miss
    # in bulk_aot_store followed by a stale compiler-cache hit.
    payload = payloads[-1]
    payload.write_bytes(b"CCCC")
    refreshed = replace(
        manifest, files=tuple(dependency(Path(d.identity)) for d in manifest.files)
    )
    before = collector.collect_cpu_closure(
        variant(text), source, compiler=compiler, toolchain=refreshed
    )
    assert (
        not before.reasons and before.closure is not None and before.recipe is not None
    )
    assert before.closure.cache_key != keys[-1]
    changed_obj = payload.parent / "changed.o"
    built = compile_controlled(
        [compiler, *before.recipe["flags"], "-c", str(source), "-o", str(changed_obj)],
        monkeypatch,
    )
    assert built.returncode == 0 and not built.timed_out, built.stderr
    after = collector.collect_cpu_closure(
        variant(text), source, compiler=compiler, toolchain=refreshed
    )
    assert not after.reasons and after.closure is not None
    assert before.closure.cache_key == after.closure.cache_key
    assert file_hash(changed_obj) != objects[-1]
    assert b"CCCC" in changed_obj.read_bytes()
    assert b"BBBB" in (payload.parent / "point.o").read_bytes()
    assert bulk_aot_store.lookup_artifact(store, after.closure).status == "miss"
    stored = bulk_aot_store.store_artifact(store, after.closure, changed_obj)
    assert stored.object_sha256 == file_hash(changed_obj)
