"""Opt-in Clang controls and real native Linux object/store qualification."""

from __future__ import annotations

import ctypes
import json
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
from generativeqc_compiler.xc import bulk_aot_closure as collector
from generativeqc_compiler.xc import bulk_aot_store
from test_bulk_aot_closure import compile_controlled, dependency, real_gcc, variant

# Export the existing fixture so pytest can discover the shared GCC producer.
__all__ = ["real_gcc"]

if TYPE_CHECKING:
    from typing import Any


@pytest.fixture
def fake_clang(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    driver = tmp_path / "clang"
    driver.write_bytes(b"\x7fELFfake-clang")
    resource = tmp_path / "resource"
    resource.mkdir()
    source = tmp_path / "point.c"
    source.write_bytes(b"int point(void) { return 1; }\n")
    monkeypatch.setattr(collector.sys, "platform", "linux")
    monkeypatch.setattr(collector.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(collector, "_system_clang", lambda: driver)
    monkeypatch.setattr(collector.shutil, "which", lambda *args, **kwargs: str(driver))

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        assert kwargs["environment"] == collector.CPU_ENVIRONMENT
        assert timeout > 0
        diagnostic = ""
        if "--version" in command:
            text = "clang version 18.1.8"
        elif "-dumpmachine" in command:
            text = "x86_64-linux-gnu"
        elif "-print-resource-dir" in command:
            text = str(resource)
        elif "-###" in command:
            text = ""
            diagnostic = (
                f'"{driver}" "-cc1" "-triple" "x86_64-linux-gnu" '
                f'"-emit-obj" "-resource-dir" "{resource}" "-x" "c" "{source}"'
            )
        elif "-E" in command:
            text = "preprocessed"
        else:
            text = "closure: " + str(source).replace(" ", "\\ ")
        return CompileResult(0, False, 0.001, text, diagnostic)

    monkeypatch.setattr(collector, "run_compiler", run)
    manifest = collector.CpuToolchainManifest((dependency(driver),), complete=True)
    return variant(source.read_text(encoding="utf-8")), source, driver, manifest


def test_explicit_clang_opt_in(fake_clang: tuple) -> None:
    item, source, driver, manifest = fake_clang
    default = collector.collect_cpu_closure(item, source, compiler=str(driver))
    assert default.reasons == ("unsupported-compiler-wrapper",)
    result = collector.collect_cpu_closure(
        item, source, compiler=str(driver), compiler_family="clang", toolchain=manifest
    )
    assert not result.reasons and result.closure is not None and result.closure.reusable
    assert result.recipe is not None
    assert result.recipe["compiler"]["executable"] == str(driver)
    assert result.recipe["flags"][:2] == ["--no-default-config", "-fintegrated-as"]
    assert result.recipe["flags"][-1].startswith("-fdebug-compilation-dir=")
    for incomplete in (None, replace(manifest, complete=False)):
        result = collector.collect_cpu_closure(
            item, source, compiler_family="clang", toolchain=incomplete
        )
        assert result.closure is not None and result.closure.cache_key is None
        assert "toolchain-manifest-incomplete" in result.reasons


@pytest.mark.parametrize(
    "fault", ["resource", "external", "frontend", "target", "config-drift", "timeout"]
)
def test_clang_model_fails_closed(
    fake_clang: tuple, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    item, source, driver, manifest = fake_clang
    original = collector.run_compiler
    plans = 0

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        nonlocal plans
        result = original(command, timeout, **kwargs)
        if fault == "timeout":
            return replace(result, timed_out=True)
        if "-print-resource-dir" in command and fault == "resource":
            return replace(result, stdout="relative/resource")
        if "-###" in command:
            plans += 1
            if fault == "external":
                return replace(
                    result, stderr=result.stderr + '\n "/usr/bin/as" "input.s"'
                )
            if fault == "frontend":
                return replace(result, stderr=result.stderr.replace("-cc1", "-cc1as"))
            if fault == "target":
                return replace(
                    result, stderr=result.stderr.replace("x86_64", "aarch64")
                )
            if fault == "config-drift" and plans == 2:
                return replace(
                    result, stderr=result.stderr + "\nchanged search selection"
                )
        return result

    monkeypatch.setattr(collector, "run_compiler", run)
    result = collector.collect_cpu_closure(
        item, source, compiler=str(driver), compiler_family="clang", toolchain=manifest
    )
    assert result.reasons
    assert result.closure is None or result.closure.cache_key is None


@pytest.fixture
def real_clang(tmp_path: Path) -> tuple[str, collector.CpuToolchainManifest]:
    if sys.platform != "linux":
        pytest.skip("real native Linux Clang qualification required separately")
    compiler = shutil.which("clang", path=collector.CPU_ENVIRONMENT["PATH"])
    assert compiler is not None, "Linux qualification requires /usr/bin/clang"
    compiler = str(Path(compiler).resolve())
    result = run_compiler(
        [compiler, "--no-default-config", "-print-resource-dir"],
        30,
        label="Clang fixture resource",
        environment=collector.CPU_ENVIRONMENT,
    )
    assert result.returncode == 0 and not result.timed_out, result.stderr
    resource = Path(result.stdout.strip())
    files = {Path(compiler), *(p for p in resource.rglob("*") if p.is_file())}
    libraries = run_compiler(
        ["/usr/bin/ldd", compiler],
        30,
        label="Clang fixture libraries",
        environment=collector.CPU_ENVIRONMENT,
    )
    assert libraries.returncode == 0
    files.update(
        Path(token).resolve()
        for token in libraries.stdout.split()
        if token.startswith("/")
    )
    config = tmp_path / "asserted snapshot contract"
    config.write_text("native Clang integrated C fixture/v1\n", encoding="utf-8")
    files.add(config)
    # Test-producer assertion: controlled image, loader inputs, all resource and
    # include namespaces, cwd and opaque inputs remain immutable inside each
    # bracket. This is not an inference from ldd/resource enumeration.
    return compiler, collector.CpuToolchainManifest(
        tuple(dependency(p) for p in sorted(files)),
        complete=True,
    )


@pytest.mark.parametrize("family", ["clang", "gcc"])
def test_real_native_object_dependencies_store_and_math(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
) -> None:
    compiler, manifest = request.getfixturevalue("real_" + family)
    root = tmp_path / "headers with spaces"
    root.mkdir()
    header = root / "value #$ header.h"
    header.write_text("#define OFFSET 0.5\n", encoding="utf-8")
    text = f'#include "{header.name}"\n#include <stddef.h>\ndouble point(double r) {{ return r*r*(r+OFFSET); }}\n'
    source = root / "point.c"
    source.write_text(text, encoding="utf-8")

    def collect(**kwargs: Any) -> collector.CpuClosureResult:
        return collector.collect_cpu_closure(
            variant(source.read_text(encoding="utf-8")),
            source,
            compiler=compiler,
            compiler_family=family,
            toolchain=kwargs.pop("toolchain", manifest),
            **kwargs,
        )

    before = collect()
    assert not before.reasons and before.closure is not None and before.closure.reusable
    assert before.recipe is not None
    obj = root / "point.o"
    depfile = root / "point.d"
    object_command = [
        compiler,
        *before.recipe["flags"],
        "-MD",
        "-MF",
        str(depfile),
        "-MT",
        "closure",
        "-H",
        "-c",
        str(source),
        "-o",
        str(obj),
    ]
    launcher = resolve_compiler_cache(60)
    built = compile_controlled(object_command, monkeypatch)
    assert built.returncode == 0 and not built.timed_out, built.stderr
    observed = {record.identity for record in before.headers}
    # Independently emitted by the object compilation, not the collector's -M.
    assert set(collector.parse_gcc_dependencies(depfile.read_text())) == observed | {
        str(source)
    }
    traced = {
        match.group(1)
        for match in re.finditer(r"^\.+ (.+)$", built.stderr, re.MULTILINE)
    }
    if family == "clang":
        assert traced == observed
    else:
        # GCC implicitly consumes this header without reporting it in -H.
        assert observed - traced == {"/usr/include/stdc-predef.h"}
    assert str(header) in observed
    assert all(
        file_hash(Path(record.identity)) == record.content_sha256
        for record in before.headers
    )
    after = collect()
    assert not after.reasons and after.closure is not None
    assert before.closure.cache_key == after.closure.cache_key
    failed_store = tmp_path / "failed-store"
    with monkeypatch.context() as failure:

        def interrupted(*args: Any, **kwargs: Any) -> None:
            raise OSError("injected publication failure")

        failure.setattr(bulk_aot_store, "_atomic_copy", interrupted)
        with pytest.raises(OSError, match="injected publication failure"):
            bulk_aot_store.store_artifact(failed_store, after.closure, obj)
    assert bulk_aot_store.lookup_artifact(failed_store, after.closure).status == "miss"
    stored = bulk_aot_store.store_artifact(tmp_path / "store", after.closure, obj)
    assert stored.status == "hit" and stored.object_sha256 == file_hash(obj)
    obj.unlink()
    hit = bulk_aot_store.lookup_artifact(tmp_path / "store", collect().closure)
    assert hit == stored and hit.artifact_path is not None
    shared = root / "point.so"
    linked = compile_controlled(
        [
            compiler,
            *(["--no-default-config"] if family == "clang" else []),
            "-shared",
            str(hit.artifact_path),
            "-o",
            str(shared),
        ],
        monkeypatch,
    )
    assert linked.returncode == 0 and not linked.timed_out, linked.stderr
    library = ctypes.CDLL(str(shared))
    library.point.argtypes = [ctypes.c_double]
    library.point.restype = ctypes.c_double
    # Independent decimal evaluation of the retained polynomial density fixture.
    numerical_rows = []
    for value in ("0", "0.125", "1", "2.75"):
        r = Decimal(value)
        reference = r**3 + r**2 / 2
        observed_value = library.point(float(r))
        assert observed_value == pytest.approx(float(reference), rel=1e-14)
        numerical_rows.append(
            {"rho": value, "observed": observed_value, "reference": str(reference)}
        )
    (root / "qualification.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "fixture": "polynomial density / Decimal reference",
                "compiler": compiler,
                "launcher": {"path": str(launcher.path), "version": launcher.version},
                "command": object_command,
                "source_text": source.read_text(),
                "fixture_header_contents": header.read_text(),
                "environment": collector.CPU_ENVIRONMENT,
                "recipe": before.recipe,
                "before": before.closure.to_payload(),
                "after": after.closure.to_payload(),
                "dependencies": depfile.read_text(),
                "header_trace": built.stderr,
                "stored_object": str(hit.artifact_path),
                "object_sha256": file_hash(hit.artifact_path),
                "shared_library_sha256": file_hash(shared),
                "numerical_rows": numerical_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    header.write_text("#define OFFSET 0.75\n", encoding="utf-8")
    changed = collect()
    assert not changed.reasons and changed.closure is not None
    assert changed.closure.cache_key != after.closure.cache_key
    assert (
        bulk_aot_store.lookup_artifact(tmp_path / "store", changed.closure).status
        == "miss"
    )
    corruption = bulk_aot_store.store_artifact(
        tmp_path / "corrupt-store", after.closure, hit.artifact_path
    )
    assert corruption.artifact_path is not None
    corruption.artifact_path.write_bytes(b"corrupt object")
    assert (
        bulk_aot_store.lookup_artifact(tmp_path / "corrupt-store", after.closure).status
        == "corrupt"
    )


def test_real_clang_opaque_inputs_invalidate_compiler_cache(
    tmp_path: Path, real_clang: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler, manifest = real_clang
    monkeypatch.chdir(tmp_path)
    payload = tmp_path / "payload.bin"
    text = '__asm__(".pushsection .rodata\\n.incbin \\"payload.bin\\"\\n.popsection");\nint point(void) { return 0; }\n'
    source = tmp_path / "point.c"
    source.write_text(text, encoding="utf-8")
    keys, objects = [], []
    for content in (b"AAAA", b"BBBB"):
        payload.write_bytes(content)
        snapshot = replace(manifest, files=(*manifest.files, dependency(payload)))
        before = collector.collect_cpu_closure(
            variant(text),
            source,
            compiler=compiler,
            compiler_family="clang",
            toolchain=snapshot,
        )
        assert (
            not before.reasons
            and before.closure is not None
            and before.recipe is not None
        )
        obj = tmp_path / "point.o"
        built = compile_controlled(
            [compiler, *before.recipe["flags"], "-c", str(source), "-o", str(obj)],
            monkeypatch,
        )
        assert built.returncode == 0 and not built.timed_out, built.stderr
        after = collector.collect_cpu_closure(
            variant(text),
            source,
            compiler=compiler,
            compiler_family="clang",
            toolchain=snapshot,
        )
        assert not after.reasons and after.closure is not None
        assert before.closure.cache_key == after.closure.cache_key
        assert content in obj.read_bytes()
        keys.append(after.closure.cache_key)
        objects.append(file_hash(obj))
    assert keys[0] != keys[1] and objects[0] != objects[1]


@pytest.mark.parametrize("name", ["foo\\", "tail ", "tail\t", "a\\ b.h", "a#$\\q.h"])
def test_real_clang_dependency_quoting_and_decoys(
    tmp_path: Path, real_clang: tuple, name: str
) -> None:
    compiler, manifest = real_clang
    header = tmp_path / name
    header.write_text("#define VALUE 1\n", encoding="utf-8")
    other = tmp_path / "z.h"
    other.write_text("/* second include */\n", encoding="utf-8")
    # Decoys cannot justify interpreting ambiguous compiler output as real input.
    (tmp_path / "foo ").write_text("/* decoy */\n", encoding="utf-8")
    if "\\" in name:
        normalized = Path(str(header).replace("\\", "/"))
        normalized.parent.mkdir(parents=True, exist_ok=True)
        normalized.write_text("/* Clang slash-normalized decoy */\n", encoding="utf-8")
    if "\t" in name:
        (tmp_path / name.split("\t", 1)[0]).write_text(
            "/* tab-split decoy */\n", encoding="utf-8"
        )
    merged = Path(str(header) + " " + str(other))
    merged.parent.mkdir(parents=True, exist_ok=True)
    merged.write_text("/* merged decoy */\n", encoding="utf-8")
    source = tmp_path / "point.c"
    text = f'#include "{name}"\n#include "z.h"\nint point(void) {{ return VALUE; }}\n'
    source.write_text(text, encoding="utf-8")
    emitted = run_compiler(
        [compiler, "--no-default-config", "-M", "-MT", "closure", str(source)],
        30,
        label="real Clang dependency quoting",
        environment=collector.CPU_ENVIRONMENT,
    )
    assert not emitted.timed_out
    if name.endswith("\\"):
        # Clang rejects a quoted header name ending in a backslash itself.
        assert emitted.returncode != 0
    else:
        assert emitted.returncode == 0, emitted.stderr
    result = collector.collect_cpu_closure(
        variant(text),
        source,
        compiler=compiler,
        compiler_family="clang",
        toolchain=manifest,
    )
    assert result.reasons and (
        result.closure is None or result.closure.cache_key is None
    )


def test_real_clang_poison_invalidation_and_failure(
    tmp_path: Path, real_clang: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler, manifest = real_clang
    source = tmp_path / "point.c"
    text = '#if __has_include("optional.h")\n#define V 2\n#else\n#define V 1\n#endif\nint point(void) { return V; }\n'
    source.write_text(text, encoding="utf-8")

    def collect(**kwargs: Any) -> collector.CpuClosureResult:
        return collector.collect_cpu_closure(
            variant(source.read_text(encoding="utf-8")),
            source,
            compiler=compiler,
            compiler_family="clang",
            toolchain=manifest,
            **kwargs,
        )

    before = collect()
    assert not before.reasons and before.closure is not None
    for key in (
        "CPATH",
        "C_INCLUDE_PATH",
        "LD_PRELOAD",
        "CCC_OVERRIDE_OPTIONS",
        "CLANG_CONFIG_FILE_USER_DIR",
    ):
        monkeypatch.setenv(key, "/invalid/poison")
    poisoned = collect()
    assert not poisoned.reasons and poisoned.closure is not None
    assert poisoned.closure.cache_key == before.closure.cache_key
    (tmp_path / "optional.h").write_text("/* never included */\n", encoding="utf-8")
    changed = collect()
    assert not changed.reasons and changed.closure is not None
    # Clang records successful __has_include probes in its -M dependencies.
    assert {header.identity for header in changed.headers} == {
        str(tmp_path / "optional.h")
    }
    assert changed.closure.cache_key != before.closure.cache_key
    reordered = collect(flags=("-O2", "-std=c99", "-fPIC"))
    assert (
        reordered.closure is not None
        and reordered.closure.cache_key != changed.closure.cache_key
    )
    source.write_text(text + "/* changed source */\n", encoding="utf-8")
    revised = collect()
    assert (
        revised.closure is not None
        and revised.closure.cache_key != changed.closure.cache_key
    )
    timed = collect(timeout=0.000001)
    assert timed.reasons == ("collection-timed-out",)
    source.write_text('#include "missing.h"\n', encoding="utf-8")
    missing = collect()
    assert missing.reasons == ("compiler-command-failed",)
    assert missing.closure is not None and missing.closure.cache_key is None
    # Restore poisoned inherited values before unrelated Python/plugin teardown.
    for key in ("LD_PRELOAD", "CCC_OVERRIDE_OPTIONS"):
        monkeypatch.delenv(key)


def test_real_clang_entry_selection_drift(
    tmp_path: Path, real_clang: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler, manifest = real_clang
    entry = tmp_path / "clang-entry"
    entry.symlink_to(compiler)
    replacement = tmp_path / "changed-driver"
    replacement.write_bytes(b"\x7fELFreplacement-not-executed")
    source = tmp_path / "point.c"
    text = "int point(void) { return 1; }\n"
    source.write_text(text, encoding="utf-8")
    original = collector.run_compiler
    changed = False

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        nonlocal changed
        result = original(command, timeout, **kwargs)
        if "-E" in command and not changed:
            entry.unlink()
            entry.symlink_to(replacement)
            changed = True
        return result

    monkeypatch.setattr(collector, "run_compiler", run)
    result = collector.collect_cpu_closure(
        variant(text),
        source,
        compiler=str(entry),
        compiler_family="clang",
        toolchain=manifest,
    )
    assert changed and result.reasons == ("compiler-selection-changed",)
    assert result.closure is not None and result.closure.cache_key is None
