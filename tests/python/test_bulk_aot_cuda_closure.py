"""Bounded NVCC closure controls and opt-in actual Linux object qualification."""

from __future__ import annotations

import json
import os
import shlex
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.common import compiler_cache
from generativeqc_compiler.common.compiler_cache import (
    resolve_compiler_cache,
    run_cached_compiler,
)
from generativeqc_compiler.common.compiler_process import CompileResult
from generativeqc_compiler.common.provenance import file_hash
from generativeqc_compiler.xc import bulk_aot_closure as collector
from generativeqc_compiler.xc import bulk_aot_store
from generativeqc_compiler.xc.bulk_aot_cache import CacheDependency
from test_bulk_aot_closure import dependency, variant

if TYPE_CHECKING:
    from typing import Any


@pytest.fixture
def fake_nvcc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    root = tmp_path / "cuda"
    driver = root / "bin/nvcc"
    host = tmp_path / "g++"
    tools = [
        driver,
        host,
        root / "bin/nvcc.profile",
        root / "bin/cudafe++",
        root / "nvvm/bin/cicc",
        root / "bin/ptxas",
        root / "bin/fatbinary",
        tmp_path / "cc1plus",
        tmp_path / "as",
        root / "nvvm/libdevice/libdevice.10.bc",
    ]
    for path in tools:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x7fELFcontrolled-input\n")
    source = tmp_path / "point.cu"
    source.write_bytes(b"__global__ void point(double *p) { *p = 1.0; }\n")
    header = tmp_path / "value.h"
    header.write_bytes(b"#define VALUE 1\n")
    monkeypatch.setattr(collector.sys, "platform", "linux")
    monkeypatch.setattr(collector.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(collector, "_system_gxx", lambda: host)
    # Path grammar itself is tested below; model controls run on Windows too.
    monkeypatch.setattr(collector, "_cuda_path", lambda path: None)
    monkeypatch.setattr(
        collector.os, "O_NONBLOCK", getattr(os, "O_NONBLOCK", 0), raising=False
    )
    monkeypatch.setattr(
        collector.os, "O_NOFOLLOW", getattr(os, "O_NOFOLLOW", 0), raising=False
    )
    monkeypatch.setattr(collector.shutil, "which", lambda name, **kwargs: name)

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        assert kwargs["environment"] == collector.CPU_ENVIRONMENT
        assert timeout > 0
        text, diagnostic = "", ""
        if "--version" in command:
            text = (
                "release 12.9, V12.9.86"
                if command[0] == str(driver)
                else "Free Software Foundation G++"
            )
        elif "--help" in command:
            text = "--frandom-seed --dryrun --generate-dependencies"
        elif "-dumpmachine" in command:
            text = "x86_64-linux-gnu"
        elif "-dumpspecs" in command:
            text = "native specs"
        elif any(arg.startswith("-print-prog-name=") for arg in command):
            text = str(tmp_path / command[-1].split("=", 1)[1])
        elif "--dryrun" in command:
            top = str(driver.parent / "..")
            diagnostic = (
                "\n".join(
                    "#$ " + key + "=" + value
                    for key, value in {
                        "_NVVM_BRANCH_": "nvvm",
                        "_SPACE_": " ",
                        "_CUDART_": "cudart",
                        "_HERE_": str(driver.parent),
                        "_THERE_": str(driver.parent),
                        "_TARGET_SIZE_": "",
                        "_TARGET_DIR_": "targets/x86_64-linux",
                        "TOP": top,
                        "CICC_PATH": top + "/nvvm/bin",
                        "NVVMIR_LIBRARY_DIR": top + "/nvvm/libdevice",
                        "LD_LIBRARY_PATH": top + "/lib:",
                        "PATH": top
                        + "/nvvm/bin:"
                        + str(driver.parent)
                        + ":/usr/bin:/bin",
                        "INCLUDES": '"-I' + top + '/targets/x86_64-linux/include"',
                        "LIBRARIES": '"-L'
                        + top
                        + '/targets/x86_64-linux/lib/stubs" "-L'
                        + top
                        + '/targets/x86_64-linux/lib"',
                        "CUDAFE_FLAGS": "",
                        "PTXAS_FLAGS": "",
                    }.items()
                )
                + "\n"
                + "\n".join(
                    "#$ " + shlex.quote(str(p)) + " -c input"
                    for p in [host, tools[3], host, *tools[4:7], host]
                )
            )
        elif "-E" in command:
            text = source.read_text(encoding="utf-8") + header.read_text(
                encoding="utf-8"
            )
        elif "-M" in command:
            text = "closure : " + " ".join(
                str(p).replace(" ", "\\ ") for p in (source, header)
            )
            if "-Xcompiler=-H" in command:
                diagnostic = ". " + str(header)
        return CompileResult(0, False, 0.001, text, diagnostic)

    monkeypatch.setattr(collector, "run_compiler", run)
    item = replace(variant(source.read_text(encoding="utf-8")), backend="cuda")
    manifest = collector.CudaToolchainManifest(
        tuple(dependency(p) for p in tools), True
    )
    return item, source, driver, host, header, manifest


def collect(model: tuple, **kwargs: Any) -> collector.CudaClosureResult:
    """Call the explicit CUDA entrance under the controlled fixture."""
    item, source, driver, host, _, manifest = model
    return collector.collect_cuda_closure(
        item,
        source,
        compiler=str(driver),
        host_compiler=str(host),
        target="sm_80",
        toolchain=kwargs.pop("toolchain", manifest),
        **kwargs,
    )


def test_complete_and_incomplete(fake_nvcc: tuple) -> None:
    result = collect(fake_nvcc)
    assert not result.reasons
    assert result.closure is not None and result.closure.reusable
    assert result.recipe is not None
    assert result.recipe["flags"][-1].startswith("--frandom-seed=")
    assert collect(fake_nvcc).closure == result.closure
    for manifest in (
        collector.CudaToolchainManifest(),
        replace(fake_nvcc[-1], complete=False),
    ):
        result = collect(fake_nvcc, toolchain=manifest)
        assert result.closure is not None and result.closure.cache_key is None
        assert "toolchain-manifest-incomplete" in result.reasons
    result = collect(fake_nvcc, toolchain=collector.CudaToolchainManifest((), True))
    assert result.closure is not None and not result.closure.reusable
    assert "downstream-toolchain-not-covered" in result.reasons


@pytest.mark.parametrize(
    "flag",
    [
        "@response",
        "-dc",
        "-dlto",
        "--use_fast_math",
        "-Xcompiler=-fplugin=bad",
        "-DVALUE=2",
        "--options-file=bad",
    ],
)
def test_unsupported_flags(fake_nvcc: tuple, flag: str) -> None:
    result = collect(fake_nvcc, flags=(flag,))
    assert result.closure is None and result.reasons == ("unsupported-cuda-flags",)


@pytest.mark.parametrize(
    "fault",
    [
        "timeout",
        "command",
        "version",
        "help",
        "job",
        "unknown-assignment",
        "shell",
        "missing-header",
        "depfile",
        "late-header",
        "late-source",
        "late-host",
        "late-device",
        "late-profile",
        "late-preprocess",
        "time-macro",
    ],
)
def test_fail_closed(
    fake_nvcc: tuple, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    _, source, driver, host, header, _manifest = fake_nvcc
    original = collector.run_compiler
    scans = 0
    if fault == "time-macro":
        header.write_bytes(b"__TIME__\n")

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        nonlocal scans
        result = original(command, timeout, **kwargs)
        if fault == "timeout":
            return replace(result, timed_out=True)
        if fault == "command":
            return replace(result, returncode=1)
        if command[0] == str(driver) and "--version" in command and fault == "version":
            return replace(result, stdout="release 13.0, V13.0.1")
        if "--help" in command and fault == "help":
            return replace(result, stdout="unsupported options")
        if "--dryrun" in command:
            if fault == "job":
                return replace(
                    result, stderr=result.stderr + "\n#$ /usr/bin/nvlink -o output"
                )
            if fault == "unknown-assignment":
                return replace(result, stderr=result.stderr + "\n#$ EVIL=1")
            if fault == "shell":
                return replace(result, stderr=result.stderr + "\n#$ PATH=$(touch evil)")
        if "-M" in command:
            scans += 1
            if fault == "missing-header":
                header.unlink(missing_ok=True)
            if fault == "depfile":
                return replace(result, stdout="other: " + str(source))
            if scans == 2:
                path = {
                    "late-header": header,
                    "late-source": source,
                    "late-host": host,
                    "late-device": driver.parent / "ptxas",
                    "late-profile": driver.parent / "nvcc.profile",
                }.get(fault)
                if path is not None:
                    path.write_bytes(b"mutated\n")
        if "-E" in command and scans and fault == "late-preprocess":
            return replace(result, stdout="changed preprocessor output")
        return result

    monkeypatch.setattr(collector, "run_compiler", run)
    result = collect(fake_nvcc)
    assert result.reasons
    assert result.closure is None or result.closure.cache_key is None


@pytest.mark.parametrize(
    "option", ["--sysroot=/opt/sdk", "--sysroot /opt/sdk", "-isysroot /opt/sdk"]
)
def test_diagnostic_alternate_sysroot(
    fake_nvcc: tuple, monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    original = collector.run_compiler

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        result = original(command, timeout, **kwargs)
        if "--dryrun" in command:
            return replace(
                result,
                stderr=result.stderr.replace(
                    " -c input", " " + option + " -c input", 1
                ),
            )
        return result

    monkeypatch.setattr(collector, "run_compiler", run)
    result = collect(fake_nvcc)
    assert result.reasons == ("unsupported-nvcc-job-model",)
    assert result.closure is None or not result.closure.reusable


@pytest.mark.parametrize(
    "changed",
    [
        "header",
        "profile",
        "device",
        "flags",
        "source",
        "nvcc",
        "host",
        "frontend",
        "assembler",
        "libdevice",
    ],
)
def test_identity_changes(fake_nvcc: tuple, changed: str) -> None:
    before = collect(fake_nvcc)
    assert before.closure is not None and before.closure.reusable
    item, source, driver, host, header, manifest = fake_nvcc
    kwargs = {}
    if changed == "flags":
        kwargs["flags"] = ("-O2",)
    else:
        path = {
            "header": header,
            "profile": driver.parent / "nvcc.profile",
            "device": driver.parent / "ptxas",
            "source": source,
            "nvcc": driver,
            "host": host,
            "frontend": host.parent / "cc1plus",
            "assembler": host.parent / "as",
            "libdevice": driver.parent.parent / "nvvm/libdevice/libdevice.10.bc",
        }[changed]
        path.write_bytes(path.read_bytes() + b"// change\n")
        if changed == "source":
            item = replace(item, source=source.read_text(encoding="utf-8"))
        manifest = replace(
            manifest, files=tuple(dependency(Path(p.identity)) for p in manifest.files)
        )
    after = collect((item, source, driver, host, header, manifest), **kwargs)
    assert after.closure is not None and after.closure.reusable
    assert before.closure.cache_key != after.closure.cache_key


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/space name",
        "/tmp/dollar$",
        "/tmp/hash#",
        "/tmp/quote'",
        "/tmp/tab\t",
        "/tmp/back\\slash",
        "/tmp/a,b",
        "relative",
    ],
)
def test_ambiguous_paths(path: str) -> None:
    with pytest.raises(ValueError, match="unsupported-cuda-path-spelling"):
        collector._cuda_path(Path(path))


@pytest.fixture
def real_nvcc() -> tuple[str, collector.CudaToolchainManifest]:
    """Require an explicitly asserted immutable producer snapshot for real tests."""
    compiler = os.environ.get("GENERATIVEQC_NVCC_CLOSURE_COMPILER")
    manifest_path = os.environ.get("GENERATIVEQC_NVCC_CLOSURE_MANIFEST")
    if not compiler and not manifest_path:
        pytest.skip("explicit Linux NVCC 12.9 producer snapshot qualification")
    assert sys.platform == "linux" and compiler and manifest_path
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest = collector.CudaToolchainManifest(
        tuple(CacheDependency(**record) for record in payload["files"]),
        complete=payload["complete"],
    )
    assert manifest.complete, "real qualification cannot use partial/guessed coverage"
    return str(Path(compiler).resolve()), manifest


def test_real_nvcc_object_store_and_opaque_input(
    tmp_path: Path,
    real_nvcc: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler, manifest = real_nvcc
    launcher = resolve_compiler_cache()
    assert launcher.name == "sccache", (
        "this integration receipt qualifies sccache 0.16+"
    )
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "point.cu"
    header = tmp_path / "value.h"
    opaque = tmp_path / "payload.bin"
    text = '#include "value.h"\n__asm__(".pushsection .rodata\\n.incbin \\"payload.bin\\"\\n.popsection");\n__global__ void point(double *p) { *p = VALUE; }\n'
    source.write_text(text, encoding="utf-8")
    header.write_text("#define VALUE 3\n", encoding="utf-8")
    item = replace(variant(text), backend="cuda")
    stats = []
    original = compiler_cache.run_compiler

    def observed_run(
        command: list[str], timeout: float, **kwargs: Any
    ) -> CompileResult:
        if command[:2] != [str(launcher.path), compiler]:
            return original(command, timeout, **kwargs)
        counters = [str(launcher.path), "--show-stats", "--stats-format=json"]
        before = original(counters, 20, **kwargs)
        result = original(command, timeout, **kwargs)
        after = original(counters, 20, **kwargs)
        assert not before.returncode and not after.returncode
        stats.append(
            {"before": json.loads(before.stdout), "after": json.loads(after.stdout)}
        )
        return result

    monkeypatch.setattr(compiler_cache, "run_compiler", observed_run)
    old_environment = dict(os.environ)
    monkeypatch.setattr(os, "environ", dict(collector.CPU_ENVIRONMENT))
    obj = tmp_path / "point.o"
    depfile = tmp_path / "point.d"
    rows = []
    for content in (b"OPAQUE_AAAA", b"OPAQUE_BBBB"):
        opaque.write_bytes(content)
        snapshot = replace(manifest, files=(*manifest.files, dependency(opaque)))
        before = collector.collect_cuda_closure(
            item,
            source,
            compiler=compiler,
            target="sm_80",
            toolchain=snapshot,
            timeout=120,
        )
        assert (
            not before.reasons
            and before.closure
            and before.closure.reusable
            and before.recipe
        )
        command = [
            compiler,
            *before.recipe["flags"],
            "-MD",
            "-MF",
            str(depfile),
            "-MT",
            "closure",
            "-c",
            str(source),
            "-o",
            str(obj),
        ]
        result = run_cached_compiler(
            command,
            120,
            label="real NVCC closure object",
            launcher=launcher,
            preserve_source_paths=True,
        )
        assert not result.timed_out and not result.returncode, result.stderr
        assert obj.read_bytes().startswith(b"\x7fELF") and content in obj.read_bytes()
        # Independent raw MD oracle for this deliberately safe-spelled fixture;
        # do not call the collector's NVCC/GCC dependency parsers here.
        target, separator, body = (
            depfile.read_text(encoding="utf-8").replace("\\\n", "").partition(":")
        )
        assert separator and target.strip() == "closure"
        assert set(body.split()) == {str(source), *(h.identity for h in before.headers)}
        after = collector.collect_cuda_closure(
            item,
            source,
            compiler=compiler,
            target="sm_80",
            toolchain=snapshot,
            timeout=120,
        )
        assert (
            not after.reasons
            and after.closure
            and before.closure.cache_key == after.closure.cache_key
        )
        store = tmp_path / "store"
        stored = bulk_aot_store.store_artifact(store, after.closure, obj)
        assert stored.artifact_path
        object_hash = file_hash(obj)
        obj.unlink()
        hit = bulk_aot_store.lookup_artifact(store, before.closure)
        assert hit.status == "hit" and hit.object_sha256 == object_hash
        replay = run_cached_compiler(
            command,
            120,
            label="real NVCC closure replay",
            launcher=launcher,
            preserve_source_paths=True,
        )
        assert not replay.returncode and file_hash(obj) == object_hash, replay.stderr
        rows.append(
            {
                "closure": after.closure.to_payload(),
                "key": after.closure.cache_key,
                "object_sha256": object_hash,
                "object_bytes": obj.stat().st_size,
                "dependency_observation": depfile.read_text(encoding="utf-8"),
                "command": command,
                "compiler_stderr": result.stderr,
            }
        )
    assert (
        rows[0]["key"] != rows[1]["key"]
        and rows[0]["object_sha256"] != rows[1]["object_sha256"]
    )
    receipt = {
        "launcher": {"path": str(launcher.path), "version": launcher.version},
        "source_sha256": file_hash(source),
        "source": text,
        "rows": rows,
        "stats": stats,
    }
    (tmp_path / "nvcc-object-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(os, "environ", old_environment)


def test_real_nvcc_cacheable_replay_and_salt(
    tmp_path: Path,
    real_nvcc: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler, manifest = real_nvcc
    launcher = resolve_compiler_cache()
    assert launcher.name == "sccache", (
        "this integration receipt qualifies sccache 0.16+"
    )
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "cacheable.cu"
    text = "__global__ void point(double *p) { *p = 4.0; }\n"
    source.write_text(text, encoding="utf-8")
    item = replace(variant(text), backend="cuda")
    config = tmp_path / "opaque-config.txt"
    stats = []
    original = compiler_cache.run_compiler

    def observed_run(
        command: list[str], timeout: float, **kwargs: Any
    ) -> CompileResult:
        if command[:2] != [str(launcher.path), compiler]:
            return original(command, timeout, **kwargs)
        counters = [str(launcher.path), "--show-stats", "--stats-format=json"]
        pre = original(counters, 20, **kwargs)
        result = original(command, timeout, **kwargs)
        post = original(counters, 20, **kwargs)
        assert not pre.returncode and not post.returncode
        stats.append(
            {"before": json.loads(pre.stdout), "after": json.loads(post.stdout)}
        )
        return result

    monkeypatch.setattr(compiler_cache, "run_compiler", observed_run)
    monkeypatch.setattr(os, "environ", dict(collector.CPU_ENVIRONMENT))
    rows = []
    for content in ("snapshot A\n", "snapshot A\n", "snapshot B\n"):
        config.write_text(content, encoding="utf-8")
        snapshot = replace(manifest, files=(*manifest.files, dependency(config)))
        before = collector.collect_cuda_closure(
            item,
            source,
            compiler=compiler,
            target="sm_80",
            toolchain=snapshot,
            timeout=120,
        )
        assert (
            not before.reasons
            and before.closure
            and before.closure.reusable
            and before.recipe
        )
        obj = tmp_path / "cacheable.o"
        result = run_cached_compiler(
            [compiler, *before.recipe["flags"], "-c", str(source), "-o", str(obj)],
            120,
            label="cacheable NVCC replay/salt",
            launcher=launcher,
        )
        assert not result.returncode and not result.timed_out, result.stderr
        after = collector.collect_cuda_closure(
            item,
            source,
            compiler=compiler,
            target="sm_80",
            toolchain=snapshot,
            timeout=120,
        )
        assert (
            not after.reasons
            and after.closure
            and before.closure.cache_key == after.closure.cache_key
        )
        rows.append(
            {
                "key": before.closure.cache_key,
                "object_sha256": file_hash(obj),
                "closure": before.closure.to_payload(),
                "recipe": before.recipe,
                "config_contents": content,
            }
        )
    assert rows[0] == rows[1] and rows[0]["key"] != rows[2]["key"]
    # Qualify actual NVCC cache hits/misses, retaining every observed stage's
    # counters. sccache may safely decline an internal host job even on this
    # ordinary source; do not promote a driver hit into a whole-object hit.
    replay = stats[1]["after"]["stats"]
    changed = stats[2]["after"]["stats"]
    assert replay["cache_hits"]["counts"].get("CUDA", 0) > 0
    assert changed["cache_misses"]["counts"].get("CUDA", 0) > 0
    (tmp_path / "nvcc-cacheable-receipt.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "stats": stats,
                "source_sha256": file_hash(source),
                "source": text,
                "launcher": {"path": str(launcher.path), "version": launcher.version},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_real_nvcc_fifo_timeout(tmp_path: Path, real_nvcc: tuple) -> None:
    compiler, manifest = real_nvcc
    fifo = tmp_path / "blocked.h"
    os.mkfifo(fifo)
    source = tmp_path / "blocked.cu"
    text = '#include "blocked.h"\n__global__ void point(double *p) { *p = 0; }\n'
    source.write_text(text, encoding="utf-8")
    result = collector.collect_cuda_closure(
        replace(variant(text), backend="cuda"),
        source,
        compiler=compiler,
        target="sm_80",
        toolchain=manifest,
        timeout=2,
    )
    assert result.reasons == ("collection-timed-out",)
    assert result.closure is None or result.closure.cache_key is None


@pytest.mark.parametrize("name", ["tab\tname.h", "back\\name.h", "space name.h"])
def test_real_nvcc_ambiguous_header_decoys(
    tmp_path: Path, real_nvcc: tuple, name: str
) -> None:
    compiler, manifest = real_nvcc
    header = tmp_path / name
    header.write_text("#define VALUE 5\n", encoding="utf-8")
    for decoy in ("tab", "name.h", "back", "space"):
        (tmp_path / decoy).write_text("/* decoy */\n", encoding="utf-8")
    source = tmp_path / "ambiguous.cu"
    text = f'#include "{name}"\n__global__ void point(double *p) {{ *p = VALUE; }}\n'
    source.write_text(text, encoding="utf-8")
    result = collector.collect_cuda_closure(
        replace(variant(text), backend="cuda"),
        source,
        compiler=compiler,
        target="sm_80",
        toolchain=manifest,
    )
    assert result.reasons and (result.closure is None or not result.closure.reusable)


def test_real_nvcc_header_selection_drift(
    tmp_path: Path, real_nvcc: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler, manifest = real_nvcc
    first, second = tmp_path / "first.h", tmp_path / "second.h"
    first.write_text("#define VALUE 3\n", encoding="utf-8")
    second.write_text("#define VALUE 4\n", encoding="utf-8")
    link = tmp_path / "selected.h"
    link.symlink_to(first)
    source = tmp_path / "selection.cu"
    text = '#include "selected.h"\n__global__ void point(double *p) { *p = VALUE; }\n'
    source.write_text(text, encoding="utf-8")
    snapshot = replace(manifest, files=(*manifest.files, dependency(link)))
    original = collector.run_compiler
    changed = False

    def run(command: list[str], timeout: float, **kwargs: Any) -> CompileResult:
        nonlocal changed
        result = original(command, timeout, **kwargs)
        if "-M" in command and not changed:
            link.unlink()
            link.symlink_to(second)
            changed = True
        return result

    monkeypatch.setattr(collector, "run_compiler", run)
    result = collector.collect_cuda_closure(
        replace(variant(text), backend="cuda"),
        source,
        compiler=compiler,
        target="sm_80",
        toolchain=snapshot,
    )
    assert changed and result.reasons
    assert result.closure is None or result.closure.cache_key is None
