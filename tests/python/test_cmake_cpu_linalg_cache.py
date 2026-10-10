"""Integration tests for the OpenBLAS capability-probe cache boundary."""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _make_openblas_provider(path: Path, *, version: str, local_threads: bool) -> None:
    """Provide linkable header-only OpenBLAS stubs without system dependencies."""
    includes = path / "include"
    includes.mkdir(parents=True, exist_ok=True)
    (path / "OpenBLASConfig.cmake").write_text(
        "set(OpenBLAS_FOUND TRUE)\n"
        f'set(OpenBLAS_VERSION "{version}")\n'
        "if(NOT TARGET OpenBLAS::OpenBLAS)\n"
        "  add_library(OpenBLAS::OpenBLAS INTERFACE IMPORTED)\n"
        "endif()\n"
        "set(OpenBLAS_LIBRARIES OpenBLAS::OpenBLAS)\n"
        'set(OpenBLAS_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/include")\n'
    )
    (includes / "cblas.h").write_text(
        "#pragma once\n"
        + (
            "#ifndef OPENBLAS_DISABLE_LOCAL_THREADS\n"
            "inline int openblas_set_num_threads_local(int) { return 0; }\n"
            "#endif\n"
            if local_threads
            else ""
        )
        + "inline int openblas_get_num_threads() { return 1; }\n"
        + "inline void openblas_set_num_threads(int) {}\n"
    )
    (includes / "lapacke.h").write_text(
        "#pragma once\n"
        "#define LAPACK_ROW_MAJOR 101\n"
        "inline int LAPACKE_dpotrf(int, char, int, double*, int) { return 0; }\n"
        "inline int LAPACKE_dsyevd(int, char, char, int, double*, "
        "int, double*) { return 0; }\n"
    )


def _capability(cache: Path, name: str) -> bool:
    result = re.search(rf"^{name}:INTERNAL=(.*)$", cache.read_text(), re.MULTILINE)
    assert result is not None, name
    return result.group(1) == "1"


def _probe_counts(output: str) -> tuple[int, int]:
    """Count real try_compile invocations, not just our status messages."""
    names = re.findall(r"^try_compile-result=(.+)$", output, re.MULTILINE)
    return (
        sum(name.startswith("GENERATIVEQC_OPENBLAS_HAS_") for name in names),
        names.count("_batch_result"),
    )


def _probe_project(
    tmp_path: Path, *, prelude: str = "", env: dict[str, str] | None = None
) -> tuple[Path, Path, Callable[..., str]]:
    src = tmp_path / "src"
    src.mkdir()
    (src / "probe.cpp").write_text("int probe() { return 0; }\n")
    (src / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(OpenBLASCacheProbe LANGUAGES CXX)\n"
        "set(CMAKE_DISABLE_FIND_PACKAGE_PkgConfig TRUE)\n"
        + prelude
        + f'include("{ROOT / "cmake/GenerativeQCCpuLinalg.cmake"}")\n'
        "add_library(probe STATIC probe.cpp)\n"
        "generativeqc_configure_cpu_linalg(probe)\n"
    )
    provider = tmp_path / "openblas"
    _make_openblas_provider(provider, version="1.0", local_threads=True)
    build = tmp_path / "build"

    def configure(*args: str, provider_path: Path = provider) -> str:
        result = subprocess.run(
            [
                "cmake",
                "--debug-trycompile",
                "--trace-expand",
                "--trace-format=json-v1",
                f"--trace-redirect={tmp_path / 'trace.jsonl'}",
                "-S",
                str(src),
                "-B",
                str(build),
                "-DGENERATIVEQC_CPU_LINALG_PROVIDER=openblas",
                f"-DOpenBLAS_DIR={provider_path}",
                *args,
            ],
            text=True,
            capture_output=True,
            check=True,
            env=env,
        )
        trace = [
            json.loads(line)
            for line in (tmp_path / "trace.jsonl").read_text().splitlines()
        ]
        invocations = "\n".join(
            "try_compile-result=" + entry["args"][0]
            for entry in trace
            if entry.get("cmd") == "try_compile"
        )
        return result.stdout + result.stderr + "\n" + invocations

    return provider, build / "CMakeCache.txt", configure


def test_openblas_capabilities_reuse_and_invalidate(tmp_path: Path) -> None:
    """Successes need one project; failures and changed capabilities re-probe."""
    _provider, cache, configure = _probe_project(tmp_path)
    assert _probe_counts(configure()) == (3, 0)
    assert _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
    assert _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LAPACKE")
    assert _probe_counts(configure()) == (0, 1)

    second = tmp_path / "openblas-2"
    _make_openblas_provider(second, version="1.0", local_threads=False)
    # A failed batch must discard every previous positive and check separately.
    assert _probe_counts(configure(provider_path=second)) == (3, 1)
    assert not _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
    assert _probe_counts(configure(provider_path=second)) == (1, 1)

    _make_openblas_provider(second, version="1.0", local_threads=True)
    assert _probe_counts(configure(provider_path=second)) == (1, 1)
    assert _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
    _make_openblas_provider(second, version="2.0", local_threads=True)
    assert _probe_counts(configure(provider_path=second)) == (0, 1)

    disabled = "-DCMAKE_CXX_FLAGS=-DOPENBLAS_DISABLE_LOCAL_THREADS"
    assert _probe_counts(configure(disabled, provider_path=second)) == (3, 1)
    assert not _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
    assert _probe_counts(configure(disabled, provider_path=second)) == (1, 1)
    assert _probe_counts(configure("-DCMAKE_CXX_FLAGS=", provider_path=second)) == (
        1,
        1,
    )
    assert _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")


@pytest.mark.parametrize("dependency", ["target", "header"])
@pytest.mark.parametrize("initially_available", [False, True])
def test_openblas_metadata_stable_transitive_change(
    tmp_path: Path, dependency: str, initially_available: bool
) -> None:
    """Transitive edits must invalidate positive and negative capability results."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "probe.cpp").write_text("int probe() { return 0; }\n")
    (src / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(OpenBLASTransitiveProbe LANGUAGES CXX)\n"
        "set(CMAKE_DISABLE_FIND_PACKAGE_PkgConfig TRUE)\n"
        f'include("{ROOT / "cmake/GenerativeQCCpuLinalg.cmake"}")\n'
        "add_library(probe STATIC probe.cpp)\n"
        "generativeqc_configure_cpu_linalg(probe)\n"
    )
    provider = tmp_path / "openblas"
    _make_openblas_provider(
        provider, version="1.0", local_threads=dependency == "target"
    )
    config = provider / "OpenBLASConfig.cmake"
    base_config = config.read_text()
    if dependency == "header":
        header = provider / "include/cblas.h"
        header.write_text(header.read_text() + '#include "local_threads.h"\n')

    def update_dependency(available: bool) -> None:
        if dependency == "target":
            # The direct target's complete set of hashed properties stays fixed.
            config.write_text(
                base_config
                + "add_library(OpenBLAS::Options INTERFACE IMPORTED)\n"
                + "set_property(TARGET OpenBLAS::OpenBLAS PROPERTY "
                "INTERFACE_LINK_LIBRARIES OpenBLAS::Options)\n"
                + (
                    "set_property(TARGET OpenBLAS::Options PROPERTY "
                    "INTERFACE_COMPILE_DEFINITIONS OPENBLAS_DISABLE_LOCAL_THREADS)\n"
                    if not available
                    else ""
                )
            )
        else:
            # Neither top-level probed header changes when this file is replaced.
            (provider / "include/local_threads.h").write_text(
                "inline int openblas_set_num_threads_local(int) { return 0; }\n"
                if available
                else "// No local-thread API in this provider.\n"
            )

    build = tmp_path / "build"
    cache = build / "CMakeCache.txt"

    def configure() -> str:
        result = subprocess.run(
            [
                "cmake",
                "-S",
                str(src),
                "-B",
                str(build),
                "-DGENERATIVEQC_CPU_LINALG_PROVIDER=openblas",
                f"-DOpenBLAS_DIR={provider}",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout + result.stderr

    update_dependency(initially_available)
    configure()
    assert (
        _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
        is initially_available
    )
    update_dependency(not initially_available)
    output = configure()
    assert (
        _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
        is not initially_available
    ), output


def _archive(path: Path, code: str) -> None:
    source = path.with_suffix(".cpp")
    obj = path.with_suffix(".o")
    source.write_text(code)
    subprocess.run(
        [*shlex.split(os.environ.get("CXX", "c++")), "-c", str(source), "-o", str(obj)],
        check=True,
        capture_output=True,
    )
    subprocess.run(["ar", "rcs", str(path), str(obj)], check=True, capture_output=True)


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("ar") is None,
    reason="requires GNU-compatible Linux compiler/linker tools",
)
@pytest.mark.parametrize("change", ["content", "release_location", "archive_order"])
@pytest.mark.parametrize("initially_available", [False, True])
def test_openblas_archive_changes(
    tmp_path: Path, change: str, initially_available: bool
) -> None:
    """Independent links detect archive changes even with stable paths/mtimes."""
    provider, cache, configure = _probe_project(tmp_path)
    config = provider / "OpenBLASConfig.cmake"
    base_config = config.read_text()
    header = provider / "include/cblas.h"
    header.write_text(
        "int openblas_set_num_threads_local(int);\n"
        "inline int openblas_get_num_threads() { return 1; }\n"
        "inline void openblas_set_num_threads(int) {}\n"
    )
    lib = provider / "openblas.a"
    if change == "archive_order":
        # An aggregate executable incorrectly rescues the local probe by pulling
        # the earlier archive for the global probe. Each check must link alone.
        header.write_text(
            "int openblas_set_num_threads_local(int);\n"
            "int openblas_get_num_threads();\nvoid openblas_set_num_threads(int);\n"
        )
        earlier = provider / "earlier.a"
        _archive(
            earlier,
            "int helper() { return 1; }\nint openblas_get_num_threads() { return 1; }\n"
            "void openblas_set_num_threads(int) {}\n",
        )
        config.write_text(
            base_config
            + f'set_property(TARGET OpenBLAS::OpenBLAS PROPERTY INTERFACE_LINK_LIBRARIES "{earlier};{lib}")\n'
        )
    elif change == "release_location":
        base_config = base_config.replace("INTERFACE IMPORTED", "STATIC IMPORTED")
    else:
        config.write_text(
            base_config
            + f'set_property(TARGET OpenBLAS::OpenBLAS PROPERTY INTERFACE_LINK_LIBRARIES "{lib}")\n'
        )

    def update(available: bool) -> None:
        target = (
            provider / f"openblas-{available}.a"
            if change == "release_location"
            else lib
        )
        old_stat = target.stat() if target.exists() else None
        code = "int openblas_set_num_threads_local(int) { return 0; }\n"
        if not available:
            code = (
                "int helper();\nint openblas_set_num_threads_local(int) { return helper(); }\n"
                if change == "archive_order"
                else "int unrelated() { return 0; }\n"
            )
        _archive(target, code)
        if old_stat:
            os.utime(target, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        if change == "release_location":
            config.write_text(
                base_config
                + "set_property(TARGET OpenBLAS::OpenBLAS PROPERTY IMPORTED_CONFIGURATIONS RELEASE)\n"
                f'set_property(TARGET OpenBLAS::OpenBLAS PROPERTY IMPORTED_LOCATION_RELEASE "{target}")\n'
            )

    flags = (
        "-DCMAKE_TRY_COMPILE_CONFIGURATION=Release",
        "-DCMAKE_CXX_FLAGS=-O3 -ffunction-sections -fdata-sections"
        + (" -flto" if change != "archive_order" else ""),
        "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--gc-sections",
    )
    update(initially_available)
    configure(*flags)
    assert (
        _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
        is initially_available
    )
    configure(*flags)
    update(not initially_available)
    output = configure(*flags)
    assert (
        _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
        is not initially_available
    ), output


@pytest.mark.parametrize("change", ["conditional", "shadow"])
def test_openblas_new_include_search_result(tmp_path: Path, change: str) -> None:
    provider, cache, configure = _probe_project(tmp_path)
    header = provider / "include/cblas.h"
    _make_openblas_provider(provider, version="1.0", local_threads=False)
    if change == "conditional":
        header.write_text(
            header.read_text()
            + '#if __has_include("optional.h")\n#include "optional.h"\n#endif\n'
        )
        child = provider / "include/optional.h"
        configure()
        assert not _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
        child.write_text(
            "inline int openblas_set_num_threads_local(int) { return 0; }\n"
        )
        configure()
        assert _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
        child.unlink()
    else:
        first = provider / "first"
        first.mkdir()
        config = provider / "OpenBLASConfig.cmake"
        config.write_text(
            config.read_text()
            + f'set(OpenBLAS_INCLUDE_DIRS "{first};{provider / "include"}")\n'
        )
        header.write_text(header.read_text() + "#include <local_threads.h>\n")
        (provider / "include/local_threads.h").write_text(
            "inline int openblas_set_num_threads_local(int) { return 0; }\n"
        )
        configure()
        assert _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
        (first / "local_threads.h").write_text("// This provider has no local API.\n")
    output = configure()
    assert not _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS"), output


def test_openblas_probe_definitions_are_isolated(tmp_path: Path) -> None:
    provider, _, configure = _probe_project(tmp_path)
    for name, allowed in (("cblas.h", "LOCAL_THREADS"), ("lapacke.h", "LAPACKE")):
        header = provider / "include" / name
        if allowed == "LAPACKE":
            guard = "#if defined(GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS) || defined(GENERATIVEQC_OPENBLAS_HAS_GLOBAL_THREADS)\n#error unrelated capability definition\n#endif\n"
        else:
            guard = "#if defined(GENERATIVEQC_OPENBLAS_HAS_LAPACKE)\n#error unrelated capability definition\n#endif\n"
        header.write_text(guard + header.read_text())
    assert _probe_counts(configure()) == (3, 0)
    assert _probe_counts(configure()) == (0, 1)


def test_openblas_unknown_project_setup_reprobes(tmp_path: Path) -> None:
    hook = tmp_path / "hook.cmake"
    hook.write_text("# User-owned project setup.\n")
    _, _, configure = _probe_project(
        tmp_path, prelude=f'set(CMAKE_PROJECT_INCLUDE "{hook}")\n'
    )
    assert _probe_counts(configure()) == (3, 0)
    assert _probe_counts(configure()) == (3, 0)


def test_openblas_independent_source_only_checks(tmp_path: Path) -> None:
    _, _, configure = _probe_project(tmp_path)
    args = ("-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY", "-DCMAKE_CXX_STANDARD=98")
    assert _probe_counts(configure(*args)) == (3, 0)
    assert _probe_counts(configure(*args)) == (0, 1)


def test_openblas_batch_rebuilds_every_independent_probe(tmp_path: Path) -> None:
    """Observe compiler commands, including a fresh link of each capability."""
    log = tmp_path / "compiler.jsonl"
    wrapper = tmp_path / "compiler.py"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    wrapper.write_text(
        "import json, subprocess, sys\n"
        f"with open({str(log)!r}, 'a') as out:\n"
        "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        f"sys.exit(subprocess.call({compiler!r} + sys.argv[1:]))\n"
    )
    env = dict(
        os.environ, CXX=f"{shlex.quote(sys.executable)} {shlex.quote(str(wrapper))}"
    )
    _, _, configure = _probe_project(tmp_path, env=env)
    configure()
    for _ in range(2):
        log.write_text("")
        assert _probe_counts(configure()) == (0, 1)
        commands = [json.loads(line) for line in log.read_text().splitlines()]
        compiles = [args for args in commands if "-c" in args]
        links = [
            args
            for args in commands
            if "-c" not in args and any(arg.endswith(".o") for arg in args)
        ]
        assert len(compiles) == 3, commands
        assert len(links) == 3, commands
        for capability in ("LOCAL_THREADS", "GLOBAL_THREADS", "LAPACKE"):
            command = next(
                args
                for args in compiles
                if any(arg.endswith(f"/{capability}.cpp") for arg in args)
            )
            assert f"-DGENERATIVEQC_OPENBLAS_HAS_{capability}" in command
            assert (
                sum(arg.startswith("-DGENERATIVEQC_OPENBLAS_HAS_") for arg in command)
                == 1
            )


@pytest.mark.parametrize("hook_behavior", ["bypass", "unexpected_targets"])
def test_openblas_incomplete_batch_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hook_behavior: str
) -> None:
    """A successful first target cannot substitute for a complete batch proof."""
    module_root = tmp_path / "module"
    module_dir = module_root / "cmake"
    module_dir.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "cmake/GenerativeQCCpuLinalg.cmake",
        module_dir / "GenerativeQCCpuLinalg.cmake",
    )
    helper = module_dir / "GenerativeQCOpenBLASProbeBatch.cmake"
    if hook_behavior == "bypass":
        helper.write_text("# A replacement hook does not validate the batch.\n")
    else:
        real_helper = ROOT / "cmake/GenerativeQCOpenBLASProbeBatch.cmake"
        helper.write_text(
            f'include("{real_helper}")\nadd_library(unexpected INTERFACE)\n'
        )
    monkeypatch.setattr(sys.modules[__name__], "ROOT", module_root)
    _, _, configure = _probe_project(tmp_path)
    assert _probe_counts(configure()) == (3, 0)
    assert _probe_counts(configure()) == (3, 1)


@pytest.mark.parametrize(
    "variable", ["CMAKE_REQUIRED_FLAGS", "CMAKE_REQUIRED_DEFINITIONS"]
)
def test_openblas_required_compile_inputs(tmp_path: Path, variable: str) -> None:
    _, cache, configure = _probe_project(tmp_path)
    configure()
    assert _probe_counts(
        configure(f"-D{variable}=-DOPENBLAS_DISABLE_LOCAL_THREADS")
    ) == (3, 1)
    assert not _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
    assert _probe_counts(configure(f"-D{variable}=")) == (1, 1)
    assert _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")


def test_openblas_source_only_required_archive_options(tmp_path: Path) -> None:
    _, cache, configure = _probe_project(tmp_path)
    source_only = "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY"
    configure(source_only)
    # CheckCXXSourceCompiles adds the result definition before creating its
    # target. Each independently built archive must receive its own options.
    option = (
        "-DCMAKE_REQUIRED_LINK_OPTIONS=$<$<IN_LIST:GENERATIVEQC_OPENBLAS_HAS_LAPACKE,"
        "$<TARGET_PROPERTY:COMPILE_DEFINITIONS>>:--invalid-generativeqc-option>"
    )
    output = configure(source_only, option)
    assert _probe_counts(output) == (3, 1), output
    assert _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LOCAL_THREADS")
    assert not _capability(cache, "GENERATIVEQC_OPENBLAS_HAS_LAPACKE"), output
