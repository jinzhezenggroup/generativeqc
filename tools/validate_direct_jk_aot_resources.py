"""Qualify one stable AOT shard's static cubin resource identity (issue #2101).

Issue #2101 established that NVCC register/stack allocation for the ordinary
RHF J/K streaming kernels (dsss/ddss/ddps, stable AOT shard 6) is sensitive to
compilation context even when the generated mathematical source is unchanged.
A default-off candidate perturbed the shared stream header and flipped
unselected kernels from 114/114/158 registers (STACK 0) to 255/154/204-206
(STACK 544) without touching their emitted math.

This tool is the fail-closed qualification gate for that failure mode. It
generates the production shards with the exact CMake generator command,
compiles the selected shard with the exact production release flags through
the verified compiler-cache launcher, dumps the per-kernel cubin resource
table with cuobjdump, and requires the measured identity and every kernel's
REG/STACK/SHARED/LOCAL to match the committed ledger under
``manifests/maintenance/``. Any identity, toolchain, or resource drift fails
with a per-kernel, attributable report. No GPU is required.

``--record-output`` measures and writes a candidate ledger for deliberate
re-qualification. It never modifies the committed ledger and is not a gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.common.compiler_cache import verify_compiler_cache
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.evidence import (
    canonical_hash,
    file_hash,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from generativeqc_compiler.common.compiler_cache import CompilerCacheLauncher

MANIFEST = (
    ROOT
    / "python"
    / "generativeqc_compiler"
    / "integral"
    / "production_shell_classes.json"
)
GENERATOR = ROOT / "tools" / "generate_shell_kernels.py"
LEDGER = ROOT / "manifests" / "maintenance" / "direct_jk_aot_resource_ledger.json"

SCHEMA = "generativeqc.direct-jk-aot-resource-ledger"
REPORT_SCHEMA = "generativeqc.direct-jk-aot-resource-report"
SCHEMA_VERSION = 1
COVERAGE = (
    "stable AOT shard 6 (sm_120): ordinary RHF J/K streaming consumers "
    "dsss/ddss/ddps and every co-located kernel"
)
ISSUE = "#2101"

QUALIFIED_SHARDS = 8
QUALIFIED_SHARD_INDEX = 6
QUALIFIED_ARCHITECTURE = "sm_120"
QUALIFIED_UNIT_MODE = "stable-shards"
QUALIFIED_STANDARD = "c++20"
QUALIFIED_FAST_COMPILE = False
QUALIFIED_HOST_COMPILER = "nvcc-default"
# The production OBJECT library sets POSITION_INDEPENDENT_CODE=ON.
QUALIFIED_HOST_OPTIONS = ("-Xcompiler=-fPIC",)
QUALIFIED_CODE_GENERATION = (
    "--generate-code=arch=compute_120,code=[compute_120,sm_120]",
)
QUALIFIED_CCACHE_VERSION = "ccache version 4.14"
# Match the normal CMake release build: split compilation changes resources.
QUALIFIED_SPLIT_COMPILE_THREADS = 1
# CMake's NVIDIA Release rule adds NDEBUG alongside the target definition.
QUALIFIED_DEFINITIONS = ("NDEBUG", "GENERATIVEQC_HAS_CUDA=1")
# Source-tree headers every production shard prelude includes; the AOT header
# contract whose perturbation triggered issue #2101.
PRELUDE_HEADERS = (
    "scf/generated_shell_task.hpp",
    "runtime/compensated_atomic.cuh",
)
RESOURCE_FIELDS = ("REG", "STACK", "SHARED", "LOCAL")
# These implementations construct the actual compile invocation. Their hashes
# invalidate execution-cache entries independently of the scientific ledger:
# changing an adapter option must not replay a cubin built with the old command.
COMPILE_IMPLEMENTATIONS = (
    "CMakeLists.txt",
    "tools/validate_direct_jk_aot_resources.py",
    "python/generativeqc_compiler/common/compiler_cache.py",
    "python/generativeqc_compiler/common/cuda_adapter.py",
)


class LedgerError(ValueError):
    """Raised when the committed ledger is missing or structurally invalid."""


def validate_ambient_nvcc_flags() -> None:
    """Reject NVCC environment overrides that can supersede explicit argv."""
    overrides = {
        name: os.environ.get(name, "")
        for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")
        if os.environ.get(name, "")
    }
    if overrides:
        raise ValueError(
            "production AOT resource qualification forbids ambient NVCC flag "
            f"overrides: {overrides!r}"
        )


def _tool_version(tool: Path) -> str:
    result = subprocess.run(
        [str(tool), "--version"], check=True, capture_output=True, text=True, timeout=30
    )
    return (result.stdout + result.stderr).strip()


def toolchain_versions(nvcc: Path) -> dict[str, str]:
    nvcc = nvcc.resolve()
    return {
        name: _tool_version(nvcc.with_name(name))
        for name in ("nvcc", "ptxas", "cuobjdump")
    }


def generator_abi() -> int:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return int(payload["architectures"][QUALIFIED_ARCHITECTURE]["generator_abi"])


def validate_production_shard_count(
    cmake_cache: Path, nvcc: Path
) -> tuple[Path, Path, str]:
    """Bind the fixed qualification to CMake's actual production generator inputs."""
    configured = cmake_cache.read_text(encoding="utf-8")
    for field, expected in (
        ("GENERATIVEQC_AOT_SHARDS", QUALIFIED_SHARDS),
        ("GENERATIVEQC_AOT_SPLIT_COMPILE_THREADS", QUALIFIED_SPLIT_COMPILE_THREADS),
    ):
        values = re.findall(
            rf"^{field}:[^=\r\n]+=([^\r\n]*)$", configured, re.MULTILINE
        )
        if values != [str(expected)]:
            raise ValueError(
                "production CMake cache must contain exactly one "
                f"{field}={expected}; found {values!r}; "
                "requalify the production compile context deliberately"
            )

    def configured_value(field: str, *, optional: bool = False) -> str:
        values = re.findall(
            rf"^{field}:[^=\r\n]+=([^\r\n]*)$", configured, re.MULTILINE
        )
        if optional and not values:
            return ""
        if len(values) != 1:
            raise ValueError(
                f"production CMake cache must contain exactly one {field}; "
                f"found {values!r}"
            )
        return values[0]

    for field in ("GENERATIVEQC_ENABLE_CUDA", "GENERATIVEQC_ENABLE_AOT_SHELLS"):
        if configured_value(field).upper() not in {"1", "ON", "YES", "TRUE", "Y"}:
            raise ValueError(f"production qualification requires {field}=ON")
    if configured_value("GENERATIVEQC_CUDA_FAST_COMPILE").upper() not in {
        "0",
        "OFF",
        "NO",
        "FALSE",
        "N",
    }:
        raise ValueError(
            "production qualification requires GENERATIVEQC_CUDA_FAST_COMPILE=OFF"
        )
    if configured_value("CMAKE_CUDA_FLAGS").strip():
        raise ValueError("production qualification requires empty CMAKE_CUDA_FLAGS")
    if shlex.split(configured_value("CMAKE_CUDA_FLAGS_RELEASE")) != [
        "-O3",
        "-DNDEBUG",
    ]:
        raise ValueError(
            "production qualification requires CMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG"
        )
    for field, allowed in (
        ("GENERATIVEQC_AOT_UNIT_MODE", {QUALIFIED_UNIT_MODE}),
        ("GENERATIVEQC_AOT_PROFILE", {"auto", QUALIFIED_ARCHITECTURE}),
        ("GENERATIVEQC_AOT_PROFILES", {"", QUALIFIED_ARCHITECTURE}),
    ):
        if configured_value(field) not in allowed:
            raise ValueError(
                f"production qualification requires {field} in {sorted(allowed)!r}"
            )

    # Use CMakeLists.txt's priority order for explicit numeric targets. An ambient CUDAARCHS
    # fallback cannot be reconstructed reliably from this cache, so fail closed.
    architecture = ""
    for field in (
        "GENERATIVEQC_CUDA_COMPILE_ARCHITECTURES",
        "GENERATIVEQC_CUDA_ARCHITECTURES",
        "CMAKE_CUDA_ARCHITECTURES",
    ):
        architecture = configured_value(field, optional=True)
        if architecture:
            break
    if architecture != QUALIFIED_ARCHITECTURE.removeprefix("sm_"):
        raise ValueError(
            "production qualification requires one explicit CUDA target 120"
        )

    manifest = configured_value("GENERATIVEQC_AOT_SHELL_MANIFEST")
    if not manifest or Path(manifest).resolve() != MANIFEST.resolve():
        raise ValueError(
            "production GENERATIVEQC_AOT_SHELL_MANIFEST must match the qualified manifest"
        )
    configured_nvcc = Path(configured_value("CMAKE_CUDA_COMPILER")).resolve()
    if configured_nvcc != nvcc.resolve():
        raise ValueError(
            "production CMAKE_CUDA_COMPILER must match the measured --nvcc; "
            f"configured {configured_nvcc}, measured {nvcc.resolve()}"
        )
    host_compilers = {
        "CMAKE_CUDA_HOST_COMPILER": configured_value(
            "CMAKE_CUDA_HOST_COMPILER", optional=True
        ),
        "GENERATIVEQC_RESOLVED_CUDA_HOST_COMPILER": configured_value(
            "GENERATIVEQC_RESOLVED_CUDA_HOST_COMPILER"
        ),
    }
    if any(host_compilers.values()):
        raise ValueError(
            "production qualification requires the nvcc-default host context; "
            f"found {host_compilers!r}"
        )
    managed_launcher = configured_value(
        "GENERATIVEQC_CUDA_COMPILER_LAUNCHER_MANAGED"
    ).upper()
    if managed_launcher not in {"1", "ON", "YES", "TRUE", "Y"}:
        raise ValueError(
            "production GENERATIVEQC_CUDA_COMPILER_LAUNCHER_MANAGED must be ON"
        )
    effective_launcher = configured_value(
        "GENERATIVEQC_EFFECTIVE_CUDA_COMPILER_LAUNCHER"
    )
    if not effective_launcher:
        raise ValueError(
            "production GENERATIVEQC_EFFECTIVE_CUDA_COMPILER_LAUNCHER must be non-empty"
        )
    cache_executable = configured_value("GENERATIVEQC_RESOLVED_COMPILER_CACHE")
    if not cache_executable:
        raise ValueError(
            "production GENERATIVEQC_RESOLVED_COMPILER_CACHE must be non-empty"
        )
    return (
        cmake_cache.resolve().parent,
        Path(cache_executable).resolve(),
        effective_launcher,
    )


def validate_effective_compiler_launcher(
    working_directory: Path,
    launcher: CompilerCacheLauncher,
    effective: str,
) -> None:
    """Require CMake's effective launcher wrapper to match this invocation."""
    if launcher.name != "ccache":
        raise ValueError(
            "production AOT resource qualification requires the canonical ccache launcher"
        )
    if launcher.version.splitlines()[0] != QUALIFIED_CCACHE_VERSION:
        raise ValueError(
            "production AOT resource qualification requires "
            f"{QUALIFIED_CCACHE_VERSION}; found {launcher.version.splitlines()[0]!r}"
        )
    fields = effective.split(";")
    expected_tail = [
        "-E",
        "env",
        f"CCACHE_BASEDIR={working_directory.resolve()}",
        str(launcher.path),
    ]
    if (
        len(fields) != 5
        or Path(fields[0]).name not in {"cmake", "cmake.exe"}
        or fields[1:] != expected_tail
    ):
        raise ValueError(
            "production CMAKE_CUDA_COMPILER_LAUNCHER does not match the "
            f"qualified ccache wrapper: {effective!r}"
        )


def validate_generated_cuda_rules(
    working_directory: Path,
    nvcc: Path,
    effective_launcher: str,
    *,
    generated_directory: Path,
) -> None:
    """Bind the qualified Ninja edge to the recorded launcher and default host.

    Support literal and per-edge CMake launchers, without evaluating arbitrary
    Ninja/shell variables or borrowing another target's flags or launcher.
    """
    rules = (working_directory / "CMakeFiles" / "rules.ninja").read_text()
    matching = re.findall(
        r"^rule (CUDA_COMPILER__generativeqc_aot_sm_120_[^\s]+)\n"
        r"(.*?)(?=^rule |\Z)",
        rules,
        re.MULTILINE | re.DOTALL,
    )
    if len(matching) != 1 or not matching[0][0].endswith("_Release"):
        raise ValueError(
            "production qualification requires one Release Ninja AOT compiler rule"
        )
    commands = re.findall(r"^[ \t]+command = ([^\r\n]+)$", matching[0][1], re.MULTILINE)
    if len(commands) != 1:
        raise ValueError(
            "production AOT compiler rule must contain exactly one command"
        )
    # Target compile options live in FLAGS on the shard's build edge, even
    # when the launcher and CMAKE_CUDA_HOST_COMPILER are literal rule tokens.
    # Read only this edge; never borrow another target's values or evaluate
    # arbitrary Ninja/shell variables.
    build = (working_directory / "build.ninja").read_text()
    edges = re.findall(
        r"^build [^\r\n]*generativeqc_generated_shell_sm120_shard_6\.cu\.o: "
        + re.escape(matching[0][0])
        + r" ([^\r\n]*)\n((?:[ \t]+[^\r\n]*\n)*)",
        build,
        re.MULTILINE,
    )
    if len(edges) != 1:
        raise ValueError(
            "production qualification requires exactly one shard-6 Ninja edge"
        )

    inputs, edge = edges[0]
    if "$" in inputs:
        raise ValueError("production shard-6 source input must be a literal Ninja path")
    input_tokens = inputs.split()
    dependency_start = next(
        (index for index, token in enumerate(input_tokens) if token in {"|", "||"}),
        len(input_tokens),
    )
    sources = input_tokens[:dependency_start]
    if len(sources) != 1:
        raise ValueError("production shard-6 edge requires one explicit source input")
    source = Path(sources[0])
    if not source.is_absolute():
        source = working_directory / source
    expected_source = (
        generated_directory
        / QUALIFIED_ARCHITECTURE
        / f"generativeqc_generated_shell_sm{QUALIFIED_ARCHITECTURE[3:]}_shard_{QUALIFIED_SHARD_INDEX}.cu"
    )
    if source.resolve() != expected_source.resolve():
        raise ValueError(
            "production shard-6 source input differs from the measured generated source"
        )

    def edge_tokens(field: str) -> list[str]:
        values = re.findall(rf"^[ \t]+{field} = ([^\r\n]*)$", edge, re.MULTILINE)
        if len(values) != 1 or "$" in values[0]:
            raise ValueError(
                f"production shard-6 edge must declare exactly one literal {field} value"
            )
        return shlex.split(values[0])

    flag_tokens = edge_tokens("FLAGS")
    if any(
        token in {"-optf", "--options-file"}
        or token.startswith(("-optf=", "--options-file="))
        for token in flag_tokens
    ):
        raise ValueError(
            "production qualification does not support options-file indirection in FLAGS"
        )
    command_text = commands[0]
    if command_text.startswith("${LAUNCHER}"):
        # Newer CMake versions put launcher variables on each build edge.
        launchers = re.findall(r"^[ \t]+LAUNCHER = ([^\r\n]+)$", edge, re.MULTILINE)
        if len(launchers) != 1:
            raise ValueError(
                "production shard-6 edge must declare exactly one LAUNCHER"
            )
        command_text = command_text.removeprefix("${LAUNCHER}")
        if command_text.startswith("${CODE_CHECK}"):
            checks = re.findall(r"^[ \t]+CODE_CHECK = ([^\r\n]*)$", edge, re.MULTILINE)
            if len(checks) > 1 or any(value.strip() for value in checks):
                raise ValueError(
                    "production qualification does not support a CODE_CHECK wrapper"
                )
            command_text = command_text.removeprefix("${CODE_CHECK}")
        command_text = launchers[0] + command_text
    command = shlex.split(command_text)
    # CUDAHOSTCXX/toolchain selections may be normal CMake variables rather
    # than cache entries. Reject their rule flags and target-option edge flags.
    if any(
        token in {"-ccbin", "--compiler-bindir"}
        or token.startswith(("-ccbin=", "--compiler-bindir="))
        for token in [*command, *flag_tokens]
    ):
        raise ValueError(
            "production qualification does not support a generated CUDA host override"
        )
    # This gate supports the stock production Release recipe, not arbitrary
    # global or target-local compiler options. A manual adapter invocation must
    # never qualify a different production optimization/architecture recipe.
    architecture = QUALIFIED_ARCHITECTURE.removeprefix("sm_")
    expected_flags = [
        "-O3",
        "-DNDEBUG",
        f"-std={QUALIFIED_STANDARD}",
        f"--generate-code=arch=compute_{architecture},code=[compute_{architecture},sm_{architecture}]",
        *QUALIFIED_HOST_OPTIONS,
    ]
    # These independent options may be reordered; require each exactly once
    # so duplicates or later overriding options cannot slip through.
    if sorted(flag_tokens) != sorted(expected_flags):
        raise ValueError(
            "production shard-6 FLAGS do not match the qualified Release recipe: "
            f"{flag_tokens!r} != {expected_flags!r}"
        )
    # NDEBUG comes from the Release FLAGS; the remaining definitions and the
    # include-search order must match the adapter and preprocessor exactly.
    if edge_tokens("DEFINES") != [
        f"-D{definition}"
        for definition in QUALIFIED_DEFINITIONS
        if definition != "NDEBUG"
    ]:
        raise ValueError("production shard-6 DEFINES differ from the qualified recipe")
    if edge_tokens("INCLUDES") != [
        f"-I{ROOT / 'src'}",
        f"-I{generated_directory.resolve()}",
    ]:
        raise ValueError("production shard-6 INCLUDES differ from the qualified recipe")
    expected_launcher = effective_launcher.split(";")
    compiler_index = len(expected_launcher)
    if (
        len(command) <= compiler_index
        or command[:compiler_index] != expected_launcher
        or not Path(command[compiler_index]).is_absolute()
        or Path(command[compiler_index]).resolve() != nvcc.resolve()
        or any("$" in token for token in command[: compiler_index + 1])
    ):
        raise ValueError(
            "production generated CUDA launcher/compiler does not match the "
            "recorded effective launcher and measured nvcc"
        )
    # Accept only the stock NVIDIA compile rule. Dependency-file bookkeeping
    # is recognized explicitly; compiler flags or options-file indirection
    # inserted by a make-rules override are not part of this qualified recipe.
    if command[compiler_index + 1 :] != [
        "-forward-unknown-to-host-compiler",
        "$DEFINES",
        "$INCLUDES",
        "$FLAGS",
        "-MD",
        "-MT",
        "$out",
        "-MF",
        "$DEP_FILE",
        "-x",
        "cu",
        "-c",
        "$in",
        "-o",
        "$out",
    ]:
        raise ValueError(
            "production CUDA rule differs from the supported compile/dependency recipe"
        )


def _production_shard_paths(output_directory: Path) -> tuple[Path, ...]:
    profile = output_directory / QUALIFIED_ARCHITECTURE
    return tuple(
        profile
        / f"generativeqc_generated_shell_sm{QUALIFIED_ARCHITECTURE[3:]}_shard_{index}.cu"
        for index in range(QUALIFIED_SHARDS)
    )


def generate_production_shards(
    working_directory: Path, output_directory: Path
) -> tuple[Path, ...]:
    """Execute CMake's production custom-command edge for the qualified shard."""
    shards = _production_shard_paths(output_directory)
    try:
        target = (
            shards[QUALIFIED_SHARD_INDEX]
            .resolve()
            .relative_to(working_directory.resolve())
        )
    except ValueError as error:
        raise ValueError(
            "production generated directory must be below the CMake build directory"
        ) from error
    arguments = [
        "cmake",
        "--build",
        str(working_directory.resolve()),
        "--target",
        target.as_posix(),
    ]
    result = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=working_directory,
    )
    if result.returncode:
        raise RuntimeError(
            "production CMake AOT generation edge failed: "
            + (result.stderr or result.stdout).strip()
        )
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise RuntimeError(
            "production AOT generation missed shard sources: " + ", ".join(missing)
        )
    return shards


def production_generated_directory(cmake_cache: Path, requested: Path | None) -> Path:
    """Return the one production generated-source directory for this CMake tree."""
    expected = cmake_cache.resolve().parent / "generated" / "production_shell_kernels"
    if requested is not None and requested.resolve() != expected:
        raise ValueError(
            "--generated-dir must match the production CMake generated-source directory"
        )
    return expected


def _relative_compile_source(shard: Path, working_directory: Path) -> Path:
    try:
        return shard.resolve().relative_to(working_directory.resolve())
    except ValueError as error:
        raise ValueError(
            "qualified shard must be below the production CMake build directory"
        ) from error


def _preprocess_command(
    nvcc: Path,
    shard: Path,
    generated: Path,
    *,
    working_directory: Path | None = None,
) -> list[str]:
    source = (
        _relative_compile_source(shard, working_directory)
        if working_directory is not None
        else shard
    )
    return [
        str(nvcc),
        f"-std={QUALIFIED_STANDARD}",
        *QUALIFIED_CODE_GENERATION,
        *QUALIFIED_HOST_OPTIONS,
        *(f"-D{definition}" for definition in QUALIFIED_DEFINITIONS),
        f"-I{ROOT / 'src'}",
        f"-I{generated}",
        "-E",
        str(source),
    ]


def preprocessed_identity_sha256(
    nvcc: Path, shard: Path, generated: Path, *, working_directory: Path
) -> str:
    """SHA256 of the preprocessed translation unit, line markers included.

    Line markers embed the compilation context whose sensitivity issue #2101
    established; the identity deliberately pins them so the gate always runs
    against the same qualified context.
    """
    result = subprocess.run(
        _preprocess_command(
            nvcc, shard, generated, working_directory=working_directory
        ),
        check=False,
        capture_output=True,
        cwd=working_directory,
        timeout=300,
    )
    if result.returncode:
        raise RuntimeError(
            "preprocessing failed: " + result.stderr.decode(errors="replace")
        )
    return hashlib.sha256(result.stdout).hexdigest()


def cuobjdump_resources(output: str) -> dict[str, dict[str, int]]:
    """Decode per-kernel cubin resources; incomplete or duplicate rows fail.

    Mirrors ``tools/generativeqc_validation/f_shell.py`` so both gates parse
    the same CUDA 12.9 ``--dump-resource-usage`` layout.
    """
    rows: dict[str, dict[str, int]] = {}
    for match in re.finditer(r"[ \t]*Function (\w+):\n([^\n]+)", output):
        fields = {
            key: int(value)
            for key, value in re.findall(r"\b(REG|STACK|SHARED|LOCAL):(\d+)", match[2])
        }
        if set(fields) != set(RESOURCE_FIELDS) or match[1] in rows:
            raise ValueError("incomplete or duplicate CUOBJDump kernel resources")
        rows[match[1]] = fields
    if not rows:
        raise ValueError("CUOBJDump reported no kernel resources")
    return rows


def qualified_flags() -> dict[str, Any]:
    return {
        "architecture": QUALIFIED_ARCHITECTURE,
        "standard": QUALIFIED_STANDARD,
        "optimization": "-O3",
        "fast_compile": QUALIFIED_FAST_COMPILE,
        "host_compiler": QUALIFIED_HOST_COMPILER,
        "host_options": list(QUALIFIED_HOST_OPTIONS),
        "code_generation": list(QUALIFIED_CODE_GENERATION),
        "split_compile_threads": QUALIFIED_SPLIT_COMPILE_THREADS,
        "definitions": list(QUALIFIED_DEFINITIONS),
        "ptxas_verbose": True,
    }


def compile_identity(
    *,
    shard_sha256: str,
    preprocessed_sha256: str,
    header_sha256: dict[str, str],
    manifest_sha256: str,
    abi: int,
    toolchain: dict[str, str],
) -> dict[str, Any]:
    return {
        "architecture": QUALIFIED_ARCHITECTURE,
        "unit_mode": QUALIFIED_UNIT_MODE,
        "shard_source_sha256": shard_sha256,
        "preprocessed_sha256": preprocessed_sha256,
        "header_sha256": dict(sorted(header_sha256.items())),
        "manifest_sha256": manifest_sha256,
        "generator_abi": abi,
        "toolchain": dict(sorted(toolchain.items())),
        "flags": qualified_flags(),
    }


def cache_statistics(launcher: CompilerCacheLauncher) -> dict[str, Any]:
    """Retain ccache totals; never inspect an unrelated shared sccache server.

    The cache helper owns and destroys sccache's private server inside compile().
    Its statistics cannot be observed here. ccache uses the same environment for
    the snapshot and compile, as in the CI job that explicitly installs ccache.
    """
    if launcher.name != "ccache":
        return {
            "status": "unavailable",
            "reason": "private sccache statistics require collection inside the cache helper",
        }
    try:
        result = subprocess.run(
            [str(launcher.path), "--show-stats"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "reason": str(error)}
    return {
        "status": "pass" if result.returncode == 0 else "fail",
        "scope": "cumulative ccache totals, not isolated per-invocation counters",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def qualified_ccache_policy(working_directory: Path) -> dict[str, str]:
    return {
        "base_dir": str(working_directory.resolve()),
        "compiler": "",
        "compiler_check": "mtime",
        "compiler_type": "auto",
        "depend_mode": "false",
        "direct_mode": "true",
        "disable": "false",
        "extra_files_to_hash": "",
        "hash_dir": "true",
        "ignore_headers_in_manifest": "",
        "ignore_options": "",
        "inode_cache": "true",
        "keep_comments_cpp": "false",
        "path": "",
        "pch_external_checksum": "false",
        "prefix_command": "",
        "prefix_command_cpp": "",
        "read_only": "false",
        "read_only_direct": "false",
        "recache": "false",
        "remote_only": "false",
        "remote_storage": "",
        "reshare": "false",
        "sloppiness": "",
        "stats": "true",
    }


def validate_ccache_configuration(
    launcher: CompilerCacheLauncher, working_directory: Path
) -> dict[str, str]:
    """Require identity-safe canonical ccache configuration before cache replay."""
    result = subprocess.run(
        [str(launcher.path), "--show-config"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode:
        raise RuntimeError(
            "ccache --show-config failed: " + (result.stderr or result.stdout).strip()
        )
    configuration = {
        match[1]: match[2]
        for match in re.finditer(
            r"^\([^\r\n)]*\)[ \t]+([a-z_]+)[ \t]*=[ \t]*(.*)$",
            result.stdout,
            re.MULTILINE,
        )
    }
    expected = qualified_ccache_policy(working_directory)
    measured = {field: configuration.get(field) for field in expected}
    if measured != expected:
        raise ValueError(
            f"ccache configuration differs from the qualified policy: {measured!r}"
        )
    return expected


@contextmanager
def _production_compile_context(directory: Path) -> Iterator[None]:
    """Match CMake's ccache cwd/base contract for one finite compile.

    This validator is a single-process CLI. Keeping the process-scoped cwd and
    environment change inside this bounded context makes ccache pass the same
    relative generated source spelling to NVCC as the configured CMake build.
    """
    working_directory = directory.resolve()
    previous_directory = Path.cwd()
    previous_basedir = os.environ.get("CCACHE_BASEDIR")
    os.chdir(working_directory)
    os.environ["CCACHE_BASEDIR"] = str(working_directory)
    try:
        yield
    finally:
        os.chdir(previous_directory)
        if previous_basedir is None:
            os.environ.pop("CCACHE_BASEDIR", None)
        else:
            os.environ["CCACHE_BASEDIR"] = previous_basedir


def compile_shard(
    *,
    nvcc: Path,
    shard: Path,
    generated: Path,
    cache: Path,
    identity: dict[str, Any],
    timeout: float,
    working_directory: Path,
    launcher: CompilerCacheLauncher,
) -> tuple[Path, Path, dict[str, Any]]:
    """Compile one shard through the verified cache launcher with a bounded
    identity-keyed object cache; successful hits require matching metadata."""
    implementation_sha256 = {
        path: file_hash(ROOT / path) for path in COMPILE_IMPLEMENTATIONS
    }
    working_directory = working_directory.resolve()
    source_argument = _relative_compile_source(shard, working_directory)
    compile_context = {
        "working_directory": str(working_directory),
        "ccache_basedir": str(working_directory),
        "source_argument": source_argument.as_posix(),
        "ccache_policy": qualified_ccache_policy(working_directory),
    }
    key = canonical_hash(
        {
            "identity": identity,
            "compile_implementation_sha256": implementation_sha256,
            "compile_context": compile_context,
        }
    )
    directory = cache / key
    directory.mkdir(parents=True, exist_ok=True)
    object_path = directory / "shard.o"
    metadata_path = directory / "compile.json"
    log_path = directory / "nvcc.log"
    metadata = None
    if metadata_path.exists():
        candidate = json.loads(metadata_path.read_text())
        if (
            candidate.get("status") == "pass"
            and candidate.get("identity") == identity
            and candidate.get("compile_implementation_sha256") == implementation_sha256
            and candidate.get("compile_context") == compile_context
            and object_path.is_file()
            and log_path.is_file()
            and candidate.get("object_hash") == file_hash(object_path)
            and candidate.get("log_hash") == file_hash(log_path)
        ):
            metadata = {**candidate, "cache_hit": True}
    if metadata is None:
        with _production_compile_context(working_directory):
            compiler = CudaCompilerAdapter(
                nvcc,
                cuda_target_info(QUALIFIED_ARCHITECTURE),
                compile_timeout=timeout,
                compiler_cache_launcher=launcher,
            )
            statistics = {
                "launcher": {
                    "name": launcher.name,
                    "path": str(launcher.path),
                    "version": launcher.version,
                },
                "context": compile_context,
                "before": cache_statistics(launcher),
                "after": {"status": "not-recorded"},
            }
            statistics_path = directory / "compiler-cache.json"
            statistics_path.write_text(
                json.dumps(statistics, indent=2, sort_keys=True) + "\n"
            )
            try:
                compiled = compiler.compile(
                    source_argument,
                    object_path,
                    includes=(ROOT / "src", generated),
                    options=(
                        *QUALIFIED_HOST_OPTIONS,
                        *(
                            (f"--split-compile={QUALIFIED_SPLIT_COMPILE_THREADS}",)
                            if QUALIFIED_SPLIT_COMPILE_THREADS != 1
                            else ()
                        ),
                        *(f"-D{definition}" for definition in QUALIFIED_DEFINITIONS),
                    ),
                    standard=QUALIFIED_STANDARD,
                    architecture_options=QUALIFIED_CODE_GENERATION,
                )
            finally:
                statistics["after"] = cache_statistics(launcher)
                statistics_path.write_text(
                    json.dumps(statistics, indent=2, sort_keys=True) + "\n"
                )
        log_path.write_text(compiled.stdout + compiled.stderr)
        metadata = {
            "status": "pass" if compiled.returncode == 0 else "fail",
            "reason": None
            if compiled.returncode == 0
            else "NVCC timeout"
            if compiled.timed_out
            else "NVCC compilation failed",
            "identity": identity,
            "compile_implementation_sha256": implementation_sha256,
            "compile_context": compile_context,
            "cache_key": key,
            "cache_hit": False,
            "seconds": compiled.duration_seconds,
            "returncode": compiled.returncode,
            "object_hash": file_hash(object_path) if compiled.returncode == 0 else None,
            "log_hash": file_hash(log_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    if metadata["status"] != "pass":
        raise RuntimeError(f"shard compilation failed: {metadata['reason']}")
    return object_path, directory, metadata


def measure_resources(
    nvcc: Path, object_path: Path, directory: Path
) -> dict[str, dict[str, int]]:
    cuobjdump = nvcc.resolve().with_name("cuobjdump")
    dumped = subprocess.run(
        [str(cuobjdump), "--dump-resource-usage", str(object_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    dump_path = directory / "cuobjdump.txt"
    dump_path.write_text(dumped.stdout)
    with TemporaryDirectory(dir=directory) as extracted:
        subprocess.run(
            [str(cuobjdump), "--extract-elf", "all", str(object_path.resolve())],
            cwd=extracted,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        cubins = [path for path in Path(extracted).glob("*.cubin")]
    if len(cubins) != 1:
        raise RuntimeError(f"expected exactly one release cubin, found {len(cubins)}")
    resources = cuobjdump_resources(dumped.stdout)
    return resources


def validate_ledger(payload: Any) -> None:
    """Strictly validate the committed ledger shape; unknown content fails."""
    if not isinstance(payload, dict):
        raise LedgerError("ledger root must be an object")
    if payload.get("schema") != SCHEMA:
        raise LedgerError("unknown ledger schema")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise LedgerError("unsupported ledger schema_version")
    if payload.get("coverage") != COVERAGE:
        raise LedgerError("ledger coverage does not match the qualified scope")
    if payload.get("issue") != ISSUE:
        raise LedgerError("ledger issue reference does not match the qualified issue")
    for field in (
        "architecture",
        "unit_mode",
        "shard_source_sha256",
        "preprocessed_sha256",
        "manifest_sha256",
    ):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise LedgerError(f"ledger field {field!r} must be a non-empty string")
    if payload.get("architecture") != QUALIFIED_ARCHITECTURE:
        raise LedgerError(
            "ledger architecture does not match the qualified architecture"
        )
    if payload.get("unit_mode") != QUALIFIED_UNIT_MODE:
        raise LedgerError("ledger unit_mode does not match the qualified unit mode")
    if payload.get("generator_abi") != generator_abi():
        raise LedgerError(
            "ledger generator_abi does not match the qualified manifest ABI"
        )
    headers = payload.get("header_sha256")
    if not isinstance(headers, dict) or set(headers) != set(PRELUDE_HEADERS):
        raise LedgerError("ledger header_sha256 must cover exactly the prelude headers")
    if not all(
        isinstance(value, str) and len(value) == 64 for value in headers.values()
    ):
        raise LedgerError("ledger header_sha256 values must be 64-character hashes")
    toolchain = payload.get("toolchain")
    if not isinstance(toolchain, dict) or set(toolchain) != {
        "nvcc",
        "ptxas",
        "cuobjdump",
    }:
        raise LedgerError("ledger toolchain must record nvcc, ptxas, and cuobjdump")
    flags = payload.get("flags")
    expected_flags = qualified_flags()
    if not isinstance(flags, dict) or set(flags) != set(expected_flags):
        raise LedgerError("ledger flags must record exactly the qualified flag fields")
    for field, expected in expected_flags.items():
        if type(flags[field]) is not type(expected) or flags[field] != expected:
            raise LedgerError(f"ledger flags.{field} must match the qualified value")
    kernels = payload.get("kernels")
    if not isinstance(kernels, dict) or not kernels:
        raise LedgerError("ledger kernels must be a non-empty object")
    for name, resources in kernels.items():
        if not isinstance(name, str) or not name.strip():
            raise LedgerError("ledger kernel names must be non-empty strings")
        if not isinstance(resources, dict) or set(resources) != set(RESOURCE_FIELDS):
            raise LedgerError(
                f"ledger kernel {name!r} must record exactly REG/STACK/SHARED/LOCAL"
            )
        for field in RESOURCE_FIELDS:
            value = resources.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LedgerError(
                    f"ledger kernel {name!r} {field} must be a non-negative integer"
                )


def compare(ledger: dict[str, Any], measurement: dict[str, Any]) -> list[str]:
    """Return attributable failures; an empty list means exact qualification."""
    validate_ledger(ledger)
    failures: list[str] = []
    toolchain = measurement["toolchain"]
    for component in ("nvcc", "ptxas", "cuobjdump"):
        if ledger["toolchain"][component] != toolchain[component]:
            failures.append(
                f"toolchain drift ({component}): measured {toolchain[component]!r} "
                f"!= qualified {ledger['toolchain'][component]!r}; re-qualify with --record-output"
            )
    identity = measurement["identity"]
    for field in (
        "architecture",
        "unit_mode",
        "shard_source_sha256",
        "preprocessed_sha256",
        "manifest_sha256",
        "generator_abi",
    ):
        if ledger.get(field) != identity.get(field):
            failures.append(
                f"identity drift ({field}): measured {identity.get(field)!r} "
                f"!= qualified {ledger.get(field)!r}; regenerate the ledger deliberately"
            )
    if ledger["header_sha256"] != identity["header_sha256"]:
        for header in PRELUDE_HEADERS:
            if ledger["header_sha256"].get(header) != identity["header_sha256"].get(
                header
            ):
                failures.append(
                    f"identity drift (header {header}): measured "
                    f"{identity['header_sha256'].get(header)!r} != qualified "
                    f"{ledger['header_sha256'].get(header)!r}; regenerate the ledger deliberately"
                )
    flags = identity.get("flags")
    if not isinstance(flags, dict):
        failures.append("identity drift (flags): measured flags must be an object")
    else:
        for field in sorted(set(ledger["flags"]) | set(flags)):
            value = ledger["flags"].get(field)
            measured = flags.get(field)
            if (
                field not in ledger["flags"]
                or field not in flags
                or type(measured) is not type(value)
                or measured != value
            ):
                failures.append(
                    f"identity drift (flags.{field}): measured {measured!r} "
                    f"!= qualified {value!r}; regenerate the ledger deliberately"
                )
    qualified_kernels = ledger["kernels"]
    measured_kernels = measurement["kernels"]
    added = sorted(set(measured_kernels) - set(qualified_kernels))
    removed = sorted(set(qualified_kernels) - set(measured_kernels))
    if added or removed:
        failures.append(f"kernel inventory drift: added={added!r} removed={removed!r}")
    for name in sorted(set(qualified_kernels) & set(measured_kernels)):
        for field in RESOURCE_FIELDS:
            qualified = qualified_kernels[name][field]
            measured = measured_kernels[name][field]
            if qualified != measured:
                failures.append(
                    f"kernel resource drift: {name} {field} {qualified} -> {measured}"
                )
    return failures


def measured_ledger(measurement: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "coverage": COVERAGE,
        "issue": ISSUE,
        **measurement["identity"],
        "kernels": dict(sorted(measurement["kernels"].items())),
    }


def run_measurement(
    *,
    nvcc: Path,
    cache: Path,
    output: Path,
    generated_directory: Path,
    cmake_cache: Path,
    timeout: float,
) -> dict[str, Any]:
    """Generate, preprocess, compile, and resource-dump the qualified shard.

    The shard must live at the production CMake source path
    (``build/cuda-release-sm120/generated/production_shell_kernels``) because
    the qualified identity deliberately pins the preprocessed line markers of
    that exact path.
    """
    generated = generated_directory.resolve()

    def validate_context() -> tuple[Path, CompilerCacheLauncher, str]:
        validate_ambient_nvcc_flags()
        working, cache_executable, effective = validate_production_shard_count(
            cmake_cache, nvcc
        )
        selected = verify_compiler_cache(cache_executable, min(timeout, 30))
        validate_effective_compiler_launcher(working, selected, effective)
        validate_generated_cuda_rules(
            working, nvcc, effective, generated_directory=generated
        )
        with _production_compile_context(working):
            validate_ccache_configuration(selected, working)
        return working, selected, effective

    # CMake's build target may regenerate its own cache/rules. Validate both the
    # configured input context and the effective post-generation context.
    working_directory, launcher, _ = validate_context()
    shards = generate_production_shards(working_directory, generated)
    working_directory, launcher, _ = validate_context()
    shard = shards[QUALIFIED_SHARD_INDEX]
    toolchain = toolchain_versions(nvcc)
    identity = compile_identity(
        shard_sha256=file_hash(shard),
        preprocessed_sha256=preprocessed_identity_sha256(
            nvcc, shard, generated, working_directory=working_directory
        ),
        header_sha256={
            header: file_hash(ROOT / "src" / header) for header in PRELUDE_HEADERS
        },
        manifest_sha256=file_hash(MANIFEST),
        abi=generator_abi(),
        toolchain=toolchain,
    )
    object_path, cache_directory, compile_metadata = compile_shard(
        nvcc=nvcc,
        shard=shard,
        generated=generated,
        cache=cache,
        identity=identity,
        timeout=timeout,
        working_directory=working_directory,
        launcher=launcher,
    )
    kernels = dict(
        sorted(measure_resources(nvcc, object_path, cache_directory).items())
    )
    return {
        "identity": identity,
        "toolchain": toolchain,
        "kernels": kernels,
        "object_path": str(object_path),
        "compile": {
            key: compile_metadata[key]
            for key in ("cache_hit", "cache_key", "seconds", "returncode")
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cmake-cache",
        type=Path,
        default=ROOT / "build" / "cuda-release-sm120" / "CMakeCache.txt",
        help="production CMakeCache.txt whose generator context and split must match",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        help=(
            "production AOT source directory (default: "
            "<cmake build>/generated/production_shell_kernels)"
        ),
    )
    parser.add_argument("--compile-timeout", type=float, default=900)
    parser.add_argument(
        "--record-output",
        type=Path,
        help="write the measured candidate ledger here instead of gating",
    )
    args = parser.parse_args(argv)

    nvcc = args.nvcc.resolve()
    if not nvcc.is_file():
        print(f"nvcc not found: {nvcc}", file=sys.stderr)
        return 2
    cache = args.cache.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    generated_directory = production_generated_directory(
        args.cmake_cache, args.generated_dir
    )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": None,
        "failures": [],
        "measurement": None,
        "ledger": str(LEDGER),
    }
    try:
        measurement = run_measurement(
            nvcc=nvcc,
            cache=cache,
            output=output,
            generated_directory=generated_directory,
            cmake_cache=args.cmake_cache,
            timeout=args.compile_timeout,
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        report["status"] = "fail"
        report["failures"] = [f"measurement failed: {error}"]
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(report["failures"][0], file=sys.stderr)
        return 1
    report["measurement"] = measurement

    if args.record_output is not None:
        record = args.record_output.resolve()
        if record == LEDGER.resolve():
            print(
                "--record-output must not overwrite the committed ledger",
                file=sys.stderr,
            )
            return 2
        record.parent.mkdir(parents=True, exist_ok=True)
        payload = measured_ledger(measurement)
        record.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        report["status"] = "recorded"
        report["recorded_kernels"] = len(payload["kernels"])
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"recorded {len(payload['kernels'])} kernels to {record}; "
            "commit the ledger deliberately to enable the gate"
        )
        return 0

    if not LEDGER.is_file():
        report["status"] = "fail"
        report["failures"] = [
            f"ledger is missing: {LEDGER}; qualify with --record-output"
        ]
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(report["failures"][0], file=sys.stderr)
        return 1

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    try:
        validate_ledger(ledger)
    except LedgerError as error:
        report["status"] = "fail"
        report["failures"] = [f"ledger validation failed: {error}"]
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(report["failures"][0], file=sys.stderr)
        return 1

    failures = compare(ledger, measurement)
    report["failures"] = failures
    report["status"] = "fail" if failures else "pass"
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print(
            f"direct J/K AOT resource qualification failed: {len(failures)} drift(s)",
            file=sys.stderr,
        )
        return 1
    print(
        f"direct J/K AOT resource qualification passed: "
        f"{len(measurement['kernels'])} kernels match the committed ledger"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
