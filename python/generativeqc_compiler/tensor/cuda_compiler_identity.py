"""Device-free, bounded dependency facts for prepared TensorIR CUDA recipes.

NVCC's -M phase only preprocesses; it does not compile or link. Never replace
its dependency closure with a guessed list of CUB/CCCL umbrella headers.
"""

from __future__ import annotations

import functools
import hashlib
import math
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from generativeqc_compiler.common.compiler_process import run_compiler
from generativeqc_compiler.common.paths import asset_path

if TYPE_CHECKING:
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter

_ENVIRONMENT = (
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
    "NVCC_CCBIN",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH",
)
# An override must not turn the read-only probe into a compile/run, redirect its
# output, or hide such switches in an options file. Ordinary -I/-D/-ccbin flags
# still go through NVCC itself, exactly as they do in compile_shared().
_UNSAFE_OPTIONS = {
    "-M",
    "--generate-dependencies",
    "-MM",
    "--generate-nonsystem-dependencies",
    "-MD",
    "--generate-dependencies-with-compile",
    "-MMD",
    "--generate-nonsystem-dependencies-with-compile",
    "-MF",
    "--dependency-output",
    "-MT",
    "--dependency-target-name",
    "-MP",
    "--generate-dependency-targets",
    "-E",
    "--preprocess",
    "-c",
    "--compile",
    "-dc",
    "--device-c",
    "-dw",
    "--device-w",
    "-dlink",
    "--device-link",
    "-link",
    "--link",
    "-lib",
    "--lib",
    "-ptx",
    "--ptx",
    "--optix-ir",
    "-optix-ir",
    "--ltoir",
    "-ltoir",
    "-cubin",
    "--cubin",
    "-fatbin",
    "--fatbin",
    "-cuda",
    "--cuda",
    "-run",
    "--run",
    "-o",
    "--output-file",
    "-odir",
    "--output-directory",
    "-optf",
    "--options-file",
    "-clean",
    "--clean-targets",
    "-dryrun",
    "--dryrun",
    "-keep",
    "--keep",
    "-save-temps",
    "--save-temps",
    "-keep-dir",
    "--keep-dir",
}
_INCLUDE_OPTIONS = {"-I", "--include-path", "-isystem", "--system-include"}
_HOST_OPTIONS = {"-ccbin", "--compiler-bindir"}
_MAX_DEPENDENCIES = 16384
_MAX_DEPENDENCY_BYTES = 256 << 20


def _file_digest(path: Path, budget: int) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(min(1 << 20, budget - size + 1)):
            size += len(chunk)
            if size > budget:
                raise ValueError("tensor compiler identity file-byte budget exceeded")
            digest.update(chunk)
    return digest.hexdigest(), size


@functools.lru_cache(maxsize=32)
def _version(path: Path, digest: str, environment: tuple[str, ...]) -> str:
    # Bytes, not only the version string or timestamp, invalidate this finite
    # process-local probe cache when a tool is replaced in place.
    result = run_compiler([str(path), "--version"], 30.0, label="compiler identity")
    if result.returncode:
        raise ValueError("tensor compiler version probe failed: " + result.stderr)
    return (result.stdout + result.stderr).strip()


def _executable(path: Path, environment: tuple[str, ...]) -> dict[str, str]:
    try:
        digest, _ = _file_digest(path, 512 << 20)
        return {"sha256": digest, "version": _version(path, digest, environment)}
    except (OSError, UnicodeError) as error:
        raise ValueError("tensor compiler executable facts unavailable") from error


def _overrides() -> tuple[dict[str, str], Path, tuple[Path, ...]]:
    environment = {name: os.environ.get(name, "") for name in _ENVIRONMENT}
    flags = [
        *shlex.split(environment["NVCC_PREPEND_FLAGS"]),
        *shlex.split(environment["NVCC_APPEND_FLAGS"]),
    ]
    host = environment["NVCC_CCBIN"] or "gcc"
    includes: list[Path] = []
    index = 0
    while index < len(flags):
        flag = flags[index]
        key, separator, value = flag.partition("=")
        if (
            key in _UNSAFE_OPTIONS
            or flag.startswith("@")
            or any(
                flag.startswith(prefix) for prefix in ("-MF", "-MT", "-optf", "-odir")
            )
            or (flag.startswith("-o") and not flag.startswith("--"))
        ):
            raise ValueError(f"unsafe NVCC identity-probe override: {key}")
        if key in {
            "-Xcompiler",
            "--compiler-options",
            "-Xpreprocessor",
            "--preprocessor-options",
        }:
            if not separator:
                index += 1
                if index == len(flags):
                    raise ValueError(f"missing NVCC override argument: {key}")
                value = flags[index]
            flags[index + 1 : index + 1] = value.split(",")
        elif key in _INCLUDE_OPTIONS | _HOST_OPTIONS:
            if not separator:
                index += 1
                if index == len(flags):
                    raise ValueError(f"missing NVCC override argument: {key}")
                value = flags[index]
            if key in _HOST_OPTIONS:
                host = value
            else:
                includes.extend(Path(item).resolve() for item in value.split(","))
        elif flag.startswith("-I"):
            includes.extend(Path(item).resolve() for item in flag[2:].split(","))
        index += 1
    for name in ("CPATH", "CPLUS_INCLUDE_PATH"):
        if environment[name]:
            includes.extend(
                Path(item).resolve() for item in environment[name].split(os.pathsep)
            )
    host_path = Path(host)
    if host_path.is_dir():
        host = str(host_path / "gcc")
    resolved = shutil.which(host)
    if not resolved:
        raise ValueError("NVCC host compiler cannot be identified")
    return environment, Path(resolved).resolve(), tuple(includes)


def _dependency_paths(text: str, source: Path) -> tuple[Path, ...]:
    # Parse one Make rule, including escaped spaces/#/$ and line continuations.
    # Reject ambiguous output instead of silently omitting a dependency.
    if len(text) > 8 << 20:
        raise ValueError("tensor dependency output exceeds byte budget")
    text = text.replace("\\\r\n", "").replace("\\\n", "")
    target, separator, dependencies = text.partition(":")
    if (
        not separator
        or target.strip() != "tensor_identity"
        or "\n" in dependencies.strip()
    ):
        raise ValueError("unrecognized NVCC dependency output")
    words, word, index = [], [], 0
    while index < len(dependencies):
        char = dependencies[index]
        if char == "\\":
            index += 1
            if index == len(dependencies):
                raise ValueError("incomplete NVCC dependency escape")
            word.append(dependencies[index])
        elif char == "$":
            index += 1
            if index == len(dependencies) or dependencies[index] != "$":
                raise ValueError("unsupported Make expansion in NVCC dependencies")
            word.append("$")
        elif char in "#:":
            raise ValueError("ambiguous NVCC dependency path")
        elif char.isspace():
            if word:
                words.append("".join(word))
                word = []
        else:
            word.append(char)
        index += 1
    if word:
        words.append("".join(word))
    if not 1 <= len(words) <= _MAX_DEPENDENCIES:
        raise ValueError("tensor dependency count exceeds budget or is empty")
    paths = tuple(dict.fromkeys(Path(word).resolve() for word in words))
    if source.resolve() not in paths:
        raise ValueError("NVCC dependency output omits its source")
    return tuple(path for path in paths if path != source.resolve())


def _dependency_probe(
    compiler: CudaCompilerAdapter,
    options: tuple[str, ...],
    roots: tuple[tuple[str, Path], ...],
    *,
    cub: bool,
) -> dict[str, str]:
    timeout = compiler.compile_timeout
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("tensor dependency probe requires a finite positive timeout")
    timeout = min(30.0, timeout)
    with tempfile.TemporaryDirectory(prefix="tensor-dependencies-") as temporary:
        source, output = Path(temporary) / "identity.cu", Path(temporary) / "identity.d"
        source.write_text(
            '#include "cuda_graph_context.cuh"\n'
            + ("#include <cub/block/block_reduce.cuh>\n" if cub else "")
        )
        result = run_compiler(
            [
                str(compiler.nvcc),
                "-std=c++17",
                f"-arch={compiler.target.architecture}",
                "-O3",
                "-Xptxas=-v",
                "-Xcompiler=-fPIC",
                f"-I{asset_path('src/tensor')}",
                *options,
                "-M",
                "-MT",
                "tensor_identity",
                "-MF",
                str(output),
                str(source),
            ],
            timeout,
            label="NVCC dependency identity",
        )
        if result.returncode:
            raise ValueError("NVCC dependency probe failed: " + result.stderr)
        with output.open("rb") as stream:
            manifest = stream.read((8 << 20) + 1).decode("utf-8")
        paths = _dependency_paths(manifest, source)
        records, remaining = {}, _MAX_DEPENDENCY_BYTES
        for path in paths:
            logical = next(
                (
                    name + "/" + path.relative_to(root).as_posix()
                    for name, root in roots
                    if path.is_relative_to(root)
                ),
                None,
            )
            if logical is None or logical in records:
                raise ValueError("tensor dependency has no unique portable identity")
            digest, size = _file_digest(path, remaining)
            records[logical], remaining = digest, remaining - size
        if not records:
            raise ValueError("NVCC dependency output contains no headers")
        return records


def _dependencies(
    compiler: CudaCompilerAdapter,
    options: tuple[str, ...],
    roots: tuple[tuple[str, Path], ...],
    *,
    cub: bool,
) -> dict[str, str] | None:
    try:
        return _dependency_probe(compiler, options, roots, cub=cub)
    except (OSError, UnicodeError, ValueError) as error:
        # A missing/malformed manifest is never permission to reuse guessed
        # dependencies. Optional CUB remains unavailable with a stable identity;
        # its diagnostics can contain randomized temporary paths, so omit them.
        if cub:
            return None
        raise ValueError(
            "tensor runtime dependency probe failed: " + str(error)
        ) from error


def compiler_dependency_facts(
    compiler: CudaCompilerAdapter, options: tuple[str, ...]
) -> dict[str, Any]:
    """Refresh actual header selection; cache only executable-bound version probes.

    Both provider possibilities use the same context before and after binding.
    Missing CUB leaves the generated provider usable; selected CUB requires a
    successful closure probe at compile time. No native runtime or GPU is loaded.
    """
    environment, host, include_roots = _overrides()
    nvcc = compiler.nvcc.resolve()
    probe_environment = tuple(environment.values()) + (os.environ.get("PATH", ""),)
    tools = {
        "nvcc": _executable(nvcc, probe_environment),
        "ptxas": _executable(nvcc.with_name("ptxas"), probe_environment),
        "host": _executable(host, probe_environment),
    }
    roots = (
        ("native", asset_path("src/tensor").parent.resolve()),
        *((f"include-{index}", root) for index, root in enumerate(include_roots)),
        ("cuda", nvcc.parent.parent),
        ("host", host.parent.parent),
        ("system", Path("/usr")),
        ("system-lib", Path("/lib").resolve()),
    )
    return {
        "toolchain": {name: tools[name]["version"] for name in ("nvcc", "ptxas")},
        "host_compiler": tools["host"]["version"],
        "compiler_executables": {
            name: facts["sha256"] for name, facts in tools.items()
        },
        "compiler_environment": environment,
        "dependencies": {
            "runtime": _dependencies(compiler, options, roots, cub=False),
            "cub": _dependencies(compiler, options, roots, cub=True),
        },
    }
