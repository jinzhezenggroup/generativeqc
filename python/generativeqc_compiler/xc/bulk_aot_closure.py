"""Opt-in native Linux dependency collection for offline CPU/CUDA AOT objects.

A GCC depfile observes headers, not the complete downstream toolchain. Reuse
requires a separately asserted, exact toolchain manifest; discovery never
upgrades that assertion. Collection alone does not qualify an existing object.
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
import re
import shlex
import shutil
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from generativeqc_compiler.common.compiler_process import run_compiler
from generativeqc_compiler.common.provenance import canonical_hash, file_hash

from .bulk_aot_cache import CacheClosure, CacheDependency, closure_from_probe_recipe

if TYPE_CHECKING:
    from typing import Any

    from .bulk_aot import SourceVariant

SCHEMA = "generativeqc.libxc-cpu-closure/v1"
# These directories are the entire child environment's executable search path.
# CPATH, GCC_EXEC_PREFIX, LD_PRELOAD, compiler-cache settings, etc. are not inherited.
CPU_ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
_SIMPLE_FLAGS = frozenset({"-std=c99", "-O0", "-O1", "-O2", "-O3", "-fPIC"})


def _system_gcc() -> Path:
    return Path("/usr/bin/gcc").resolve()


def _system_clang() -> Path:
    return Path("/usr/bin/clang").resolve()


def _system_gxx() -> Path:
    return Path("/usr/bin/g++").resolve()


@dataclass(frozen=True)
class CpuToolchainManifest:
    """Caller-owned trust boundary for inputs a C depfile cannot enumerate.

    Every record has role ``toolchain-file``, an absolute file identity and its
    expected hash. ``complete`` asserts an immutable compilation snapshot covers
    all downstream executables, loaded libraries, specs/configuration and header
    search namespaces (including absent/shadowing headers and PCH). It must not
    be set from a guessed list or from the depfile. The collector verifies listed
    files and requires the actual driver and selected executable jobs (GCC cc1/as
    or the admitted Clang integrated frontend/assembler), but cannot prove the
    caller's coverage assertion. An incomplete manifest is still diagnostic.
    """

    files: tuple[CacheDependency, ...] = ()
    complete: bool = False

    def __post_init__(self) -> None:
        if type(self.complete) is not bool or not isinstance(self.files, tuple):
            raise ValueError("manifest requires tuple files and bool complete")
        identities = []
        for item in self.files:
            if (
                not isinstance(item, CacheDependency)
                or item.role != "toolchain-file"
                or not Path(item.identity).is_absolute()
            ):
                raise ValueError("toolchain files require absolute file identities")
            identities.append(item.identity)
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate toolchain file identity")


@dataclass(frozen=True)
class CpuClosureResult:
    """Observed recipe/closure plus deterministic fail-closed diagnostics."""

    closure: CacheClosure | None
    recipe: dict[str, Any] | None
    reasons: tuple[str, ...]
    headers: tuple[CacheDependency, ...] = ()


def parse_gcc_dependencies(text: str) -> tuple[str, ...]:
    """Parse one fixed-target GCC -M rule; reject unsupported make syntax.

    GCC escapes spaces/tabs and #, doubles dollars, and preserves backslashes
    except for GNU make's 2N+1 backslash quoting before whitespace.
    The target is forced to ``closure`` so colons in file paths are unambiguous.
    Ambiguous terminal backslashes, empty/extra rules, unescaped comments and
    dangling escapes fail closed, even if a misparsed decoy file exists.
    """
    # A filename's terminal backslash must not masquerade as a continuation.
    # GCC inserts an unescaped separator before genuine rule continuations.
    for continuation in re.finditer(r"\\\r?\n", text):
        separator = continuation.start() - 1
        if separator < 0 or text[separator] not in " \t":
            raise ValueError("ambiguous trailing dependency backslashes")
        escapes = 0
        while separator > 0 and text[separator - 1] == "\\":
            separator -= 1
            escapes += 1
        if escapes % 2:
            raise ValueError("ambiguous trailing dependency backslashes")
    text = re.sub(r"\\\r?\n", "", text)
    if not text.startswith("closure:"):
        raise ValueError("unexpected dependency target")
    body = text[len("closure:") :].rstrip("\r\n")
    if "\n" in body or "\r" in body:
        raise ValueError("multiple dependency rules")
    words: list[str] = []
    word = ""
    index = 0
    while index < len(body):
        char = body[index]
        if char == "\\":
            begin = index
            while index < len(body) and body[index] == "\\":
                index += 1
            if index == len(body):
                raise ValueError("dangling dependency escape")
            count = index - begin
            escaped = body[index]
            if escaped in " \t":
                # GCC does not quote terminal filename backslashes before a
                # dependency separator. Even a real merged decoy file cannot
                # disambiguate that output from an escaped filename space.
                if count % 2 == 0 or (
                    index + 1 < len(body) and body[index + 1] in "/ \t"
                ):
                    raise ValueError("ambiguous trailing dependency backslashes")
                word += "\\" * (count // 2) + escaped
            elif escaped == "#":
                word += "\\" * (count - 1) + "#"
            else:
                word += "\\" * count
                # The following character still needs ordinary make decoding.
                continue
        elif char == "$":
            index += 1
            if index == len(body) or body[index] != "$":
                raise ValueError("unsupported make expansion")
            word += "$"
        elif char == "#":
            raise ValueError("unescaped dependency comment")
        elif char in " \t":
            if word:
                words.append(word)
                word = ""
        else:
            word += char
        index += 1
    if word:
        words.append(word)
    if not words:
        raise ValueError("empty dependency rule")
    return tuple(sorted(set(words)))


def _validate_flags(flags: tuple[str, ...]) -> None:
    if not isinstance(flags, tuple):
        raise TypeError("flags must be a tuple")
    for flag in flags:
        if isinstance(flag, str) and flag in _SIMPLE_FLAGS:
            continue
        if isinstance(flag, str) and flag.startswith("-I"):
            directory = Path(flag[2:])
            if directory.is_absolute() and directory.is_dir():
                continue
        raise ValueError("unsupported CPU compiler flag")


def collect_cpu_closure(
    variant: SourceVariant,
    source_path: Path,
    *,
    compiler: str = "gcc",
    compiler_family: str = "gcc",
    flags: tuple[str, ...] = ("-std=c99", "-O1", "-fPIC"),
    toolchain: CpuToolchainManifest | None = None,
    timeout: float = 60,
) -> CpuClosureResult:
    """Collect actual system/user headers and verify an explicit toolchain.

    Supports native Linux ELF GCC C by default; ``compiler_family="clang"``
    explicitly admits system Clang C with disabled default configuration and an
    observed integrated frontend/assembler job. Supports plain flags and absolute -I
    directories, under ``CPU_ENVIRONMENT``. Wrappers, plugins, response files,
    alternate targets/sysroots, inherited environments and other compilers fail
    closed. The recipe includes the physical source path: __FILE__ and quoted
    include resolution make it part of object identity. Time macros are errors.

    The caller must hold the asserted immutable snapshot through compilation and
    collection, use the returned flags/compiler/environment/working directory,
    and check
    matching reusable keys before/after compiling an object. A previously built
    ``compile_probe`` object cannot be retroactively qualified by this function.
    Nothing here compiles, stores, loads, or activates an object at import/runtime.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    manifest = toolchain or CpuToolchainManifest()
    if not isinstance(manifest, CpuToolchainManifest):
        raise TypeError("toolchain must be CpuToolchainManifest")
    if sys.platform != "linux" or variant.backend != "cpu":
        return CpuClosureResult(None, None, ("unsupported-cpu-platform-or-backend",))
    if compiler_family not in {"gcc", "clang"}:
        return CpuClosureResult(None, None, ("unsupported-compiler-family",))
    try:
        _validate_flags(flags)
    except (ValueError, TypeError):
        return CpuClosureResult(None, None, ("unsupported-cpu-flags",))
    executable = shutil.which(compiler, path=CPU_ENVIRONMENT["PATH"])
    if executable is None:
        return CpuClosureResult(None, None, ("compiler-not-found",))
    path = Path(executable).resolve()
    source = source_path.absolute()
    is_clang = compiler_family == "clang"
    if path != (_system_clang() if is_clang else _system_gcc()):
        return CpuClosureResult(None, None, ("unsupported-compiler-wrapper",))
    if source.suffix != ".c":
        return CpuClosureResult(None, None, ("unsupported-source-language",))
    deadline = time.monotonic() + timeout
    working_directory = str(Path.cwd())
    recipe = None
    headers: tuple[CacheDependency, ...] = ()
    dependencies: tuple[CacheDependency, ...] = ()
    target = "unverified-native-cpu"

    def failure(reason: str) -> CpuClosureResult:
        closure = None
        if recipe is not None:
            closure = closure_from_probe_recipe(
                recipe,
                backend="cpu",
                target=target,
                dependencies=dependencies,
                complete=False,
            )
        return CpuClosureResult(closure, recipe, (reason,), headers)

    def check_working_directory() -> None:
        if str(Path.cwd()) != working_directory:
            raise ValueError("working-directory-changed")

    def check_clang_selection() -> None:
        if not is_clang:
            return
        selected = shutil.which(compiler, path=CPU_ENVIRONMENT["PATH"])
        if (
            selected is None
            or Path(selected).resolve() != path
            or _system_clang() != path
        ):
            raise ValueError("compiler-selection-changed")

    def query(arguments: list[str], *, diagnostic: bool = False) -> str:
        check_working_directory()
        check_clang_selection()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        if is_clang and "--no-default-config" not in arguments:
            arguments = ["--no-default-config", *arguments]
        result = run_compiler(
            [str(path), *arguments],
            remaining,
            label="CPU AOT closure",
            environment=dict(CPU_ENVIRONMENT),
        )
        check_working_directory()
        check_clang_selection()
        if result.timed_out:
            raise TimeoutError
        if result.returncode:
            raise ValueError("compiler-command-failed")
        # Remove record terminators only: GCC quotes a filename's terminal
        # space/tab, and trimming that whitespace destroys dependency identity.
        return (result.stderr if diagnostic else result.stdout).rstrip("\r\n")

    def snapshot(files: tuple[CacheDependency, ...]) -> None:
        for item in files:
            if hash_input(Path(item.identity)) != item.content_sha256:
                raise ValueError("dependency-changed")

    def hash_input(input_path: Path) -> str:
        if time.monotonic() >= deadline:
            raise TimeoutError
        if not stat.S_ISREG(input_path.stat().st_mode):
            raise ValueError("nonregular-dependency-file")
        digest = file_hash(input_path)
        if time.monotonic() >= deadline:
            raise TimeoutError
        return digest

    def check_time_macros(input_path: Path) -> None:
        # Reject even commented spellings conservatively; never key an object
        # on wall-clock-dependent expansions or rely on a suppressible warning.
        with input_path.open("rb") as stream:
            carry = b""
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                data = carry + chunk
                if re.search(rb"__(?:DATE|TIME|TIMESTAMP)__", data):
                    raise ValueError("unsupported-time-macro")
                carry = data[-32:]

    try:
        # A named FIFO can satisfy executable lookup; never open a nonregular
        # Clang input while waiting for ELF bytes outside the process deadline.
        executable_hash = hash_input(path) if is_clang else None
        with path.open("rb") as stream:
            magic = stream.read(4)
        if magic != b"\x7fELF":
            return failure("unsupported-compiler-wrapper")
        if executable_hash is None:
            executable_hash = hash_input(path)
        version = query(["--version"])
        # __GNUC__ is also defined by clang; require GCC's driver banner/query.
        if is_clang:
            if not re.search(r"\bclang version [0-9]+\.", version):
                return failure("unsupported-compiler")
        elif "Free Software Foundation" not in version or not re.fullmatch(
            r"[0-9]+(?:\.[0-9]+)*", query(["-dumpfullversion"])
        ):
            return failure("unsupported-compiler")
        target = query(["-dumpmachine"])
        if not re.fullmatch(r"(?:x86_64|aarch64)-[^\s]*linux[^\s]*", target):
            return failure("unsupported-cpu-target")
        if target.split("-", 1)[0] != platform.machine():
            return failure("unsupported-cpu-target")
        effective_flags = (*flags, "-Werror=date-time")
        if is_clang:
            # Do not enter Clang's user/system/default configuration domain.
            # Explicitly select its integrated assembler; verify the actual job.
            effective_flags = (
                "--no-default-config",
                "-fintegrated-as",
                *effective_flags,
            )
        if hash_input(source) != variant.source_sha256:
            return failure("source-mismatch")
        check_time_macros(source)
        recipe = {
            "emission_identity": variant.emission_identity,
            "translation_unit_sha256": variant.source_sha256,
            "compiler": {"executable_sha256": executable_hash, "version": version},
            "flags": list(effective_flags),
            "working_directory": working_directory,
        }
        if is_clang:
            recipe["compiler"]["executable"] = str(path)
        snapshot(manifest.files)
        required = {str(path)}
        downstream = {}
        for name in () if is_clang else ("cc1", "as"):
            tool = query(["-print-prog-name=" + name])
            found = shutil.which(tool, path=CPU_ENVIRONMENT["PATH"])
            if found is None:
                return failure("downstream-tool-not-found")
            resolved = str(Path(found).resolve())
            required.add(resolved)
            downstream[name] = resolved
        observed = {str(Path(item.identity).resolve()) for item in manifest.files}

        def clang_context() -> dict[str, Any]:
            resource = query(["--no-default-config", "-print-resource-dir"])
            if not Path(resource).is_absolute() or not Path(resource).is_dir():
                raise ValueError("unsupported-clang-resource-directory")
            plan = query(
                [*effective_flags, "-###", "-c", str(source), "-o", "/dev/null"],
                diagnostic=True,
            )
            # -### is a diagnostic protocol, not a shell script to execute.
            # Admit exactly one quoted cc1 object job in the driver executable.
            lines = [
                line for line in plan.splitlines() if line.lstrip().startswith('"')
            ]
            if len(lines) != 1:
                raise ValueError("unsupported-clang-job-model")
            job = shlex.split(lines[0])

            def selected(option: str) -> str | None:
                if job.count(option) != 1:
                    return None
                index = job.index(option) + 1
                return job[index] if index < len(job) else None

            if (
                len(job) < 2
                or not Path(job[0]).is_absolute()
                or Path(job[0]).resolve() != path
                or job[1] != "-cc1"
                or "-emit-obj" not in job
                or "-no-integrated-as" in job
                or "-fno-integrated-as" in job
                or selected("-triple") != target
                or selected("-resource-dir") != resource
                or selected("-x") != "c"
                or any(
                    arg in job for arg in ("-load", "-plugin", "-isysroot", "-fmodules")
                )
                or job[-1] != str(source)
            ):
                raise ValueError("unsupported-clang-job-model")
            return {
                "resource_directory": resource,
                "object_job": job,
                "driver_plan": plan,
            }

        specs = clang_context() if is_clang else query(["-dumpspecs"])
        # -M alone cannot observe __has_include branches which include no file,
        # or diagnostics/time macro expansions. Bind actual preprocessor output
        # too, without confusing it with the downstream toolchain closure.
        preprocessed = canonical_hash(
            {"preprocessed": query([*effective_flags, "-E", str(source)])}
        )

        def scan() -> tuple[str, ...]:
            names = parse_gcc_dependencies(
                query(
                    [
                        *effective_flags,
                        "-M",
                        "-MT",
                        "closure",
                        str(source),
                    ]
                )
            )
            if any(not Path(name).is_absolute() for name in names):
                raise ValueError("relative-header-path")
            if str(source) not in names:
                raise ValueError("source-missing-from-dependencies")
            return names

        def clang_headers() -> set[str]:
            trace = query([*effective_flags, "-E", "-H", str(source)], diagnostic=True)
            traced = set()
            for line in trace.splitlines():
                match = re.fullmatch(r"\.+ (.+)", line)
                if match is None:
                    continue
                header_path = match[1]
                # Clang's make output normalizes literal backslashes to '/'
                # and leaves filename tabs unquoted. A real normalized/split
                # decoy is not evidence of the header actually opened. Admit
                # only trace paths with an unambiguous make representation.
                if (
                    "\\" in header_path
                    or any(ord(char) < 32 or ord(char) == 127 for char in header_path)
                    or header_path.endswith(" ")
                    or not Path(header_path).is_absolute()
                ):
                    raise ValueError("unsupported-clang-dependency-path")
                traced.add(header_path)
            return traced

        names = scan()
        traced = clang_headers() if is_clang else set()
        if is_clang and traced != set(names) - {str(source)}:
            return failure("clang-header-trace-mismatch")
        headers = tuple(
            CacheDependency("header", name, hash_input(Path(name)))
            for name in names
            if name != str(source)
        )
        for header in headers:
            check_time_macros(Path(header.identity))
        dependencies = (
            *headers,
            *manifest.files,
            CacheDependency(
                "system-header-manifest",
                "clang-M" if is_clang else "gcc-M",
                canonical_hash([item.to_payload() for item in headers]),
            ),
            CacheDependency(
                "cpu-toolchain-manifest",
                "explicit-caller-snapshot",
                canonical_hash([item.to_payload() for item in sorted(manifest.files)]),
            ),
            CacheDependency(
                "cpu-build-context",
                "clang-native" if is_clang else "gcc-native",
                canonical_hash(
                    {
                        "schema": SCHEMA,
                        "compiler": str(path),
                        "source": str(source),
                        "working_directory": working_directory,
                        "environment": CPU_ENVIRONMENT,
                        "target": target,
                        "specs": specs,
                        "downstream": downstream,
                        "preprocessed_sha256": preprocessed,
                    }
                ),
            ),
        )
        if scan() != names:
            return failure("dependency-set-changed")
        snapshot((*headers, *manifest.files))
        if (
            hash_input(source) != variant.source_sha256
            or hash_input(path) != executable_hash
        ):
            return failure("source-or-compiler-changed")
        if (
            query(["--version"]) != version
            or (clang_context() if is_clang else query(["-dumpspecs"])) != specs
        ):
            return failure("compiler-configuration-changed")
        if (
            canonical_hash(
                {"preprocessed": query([*effective_flags, "-E", str(source)])}
            )
            != preprocessed
        ):
            return failure("preprocessor-output-changed")
        if is_clang and clang_headers() != traced:
            return failure("clang-header-trace-changed")
        for name, resolved in downstream.items():
            found = shutil.which(
                query(["-print-prog-name=" + name]), path=CPU_ENVIRONMENT["PATH"]
            )
            if found is None or str(Path(found).resolve()) != resolved:
                return failure("downstream-selection-changed")
        snapshot((*headers, *manifest.files))
        if (
            hash_input(source) != variant.source_sha256
            or hash_input(path) != executable_hash
        ):
            return failure("source-or-compiler-changed")
        reasons = []
        check_working_directory()
        check_clang_selection()
        if not manifest.complete:
            reasons.append("toolchain-manifest-incomplete")
        if not required <= observed:
            reasons.append("downstream-toolchain-not-covered")
        closure = closure_from_probe_recipe(
            recipe,
            backend="cpu",
            target=target,
            dependencies=dependencies,
            complete=not reasons,
        )
        # A compiler cache may not enumerate cc1/config/opaque assembler inputs.
        # Transport the observed closure identity into a real driver option
        # so those changes invalidate its command identity as well as our store.
        # Hash the unsalted payload once; the final key includes the returned flag
        # without any self-reference. Partial observations remain non-reusable.
        salt = canonical_hash(closure.to_payload())
        recipe["flags"].append(
            # No debug flags are admitted. This Clang option supplies command
            # identity without injecting a macro into scientific C expressions.
            "-fdebug-compilation-dir=generativeqc-closure-" + salt
            if is_clang
            else "-frandom-seed=" + salt
        )
        closure = closure_from_probe_recipe(
            recipe,
            backend="cpu",
            target=target,
            dependencies=dependencies,
            complete=not reasons,
        )
        return CpuClosureResult(closure, recipe, tuple(reasons), headers)
    except TimeoutError:
        return failure("collection-timed-out")
    except OSError:
        return failure("dependency-file-unavailable")
    except (ValueError, UnicodeError) as error:
        return failure(str(error))


@dataclass(frozen=True)
class CudaToolchainManifest(CpuToolchainManifest):
    """Explicit immutable NVCC/host/device/header/search-namespace snapshot.

    The inherited file records bind exact bytes. ``complete`` also asserts all
    loaded libraries, configuration, downstream tools, absent/shadowing/PCH
    namespaces and opaque inputs are covered and remain immutable through the
    pre/object/post bracket. Neither dryrun, depfiles nor ldd prove coverage.
    """


@dataclass(frozen=True)
class CudaClosureResult(CpuClosureResult):
    """Observed CUDA recipe and closure; reasons prevent reusable identity."""


_CUDA_FLAGS = frozenset({"-std=c++17", "-O0", "-O1", "-O2", "-O3", "--fmad=false"})
_CUDA_PATH = re.compile(r"/[A-Za-z0-9_./+\-]+")
_CUDA_ASSIGNMENTS = frozenset(
    {
        "_HERE_",
        "_NVVM_BRANCH_",
        "_SPACE_",
        "_CUDART_",
        "_THERE_",
        "_TARGET_SIZE_",
        "_TARGET_DIR_",
        "TOP",
        "CICC_PATH",
        "NVVMIR_LIBRARY_DIR",
        "LD_LIBRARY_PATH",
        "PATH",
        "INCLUDES",
        "LIBRARIES",
        "CUDAFE_FLAGS",
        "PTXAS_FLAGS",
    }
)


def _cuda_path(path: Path) -> None:
    if not path.is_absolute() or _CUDA_PATH.fullmatch(str(path)) is None:
        raise ValueError("unsupported-cuda-path-spelling")


def _nvcc_dependencies(text: str) -> tuple[str, ...]:
    # NVCC 12.9 emits an extra target separator space. Preserve the GCC parser
    # contract; normalize only this fixed diagnostic target, never filenames.
    if text.startswith("closure :"):
        text = "closure:" + text[len("closure :") :]
    names = parse_gcc_dependencies(text)
    for name in names:
        _cuda_path(Path(name))
    return names


def _cuda_jobs(plan: str, nvcc: Path, host: Path) -> tuple[str, ...]:
    """Read the admitted 12.9 diagnostic model without executing shell text."""
    device = {
        (nvcc.parent / ".." / "nvvm" / "bin" / "cicc").resolve(): "cicc",
        (nvcc.parent / "cudafe++").resolve(): "cudafe++",
        (nvcc.parent / "ptxas").resolve(): "ptxas",
        (nvcc.parent / "fatbinary").resolve(): "fatbinary",
    }
    jobs = []
    assignments = {}
    for line in plan.splitlines():
        if not line.startswith("#$ "):
            raise ValueError("unsupported-nvcc-diagnostic")
        body = line[3:]
        assignment = re.match(r"([A-Za-z_][A-Za-z_0-9]*)=", body)
        if assignment:
            if (
                jobs
                or assignment[1] not in _CUDA_ASSIGNMENTS
                or any(marker in body for marker in ("`", "$(", "\n", ";", "&&", "||"))
            ):
                raise ValueError("unsupported-nvcc-configuration")
            assignments[assignment[1]] = body[assignment.end() :]
            continue
        words = shlex.split(body)
        if not words or any(
            marker in body for marker in ("`", ";", "&&", "||", "|", "<", ">")
        ):
            raise ValueError("unsupported-nvcc-job-model")
        command = words[0]
        if command == "$CICC_PATH/cicc":
            command = str(nvcc.parent / ".." / "nvvm" / "bin" / "cicc")
        elif command in {"cudafe++", "ptxas", "fatbinary"}:
            command = str(nvcc.parent / command)
        if "$" in command or any("$" in word for word in words[1:]):
            raise ValueError("unsupported-nvcc-job-model")
        program = Path(command)
        if not program.is_absolute():
            raise ValueError("unsupported-nvcc-job-model")
        program = program.resolve()
        if program != host and program not in device:
            raise ValueError("unsupported-nvcc-job-model")
        if any(
            word.startswith(("@", "-fplugin", "--options-file", "-isysroot"))
            or word in {"-rdc=true", "-dc", "-dlink", "-dlto"}
            for word in words[1:]
        ):
            raise ValueError("unsupported-nvcc-job-model")
        jobs.append(str(program))
    # Whole-program object: host preprocessing, device frontend/assembler,
    # fatbinary embedding and host object compilation; no linker/nvlink job.
    expected_jobs = (
        str(host),
        str((nvcc.parent / "cudafe++").resolve()),
        str(host),
        str((nvcc.parent / "../nvvm/bin/cicc").resolve()),
        str((nvcc.parent / "ptxas").resolve()),
        str((nvcc.parent / "fatbinary").resolve()),
        str(host),
    )
    if tuple(jobs) != expected_jobs:
        raise ValueError("unsupported-nvcc-job-model")
    # Only the stock 12.9 whole-program diagnostic environment is admitted.
    # No shell expansion/evaluation of these values takes place here.
    top = str(nvcc.parent / "..")
    target_dir = "targets/" + platform.machine() + "-linux"
    expected_assignments = {
        "_NVVM_BRANCH_": "nvvm",
        "_SPACE_": " ",
        "_CUDART_": "cudart",
        "_HERE_": str(nvcc.parent),
        "_THERE_": str(nvcc.parent),
        "_TARGET_SIZE_": "",
        "_TARGET_DIR_": target_dir,
        "TOP": top,
        "CICC_PATH": top + "/nvvm/bin",
        "NVVMIR_LIBRARY_DIR": top + "/nvvm/libdevice",
        "LD_LIBRARY_PATH": top + "/lib:",
        "PATH": top + "/nvvm/bin:" + str(nvcc.parent) + ":" + CPU_ENVIRONMENT["PATH"],
        "INCLUDES": '"-I' + top + "/" + target_dir + '/include"',
        "LIBRARIES": '"-L'
        + top
        + "/"
        + target_dir
        + '/lib/stubs" "-L'
        + top
        + "/"
        + target_dir
        + '/lib"',
        "CUDAFE_FLAGS": "",
        "PTXAS_FLAGS": "",
    }
    if {k: v.strip() for k, v in assignments.items()} != {
        k: v.strip() for k, v in expected_assignments.items()
    }:
        raise ValueError("unsupported-nvcc-configuration")
    return tuple(jobs)


def collect_cuda_closure(
    variant: SourceVariant,
    source_path: Path,
    *,
    compiler: str,
    target: str,
    host_compiler: str = "/usr/bin/g++",
    flags: tuple[str, ...] = ("-std=c++17", "-O1", "--fmad=false"),
    toolchain: CudaToolchainManifest | None = None,
    timeout: float = 60,
) -> CudaClosureResult:
    """Observe an explicit Linux ELF NVCC 12.9 whole-program sm_80 object recipe.

    Accept only the native system G++ host driver and bounded plain flags/absolute
    include paths. The caller must compile with the returned canonical driver,
    flags, environment and cwd under its immutable complete snapshot, and require
    equal reusable pre/post keys. No object is compiled, stored or retroactively
    qualified here. Dryrun jobs are diagnostics, never executable build scripts.
    Other releases/targets, wrappers, RDC/LTO, response/plugin/sysroot domains and
    ambiguous filenames fail closed. Ordinary runtime/import never calls NVCC.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    manifest = toolchain or CudaToolchainManifest()
    if not isinstance(manifest, CudaToolchainManifest):
        raise TypeError("toolchain must be CudaToolchainManifest")
    if (
        sys.platform != "linux"
        or platform.machine() != "x86_64"
        or variant.backend != "cuda"
    ):
        return CudaClosureResult(None, None, ("unsupported-cuda-platform-or-backend",))
    if target != "sm_80":
        return CudaClosureResult(None, None, ("unsupported-cuda-target",))
    deadline = time.monotonic() + timeout
    cwd = str(Path.cwd())
    source = source_path.absolute()
    entry = Path(compiler)
    host_entry = Path(host_compiler)
    recipe = None
    headers: tuple[CacheDependency, ...] = ()
    dependencies: tuple[CacheDependency, ...] = ()
    selections: dict[Path, Path] = {}
    identities: dict[Path, str] = {}
    prefixes: dict[Path, bytes] = {}
    macro_inputs = {source}

    def failure(reason: str) -> CudaClosureResult:
        closure = (
            None
            if recipe is None
            else closure_from_probe_recipe(
                recipe,
                backend="cuda",
                target=target,
                dependencies=dependencies,
                complete=False,
            )
        )
        return CudaClosureResult(closure, recipe, (reason,), headers)

    def check_context() -> None:
        if str(Path.cwd()) != cwd:
            raise ValueError("working-directory-changed")
        for selected, resolved in selections.items():
            if selected.resolve() != resolved:
                raise ValueError("dependency-selection-changed")

    def digest(path: Path) -> str:
        check_context()
        if time.monotonic() >= deadline:
            raise TimeoutError
        _cuda_path(path)
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("nonregular-dependency-file")
        selections.setdefault(path, path.resolve())
        # A regular input can be replaced with a FIFO between stat and open.
        # Never wait for a writer outside the finite collection deadline.
        descriptor = os.open(
            path.resolve(), os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
        )
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("nonregular-dependency-file")
            checksum = hashlib.sha256()
            prefix = b""
            carry = b""
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                if time.monotonic() >= deadline:
                    raise TimeoutError
                checksum.update(chunk)
                prefix = (prefix + chunk)[:4]
                if path in macro_inputs and re.search(
                    rb"__(?:DATE|TIME|TIMESTAMP)__", carry + chunk
                ):
                    raise ValueError("unsupported-time-macro")
                carry = chunk[-32:]
            value = checksum.hexdigest()
            prefixes[path] = prefix
        if time.monotonic() >= deadline:
            raise TimeoutError
        check_context()
        return value

    def remember(path: Path) -> str:
        value = digest(path)
        identities[path] = value
        return value

    def snapshot() -> None:
        for path, value in identities.items():
            if digest(path) != value:
                raise ValueError("dependency-changed")
        for item in manifest.files:
            if digest(Path(item.identity)) != item.content_sha256:
                raise ValueError("dependency-changed")

    def query(driver: Path, arguments: list[str], diagnostic: bool = False) -> str:
        check_context()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        result = run_compiler(
            [str(driver), *arguments],
            remaining,
            label="CUDA AOT closure",
            environment=dict(CPU_ENVIRONMENT),
        )
        check_context()
        if result.timed_out:
            raise TimeoutError
        if result.returncode:
            raise ValueError("compiler-command-failed")
        return (result.stderr if diagnostic else result.stdout).rstrip("\r\n")

    try:
        _cuda_path(Path(cwd))
        _cuda_path(source)
        _cuda_path(entry)
        _cuda_path(host_entry)
        if source.suffix != ".cu" or source.resolve() != source:
            return failure("unsupported-source-language-or-selection")
        if not isinstance(flags, tuple):
            return failure("unsupported-cuda-flags")
        for flag in flags:
            if isinstance(flag, str) and flag in _CUDA_FLAGS:
                continue
            if isinstance(flag, str) and flag.startswith("-I"):
                directory = Path(flag[2:])
                _cuda_path(directory)
                if directory.is_dir():
                    selections[directory] = directory.resolve()
                    continue
            return failure("unsupported-cuda-flags")
        nvcc = entry.resolve()
        host = host_entry.resolve()
        if host != _system_gxx() or nvcc.name != "nvcc":
            return failure("unsupported-compiler-wrapper")
        for path in (entry, host_entry, nvcc, host):
            remember(path)
            if prefixes[path] != b"\x7fELF":
                return failure("unsupported-compiler-wrapper")
        version = query(nvcc, ["--version"])
        if not re.search(r"release 12\.9, V12\.9\.[0-9]+", version):
            return failure("unsupported-nvcc-version")
        help_text = query(nvcc, ["--help"])
        if any(
            option not in help_text
            for option in (
                "--frandom-seed",
                "--dryrun",
                "--generate-dependencies",
            )
        ):
            return failure("unsupported-nvcc-options")
        host_version = query(host, ["--version"])
        host_target = query(host, ["-dumpmachine"])
        if (
            "Free Software Foundation" not in host_version
            or not re.fullmatch(r"(?:x86_64|aarch64)-[^\s]*linux[^\s]*", host_target)
            or host_target.split("-", 1)[0] != platform.machine()
        ):
            return failure("unsupported-host-compiler")
        host_specs = query(host, ["-dumpspecs"])
        required = {str(nvcc), str(host)}
        for name in ("cc1plus", "as"):
            selected = query(host, ["-print-prog-name=" + name])
            found = shutil.which(selected, path=CPU_ENVIRONMENT["PATH"])
            if found is None:
                return failure("downstream-tool-not-found")
            path = Path(found)
            remember(path)
            if prefixes[path] != b"\x7fELF":
                return failure("unsupported-downstream-wrapper")
            required.add(str(path.resolve()))
        profile = nvcc.parent / "nvcc.profile"
        remember(profile)
        required.add(str(profile.resolve()))
        libdevice = (nvcc.parent / ".." / "nvvm/libdevice/libdevice.10.bc").resolve()
        remember(libdevice)
        required.add(str(libdevice))
        effective = [
            "--compiler-bindir=" + str(host),
            "--gpu-architecture=" + target,
            *flags,
            "-Xcompiler=-Werror=date-time",
        ]
        if remember(source) != variant.source_sha256:
            return failure("source-mismatch")
        recipe = {
            "emission_identity": variant.emission_identity,
            "translation_unit_sha256": variant.source_sha256,
            "compiler": {
                "executable": str(nvcc),
                "executable_sha256": identities[nvcc],
                "version": version,
            },
            "flags": effective,
            "working_directory": cwd,
            "environment": dict(CPU_ENVIRONMENT),
            "source": str(source),
        }
        snapshot()

        def context() -> dict[str, Any]:
            plan = query(
                nvcc,
                [
                    *effective,
                    "--dryrun",
                    "--keep",
                    "-c",
                    str(source),
                    "-o",
                    "/dev/null",
                ],
                True,
            )
            jobs = _cuda_jobs(plan, nvcc, host)
            for job in jobs:
                path = Path(job)
                if path not in identities:
                    remember(path)
                if prefixes[path] != b"\x7fELF":
                    raise ValueError("unsupported-downstream-wrapper")
                required.add(job)
            return {
                "plan": plan,
                "jobs": jobs,
                "host_version": host_version,
                "host_target": host_target,
                "host_specs": host_specs,
                "preprocessed": query(nvcc, [*effective, "-E", str(source)]),
            }

        observed_context = context()

        def scan() -> tuple[str, ...]:
            names = _nvcc_dependencies(
                query(nvcc, [*effective, "-M", "-MT", "closure", str(source)])
            )
            if str(source) not in names:
                raise ValueError("source-dependency-missing")
            return names

        names = scan()

        def trace_headers() -> tuple[str, ...]:
            diagnostic = query(
                nvcc,
                [*effective, "-Xcompiler=-H", "-M", "-MT", "closure", str(source)],
                True,
            )
            traced = set()
            for line in diagnostic.splitlines():
                match = re.fullmatch(r"\.+ (.+)", line)
                if match is not None:
                    _cuda_path(Path(match[1]))
                    traced.add(match[1])
            # NVCC's implicit runtime census is not fully exposed by host -H.
            # Every witnessed user/device header must nevertheless be in -M,
            # with unambiguous original spelling even when decoys exist.
            if not traced <= set(names) - {str(source)}:
                raise ValueError("nvcc-header-trace-mismatch")
            return tuple(sorted(traced))

        traced = trace_headers()
        macro_inputs.update(Path(name) for name in names)
        headers = tuple(
            CacheDependency("header", name, remember(Path(name)))
            for name in names
            if name != str(source)
        )
        dependencies = (
            *headers,
            *manifest.files,
            CacheDependency(
                "system-header-manifest",
                "nvcc-12.9-M",
                canonical_hash([item.to_payload() for item in headers]),
            ),
            CacheDependency(
                "cuda-device-toolchain-manifest",
                "explicit-caller-snapshot",
                canonical_hash([item.to_payload() for item in sorted(manifest.files)]),
            ),
            CacheDependency(
                "cuda-build-context",
                "nvcc-12.9-native",
                canonical_hash(
                    {
                        "schema": "generativeqc.libxc-cuda-closure/v1",
                        "source": str(source),
                        "cwd": cwd,
                        "environment": CPU_ENVIRONMENT,
                        "context": observed_context,
                        "traced_headers": traced,
                        "selected": {str(k): str(v) for k, v in selections.items()},
                        "files": {str(k): v for k, v in identities.items()},
                    }
                ),
            ),
        )
        snapshot()
        if (
            scan() != names
            or context() != observed_context
            or trace_headers() != traced
        ):
            return failure("compiler-context-changed")
        if (
            query(nvcc, ["--version"]) != version
            or query(host, ["--version"]) != host_version
        ):
            return failure("compiler-version-changed")
        if query(host, ["-dumpspecs"]) != host_specs:
            return failure("host-configuration-changed")
        snapshot()
        observed = {str(Path(item.identity).resolve()) for item in manifest.files}
        reasons = []
        if not manifest.complete:
            reasons.append("toolchain-manifest-incomplete")
        if not required <= observed:
            reasons.append("downstream-toolchain-not-covered")
        closure = closure_from_probe_recipe(
            recipe,
            backend="cuda",
            target=target,
            dependencies=dependencies,
            complete=not reasons,
        )
        effective.append("--frandom-seed=" + canonical_hash(closure.to_payload()))
        closure = closure_from_probe_recipe(
            recipe,
            backend="cuda",
            target=target,
            dependencies=dependencies,
            complete=not reasons,
        )
        return CudaClosureResult(closure, recipe, tuple(reasons), headers)
    except TimeoutError:
        return failure("collection-timed-out")
    except OSError:
        return failure("dependency-file-unavailable")
    except (ValueError, UnicodeError) as error:
        return failure(str(error))
