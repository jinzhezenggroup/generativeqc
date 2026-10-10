"""Opt-in Linux GCC dependency collection for offline CPU AOT objects.

A GCC depfile observes headers, not the complete downstream toolchain. Reuse
requires a separately asserted, exact toolchain manifest; discovery never
upgrades that assertion. Collection alone does not qualify an existing object.
"""

from __future__ import annotations

import math
import platform
import re
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


@dataclass(frozen=True)
class CpuToolchainManifest:
    """Caller-owned trust boundary for inputs GCC cannot enumerate via -M.

    Every record has role ``toolchain-file``, an absolute file identity and its
    expected hash. ``complete`` asserts an immutable compilation snapshot covers
    all downstream executables, loaded libraries, specs/configuration and header
    search namespaces (including absent/shadowing headers and PCH). It must not
    be set from a guessed list or from the depfile. The collector verifies listed
    files and requires the actual driver, cc1 and assembler, but cannot prove the
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
    Empty rules, extra rules, unescaped comments and dangling escapes fail closed.
    """
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
                if count % 2 == 0:
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
    flags: tuple[str, ...] = ("-std=c99", "-O1", "-fPIC"),
    toolchain: CpuToolchainManifest | None = None,
    timeout: float = 60,
) -> CpuClosureResult:
    """Collect actual system/user headers and verify an explicit toolchain.

    Supports only native Linux ELF GCC C, plain object flags and absolute -I
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
    try:
        _validate_flags(flags)
    except (ValueError, TypeError):
        return CpuClosureResult(None, None, ("unsupported-cpu-flags",))
    executable = shutil.which(compiler, path=CPU_ENVIRONMENT["PATH"])
    if executable is None:
        return CpuClosureResult(None, None, ("compiler-not-found",))
    path = Path(executable).resolve()
    source = source_path.absolute()
    if path != _system_gcc():
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

    def query(arguments: list[str]) -> str:
        check_working_directory()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        result = run_compiler(
            [str(path), *arguments],
            remaining,
            label="CPU AOT closure",
            environment=dict(CPU_ENVIRONMENT),
        )
        check_working_directory()
        if result.timed_out:
            raise TimeoutError
        if result.returncode:
            raise ValueError("compiler-command-failed")
        return result.stdout.strip()

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
        with path.open("rb") as stream:
            magic = stream.read(4)
        if magic != b"\x7fELF":
            return failure("unsupported-compiler-wrapper")
        executable_hash = hash_input(path)
        version = query(["--version"])
        # __GNUC__ is also defined by clang; require GCC's driver banner/query.
        if "Free Software Foundation" not in version or not re.fullmatch(
            r"[0-9]+(?:\.[0-9]+)*", query(["-dumpfullversion"])
        ):
            return failure("unsupported-compiler")
        target = query(["-dumpmachine"])
        if not re.fullmatch(r"(?:x86_64|aarch64)-[^\s]*linux[^\s]*", target):
            return failure("unsupported-cpu-target")
        if target.split("-", 1)[0] != platform.machine():
            return failure("unsupported-cpu-target")
        effective_flags = (*flags, "-Werror=date-time")
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
        snapshot(manifest.files)
        required = {str(path)}
        downstream = {}
        for name in ("cc1", "as"):
            tool = query(["-print-prog-name=" + name])
            found = shutil.which(tool, path=CPU_ENVIRONMENT["PATH"])
            if found is None:
                return failure("downstream-tool-not-found")
            resolved = str(Path(found).resolve())
            required.add(resolved)
            downstream[name] = resolved
        observed = {str(Path(item.identity).resolve()) for item in manifest.files}
        specs = query(["-dumpspecs"])
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

        names = scan()
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
                "gcc-M",
                canonical_hash([item.to_payload() for item in headers]),
            ),
            CacheDependency(
                "cpu-toolchain-manifest",
                "explicit-caller-snapshot",
                canonical_hash([item.to_payload() for item in sorted(manifest.files)]),
            ),
            CacheDependency(
                "cpu-build-context",
                "gcc-native",
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
        if query(["--version"]) != version or query(["-dumpspecs"]) != specs:
            return failure("compiler-configuration-changed")
        if (
            canonical_hash(
                {"preprocessed": query([*effective_flags, "-E", str(source)])}
            )
            != preprocessed
        ):
            return failure("preprocessor-output-changed")
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
        # Transport the observed closure identity into a real GCC compile option
        # so those changes invalidate its command identity as well as our store.
        # Hash the unsalted payload once; the final key includes the returned flag
        # without any self-reference. Partial observations remain non-reusable.
        recipe["flags"].append("-frandom-seed=" + canonical_hash(closure.to_payload()))
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
