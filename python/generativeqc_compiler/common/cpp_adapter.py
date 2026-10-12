"""Explicit CPU C++ compilation using the same finite process adapter as CUDA."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
import time
import typing
from dataclasses import dataclass
from pathlib import Path

from .backend import TargetInfo
from .compiler_cache import resolve_compiler_cache, run_cached_compiler
from .compiler_process import CompileResult


def _object_compilation_supported(
    sources: tuple[Path, ...], options: tuple[object, ...]
) -> bool:
    """Only split driver arguments whose compile/link phases are unambiguous.

    Opaque response files, language switches, additional inputs and linker state
    retain the original combined command. Extend this list only with executable
    phase-equivalence coverage; it is deliberately not a general driver parser.
    """
    if any(
        source.suffix not in {".c", ".cc", ".cp", ".cpp", ".CPP", ".cxx", ".c++", ".C"}
        for source in sources
    ):
        return False
    arguments = iter(options)
    for option in arguments:
        if not isinstance(option, str):
            return False
        if option == "-ffp-contract=off":
            continue
        if option in {"-I", "-D", "-U"}:
            value = next(arguments, "")
            if not isinstance(value, str) or not value or value.startswith("-"):
                return False
        elif not (option.startswith(("-I", "-D", "-U")) and len(option) > 2):
            return False
    return True


@dataclass(frozen=True, slots=True)
class CppCompilerAdapter:
    """Compile native CPU shared libraries; selecting CUDA never uses this path."""

    cxx: Path
    compile_timeout: float = 300.0

    def __post_init__(self) -> None:
        executable = shutil.which(str(self.cxx))
        if executable is None:
            raise ValueError("the requested C++ compiler is unavailable")
        if (
            isinstance(self.compile_timeout, bool)
            or not math.isfinite(self.compile_timeout)
            or self.compile_timeout <= 0
        ):
            raise ValueError("compiler timeout must be finite and positive")
        # Preserve clang++/g++ invocation spelling: resolving a symlink to a
        # generic driver can change its language and standard-library defaults.
        object.__setattr__(self, "cxx", Path(executable).absolute())

    @property
    def target(self) -> TargetInfo:
        """Record the compiler's actual target triple without executing a binary."""
        triple = subprocess.check_output(
            [str(self.cxx), "-dumpmachine"], text=True, timeout=30
        ).strip()
        return TargetInfo("cpu", triple, None, 1, None)

    def compile_shared(
        self,
        source: Path,
        output: Path,
        *,
        includes: typing.Any = (),
        libraries: typing.Any = (),
        options: typing.Any = (),
    ) -> CompileResult:
        """Compile one translation unit into a shared library."""
        return self.compile_shared_many(
            (source,),
            output,
            includes=includes,
            libraries=libraries,
            options=options,
        )

    def compile_shared_many(
        self,
        sources: typing.Iterable[Path],
        output: Path,
        *,
        includes: typing.Any = (),
        libraries: typing.Any = (),
        options: typing.Any = (),
    ) -> CompileResult:
        """Compile explicit translation units into one shared library."""
        sources = tuple(Path(source) for source in sources)
        if not sources:
            raise ValueError("shared-library compilation requires at least one source")
        output = Path(output)
        options = tuple(options)
        flags = [
            "-std=c++17",
            "-O3",
            "-fPIC",
            *(f"-I{path}" for path in includes),
            *options,
        ]
        libraries = tuple(f"-l{name}" for name in libraries)

        def shared_command(
            inputs: typing.Iterable[Path], destination: Path
        ) -> list[str]:
            return [
                str(self.cxx),
                *flags[:2],
                "-shared",
                *flags[2:],
                *(str(source) for source in inputs),
                *libraries,
                "-o",
                str(destination),
            ]

        if not _object_compilation_supported(sources, options):
            # Opaque options can place side outputs relative to -o. Preserve the
            # exact original invocation, including its failure/output behavior.
            return run_cached_compiler(
                shared_command(sources, output), self.compile_timeout, label="C++"
            )

        started = time.monotonic()
        deadline = started + self.compile_timeout
        stdout: list[str] = []
        stderr: list[str] = []
        result = CompileResult(124, True, 0.0, "", "")

        def remaining() -> float:
            value = deadline - time.monotonic()
            if value <= 0:
                raise subprocess.TimeoutExpired("C++ compilation", self.compile_timeout)
            return value

        # Publish only a complete library. In particular, a failed later TU or
        # link must neither expose a partial output nor remove a caller's prior
        # library. Sources retain their original paths for includes/__FILE__.
        try:
            temporary = tempfile.TemporaryDirectory(
                prefix=".gqc-cpp-", dir=output.parent
            )
        except OSError as error:
            return CompileResult(
                1,
                False,
                time.monotonic() - started,
                "",
                f"C++ output preparation failed: {error}\n",
            )
        with temporary:
            folder = Path(temporary.name)
            library = folder / output.name
            try:
                launcher = resolve_compiler_cache(remaining())
                objects = [folder / f"unit-{index}.o" for index in range(len(sources))]
                commands = [
                    [str(self.cxx), *flags, "-c", str(source), "-o", str(obj)]
                    for source, obj in zip(sources, objects, strict=True)
                ]
                # Keep all driver flags and ordered library operands at link
                # time too. The launcher also owns this uncacheable link, as it
                # did before; only individual -c invocations can earn cache hits.
                commands.append(shared_command(objects, library))
                for command in commands:
                    result = run_cached_compiler(
                        command,
                        remaining(),
                        label="C++",
                        launcher=launcher,
                        preserve_source_paths=True,
                    )
                    stdout.append(result.stdout)
                    stderr.append(result.stderr)
                    if result.returncode:
                        break
                if result.returncode == 0:
                    remaining()
                    try:
                        library.replace(output)
                    except OSError as error:
                        result = CompileResult(1, False, 0.0, "", "")
                        stderr.append(f"C++ output publication failed: {error}\n")
            except subprocess.TimeoutExpired:
                result = CompileResult(124, True, 0.0, "", "")
        if result.timed_out:
            stderr.append(
                f"C++ compilation timed out after {self.compile_timeout:g} seconds (total budget)\n"
            )
        return CompileResult(
            result.returncode,
            result.timed_out,
            time.monotonic() - started,
            "".join(stdout),
            "".join(stderr),
        )
