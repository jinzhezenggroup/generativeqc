"""Shared cached native compiler fixture for Python tests."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(frozen=True)
class NativeCxx:
    """Compile standalone native test probes through ccache, then link normally."""

    compiler: str
    cache: str
    base_dir: Path

    def _env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {**os.environ, "CCACHE_BASEDIR": str(self.base_dir)}
        if extra:
            env.update(extra)
        return env

    @staticmethod
    def _absolute_include_args(args: Sequence[str]) -> list[str]:
        """Keep include directories relative to the caller when changing cwd."""

        def absolute(directory: str) -> str:
            if directory == "-" or directory.startswith(("=", "$SYSROOT")):
                return directory
            return str(Path(directory).absolute())

        options = ("-I", "-isystem", "-iquote", "-idirafter")
        result = []
        directory_next = False
        for arg in args:
            if directory_next:
                result.append(absolute(arg))
                directory_next = False
            elif arg in options:
                result.append(arg)
                directory_next = True
            else:
                option = next((item for item in options if arg.startswith(item)), None)
                if option and len(arg) > len(option):
                    result.append(option + absolute(arg[len(option) :]))
                else:
                    result.append(arg)
        return result

    def compile_object(
        self,
        source: Path,
        output: Path,
        *,
        args: Sequence[str] = (),
        timeout: int = 120,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Compile one translation unit so ccache can cache it."""
        return subprocess.run(
            [
                self.cache,
                self.compiler,
                *self._absolute_include_args(args),
                "-c",
                str(source.absolute()),
                "-o",
                str(output.absolute()),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=self._env(env),
            # ccache rewrites paths relative to cwd, not to CCACHE_BASEDIR.
            # Both must be this worker's pytest temporary root for generated
            # probes to reuse objects across session and worker directories.
            cwd=self.base_dir,
        )

    def link(
        self,
        objects: Iterable[Path],
        output: Path,
        *,
        args: Sequence[str] = (),
        timeout: int = 60,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Link cached objects without routing the non-cacheable link through ccache."""
        return subprocess.run(
            [self.compiler, *map(str, objects), *args, "-o", str(output)],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **(env or {})},
        )

    def build_executable(
        self,
        sources: Sequence[Path],
        output: Path,
        *,
        compile_args: Sequence[str] = (),
        link_args: Sequence[str] = (),
        compile_timeout: int = 120,
        link_timeout: int = 60,
        env: dict[str, str] | None = None,
    ) -> Path:
        """Compile each source independently through ccache and link an executable."""
        objects = []
        for index, source in enumerate(sources):
            obj = output.parent / f".{output.name}.{index}.o"
            self.compile_object(
                source,
                obj,
                args=compile_args,
                timeout=compile_timeout,
                env=env,
            )
            objects.append(obj)
        self.link(objects, output, args=link_args, timeout=link_timeout, env=env)
        return output

    def build_shared(
        self,
        sources: Sequence[Path],
        output: Path,
        *,
        compile_args: Sequence[str] = (),
        link_args: Sequence[str] = (),
        compile_timeout: int = 120,
        link_timeout: int = 60,
        env: dict[str, str] | None = None,
    ) -> Path:
        """Compile cached PIC objects and link a shared library."""
        objects = []
        for index, source in enumerate(sources):
            obj = output.parent / f".{output.name}.{index}.o"
            self.compile_object(
                source,
                obj,
                args=(*compile_args, "-fPIC"),
                timeout=compile_timeout,
                env=env,
            )
            objects.append(obj)
        self.link(
            objects,
            output,
            args=("-shared", *link_args),
            timeout=link_timeout,
            env=env,
        )
        return output


def _native_cxx(
    tmp_path_factory: pytest.TempPathFactory, *, required: bool
) -> NativeCxx:
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    cache = shutil.which(os.environ.get("CCACHE", "ccache"))
    if compiler is None or cache is None:
        message = "native probe requires a host C++ compiler and ccache"
        if required:
            pytest.fail(message)
        pytest.skip(message)
    # which() may retain a relative spelling. Preserve the executable basename
    # (e.g. clang++), but make it independent of the probe compilation cwd.
    compiler = os.path.abspath(compiler)
    cache = os.path.abspath(cache)
    subprocess.run([cache, "--version"], check=True, capture_output=True, text=True)
    return NativeCxx(compiler, cache, tmp_path_factory.getbasetemp())


@pytest.fixture(scope="session")
def native_cxx(tmp_path_factory: pytest.TempPathFactory) -> NativeCxx:
    """Provide a cached host compiler for probes with optional prerequisites."""
    return _native_cxx(tmp_path_factory, required=False)


@pytest.fixture(scope="session")
def required_native_cxx(tmp_path_factory: pytest.TempPathFactory) -> NativeCxx:
    """Fail mandatory native gates when compiler/cache prerequisites are absent."""
    return _native_cxx(tmp_path_factory, required=True)
