"""Shared cached native compiler fixture for Python tests."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path


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
                *args,
                "-c",
                str(source),
                "-o",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=self._env(env),
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


@pytest.fixture(scope="session")
def native_cxx(tmp_path_factory: pytest.TempPathFactory) -> NativeCxx:
    """Provide a required ccache-backed host compiler for native Python probes."""
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    cache = shutil.which(os.environ.get("CCACHE", "ccache"))
    if compiler is None or cache is None:
        pytest.skip("native probe requires a host C++ compiler and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True, text=True)
    return NativeCxx(compiler, cache, tmp_path_factory.getbasetemp())
