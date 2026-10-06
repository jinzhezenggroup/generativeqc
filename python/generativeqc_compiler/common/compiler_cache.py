"""Verified compiler-cache launcher discovery for runtime compilation."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CompilerCacheLauncher:
    """One verified compiler-cache executable and its observed version."""

    path: Path
    name: str
    version: str

    def wrap(self, command: list[str]) -> list[str]:
        """Prefix one compiler command with this verified launcher."""
        if not command:
            raise ValueError("compiler command must be non-empty")
        return [str(self.path), *command]


@functools.lru_cache(maxsize=8)
def _resolve_compiler_cache(path_environment: str) -> CompilerCacheLauncher:
    """Resolve the CMake-compatible sccache/ccache preference for one PATH."""

    failures: list[str] = []
    for name in ("sccache", "ccache"):
        executable = shutil.which(name, path=path_environment)
        if executable is None:
            continue
        path = Path(executable).absolute()
        try:
            result = subprocess.run(
                [str(path), "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            failures.append(f"{name}: {error}")
            continue
        if result.returncode:
            diagnostic = (result.stderr or result.stdout).strip()
            failures.append(
                f"{name}: --version exited {result.returncode}"
                + (f" ({diagnostic})" if diagnostic else "")
            )
            continue
        version = (result.stdout + result.stderr).strip()
        return CompilerCacheLauncher(path=path, name=name, version=version)

    detail = f" ({'; '.join(failures)})" if failures else ""
    raise RuntimeError(
        "no usable compiler cache launcher found; install sccache with "
        "`python -m pip install sccache` or install ccache" + detail
    )


def resolve_compiler_cache() -> CompilerCacheLauncher:
    """Return the verified process-local compiler-cache launcher."""
    return _resolve_compiler_cache(os.environ.get("PATH", ""))


def cached_compiler_command(command: list[str]) -> list[str]:
    """Wrap a cache-miss compiler command; never silently compile uncached."""
    return resolve_compiler_cache().wrap(command)
