"""Verified compiler-cache launcher discovery for runtime compilation."""

from __future__ import annotations

import functools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .compiler_process import CompileResult, kill_compiler_group, run_compiler

if TYPE_CHECKING:
    from collections.abc import Iterator


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
def _resolve_compiler_cache(
    path_environment: str, timeout: float = 30.0
) -> CompilerCacheLauncher:
    """Resolve the CMake-compatible sccache/ccache preference for one PATH."""

    deadline = time.monotonic() + timeout
    failures: list[str] = []
    for name in ("sccache", "ccache"):
        executable = shutil.which(name, path=path_environment)
        if executable is None:
            continue
        path = Path(executable).absolute()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("compiler-cache discovery", timeout)
        try:
            result = subprocess.run(
                [str(path), "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=min(30, remaining),
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
        if name == "sccache":
            match = re.search(r"sccache (\d+)\.(\d+)\.(\d+)", version)
            if match is None or tuple(map(int, match.groups())) < (0, 16, 0):
                failures.append("sccache: version 0.16.0 or later is required")
                continue
        return CompilerCacheLauncher(path=path, name=name, version=version)

    if time.monotonic() >= deadline:
        raise subprocess.TimeoutExpired("compiler-cache discovery", timeout)
    detail = f" ({'; '.join(failures)})" if failures else ""
    raise RuntimeError(
        "no usable compiler cache launcher found; install sccache with "
        "`python -m pip install sccache` or install ccache" + detail
    )


def resolve_compiler_cache(timeout: float = 30.0) -> CompilerCacheLauncher:
    """Return the verified process-local compiler-cache launcher."""
    return _resolve_compiler_cache(os.environ.get("PATH", ""), timeout)


def cached_compiler_command(command: list[str]) -> list[str]:
    """Wrap a cache-miss compiler command; never silently compile uncached."""
    return resolve_compiler_cache().wrap(command)


def run_cached_compiler(
    command: list[str], timeout: float, *, label: str
) -> CompileResult:
    """Cache compilation while owning every local compiler's finite lifetime.

    sccache's ordinary shared daemon is outside the client's process group.
    Use a private foreground server and an exclusively leased persistent JIT
    cache slot. Never connect to or shut down the user's shared server.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("compiler timeout must be finite and positive")
    started = time.monotonic()
    deadline = started + timeout

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise subprocess.TimeoutExpired(command, timeout)
        return value

    def outcome(result: CompileResult) -> CompileResult:
        stderr = result.stderr
        if result.timed_out and "compilation timed out" not in stderr:
            stderr += f"{label} compilation timed out after {timeout:g} seconds\n"
        return CompileResult(
            result.returncode,
            result.timed_out,
            time.monotonic() - started,
            result.stdout,
            stderr,
        )

    try:
        launcher = resolve_compiler_cache(timeout)
        if launcher.name != "sccache":
            result = run_compiler(launcher.wrap(command), remaining(), label=label)
            return outcome(result)
        result = _run_sccache(launcher, command, deadline, label=label)
        return outcome(result)
    except subprocess.TimeoutExpired:
        return outcome(CompileResult(124, True, 0.0, "", ""))


def _shared_sccache_disk_roots() -> tuple[Path, ...]:
    """Conservatively exclude every configured/default shared disk-cache tree.

    sccache disk size/mode environment settings can replace the file config's
    entire disk section, including its path. Excluding both paths avoids trying
    to reproduce that precedence and protects already-running shared servers.
    """
    roots = []
    if "SCCACHE_DIR" in os.environ:
        roots.append(Path(os.environ["SCCACHE_DIR"]).resolve())
    home = Path.home()
    if sys.platform == "darwin":
        default = home / "Library/Caches/Mozilla.sccache"
        candidates = [
            home / "Library/Application Support/Mozilla.sccache/config",
            home / "Library/Preferences/Mozilla.sccache/config",
        ]
    else:

        def xdg(name: str, fallback: Path) -> Path:
            value = Path(os.environ.get(name, ""))
            return value if value.is_absolute() else fallback

        default = xdg("XDG_CACHE_HOME", home / ".cache") / "sccache"
        candidates = [xdg("XDG_CONFIG_HOME", home / ".config") / "sccache/config"]
    config = (
        Path(os.environ["SCCACHE_CONF"])
        if "SCCACHE_CONF" in os.environ
        else next((path for path in candidates if path.exists()), candidates[0])
    )
    if config.is_file():
        content = config.read_text()
        if config.suffix == ".json":
            settings = json.loads(content)
        else:
            if sys.version_info >= (3, 11):
                import tomllib
            else:
                try:
                    import tomli as tomllib
                except ImportError as error:
                    raise RuntimeError(
                        "reading sccache TOML on Python 3.10 requires `python -m pip install tomli`"
                    ) from error
            settings = tomllib.loads(content)
        directory = settings.get("cache", {}).get("disk", {}).get("dir")
        if directory is not None:
            roots.append(Path(directory).resolve())
    roots.append(default.resolve())
    return tuple(dict.fromkeys(roots))


@contextmanager
def _sccache_slot(deadline: float) -> Iterator[tuple[Path, int]]:
    """Lease the lowest available persistent cache; sccache's disk LRU is single-owner."""
    import fcntl

    configured = os.environ.get("GENERATIVEQC_JIT_SCCACHE_ROOT")
    root = (
        Path(configured).expanduser()
        if configured
        else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "generativeqc"
        / "sccache-jit"
    ).resolve()
    for shared_path in _shared_sccache_disk_roots():
        if (
            root == shared_path
            or root in shared_path.parents
            or shared_path in root.parents
        ):
            raise ValueError(
                "JIT sccache pool must be outside the shared disk-cache tree"
            )
    root.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired("sccache cache-slot acquisition", 0)
        # Lock files live outside cache directories: sccache scans/removes its
        # own temporary files at startup and must never own the lease metadata.
        lease = (root / f"slot-{index}.lock").open("a")
        try:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                index += 1
                continue
            slot = root / f"slot-{index}"
            slot.mkdir(exist_ok=True)
            yield slot, lease.fileno()
            return
        finally:
            # The lease stays held until the foreground server AND all its
            # compiler children have been killed/reaped by _run_sccache.
            lease.close()


def _run_sccache(
    launcher: CompilerCacheLauncher,
    command: list[str],
    deadline: float,
    *,
    label: str,
) -> CompileResult:
    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise subprocess.TimeoutExpired(command, 0)
        return value

    # The finite process runner is POSIX-based. A short private directory
    # also avoids the Unix-domain socket path length limit in deep caches.
    with (
        _sccache_slot(deadline) as (slot, lease_fd),
        tempfile.TemporaryDirectory(prefix="gqc-scc-", dir="/tmp") as folder,
    ):
        endpoint = Path(folder) / "server.sock"
        environment: dict[str, str] = {
            **os.environ,
            "SCCACHE_SERVER_UDS": str(endpoint),
            "SCCACHE_DIR": str(slot / "cache"),
            "SCCACHE_CACHED_CONF": str(slot / "cached-config"),
            "SCCACHE_NO_DAEMON": "1",
            "SCCACHE_IDLE_TIMEOUT": str(max(1, math.ceil(remaining()))),
        }
        # Do not inherit another server's startup notification or mode.
        environment.pop("SCCACHE_STARTUP_NOTIFY", None)
        environment.pop("SCCACHE_START_SERVER", None)
        with tempfile.TemporaryFile(mode="w+") as server_log:
            server = subprocess.Popen(
                [str(launcher.path)],
                env={**environment, "SCCACHE_START_SERVER": "1"},
                stdin=subprocess.DEVNULL,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
                # Keep the cache leased even if the Python caller is killed.
                # Closing the parent FD does not unlock a live server's copy.
                pass_fds=(lease_fd,),
            )
            try:
                while not endpoint.exists():
                    if server.poll() is not None:
                        server_log.seek(0)
                        return CompileResult(
                            server.returncode or 1, False, 0.0, "", server_log.read()
                        )
                    time.sleep(min(0.01, remaining()))
                # Remote workers cannot be owned by this local process group.
                # Refuse that configuration before sending any compilation.
                status = run_compiler(
                    [str(launcher.path), "--dist-status"],
                    remaining(),
                    label=label,
                    environment=environment,
                    pass_fds=(lease_fd,),
                )
                if status.returncode:
                    return status
                try:
                    distribution = json.loads(status.stdout)
                except ValueError:
                    distribution = None
                if not isinstance(distribution, dict) or set(distribution) != {
                    "Disabled"
                }:
                    return CompileResult(
                        1,
                        False,
                        0.0,
                        "",
                        "finite runtime compilation requires local sccache workers\n",
                    )
                result = run_compiler(
                    launcher.wrap(command),
                    remaining(),
                    label=label,
                    environment=environment,
                    pass_fds=(lease_fd,),
                )
            finally:
                # This PGID belongs only to our private foreground server.
                # SIGKILL at the deadline is bounded and also catches children
                # which ignore SIGTERM. Successful responses include cache writes.
                kill_compiler_group(server)
            return result
