"""Finite compiler process-tree execution shared by CPU and CUDA adapters."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompileResult:
    """Compiler outcome including deterministic timeout diagnostics."""

    returncode: int
    timed_out: bool
    duration_seconds: float
    stdout: str
    stderr: str


def kill_compiler_group(process: subprocess.Popen[str]) -> None:
    """Stop only a process group created by this invocation, then reap its leader."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_compiler(
    command: list[str],
    timeout: float,
    *,
    label: str,
    environment: dict[str, str] | None = None,
    pass_fds: tuple[int, ...] = (),
) -> CompileResult:
    """Capture diagnostics and terminate all compiler children on timeout."""
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=environment,
        pass_fds=pass_fds,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(
            timeout=max(0.0, timeout - (time.monotonic() - started))
        )
    except subprocess.TimeoutExpired:
        timed_out = True
        # The deadline is already exhausted. A fresh five-second grace period
        # would extend the caller's finite budget and leave compilers running.
        kill_compiler_group(process)
        stdout, stderr = process.communicate()
    finally:
        # A launcher can exit while a compiler child keeps running with closed
        # output pipes. Reclaim our group on success and failure as well.
        kill_compiler_group(process)
    duration = time.monotonic() - started
    if timed_out:
        stderr += f"{label} compilation timed out after {timeout:g} seconds\n"
    return CompileResult(
        returncode=124 if timed_out else process.returncode,
        timed_out=timed_out,
        duration_seconds=duration,
        stdout=stdout,
        stderr=stderr,
    )
