#!/usr/bin/env python3
"""Run an unchanged CI command with bounded, best-effort host resource samples.

Samples are flushed to the live job log, not deferred to post-build steps which
cannot run after runner loss. They diagnose pressure, not its cause or an OOM.
The sampler is a daemon thread, never a background process. Build descendants
stay in the runner's existing process group; signals target only our live child.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import FrameType
    from typing import TextIO

MAX_SAMPLES = 512  # 30-second samples cover the existing 210-minute job budget.
MAX_PROCESSES = 4096
MAX_RECORD_BYTES = 4096
COMPILERS = frozenset(
    {"nvcc", "cicc", "ptxas", "fatbinary", "cc1", "cc1plus", "clang", "clang++", "lto1"}
)


def _read(path: Path, limit: int = 4096) -> str:
    with path.open(encoding="ascii", errors="replace") as stream:
        return stream.read(limit)


def snapshot(proc: Path = Path("/proc"), disk: Path = Path(".")) -> dict[str, object]:
    """Read allowlisted counters only, never argv, environment or arbitrary text."""
    result: dict[str, object] = {}
    try:
        wanted = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
        result["memory_kib"] = {
            key: int(value.split()[0])
            for line in _read(proc / "meminfo").splitlines()
            for key, value in [line.split(":", 1)]
            if key in wanted
        }
    except (OSError, ValueError, IndexError):
        result["memory_kib"] = None
    for resource in ("cpu", "memory"):
        try:
            pressure = {}
            for line in _read(proc / "pressure" / resource).splitlines():
                kind, *fields = line.split()
                if kind not in {"some", "full"}:
                    continue
                values = {}
                for field in fields:
                    key, value = field.split("=", 1)
                    if key in {"avg10", "avg60", "avg300", "total"}:
                        number = int(value) if key == "total" else float(value)
                        if math.isfinite(number) and number >= 0:
                            values[key] = number
                pressure[kind] = values
            result[f"{resource}_pressure"] = pressure
        except (OSError, ValueError):
            result[f"{resource}_pressure"] = None
    try:
        usage = os.statvfs(disk)
        result["workspace_disk_bytes"] = {
            "total": usage.f_blocks * usage.f_frsize,
            "available": usage.f_bavail * usage.f_frsize,
        }
    except (OSError, AttributeError):
        result["workspace_disk_bytes"] = None
    try:
        count = total = peak = skipped = 0
        truncated = False
        pagesize = os.sysconf("SC_PAGE_SIZE")
        with os.scandir(proc) as entries:
            for index, entry in enumerate(entries):
                if index >= MAX_PROCESSES:
                    truncated = True
                    break
                if not entry.name.isdecimal():
                    continue
                try:
                    path = Path(entry.path)
                    if _read(path / "comm", 64).strip() not in COMPILERS:
                        continue
                    resident = int(_read(path / "statm", 256).split()[1]) * pagesize
                    if resident < 0:
                        continue
                    count += 1
                    total += resident
                    peak = max(peak, resident)
                except (OSError, ValueError, IndexError):
                    skipped += 1  # Includes processes exiting during sampling.
        result["host_compiler_rss_bytes"] = {
            "count": count,
            "sum": total,
            "max": peak,
            "unreadable_processes": skipped,
            "scan_truncated": truncated,
        }
    except (OSError, ValueError, AttributeError):
        result["host_compiler_rss_bytes"] = None
    return result


class Sampler:
    """No subprocesses, unbounded log growth, or required diagnostic output."""

    def __init__(self, interval: float, stream: TextIO) -> None:
        self.interval = interval
        self.fd = stream.fileno()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        started = time.monotonic()
        for index in range(MAX_SAMPLES):
            try:
                record = {
                    "sample": index,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    **snapshot(),
                }
                line = "CI_HOST_RESOURCES " + json.dumps(record, allow_nan=False) + "\n"
                if len(line.encode()) > MAX_RECORD_BYTES or self.stop.is_set():
                    return
                # Avoid buffered-I/O locks: a stalled log pipe must not hold
                # Python stdout during interpreter shutdown after the build.
                os.write(self.fd, line.encode())
            except Exception:  # noqa: BLE001 - diagnostics must not replace build results
                return
            if self.stop.wait(self.interval):
                return

    def close(self) -> None:
        self.stop.set()
        # An unavailable metric/output sink must not delay the build's result.
        if self.thread.ident is not None:
            self.thread.join(timeout=1)


def run(command: list[str], interval: float = 30.0) -> int:
    """Preserve argv, environment, I/O and shell-style exit/signal status."""
    sampler = None
    process = None
    pending = None

    def forward(signum: int, _frame: FrameType | None) -> None:
        nonlocal pending
        pending = signum
        if process is not None:
            # Popen checks/reaps its own child before signaling, protecting
            # against PID reuse. Never signal the runner or a shared compiler cache.
            process.send_signal(signum)

    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, forward)
        try:
            sampler = Sampler(interval, sys.stdout)
            sampler.thread.start()
        except Exception:  # noqa: BLE001, S110 - optional diagnostics cannot skip the build
            pass
        if pending is not None:
            return 128 + pending  # Cancellation during setup must not start a build.
        process = subprocess.Popen(command)
        if pending is not None:
            process.send_signal(pending)
        status = process.wait()
        return status if status >= 0 else 128 - status
    finally:
        if sampler is not None:
            try:
                sampler.close()
            except Exception:  # noqa: BLE001, S110 - preserve the build outcome
                pass
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if not math.isfinite(args.interval_seconds) or args.interval_seconds < 0.05:
        parser.error("interval must be finite and at least 0.05 seconds")
    try:
        return run(command, args.interval_seconds)
    except FileNotFoundError:
        return 127
    except OSError:
        return 126


if __name__ == "__main__":
    sys.exit(main())
