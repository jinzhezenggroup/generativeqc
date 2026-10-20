"""GIL-held timed tracebacks and bounded, passive CI worker-exit diagnostics.

The C dump_traceback_later watchdog walks changing frames without the GIL and
can itself fault (CPython #116008). A Python thread holds the GIL while calling
this synchronous dump. It can be delayed by native code that holds the GIL;
controller-side exit/resource records and the CI job deadline remain independent.
"""

from __future__ import annotations

import faulthandler
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

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# Soft cap: one final dump can cross it. CPython 3.11 bounds each dump to
# 100 threads x 100 frames (each filename/function string is capped at 500 bytes).
_LIMIT = 4 * 1024 * 1024
_STATE: pytest.StashKey[Watchdog | ExitRecorder] = pytest.StashKey()


def _notice(error: BaseException) -> None:
    try:
        print(
            f"CI diagnostics unavailable: {type(error).__name__}", file=sys.__stderr__
        )
    except Exception:  # noqa: BLE001, S110 - stderr itself may be unavailable
        pass


class Watchdog:
    """One sleeping Python thread, with a cancellable one-shot test deadline."""

    def __init__(self, directory: Path, timeout: float) -> None:
        self.timeout = timeout
        self.condition = threading.Condition()
        self.deadline = None
        self.nodeid = None
        self.stopped = False
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"worker-{worker}-{os.getpid()}.traceback.log"
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            self.stderr_fd = os.dup(sys.__stderr__.fileno())
        except Exception:  # noqa: BLE001 - file diagnostics remain available
            self.stderr_fd = None
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="ci-traceback"
        )
        try:
            self.thread.start()
        except BaseException:
            os.close(self.fd)
            if self.stderr_fd is not None:
                os.close(self.stderr_fd)
            raise

    def arm(self, nodeid: str) -> None:
        with self.condition:
            if not self.stopped:
                self.nodeid = nodeid[:2048]
                self.deadline = time.monotonic() + self.timeout
                self.condition.notify()

    def cancel(self) -> None:
        # Serialization with the dump ensures no stale callback reaches the next test.
        with self.condition:
            self.deadline = None
            self.nodeid = None
            self.condition.notify()

    def _run(self) -> None:
        with self.condition:
            while not self.stopped:
                if self.deadline is None:
                    self.condition.wait()
                    continue
                delay = self.deadline - time.monotonic()
                if delay > 0:
                    self.condition.wait(delay)
                    continue
                nodeid, deadline = self.nodeid, self.deadline
                self.deadline = None  # One dump per test, even if it stays blocked.
                try:
                    if os.fstat(self.fd).st_size >= _LIMIT:
                        continue
                    header = (
                        json.dumps(
                            {
                                "event": "test_diagnostic",
                                "nodeid": nodeid,
                                "threshold_seconds": self.timeout,
                                "delay_seconds": max(0.0, time.monotonic() - deadline),
                            }
                        )
                        + "\n"
                    )
                    os.write(self.fd, header.encode())
                    if self.stderr_fd is not None:
                        try:
                            os.write(
                                self.stderr_fd,
                                ("CI timed traceback: " + header).encode(),
                            )
                        except OSError as error:
                            _notice(error)
                    # This Python callback owns the GIL throughout the synchronous dump.
                    faulthandler.dump_traceback(file=self.fd, all_threads=True)
                except Exception as error:  # noqa: BLE001 - diagnostic failures are nonfatal
                    _notice(error)

    def close(self) -> None:
        with self.condition:
            if self.stopped:
                return
            self.stopped = True
            self.condition.notify()
        self.thread.join(timeout=1)
        # Never close a descriptor while a callback might still be using it.
        if not self.thread.is_alive():
            os.close(self.fd)
            if self.stderr_fd is not None:
                os.close(self.stderr_fd)


def resource_snapshot() -> dict[str, object]:
    """Read only allowlisted numeric host/cgroup counters, never environment data."""
    result: dict[str, object] = {}
    try:
        wanted = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
        result["meminfo_kib"] = {
            key: int(value.split()[0])
            for line in Path("/proc/meminfo").read_text().splitlines()
            for key, value in [line.split(":", 1)]
            if key in wanted
        }
        unified = next(
            line.split(":", 2)[2]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        root = Path("/sys/fs/cgroup") / unified.lstrip("/")
        for name in ("memory.events", "memory.current", "memory.peak", "memory.max"):
            try:
                result[name] = (root / name).read_text()[:2048].strip()
            except OSError as error:
                result[name] = {"unavailable": type(error).__name__}
    except (OSError, ValueError, StopIteration) as error:
        result["unavailable"] = type(error).__name__
    return result


class ExitRecorder:
    """Lives in the xdist controller, independently of each worker's GIL."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"controller-{os.getpid()}.jsonl"
        self.stream = self.path.open("w", encoding="utf-8")
        self.pending: list[tuple[object, object]] = []
        self.write("session")

    def write(self, event: str, **fields: object) -> None:
        try:
            if self.stream.tell() >= _LIMIT:
                return
            self.stream.write(
                json.dumps(
                    {
                        "event": event,
                        "monotonic_ns": time.monotonic_ns(),
                        "resources": resource_snapshot(),
                        **fields,
                    }
                )
                + "\n"
            )
            self.stream.flush()
        except Exception as error:  # noqa: BLE001 - diagnostics cannot replace outcomes
            _notice(error)

    def node_down(self, node: object, error: object, *, final: bool = False) -> None:
        gateway = getattr(node, "gateway", None)
        worker = str(getattr(gateway, "id", "unknown"))[:80]
        # xdist has no public OS-returncode hook. Local execnet popen is optional:
        # record unknown on unsupported transports, API changes, or a live process.
        process = getattr(getattr(gateway, "_io", None), "popen", None)
        returncode = None
        reason = "unavailable_transport"
        try:
            if isinstance(process, subprocess.Popen):
                returncode = process.poll()
                if returncode is None and error is not None:
                    returncode = process.wait(timeout=0.2)
                reason = (
                    "local_popen" if returncode is not None else "process_still_running"
                )
        except subprocess.TimeoutExpired:
            reason = "process_still_running"
        except Exception as failure:  # noqa: BLE001 - preserve the original node failure
            reason = type(failure).__name__
        if returncode is None and not final and isinstance(process, subprocess.Popen):
            self.pending.append((node, error))
        signum = -returncode if isinstance(returncode, int) and returncode < 0 else None
        try:
            signal_name = signal.Signals(signum).name if signum is not None else None
        except ValueError:
            signal_name = None
        self.write(
            "worker_exit",
            worker=worker,
            pid=getattr(process, "pid", None),
            returncode=returncode,
            signal=signum,
            signal_name=signal_name,
            returncode_source=reason,
            node_error=type(error).__name__ if error else None,
        )

    def close(self) -> None:
        # Orderly xdist worker reports can precede process exit. Recheck after
        # xdist teardown without killing or waiting indefinitely for any worker.
        for node, error in self.pending:
            self.node_down(node, error, final=True)
        self.pending.clear()
        self.write("finish")
        self.stream.close()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("safe CI diagnostics")
    group.addoption("--diagnostic-timeout", type=float, default=0.0)
    group.addoption("--diagnostics-dir", default=None)


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    timeout = config.getoption("diagnostic_timeout")
    if not math.isfinite(timeout) or timeout < 0:
        raise pytest.UsageError("--diagnostic-timeout must be finite and nonnegative")
    if not timeout:
        return
    directory = config.getoption("diagnostics_dir")
    if not directory:
        raise pytest.UsageError("--diagnostic-timeout requires --diagnostics-dir")
    if float(config.getini("faulthandler_timeout") or 0) != 0:
        raise pytest.UsageError("Safe CI diagnostics requires faulthandler_timeout=0")
    try:
        if getattr(config.option, "numprocesses", 0) and not hasattr(
            config, "workerinput"
        ):
            state = ExitRecorder(Path(directory))
        else:
            state = Watchdog(Path(directory), timeout)
        config.stash[_STATE] = state
    except Exception as error:  # noqa: BLE001 - unavailable diagnostics do not fail tests
        _notice(error)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item) -> Generator[None, object, object]:
    state = item.config.stash.get(_STATE, None)
    if isinstance(state, Watchdog):
        state.arm(item.nodeid)
    try:
        return (yield)
    finally:
        if isinstance(state, Watchdog):
            state.cancel()


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: object, error: object) -> None:
    config = getattr(node, "config", None)
    if config is not None:
        state = config.stash.get(_STATE, None)
        if isinstance(state, ExitRecorder):
            try:
                state.node_down(node, error)
            except Exception as failure:  # noqa: BLE001 - preserve xdist's outcome
                _notice(failure)


@pytest.hookimpl(tryfirst=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    state = config.stash.get(_STATE, None)
    if state is not None:
        try:
            state.close()
        except Exception as error:  # noqa: BLE001 - teardown diagnostics are nonfatal
            _notice(error)
        del config.stash[_STATE]
