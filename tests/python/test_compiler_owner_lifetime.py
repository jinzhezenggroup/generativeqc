"""An explicit outer owner must not weaken normal compiler process semantics."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from generativeqc_compiler.common import compiler_lifetime
from generativeqc_compiler.common.compiler_process import (
    kill_compiler_group,
    run_compiler,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def owned(monkeypatch: pytest.MonkeyPatch) -> Iterator[int]:
    reader, writer = os.pipe()
    monkeypatch.setenv(compiler_lifetime.OWNER_FD_ENV, str(reader))
    try:
        yield reader
    finally:
        os.close(reader)
        os.close(writer)


def _running(pid: int) -> bool:
    try:
        if Path(f"/proc/{pid}/stat").read_text().split()[2] == "Z":
            return False
    except (FileNotFoundError, ProcessLookupError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_watchdog_ignores_poisoned_site_and_preserves_cwd_env_stdio(
    owned: int, tmp_path: Path
) -> None:
    marker = tmp_path / "site-ran"
    (tmp_path / "sitecustomize.py").write_text(
        f"import pathlib,time; pathlib.Path({str(marker)!r}).touch(); time.sleep(60)"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "OWNER_TEST_VALUE": "preserved",
    }
    process = compiler_lifetime.start_owned_process(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            (
                "import json,os,sys; "
                "print(json.dumps([os.getcwd(),os.environ['OWNER_TEST_VALUE'],"
                "os.environ['PYTHONPATH']])); sys.stderr.write('diagnostic')"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=2)
    finally:
        kill_compiler_group(process)
        kill_compiler_group(process)
    assert process.returncode == 0
    assert json.loads(stdout) == [str(tmp_path), "preserved", str(tmp_path)]
    assert stderr == "diagnostic"
    assert not marker.exists()


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL])
def test_owned_command_preserves_signal_returncode(owned: int, sig: int) -> None:
    result = run_compiler(
        [sys.executable, "-I", "-S", "-c", f"import os;os.kill(os.getpid(),{sig})"],
        2,
        label="signal",
    )
    assert result.returncode == -sig


def test_owned_success_closes_its_extra_descriptors(owned: int) -> None:
    if not Path("/proc/self/fd").exists():
        pytest.skip("descriptor inventory requires procfs")
    before = len(list(Path("/proc/self/fd").iterdir()))
    for _ in range(5):
        assert (
            run_compiler(
                [sys.executable, "-I", "-S", "-c", "pass"], 2, label="fd"
            ).returncode
            == 0
        )
    assert len(list(Path("/proc/self/fd").iterdir())) == before


def test_target_launch_failure_remains_bounded(owned: int) -> None:
    started = time.monotonic()
    result = run_compiler(["/no/such/compiler"], 2, label="missing target")
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
    assert time.monotonic() - started < 2


def test_watchdog_launch_failure_closes_descriptors(
    owned: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not Path("/proc/self/fd").exists():
        pytest.skip("descriptor inventory requires procfs")
    before = len(list(Path("/proc/self/fd").iterdir()))
    monkeypatch.setattr(compiler_lifetime.sys, "executable", "/no/such/python")
    with pytest.raises(FileNotFoundError):
        compiler_lifetime.start_owned_process(["unused"], start_new_session=True)
    assert len(list(Path("/proc/self/fd").iterdir())) == before


@pytest.mark.parametrize("kind", ["regular", "writable-pipe", "closed", "invalid"])
def test_false_owner_descriptors_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    reader, writer = os.pipe()
    regular = (tmp_path / "ordinary-file").open("w+")
    closed = os.dup(reader)
    os.close(closed)
    descriptor = {
        "regular": str(regular.fileno()),
        "writable-pipe": str(writer),
        "closed": str(closed),
        "invalid": "not-an-fd",
    }[kind]
    monkeypatch.setenv(compiler_lifetime.OWNER_FD_ENV, descriptor)
    try:
        with pytest.raises((ValueError, OSError)):
            compiler_lifetime.start_owned_process(
                ["must-not-run"], start_new_session=True
            )
    finally:
        regular.close()
        os.close(reader)
        os.close(writer)


def test_stopped_watchdog_cancellation_is_bounded_and_reclaims_registered_group(
    owned: int, tmp_path: Path
) -> None:
    pid_file = tmp_path / "target.pid"
    process = compiler_lifetime.start_owned_process(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            f"import os,pathlib,time;pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()));time.sleep(60)",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid = None
    try:
        deadline = time.monotonic() + 2
        while not pid_file.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        pid = int(pid_file.read_text())
        os.kill(process.pid, signal.SIGSTOP)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="acknowledge cancellation"):
            kill_compiler_group(process)
        assert time.monotonic() - started < 2.5
        kill_compiler_group(process)
        deadline = time.monotonic() + 1
        while _running(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _running(pid)
    finally:
        if pid is not None and _running(pid):
            os.killpg(pid, signal.SIGKILL)
        if process.poll() is None:
            process.kill()
        process.wait()


def test_channel_allocation_failure_does_not_leak_first_pipe(
    owned: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not Path("/proc/self/fd").exists():
        pytest.skip("descriptor inventory requires procfs")
    before = len(list(Path("/proc/self/fd").iterdir()))
    original = os.pipe
    calls = 0

    def pipe() -> tuple[int, int]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected descriptor capacity failure")
        return original()

    monkeypatch.setattr(compiler_lifetime.os, "pipe", pipe)
    with pytest.raises(OSError, match="capacity failure"):
        compiler_lifetime.start_owned_process(["must-not-run"], start_new_session=True)
    assert len(list(Path("/proc/self/fd").iterdir())) == before
