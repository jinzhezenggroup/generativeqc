"""Owned-session outer bounds; never signal the caller or unrelated actors."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import cc_cupti_execution_guard as execution

linux_only = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux PID/start/session ownership"
)


@linux_only
def test_guard_is_ready_before_exec_and_normal_exit_drains(tmp_path: Path) -> None:
    process, guard, identity = execution.launch_owned(
        [sys.executable, "-c", "pass"], tmp_path / "normal", dict(os.environ), 10
    )
    assert process.wait(timeout=5) == 0
    assert guard.wait(timeout=5) == 0
    receipt = json.loads((tmp_path / "normal/outer-guard.json").read_text())
    assert receipt["attached_after_launch"] is False
    assert receipt["drained"] and not receipt["timed_out"] and not receipt["errors"]
    assert identity["pgid"] != os.getpgrp()
    assert receipt["guard_pgid"] == receipt["guard_pid"] != identity["pgid"]


@linux_only
def test_timeout_drains_worker_and_native_descendant_preserving_other_session(
    tmp_path: Path,
) -> None:
    sentinel = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        child = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); time.sleep(30)"
        process, guard, _ = execution.launch_owned(
            [sys.executable, "-c", child], tmp_path / "timeout", dict(os.environ), 0.3
        )
        assert guard.wait(timeout=15) == 1
        assert process.wait(timeout=5) != 0
        receipt = json.loads((tmp_path / "timeout/outer-guard.json").read_text())
        assert receipt["timed_out"] and receipt["drained"] and not receipt["current"]
        assert len(receipt["seen"]) >= 2
        assert sentinel.poll() is None
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


@linux_only
def test_wrong_leader_start_is_rejected_without_signal(tmp_path: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        fields = Path(f"/proc/{process.pid}/stat").read_text().split(") ", 1)[1].split()
        plan = {
            "leader": {
                "pid": process.pid,
                "starttime": str(int(fields[19]) + 1),
                "pgid": process.pid,
                "session": process.pid,
            },
            "limit_seconds": 0.01,
            "attached_after_launch": False,
            "output": str(tmp_path / "wrong.json"),
        }
        with pytest.raises(ValueError, match="reused"):
            execution.supervise(plan)
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


@linux_only
def test_postspawn_identity_failure_never_releases_exec_or_leaks_actor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actors = []
    real_popen, real_read = subprocess.Popen, Path.read_text

    def remember(*args: object, **kwargs: object) -> subprocess.Popen:
        process = real_popen(*args, **kwargs)
        actors.append(process)
        return process

    def unavailable(path: Path, *args: object, **kwargs: object) -> str:
        if str(path).startswith("/proc/") and path.name == "stat":
            raise PermissionError("controlled identity unavailable")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", remember)
    monkeypatch.setattr(Path, "read_text", unavailable)
    marker = tmp_path / "must-not-exec"
    with pytest.raises(PermissionError):
        execution.launch_owned(
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).write_text('bad')",
            ],
            tmp_path / "unavailable",
            dict(os.environ),
            10,
        )
    assert not marker.exists() and actors and all(p.poll() is not None for p in actors)


@pytest.mark.parametrize("defect", ["missing_ack", "wrong_session", "errors"])
def test_unacknowledged_guard_never_releases_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    class Gate:
        closed = False
        writes = b""

        def write(self, data: bytes) -> None:
            self.writes += data

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    class Actor:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.stdin = Gate()
            self.returncode = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def wait(self, timeout: float) -> int | None:
            return self.returncode

    worker, guard = Actor(101), Actor(202)
    spawned = iter((worker, guard))
    leader = {"pid": 101, "starttime": "123", "pgid": 101, "session": 101}
    ack = dict(leader)
    if defect == "wrong_session":
        ack["session"] = 999
    receipt = {
        "leader": leader,
        "guard_pid": 202,
        "guard_pgid": 202,
        "attached_after_launch": False,
        "timed_out": False,
        "errors": ["controlled failure"] if defect == "errors" else [],
        "seen": {} if defect == "missing_ack" else {"101:123": ack},
        "current": [ack],
    }
    real_read = Path.read_text

    def read(path: Path, *args: object, **kwargs: object) -> str:
        if str(path).replace("\\", "/").startswith("/proc/"):
            fields = ["S", "0", "101", "101"] + ["0"] * 15 + ["123"]
            return "101 (held) " + " ".join(fields)
        if path.name == "outer-guard.json":
            return json.dumps(receipt)
        return real_read(path, *args, **kwargs)

    ticks = iter((0.0, 1.0, 11.0))
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **k: next(spawned))
    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(Path, "is_file", lambda path: True)
    monkeypatch.setattr(execution.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(execution.time, "sleep", lambda _: None)
    monkeypatch.setattr(execution, "current", lambda _: True)
    with pytest.raises(RuntimeError, match="not ready"):
        execution.launch_owned(["held"], tmp_path / defect, {}, 20)
    assert worker.stdin.writes == b"" and worker.stdin.closed
    assert worker.poll() is not None and guard.poll() is not None
