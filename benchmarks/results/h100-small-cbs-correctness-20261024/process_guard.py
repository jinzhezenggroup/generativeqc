"""Linux lifecycle guard for one task-owned command, without scientific imports.

The guard stays the session leader and subreaper until its owned descendants
are reaped. It never searches by command name or signals an existing service.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from transport import read, tree_bytes, write_new


def process_identity(pid: int) -> dict | None:
    try:
        tail = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    if len(tail) < 20:
        raise RuntimeError("process start identity unavailable")
    return {
        "pid": pid,
        "state": tail[0],
        "ppid": int(tail[1]),
        "pgid": int(tail[2]),
        "sid": int(tail[3]),
        "start_ticks": int(tail[19]),
    }


def enable_subreaper() -> None:
    if (
        not Path("/proc/self/stat").is_file()
        or not hasattr(os, "pidfd_open")
        or not hasattr(signal, "pidfd_send_signal")
    ):
        raise RuntimeError(
            "Linux proc/subreaper/pidfd lifecycle prerequisites unavailable"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    if (
        libc.prctl(36, 1, 0, 0, 0) != 0
    ):  # PR_SET_CHILD_SUBREAPER; libc/kernel only, no GQC DSO.
        raise OSError(ctypes.get_errno(), "cannot establish task subreaper")
    descriptor = os.pidfd_open(os.getpid(), 0)
    try:
        signal.pidfd_send_signal(descriptor, 0, None, 0)
    finally:
        os.close(descriptor)


def descendants(guard: dict) -> dict[int, dict]:
    current = process_identity(guard["pid"])
    if current is None or any(
        current[k] != guard[k] for k in ("pid", "pgid", "sid", "start_ticks")
    ):
        raise RuntimeError("live guard identity changed; no signal authorized")
    found = {}
    queue = [guard]
    while queue:
        parent = queue.pop()
        task_root = Path(f"/proc/{parent['pid']}/task")
        try:
            tasks = list(task_root.iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        for task in tasks:
            try:
                children = (task / "children").read_text().split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            for text in children:
                pid = int(text)
                if pid in found or pid == guard["pid"]:
                    continue
                child = process_identity(pid)
                if (
                    child is not None
                    and child["ppid"] == parent["pid"]
                    and child["start_ticks"] >= guard["start_ticks"]
                ):
                    found[pid] = child
                    queue.append(child)
    return found


def signal_descendants(guard: dict, signum: int) -> list[dict]:
    records = []
    owned = descendants(guard)
    for pid, before in owned.items():
        if before["state"] == "Z":
            continue
        descriptor = None
        try:
            descriptor = os.pidfd_open(pid, 0)
            # PIDFD prevents retargeting after this check; fresh ancestry avoids
            # attributing an already reused PID to an earlier descendant.
            fresh = descendants(guard).get(pid)
            if fresh is None or fresh["start_ticks"] != before["start_ticks"]:
                continue
            signal.pidfd_send_signal(descriptor, signum, None, 0)
            records.append({**fresh, "signal": signum})
        except ProcessLookupError:
            continue
        finally:
            if descriptor is not None:
                os.close(descriptor)
    return records


def reap(command: subprocess.Popen) -> None:
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return
        if pid == command.pid:
            command.returncode = os.waitstatus_to_exitcode(status)


def run(args: Any) -> int:
    if (
        not 0 < args.grace_seconds <= 30
        or not 0 < args.kill_reap_seconds <= 5
        or not args.command
    ):
        raise ValueError("invalid finite lifecycle contract")
    args.control.mkdir(exist_ok=False)
    guard = process_identity(os.getpid())
    if guard is None or guard["sid"] != guard["pid"] or guard["pgid"] != guard["pid"]:
        raise RuntimeError("guard must be a fresh exclusive session leader")
    enable_subreaper()
    write_new(args.control / "identity.json", {"nonce": args.nonce, **guard})
    signalled = []

    def requested(signum: int, unused: Any) -> None:
        signalled.append(signum)

    signal.signal(signal.SIGTERM, requested)
    signal.signal(signal.SIGINT, requested)
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    started = time.monotonic()
    command_digest = hashlib.sha256(
        json.dumps(args.command, separators=(",", ":")).encode()
    ).hexdigest()
    ready_deadline = min(args.deadline, started + 10)
    while not (args.control / "start-command.json").exists():
        if (
            signalled
            or (args.control / "stop-request.json").exists()
            or time.monotonic() >= ready_deadline
        ):
            write_new(
                args.control / "guard-outcome.json",
                {
                    "nonce": args.nonce,
                    "status": "CONTROL_ERROR",
                    "trigger": "COMMAND_NOT_ACKNOWLEDGED",
                    "command_started": False,
                    "cleanup_success": True,
                    "remaining_owned": [],
                    "guard_identity": guard,
                },
            )
            return 125
        time.sleep(0.02)
    acknowledgement = read(args.control / "start-command.json")
    if (
        acknowledgement.get("nonce") != args.nonce
        or acknowledgement.get("guard_identity") != guard
        or acknowledgement.get("command_sha256") != command_digest
        or time.monotonic() >= args.deadline
    ):
        raise ValueError("command handshake differs from live guard identity")
    command = subprocess.Popen(args.command, stdin=subprocess.DEVNULL)
    command_id = process_identity(command.pid)
    if command_id is None:
        raise RuntimeError("child process birth identity unavailable")
    write_new(
        args.control / "command-identity.json", {"nonce": args.nonce, **command_id}
    )
    result = {
        "nonce": args.nonce,
        "guard_identity": guard,
        "command_identity": command_id,
        "command": args.command,
        "status": "STARTING",
        "signal_records": [],
    }
    cause = None
    cleanup_started = None
    try:
        while True:
            reap(command)
            owned = descendants(guard)
            request = args.control / "stop-request.json"
            if request.exists():
                payload = read(request)
                cause = (
                    payload.get("cause")
                    if payload.get("nonce") == args.nonce
                    else "CONTROL_IDENTITY_ERROR"
                )
                break
            if signalled:
                cause = "PARENT_SIGNAL"
                break
            if time.monotonic() >= args.deadline:
                cause = "TIMEOUT"
                break
            if (
                tree_bytes(args.output_root) > args.output_cap
                or tree_bytes(args.jit_root) > args.jit_cap
                or (
                    args.task_cap is not None
                    and tree_bytes(args.output_root.parent) > args.task_cap
                )
                or (
                    args.tmp_cache_cap is not None
                    and sum(
                        tree_bytes(args.output_root.parent / leaf)
                        for leaf in ("tmp", "cache", "cupy-cache", "pip-cache")
                    )
                    > args.tmp_cache_cap
                )
            ):
                cause = "STORAGE_LIMIT"
                break
            if command.returncode is not None:
                if owned:
                    cause = "DESCENDANT_LEFT_AFTER_LEADER"
                break
            time.sleep(0.05)
        if cause is not None:
            result["trigger"] = cause
            cleanup_started = time.monotonic()
            write_new(
                args.control / "cleanup-start.json",
                {
                    "nonce": args.nonce,
                    "trigger": cause,
                    "started_monotonic": cleanup_started,
                },
            )
            term_deadline = cleanup_started + args.grace_seconds
            while descendants(guard) and time.monotonic() < term_deadline:
                result["signal_records"] += signal_descendants(guard, signal.SIGTERM)
                reap(command)
                time.sleep(0.05)
            kill_deadline = term_deadline + args.kill_reap_seconds
            while descendants(guard) and time.monotonic() < kill_deadline:
                result["signal_records"] += signal_descendants(guard, signal.SIGKILL)
                reap(command)
                time.sleep(0.05)
        reap(command)
        remaining = descendants(guard)
        result.update(
            returncode=command.returncode,
            remaining_owned=list(remaining.values()),
            cleanup_success=not remaining,
        )
        if remaining or command.returncode is None:
            result["status"] = "CLEANUP_FAILED"
        elif cause in ("TIMEOUT", "STORAGE_LIMIT"):
            result["status"] = cause
        elif cause is not None:
            result["status"] = "CONTROL_ERROR"
        else:
            result["status"] = "RETURNED" if command.returncode == 0 else "FAILED"
    except BaseException as error:  # noqa: BLE001 -- retain the control error and bounded cleanup result
        result.update(status="CONTROL_ERROR", error=f"{type(error).__name__}: {error}")
        # Attempt only freshly proven descendants. Loss of /proc/pidfd evidence
        # leaves an explicit cleanup failure, never a guessed group-wide kill.
        try:
            if cleanup_started is None:
                cleanup_started = time.monotonic()
            end = cleanup_started + args.grace_seconds
            while descendants(guard) and time.monotonic() < end:
                result["signal_records"] += signal_descendants(guard, signal.SIGTERM)
                reap(command)
                time.sleep(0.05)
            end = cleanup_started + args.grace_seconds + args.kill_reap_seconds
            while descendants(guard) and time.monotonic() < end:
                result["signal_records"] += signal_descendants(guard, signal.SIGKILL)
                reap(command)
                time.sleep(0.05)
            remaining = descendants(guard)
            result.update(
                cleanup_success=not remaining,
                remaining_owned=list(remaining.values()),
                returncode=command.returncode,
            )
        except BaseException as cleanup_error:  # noqa: BLE001 -- preserve both failures and refuse success
            result.update(
                status="CLEANUP_FAILED",
                cleanup_success=False,
                remaining_owned="UNAVAILABLE",
                cleanup_error=f"{type(cleanup_error).__name__}: {cleanup_error}",
            )
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write_new(args.control / "guard-outcome.json", result)
    return 0 if result["status"] == "RETURNED" else 125


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "output-root", "jit-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    for name in (
        "seconds",
        "grace-seconds",
        "kill-reap-seconds",
        "output-cap",
        "jit-cap",
    ):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--task-cap", type=int)
    parser.add_argument("--tmp-cache-cap", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not math.isfinite(args.deadline):
        parser.error("finite monotonic deadline required")
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    raise SystemExit(run(args))
