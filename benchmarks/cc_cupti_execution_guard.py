"""Finite execution supervisor for one sealed PID/start/session process group."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def actors(group: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    errors = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().split(") ", 1)[1].split()
            if int(fields[2]) == group and fields[0] != "Z":
                rows.append(
                    {
                        "pid": int(path.parent.name),
                        "starttime": fields[19],
                        "pgid": int(fields[2]),
                        "session": int(fields[3]),
                    }
                )
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, ValueError, IndexError) as error:
            errors.append({"pid": path.parent.name, "type": type(error).__name__})
    return rows, errors


def current(identity: dict[str, Any]) -> bool:
    try:
        fields = (
            Path(f"/proc/{identity['pid']}/stat").read_text().split(") ", 1)[1].split()
        )
        return (
            fields[19] == identity["starttime"]
            and fields[0] != "Z"
            and int(fields[2]) == identity["pgid"]
            and int(fields[3]) == identity["session"]
        )
    except (FileNotFoundError, ProcessLookupError):
        return False


def supervise(plan: dict[str, Any]) -> dict[str, Any]:
    leader = plan["leader"]
    group = leader["pgid"]
    if group == os.getpgrp() or group != leader["pid"] or group != leader["session"]:
        raise ValueError("guard must remain outside the sealed owned session")
    initial, initial_errors = actors(group)
    if initial_errors:
        raise RuntimeError("owned session liveness unavailable before guard admission")
    leaders = [row for row in initial if row["pid"] == leader["pid"]]
    if leaders and leaders[0]["starttime"] != leader["starttime"]:
        raise ValueError("sealed leader identity was reused")
    if not leaders and initial:
        raise ValueError(
            "cannot newly admit an orphan group without a live sealed leader"
        )
    limit = plan["limit_seconds"]
    if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
        raise ValueError("invalid finite execution limit")
    deadline = int(leader["starttime"]) / os.sysconf("SC_CLK_TCK") + limit
    output = Path(plan["output"])
    record = {
        "scope": "owned process execution bound only; never Notebook/cost timer",
        "leader": leader,
        "attached_after_launch": plan["attached_after_launch"],
        "deadline_monotonic": deadline,
        "guard_pid": os.getpid(),
        "guard_pgid": os.getpgrp(),
        "timed_out": False,
        "errors": [],
        "seen": {},
    }

    def save() -> None:
        temporary = output.with_name(output.name + ".pending")
        with temporary.open("w") as stream:
            json.dump(record, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)

    if initial:
        record["seen"][str(leader["pid"]) + ":" + leader["starttime"]] = leader
    empty = 0
    while True:
        rows, errors = actors(group)
        record["errors"].extend(errors)
        for row in rows:
            if row["session"] != group:
                record["errors"].append(
                    {"type": "unowned_group_member", "pid": row["pid"]}
                )
            record["seen"][str(row["pid"]) + ":" + row["starttime"]] = row
        record["current"] = rows
        empty = empty + 1 if not rows and not errors else 0
        save()
        if empty >= 2:
            record["drained"] = not record["errors"]
            save()
            return record
        if time.monotonic() >= deadline:
            record["timed_out"] = True
            for sig in (signal.SIGTERM, signal.SIGKILL):
                for row in rows:
                    try:
                        if row["session"] == group and current(row):
                            os.kill(row["pid"], sig)
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    except OSError as error:
                        record["errors"].append(
                            {"type": type(error).__name__, "pid": row["pid"]}
                        )
                until = time.monotonic() + 5
                while time.monotonic() < until:
                    rows, errors = actors(group)
                    record["errors"].extend(errors)
                    if not rows:
                        break
                    time.sleep(0.1)
            record["current"] = rows
            record["drained"] = not rows and not record["errors"]
            save()
            return record
        time.sleep(1)


def launch_owned(
    argv: list[str], root: Path, environment: dict[str, str], limit_seconds: float
) -> tuple[subprocess.Popen[bytes], subprocess.Popen[bytes], dict[str, Any]]:
    """Seal a new session and its independent guard before releasing exec.

    This repaired launch protocol is intentionally distinct from the historical
    three-child run whose late guard saw an already-empty session.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    if (
        type(limit_seconds) not in (int, float)
        or not math.isfinite(limit_seconds)
        or limit_seconds <= 0
    ):
        raise ValueError("positive finite execution limit required")
    if not argv or not all(
        isinstance(value, str) and "\0" not in value for value in argv
    ):
        raise ValueError("invalid frozen exec argv")
    gate = "import os,json,sys; a=json.loads(sys.argv[1]); g=sys.stdin.buffer.read(1); sys.exit(125) if g!=b'G' else os.execve(a[0],a,os.environ)"
    process = None
    guard = None
    try:
        with (
            (root / "worker.stdout").open("xb") as out,
            (root / "worker.stderr").open("xb") as err,
        ):
            process = subprocess.Popen(
                [sys.executable, "-B", "-c", gate, json.dumps(argv)],
                cwd=root,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
        fields = Path(f"/proc/{process.pid}/stat").read_text().split(") ", 1)[1].split()
        leader = {
            "pid": process.pid,
            "starttime": fields[19],
            "pgid": int(fields[2]),
            "session": int(fields[3]),
        }
        if leader["pgid"] != process.pid or leader["session"] != process.pid:
            raise ValueError("new worker did not own its session")
        receipt = root / "outer-guard.json"
        plan = {
            "leader": leader,
            "limit_seconds": limit_seconds,
            "attached_after_launch": False,
            "output": str(receipt),
        }
        plan_path = root / "outer-plan.json"
        plan_path.write_text(json.dumps(plan))
        with (
            (root / "guard.stdout").open("xb") as out,
            (root / "guard.stderr").open("xb") as err,
        ):
            guard = subprocess.Popen(
                [sys.executable, "-B", str(Path(__file__).resolve()), str(plan_path)],
                cwd=root,
                env=environment,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
        until = time.monotonic() + 10
        acknowledged = False
        while time.monotonic() < until and guard.poll() is None:
            if receipt.is_file():
                ready = json.loads(receipt.read_text())
                if (
                    ready.get("leader") == leader
                    and ready.get("guard_pid") == guard.pid
                    and ready.get("guard_pgid") == guard.pid
                    and ready.get("attached_after_launch") is False
                    and ready.get("timed_out") is False
                    and ready.get("errors") == []
                    and ready.get("seen", {}).get(
                        str(process.pid) + ":" + leader["starttime"]
                    )
                    == leader
                    and leader in ready.get("current", [])
                ):
                    acknowledged = True
                    break
            time.sleep(0.01)
        if not acknowledged or guard.poll() is not None or not current(leader):
            raise RuntimeError("independent guard was not ready before exec")
        process.stdin.write(b"G")
        process.stdin.flush()
        process.stdin.close()
        return process, guard, leader
    except BaseException as primary:
        # The gate has not been released on admission failure. Reap both actors
        # owned by this launch, retaining the admission failure if cleanup fails.
        for actor in (process, guard):
            if actor is None:
                continue
            try:
                if actor.stdin is not None and not actor.stdin.closed:
                    actor.stdin.close()
                if actor.poll() is None:
                    actor.terminate()
                try:
                    actor.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    actor.kill()
                    actor.wait(timeout=5)
            except BaseException as error:  # noqa: BLE001 - retain primary on interrupted cleanup
                primary.add_note("owned launch cleanup failed: " + repr(error))
        raise


if __name__ == "__main__":
    result = supervise(json.loads(Path(sys.argv[1]).read_text()))
    raise SystemExit(0 if result.get("drained") and not result["timed_out"] else 1)
