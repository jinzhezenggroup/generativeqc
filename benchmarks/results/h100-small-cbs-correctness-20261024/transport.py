"""Task-owned, finite process and raw-return primitives; no solver imports.

The supervisor entrypoint is intentionally separate from reference admission.
An externally supplied root release must bind the frozen plan and real qz proof.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import struct
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

if not __debug__:
    raise RuntimeError("H100 transport requires non-optimized Python")

ROOT_CHAT = "01a10e8f-7b0a-7b52-a330-b2de34309210"


def inherit_no_bytecode_writes() -> None:
    """-B is process-local; the targeted environment control also reaches children."""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True


def creation_identity(value: Any) -> str:
    """Canonical comparison token; never rewrite the original API envelope."""
    if type(value) is int:
        if 0 < value < (1 << 63):
            return str(value)
    elif (
        type(value) is str
        and re.fullmatch(r"[1-9][0-9]{0,18}", value)
        and int(value) < (1 << 63)
    ):
        return value
    raise ValueError("creation identity requires a positive canonical ASCII integer")


def can_continue_scientific_failure(process: dict, worker: dict) -> bool:
    allowed = {
        ("SCI_NATIVE", "SCIENTIFIC_REQUEST"),
        ("SCI_NUMERIC", "SCIENTIFIC_NUMERIC"),
    }
    return (
        process.get("status") == "FAILED"
        and not process.get("trigger")
        and not process.get("cleanup_failed")
        and type(process.get("returncode")) is int
        and process["returncode"] > 0
        and worker.get("status") == "FAILED"
        and (worker.get("stage"), worker.get("failure_kind")) in allowed
        and type(worker.get("calls_attempted")) is int
        and worker["calls_attempted"] > 0
    )


@contextmanager
def phase_deadline(seconds: int) -> Iterator[None]:
    if os.name != "posix":
        raise RuntimeError("POSIX phase deadline required for actual runtime admission")
    if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        raise RuntimeError("refusing to replace an existing process timer")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(*unused: object) -> None:
        raise TimeoutError("finite metadata admission deadline exceeded")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def sha(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def checked(path: Path, expected: str) -> Path:
    if len(expected) != 64 or sha(path) != expected:
        raise ValueError("hash-bound input changed: " + path.name)
    return path.resolve(strict=True)


def read(path: Path) -> dict:
    def reject(value: str) -> None:
        raise ValueError("nonfinite JSON: " + value)

    value = json.loads(path.read_bytes(), parse_constant=reject)
    if not isinstance(value, dict):
        raise TypeError("expected JSON object")
    return value


def plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"raw_nonfinite": repr(value)}
    return value


def write_new(path: Path, value: dict | list) -> None:
    temporary = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(plain(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Linking a complete inode is atomic and fails if any publisher won first.
        os.link(temporary, path)
        if os.name == "posix":
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def retain_return(directory: Path, sample: dict) -> dict:
    """Persist IEEE originals and their index before callers inspect results."""
    directory.mkdir(exist_ok=False)
    binary = directory / "values.bin"
    index = []
    with binary.open("xb") as stream:

        def walk(value: Any, field: list) -> None:
            if hasattr(value, "dtype") and hasattr(value, "tobytes"):
                if value.dtype.hasobject:
                    raise ValueError(
                        "object arrays cannot be original numeric evidence"
                    )
                payload, dtype, shape = (
                    value.tobytes(order="C"),
                    value.dtype.str,
                    list(value.shape),
                )
            elif isinstance(value, float):
                payload, dtype, shape = struct.pack("<d", value), "<f8", []
            elif isinstance(value, dict):
                for k, v in value.items():
                    walk(v, [*field, k])
                return
            elif isinstance(value, (list, tuple)):
                for k, v in enumerate(value):
                    walk(v, [*field, k])
                return
            else:
                return
            index.append(
                {
                    "field": field,
                    "offset": stream.tell(),
                    "bytes": len(payload),
                    "dtype": dtype,
                    "shape": shape,
                }
            )
            stream.write(payload)

        walk(sample, [])
        stream.flush()
        os.fsync(stream.fileno())
    write_new(directory / "values.index.json", index)
    write_new(directory / "return.json", sample)
    return {p.name: sha(p) for p in directory.iterdir()}


def release(plan_path: Path, release_path: Path) -> tuple[dict, dict]:
    """Reject stale/foreign releases before native imports or output mutation."""
    plan = read(plan_path)
    approval = read(release_path)
    if (
        approval.get("schema") != "generativeqc.h100-small-cbs-root-release.v2"
        or approval.get("root_coordinator_chat") != ROOT_CHAT
        or approval.get("prepared_plan_sha256") != sha(plan_path)
        or approval.get("expected_job_name") != plan.get("expected_job_name")
        or approval.get("authorized_calls") != 32
        or approval.get("authorized_science_processes") != 4
        or approval.get("authority") != "ONE_FINITE_BATCH"
    ):
        raise ValueError("missing exact root release for this finite plan")
    if plan.get("execution_authority") != "SCI_NOT_AUTHORIZED":
        raise ValueError(
            "prepared plan authority must remain unchanged; release is separate"
        )
    proof = read(
        checked(Path(approval["allocation_proof"]), approval["allocation_proof_sha256"])
    )
    if (
        proof.get("name") != plan["expected_job_name"]
        or proof.get("provider") != "qz"
        or proof.get("status") != "RUNNING"
        or proof.get("gpu_count") != 1
        or not proof.get("platform_status")
        or not proof.get("instances")
        or not proof.get("resources")
    ):
        raise ValueError("incomplete actual single-H100 qz allocation proof")
    raw_created = proof["platform_status"].get("created_at")
    canonical = creation_identity(raw_created)
    if (
        approval.get("allocation_created_at") != canonical
        or approval.get("expected_created_at_canonical") != canonical
        or type(approval.get("allocation_created_at_raw")) is not type(raw_created)
        or approval.get("allocation_created_at_raw") != raw_created
    ):
        raise ValueError("root release does not bind actual Job creation identity")
    window = approval.get("submission_window_ms", {})
    low = creation_identity(window.get("not_before"))
    high = creation_identity(window.get("not_after"))
    if not int(low) <= int(canonical) <= int(high):
        raise ValueError("actual creation is outside root submission window")
    now = time.time()
    start = approval.get("released_unix_seconds")
    if (
        not isinstance(start, (int, float))
        or not math.isfinite(start)
        or not 0 <= now - start < plan["caps"]["job_seconds"]
    ):
        raise ValueError("root release expired or timestamp invalid")
    for name, expected in plan["task_files"].items():
        path = plan_path.parent / name
        if path.name != name or path.resolve().parent != plan_path.parent.resolve():
            raise ValueError("invalid task closure path")
        checked(path, expected)
    if (
        len(plan["calls"]) != 32
        or len(plan["processes"]) != 4
        or [c["index"] for c in plan["calls"]] != list(range(32))
    ):
        raise ValueError("finite call inventory changed")
    return plan, proof


def tree_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError("task output/JIT slots must not escape through symlinks")
        if item.is_file():
            total += item.stat().st_size
    return total


def bounded(
    command: list[str],
    directory: Path,
    output_root: Path,
    jit_root: Path,
    seconds: int,
    output_cap: int,
    jit_cap: int,
    task_cap: int | None = None,
    *,
    grace_seconds: int = 30,
    kill_reap_seconds: int = 5,
    tmp_cache_cap: int | None = None,
) -> dict:
    """Run once behind a live session/subreaper guard; rc0 alone is insufficient."""
    if os.name != "posix":
        raise RuntimeError(
            "real child-group execution requires POSIX; no Windows simulation"
        )
    if not 0 < grace_seconds <= 30 or not 0 < kill_reap_seconds <= 5 or seconds <= 0:
        raise ValueError("finite lifecycle limits required")
    from process_guard import process_identity

    inherit_no_bytecode_writes()

    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    nonce = uuid.uuid4().hex
    control = directory / "control"
    deadline = started + seconds
    cleanup_budget = grace_seconds + kill_reap_seconds + 2
    outer_deadline = deadline + cleanup_budget
    outcome = {
        "command": command,
        "status": "STARTING",
        "retry": False,
        "nonce": nonce,
        "cleanup_budget_seconds": cleanup_budget,
    }
    process = None
    born = None

    def signal_guard(signum: int) -> None:
        if process is None or born is None or process.poll() is not None:
            return
        descriptor = os.pidfd_open(process.pid, 0)
        try:
            now = process_identity(process.pid)
            if now is None or any(
                now[k] != born[k] for k in ("pid", "ppid", "pgid", "sid", "start_ticks")
            ):
                raise RuntimeError("guard birth identity changed; signal refused")
            signal.pidfd_send_signal(descriptor, signum, None, 0)
        except ProcessLookupError:
            pass
        finally:
            os.close(descriptor)

    def stop_request(cause: str) -> None:
        outcome["trigger"] = cause
        if control.is_dir() and not (control / "stop-request.json").exists():
            write_new(control / "stop-request.json", {"nonce": nonce, "cause": cause})
        signal_guard(signal.SIGTERM)

    try:
        with (
            (directory / "stdout.txt").open("xb") as stdout,
            (directory / "stderr.txt").open("xb") as stderr,
        ):
            guarded = [
                sys.executable,
                "-B",
                str(Path(__file__).with_name("process_guard.py")),
                "--control",
                str(control),
                "--nonce",
                nonce,
                "--deadline",
                str(deadline),
                "--seconds",
                str(seconds),
                "--grace-seconds",
                str(grace_seconds),
                "--kill-reap-seconds",
                str(kill_reap_seconds),
                "--output-root",
                str(output_root),
                "--jit-root",
                str(jit_root),
                "--output-cap",
                str(output_cap),
                "--jit-cap",
                str(jit_cap),
            ]
            if task_cap is not None:
                guarded += ["--task-cap", str(task_cap)]
            if tmp_cache_cap is not None:
                guarded += ["--tmp-cache-cap", str(tmp_cache_cap)]
            process = subprocess.Popen(
                [*guarded, "--", *command],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            outcome["guard_pid"] = process.pid
            born = process_identity(process.pid)
            if (
                born is None
                or born["ppid"] != os.getpid()
                or born["sid"] != process.pid
                or born["pgid"] != process.pid
            ):
                raise RuntimeError("actual exclusive guard birth not established")
            ready = min(deadline, started + 10)
            while (
                not (control / "identity.json").exists()
                and process.poll() is None
                and time.monotonic() < ready
            ):
                time.sleep(0.02)
            if not (control / "identity.json").exists():
                raise RuntimeError("guard failed before command acknowledgement")
            identity = read(control / "identity.json")
            if identity.get("nonce") != nonce or any(
                identity.get(k) != born[k]
                for k in ("pid", "ppid", "pgid", "sid", "start_ticks")
            ):
                raise RuntimeError("guard startup handshake identity differs")
            write_new(
                control / "start-command.json",
                {
                    "nonce": nonce,
                    "guard_identity": {
                        k: v for k, v in identity.items() if k != "nonce"
                    },
                    "command_sha256": hashlib.sha256(
                        json.dumps(command, separators=(",", ":")).encode()
                    ).hexdigest(),
                },
            )
            while process.poll() is None:
                if time.monotonic() >= outer_deadline:
                    outcome.update(
                        status="CLEANUP_FAILED",
                        cleanup_failed=True,
                        cleanup_success=False,
                        cleanup_error="guard exceeded command plus finite cleanup/report budget",
                    )
                    signal_guard(signal.SIGKILL)
                    break
                time.sleep(0.05)
            outcome["guard_returncode"] = process.poll()
            if (control / "guard-outcome.json").exists():
                result = read(control / "guard-outcome.json")
                if result.get("nonce") != nonce:
                    raise RuntimeError("guard outcome nonce differs")
                outcome.update(
                    {
                        k: result[k]
                        for k in (
                            "status",
                            "trigger",
                            "returncode",
                            "cleanup_success",
                            "remaining_owned",
                            "error",
                            "cleanup_error",
                        )
                        if k in result
                    }
                )
                outcome["pid"] = result.get("command_identity", {}).get("pid")
                if outcome["status"] == "RETURNED" and (
                    outcome.get("returncode") != 0
                    or not outcome.get("cleanup_success")
                    or outcome.get("remaining_owned")
                    or outcome["guard_returncode"] != 0
                ):
                    outcome.update(
                        status="CONTROL_ERROR",
                        error="inconsistent clean-return guard evidence",
                    )
            elif outcome["status"] == "STARTING":
                outcome.update(
                    status="CONTROL_ERROR",
                    cleanup_success=False,
                    remaining_owned="UNAVAILABLE",
                    error="guard exited without final lifecycle evidence",
                )
            stdout.flush()
            os.fsync(stdout.fileno())
            stderr.flush()
            os.fsync(stderr.fileno())
    except BaseException as error:
        outcome.update(status="CONTROL_ERROR", error=f"{type(error).__name__}: {error}")
        if process is not None and process.poll() is None:
            stop_request("PARENT_CONTROL_ERROR")
        raise
    finally:
        if process is not None and process.poll() is None:
            end = min(outer_deadline, time.monotonic() + cleanup_budget)
            while process.poll() is None and time.monotonic() < end:
                time.sleep(0.05)
            if process.poll() is None:
                signal_guard(signal.SIGKILL)
                outcome.update(
                    cleanup_failed=True,
                    cleanup_success=False,
                    remaining_owned="UNAVAILABLE",
                )
        if process is not None:
            outcome["guard_returncode"] = process.poll()
        outcome["elapsed_seconds"] = time.monotonic() - started
        write_new(directory / "process-outcome.json", outcome)
    return outcome
