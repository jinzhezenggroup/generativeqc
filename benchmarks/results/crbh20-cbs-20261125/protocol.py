"""Shared endpoint protocol and durable raw-record acquisition."""

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parent
REVISION = "ae1086d511060662b7d728eb80d3e53981443ec1"
RECIPES = {"DT_D": (2, 3), "TQ_D": (3, 4)}
TIME_LIMIT_SECONDS = 7200
NUMERIC_BYTES = 64 << 30
MAX_SCALAR_WORK = 50_000_000_000_000
WORK_ENV = "GENERATIVEQC_CORRELATION_MAX_SCALAR_WORK"
KCAL_PER_HARTREE = 627.5094740631


def basis(cardinal: int) -> str:
    return f"cc-pv{'dtq'[cardinal - 2]}z"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def serializable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if hasattr(value, "tolist"):
        return serializable(value.tolist())
    return value


def append(stream: TextIO, row: dict) -> None:
    """Preserve completed records even if a later endpoint is interrupted."""
    stream.write(json.dumps(serializable(row), allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def outcome(status: str, error: str = "", returncode: int | None = None) -> str:
    message = error.lower()
    if status == "ok":
        return "ok"
    if status in ("timeout", "unfinished", "not_run"):
        return "protocol_limit" if status == "timeout" else status
    if any(
        text in message
        for text in (
            "out of memory",
            "out_of_memory",
            "memory budget",
            "resource limit",
            "resource_limit",
            "work limit",
            "scalar-work",
            "capacity exceeded",
            "no space left",
            "allocfailed",
            "memoryerror",
        )
    ):
        return "resource_limit"
    if any(
        text in message
        for text in (
            "did not converge",
            "did not complete scf",
            "non-convergen",
            "nonconvergen",
            "convergence failure",
            "not converged",
            "cc amplitudes failed",
        )
    ):
        return "convergence"
    if any(
        text in message
        for text in (
            "supports only l<=",
            "angular momentum",
            "not implemented",
            "not supported",
            "unsupported",
            "unavailable in this build",
            "eigensolve input",
        )
    ):
        return "development_required"
    if returncode in (-signal.SIGKILL, -signal.SIGTERM):
        return "external_termination"
    return "unresolved"


def endpoint(driver: Path, args: list[str], result_path: Path, log: TextIO) -> dict:
    """Use a fresh process and end its process group when the cap expires."""
    result_path.unlink(missing_ok=True)
    start = time.perf_counter()
    with subprocess.Popen(
        [sys.executable, str(driver), "point", *args, str(result_path)],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    ) as process:
        try:
            code = process.wait(timeout=TIME_LIMIT_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            row = {"status": "timeout", "limit_seconds": TIME_LIMIT_SECONDS}
        else:
            if code == 0 and result_path.is_file():
                row = load(result_path)
            else:
                row = {"status": "process_failed", "returncode": code}
    row["process_wall_seconds"] = time.perf_counter() - start
    row["classification"] = outcome(
        row["status"], row.get("error", ""), row.get("returncode")
    )
    return row
