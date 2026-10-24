"""Task-local build harness checks; does not operate shared cache servers."""

from __future__ import annotations

import json
import os
import signal
import subprocess


def local_workers_only(raw_status: str) -> bool:
    try:
        status = json.loads(raw_status)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(status, dict)
        and set(status) == {"Disabled"}
        and (status["Disabled"] is None or isinstance(status["Disabled"], str))
    )


def stop_owned_process_group(process: subprocess.Popen | None) -> None:
    """Stop only a still-owned child created with start_new_session=True."""
    if process is None or process.poll() is not None:
        return
    if os.getpgid(process.pid) != process.pid:
        raise RuntimeError("refusing to signal an unowned process group")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)
