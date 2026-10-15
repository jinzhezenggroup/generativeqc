"""Opt-in owner-death cleanup for isolated compiler process groups.

A native parent can lend a read-only pipe whose writer it exclusively owns.
The default compiler launch policy is unchanged when that descriptor is absent.
"""

from __future__ import annotations

import json
import os
import select
import signal
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

OWNER_FD_ENV = "GENERATIVEQC_COMPILER_OWNER_FD"


def start_owned_process(
    command: list[str], *, pass_fds: tuple[int, ...] = (), **options: Any
) -> subprocess.Popen:
    """Preserve normal Popen behavior unless an explicit lifetime owner exists."""
    environment = options.get("env")
    descriptor = (os.environ if environment is None else environment).get(OWNER_FD_ENV)
    if descriptor is None:
        return subprocess.Popen(command, pass_fds=pass_fds, **options)
    owner = int(descriptor)
    if owner < 0 or not stat.S_ISFIFO(os.fstat(owner).st_mode):
        raise ValueError("compiler lifetime owner must be an open pipe descriptor")
    import fcntl

    if fcntl.fcntl(owner, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
        raise ValueError("compiler lifetime owner pipe must be read-only")
    cancel_read, cancel_write = os.pipe()
    try:
        group_read, group_write = os.pipe()
    except BaseException:
        os.close(cancel_read)
        os.close(cancel_write)
        raise
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                str(Path(__file__).resolve()),
                str(owner),
                str(cancel_read),
                str(group_write),
                json.dumps(list(pass_fds)),
                *command,
            ],
            pass_fds=tuple(dict.fromkeys((*pass_fds, owner, cancel_read, group_write))),
            **options,
        )
    except BaseException:
        os.close(cancel_write)
        os.close(group_read)
        raise
    finally:
        os.close(cancel_read)
        os.close(group_write)
    # Only the Python owner holds this writer. Its death also requests cleanup.
    process._generativeqc_cancel_writer = cancel_write
    process._generativeqc_group_reader = group_read
    return process


def cancel_owned_process(process: subprocess.Popen) -> bool:
    """Ask a watchdog to reclaim its group; return false for ordinary launches."""
    if not hasattr(process, "_generativeqc_cancel_writer"):
        return False
    descriptor = process._generativeqc_cancel_writer
    if descriptor is not None:
        process._generativeqc_cancel_writer = None
        os.close(descriptor)
    reader = process._generativeqc_group_reader
    try:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            # A stopped/failed watchdog must not turn cancellation into an
            # unbounded wait. Only its explicitly registered child group is ours.
            encoded = b""
            if reader is not None:
                os.set_blocking(reader, False)
                try:
                    encoded = os.read(reader, 64)
                except BlockingIOError:
                    pass
            if encoded.endswith(b"\n") and encoded[:-1].isdigit() and int(encoded) > 0:
                try:
                    os.killpg(int(encoded), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.kill()
            process.wait(timeout=1)
            raise RuntimeError(
                "compiler lifetime watchdog failed to acknowledge cancellation"
            ) from None
    finally:
        if reader is not None:
            os.close(reader)
            process._generativeqc_group_reader = None
    return True


def _supervise(
    owner: int, cancel: int, group: int, inherited: tuple[int, ...], command: list[str]
) -> int:
    """Remain outside the compiler group and cancel it even if its caller dies."""
    descriptors = (owner, cancel)
    # A parent may have exited between spawning us and our interpreter startup.
    if select.select(descriptors, (), (), 0)[0]:
        return 125
    process = subprocess.Popen(
        command,
        start_new_session=True,
        pass_fds=tuple(dict.fromkeys((*inherited, owner))),
    )
    try:
        os.write(group, f"{process.pid}\n".encode())
        os.close(group)
        while process.poll() is None:
            if select.select(descriptors, (), (), 0.01)[0]:
                return 125
        return process.returncode
    finally:
        # Retain the launcher's group isolation on success, failure and owner death.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


if __name__ == "__main__":
    result = _supervise(
        int(sys.argv[1]),
        int(sys.argv[2]),
        int(sys.argv[3]),
        tuple(json.loads(sys.argv[4])),
        sys.argv[5:],
    )
    if result < 0:
        # Preserve Popen's signal return codes for the supervised command.
        if -result not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(-result, signal.SIG_DFL)
        os.kill(os.getpid(), -result)
    raise SystemExit(result)
