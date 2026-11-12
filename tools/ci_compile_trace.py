#!/usr/bin/env python3
"""Associate CI compiler descendants with public source paths, then exec unchanged.

Used only as a ccache prefix, so ccache still identifies the real compiler and
owns hits/misses. Never log argv or environment. Optional records are consumed
by the live host sampler because Ninja buffers ordinary compiler stdout.
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE_ENV = "GENERATIVEQC_CI_COMPILE_TRACE_DIR"
MAX_SETUP_SECONDS = 1.0
SAFE_SOURCE = re.compile(r"[A-Za-z0-9_-]+(?:[./][A-Za-z0-9_-]+)*")


def process_identity(path: Path) -> tuple[int, int]:
    """Read only numeric parent PID/start ticks, never proc argv or environment."""
    with (path / "stat").open(encoding="ascii", errors="replace") as stream:
        fields = stream.read(4096).rpartition(") ")[2].split()
    parent, started = int(fields[1]), int(fields[19])
    if parent < 0 or started < 0:
        raise ValueError("invalid process identity")
    return parent, started


def public_source(value: str | Path, root: Path = ROOT) -> str | None:
    """Accept existing source files within the two explicit CLI source roots."""
    if len(os.fspath(value)) > 4096:
        return None
    try:
        relative = Path(value).resolve(strict=True).relative_to(root.resolve())
    except (OSError, ValueError, RuntimeError):
        return None
    name = relative.as_posix()
    if (
        len(name) > 180
        or not SAFE_SOURCE.fullmatch(name)
        or not name.startswith(("src/", "build/cuda-release-cli/generated/"))
        or relative.suffix not in {".c", ".cc", ".cpp", ".cxx", ".cu"}
        or not (root / relative).is_file()
    ):
        return None
    return name


def record(command: list[str]) -> None:
    """Atomically publish a bounded optional PID record before replacing ourselves."""
    destination = os.environ.get(TRACE_ENV)
    if not destination:
        return
    sources = {
        source
        for argument in command[1:4097]
        if not argument.startswith("-")
        and Path(argument).suffix in {".c", ".cc", ".cpp", ".cxx", ".cu"}
        and (source := public_source(argument, ROOT)) is not None
    }
    if len(sources) != 1:
        return
    _, started = process_identity(Path("/proc") / str(os.getpid()))
    payload = json.dumps({"start_ticks": started, "source": sources.pop()})
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=directory, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, directory / f"{os.getpid()}.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    command = sys.argv[1:]
    if not command:
        return 2

    def optional_record() -> None:
        try:
            record(command)
        except Exception:  # noqa: BLE001, S110 - diagnostics cannot skip compilation
            pass

    try:
        writer = threading.Thread(target=optional_record, daemon=True)
        writer.start()
        writer.join(timeout=MAX_SETUP_SECONDS)
    except Exception:  # noqa: BLE001, S110 - initialization is optional too
        pass
    # exec replaces any stalled diagnostic thread as well as this interpreter.
    # A missing record is preferable to delaying the compiler on a blocked sink.
    # Match subprocess's restore_signals=True: Python ignores these at startup,
    # but an exec'd native compiler must retain normal pipe/file-limit failures.
    for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ"):
        if (signum := getattr(signal, name, None)) is not None:
            signal.signal(signum, signal.SIG_DFL)
    try:
        os.execvp(command[0], command)  # noqa: S606 - preserve the original compiler invocation
    except FileNotFoundError:
        return 127
    except OSError:
        return 126


if __name__ == "__main__":
    sys.exit(main())
