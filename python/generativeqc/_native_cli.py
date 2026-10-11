"""Route installed runtime commands to the packaged executable, not Python science."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def dispatch(arguments: list[str]) -> int | None:
    """Inherit scheduler visibility and propagate the native exit status unchanged.

    Source-only compiler installations retain their existing administration
    interface. A configured executable is authoritative: a bad override never
    triggers fallback to another implementation or a PATH console-script loop.
    """
    if not arguments:
        return None
    runtime = arguments[0] in (
        "run",
        "resources",
        "profile",
        "basis",
        "--version",
        "version",
    )
    runtime |= arguments[0] == "methods"
    runtime |= arguments[0] == "autotune" and any(
        flag in arguments for flag in ("--show-profile", "--clear-profile")
    )
    if not runtime:
        return None
    configured = os.environ.get("GENERATIVEQC_NATIVE_CLI")
    executable = (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parent / "bin/generativeqc"
    )
    if not executable.is_file() or not os.access(executable, os.X_OK):
        if configured or arguments[0] in ("run", "basis", "--version", "version"):
            raise RuntimeError(
                "native CLI was not found; build/install the native SDK or set "
                "GENERATIVEQC_NATIVE_CLI to its executable"
            )
        return None
    return subprocess.call([str(executable), *arguments])
