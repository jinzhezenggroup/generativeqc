"""Short-lived CUDA discovery for tuners whose endpoints run in subprocesses."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path


def inspect_library(
    library: Path, device_id: int, *, select: bool = False, timeout: float = 60
) -> dict:
    """Return device/profile metadata without retaining a parent CUDA context.

    The native probe may initialize CUDA through cudaSetDevice. A persistent
    coordinator context prevents endpoint/oracle children from using an
    exclusive-process GPU. Process exit releases only the probe's own context;
    never reset a context that could belong to another in-process caller.
    """
    command = [
        sys.executable,
        "-m",
        "generativeqc._autotune_device",
        str(library),
        str(device_id),
    ]
    if select:
        command.append("--select")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env={**os.environ, "GENERATIVEQC_LIBRARY": str(library)},
    )
    if result.returncode:
        raise RuntimeError(
            f"isolated CUDA tuning discovery failed: {result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def main() -> None:
    """Only this disposable process loads and probes the CUDA library."""
    from .profiles import probe_device, select_library

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("device_id", type=int)
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    library = ctypes.CDLL(str(args.library))
    probe = probe_device(library, args.device_id)
    diagnostics = None
    if args.select:
        library, diagnostics = select_library(library, args.device_id)
    print(
        json.dumps(
            {
                "probe": probe,
                "library": str(Path(library._name).resolve()),
                "diagnostics": diagnostics,
            }
        )
    )


if __name__ == "__main__":
    main()
