"""Authenticate and restore the complete historical augmented-QZ campaign."""

from __future__ import annotations

import argparse
import runpy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks._exact_fp64_evidence import load


def restore(destination: Path) -> None:
    """Restore authenticated original text bytes into a new empty directory."""
    _, files = load(Path(__file__).resolve().parent)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Restore destination must be empty")
    for name in files:
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Campaign receipt must have a plain filename")
    destination.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (destination / name).write_bytes(text.encode("utf-8"))


def replay(destination: Path) -> None:
    """Run the original independent E/F gates with assertions enabled."""
    if not __debug__:
        raise ValueError("Historical scientific replay requires assertions enabled")
    restore(destination)
    namespace = runpy.run_path(str(destination / "summarize.py"))
    for case in namespace["CASES"]:
        result = namespace["summarize"](case)
        print(case, result["complete_endpoint_seconds"], "independent E/F passed")


def main() -> None:
    """Restore persistently on request, or replay in temporary storage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        replay(args.output)
    else:
        with tempfile.TemporaryDirectory(prefix="generativeqc-aug-qz-") as folder:
            replay(Path(folder))


if __name__ == "__main__":
    main()
