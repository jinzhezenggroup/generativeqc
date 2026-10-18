"""Post-measurement output protection; no scientific or GPU imports."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys


def check_destination(value):
    """Reject existing objects, aliases and retained evidence before any write."""
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise ValueError("output traversal is not allowed")
    path = Path(os.path.abspath(path))
    if any(a == "benchmarks" and b == "results"
           for a, b in zip(path.parts, path.parts[1:])):
        raise ValueError("raw output cannot target retained benchmark evidence")
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("output path cannot contain a symlink")
    if os.path.lexists(path):
        raise FileExistsError(f"refusing existing output: {path}")
    return path


def check_outputs(*values):
    """Validate all requested destinations before reserving any of them."""
    paths = [check_destination(value) for value in values if value is not None]
    for index, first in enumerate(paths):
        for second in paths[index + 1:]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError("output destinations overlap")
    return paths


def claim_directory(value):
    """Exclusively create a fresh output directory; existing data is untouched."""
    path = check_destination(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def claim_file(value):
    """Exclusively open a new output, including dangling-symlink protection."""
    path = check_destination(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("x", encoding="utf-8")


def write_json(handle, value):
    """Replace only bytes in a handle exclusively owned by this run."""
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    handle.seek(0)
    handle.write(payload)
    handle.truncate()
    handle.flush()


@contextmanager
def json_output(value):
    """Retain setup/calculation failures while preserving the original error."""
    handle = claim_file(value)
    try:
        write_json(handle, {"status": "running"})
        try:
            yield lambda result: write_json(handle, result)
        except BaseException as error:
            try:
                write_json(handle, {"status": "failed",
                                   "exception": type(error).__name__,
                                   "detail": str(error)})
            except BaseException:
                pass
            raise
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            handle.close()
        except BaseException:
            if not active_error:
                raise


@contextmanager
def run_receipt(directory, label):
    """A secondary receipt must never mask the calculation's failure."""
    destination = Path(directory) / (label + "-exit.json")
    try:
        yield
    except BaseException as error:
        code = error.code if isinstance(error, SystemExit) else 1
        try:
            with claim_file(destination) as handle:
                write_json(handle, {"exit_code": code, "exception":
                                    type(error).__name__, "detail": str(error)})
        except BaseException:
            pass
        raise
    else:
        with claim_file(destination) as handle:
            write_json(handle, {"exit_code": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("directory", "status"))
    parser.add_argument("path")
    parser.add_argument("exit_code", nargs="?", type=int)
    args = parser.parse_args()
    if args.operation == "directory":
        claim_directory(args.path)
    else:
        with claim_file(args.path) as handle:
            write_json(handle, {"exit_code": args.exit_code})


if __name__ == "__main__":
    main()
