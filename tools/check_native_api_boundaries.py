#!/usr/bin/env python3
"""Enforce the native-core -> method -> public-API dependency direction."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE_EXTENSIONS = {".h", ".hpp", ".cuh", ".cpp", ".cc", ".cu"}
HANDLE_INCLUDE = re.compile(r'^\s*#\s*include\s*["<]api/handles\.hpp[">]', re.MULTILINE)
PUBLIC_INCLUDE = re.compile(
    r'^\s*#\s*include\s*["<]generativeqc/generativeqc\.h[">]', re.MULTILINE
)
HANDLE_DEREFERENCE = re.compile(r"\bdensity_fitting_auxiliary_basis\s*->\s*data\b")


def audit_native_api_boundaries(root: Path = ROOT) -> list[str]:
    """Report forbidden C-handle dependencies without a compiler or GPU."""
    errors: list[str] = []
    native_core = root / "src/core/types.hpp"
    if not native_core.is_file():
        errors.append("src/core/types.hpp: missing native core header")
    elif PUBLIC_INCLUDE.search(native_core.read_text()):
        errors.append("src/core/types.hpp: public C ABI must not enter native core")

    method_root = root / "src/methods"
    if not method_root.is_dir():
        errors.append("src/methods: missing method sources")
        return errors
    for path in sorted(method_root.rglob("*")):
        if not path.is_file() or path.suffix not in NATIVE_EXTENSIONS:
            continue
        source = path.read_text()
        relative = path.relative_to(root).as_posix()
        for line_number, line in enumerate(source.splitlines(), 1):
            if HANDLE_INCLUDE.match(line):
                errors.append(
                    f"{relative}:{line_number}: methods must not include API handles"
                )
            if HANDLE_DEREFERENCE.search(line) and not line.lstrip().startswith(
                ("//", "*")
            ):
                errors.append(
                    f"{relative}:{line_number}: method must use a native auxiliary system"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    errors = audit_native_api_boundaries(args.root)
    for message in errors:
        print(message)
    if errors:
        return 1
    print("native API dependency boundaries: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
