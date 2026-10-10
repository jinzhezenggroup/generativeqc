#!/usr/bin/env python3
"""Inventory the fixed GCC host closure used by rank-k CUDA qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

SCHEMA = "generativeqc.rank-k-host-toolchain.v2"
PROGRAMS = ("cc1plus", "as", "collect2", "ld", "lto-wrapper")
LINKER_PLUGINS = ("liblto_plugin.so",)
LINK_INPUTS = (
    "Scrt1.o",
    "crt1.o",
    "crti.o",
    "crtbeginS.o",
    "crtendS.o",
    "crtn.o",
    "libstdc++.so",
    "libstdc++.so.6",
    "libm.so",
    "libm.so.6",
    "libmvec.so.1",
    "libmvec_nonshared.a",
    "libgcc_s.so",
    "libgcc_s.so.1",
    "libgcc.a",
    "libc.so",
    "libc.so.6",
    "libc_nonshared.a",
    "ld-linux-x86-64.so.2",
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(
    arguments: list[str], *, stdin: str | None = None
) -> subprocess.CompletedProcess:
    environment = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    return subprocess.run(
        arguments,
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env=environment,
    )


def _program(compiler: Path, name: str) -> Path:
    value: str = _run([str(compiler), f"-print-prog-name={name}"]).stdout.strip()
    candidate = Path(value)
    if not candidate.is_absolute():
        resolved = Path(shutil.which(value) or "")
    else:
        resolved = candidate
    if not resolved.is_file():
        raise FileNotFoundError(f"missing GCC host program: {name}")
    return resolved.resolve()


def _include_roots(compiler: Path) -> tuple[Path, ...]:
    probe = _run([str(compiler), "-E", "-x", "c++", "-v", "-"], stdin="")
    active = False
    roots = []
    for line in probe.stderr.splitlines():
        text = line.strip()
        if text == "#include <...> search starts here:":
            active = True
            continue
        if text == "End of search list.":
            break
        if active:
            text = text.removesuffix(" (framework directory)")
            path = Path(text).resolve()
            if path.is_dir() and path not in roots:
                roots.append(path)
    if not roots:
        raise RuntimeError("GCC reported no C++ include search roots")
    return tuple(roots)


def _link_input(compiler: Path, name: str) -> Path | None:
    value = _run([str(compiler), f"-print-file-name={name}"]).stdout.strip()
    path = Path(value)
    return path.resolve() if value != name and path.is_file() else None


def _ldd(path: Path) -> tuple[Path, ...]:
    result = _run(["ldd", str(path)])
    dependencies = []
    for line in result.stdout.splitlines():
        match = re.search(r"(?:=>\s+)?(/[^\s]+)", line)
        if match:
            dependency = Path(match.group(1)).resolve()
            if dependency.is_file() and dependency not in dependencies:
                dependencies.append(dependency)
    return tuple(dependencies)


def inventory(compiler: Path) -> dict:
    """Return path-independent roles and hashes for one fixed GCC installation."""
    compiler = compiler.resolve()
    if not compiler.is_file():
        raise FileNotFoundError("rank-k host compiler is missing")
    entries: dict[str, str] = {}

    def add_file(role: str, path: Path) -> None:
        if role in entries:
            raise ValueError(f"duplicate host-toolchain role: {role}")
        entries[role] = _file_digest(path)

    programs = {"driver": compiler}
    programs.update({name: _program(compiler, name) for name in PROGRAMS})
    for role, path in programs.items():
        add_file(f"program:{role}", path)
        for dependency in _ldd(path):
            add_file(f"dependency:{role}:{dependency.name}", dependency)

    for name in LINKER_PLUGINS:
        path = _link_input(compiler, name)
        if path is None:
            raise FileNotFoundError(f"missing GCC host linker plugin: {name}")
        add_file(f"linker-plugin:{name}", path)
        for dependency in _ldd(path):
            add_file(f"dependency:linker-plugin:{name}:{dependency.name}", dependency)

    roots = _include_roots(compiler)
    for index, root in enumerate(roots):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                add_file(f"header:{index}:{path.relative_to(root).as_posix()}", path)

    for name in LINK_INPUTS:
        path = _link_input(compiler, name)
        if path is not None:
            add_file(f"link-input:{name}", path)
    for required in ("libstdc++.so", "libgcc.a", "crtbeginS.o"):
        if f"link-input:{required}" not in entries:
            raise FileNotFoundError(f"missing GCC host link input: {required}")

    entries["config:gcc-specs"] = _digest(
        _run([str(compiler), "-dumpspecs"]).stdout.encode()
    )
    entries["config:ld-default-script"] = _digest(
        _run([str(programs["ld"]), "--verbose"]).stdout.encode()
    )
    return {
        "schema": SCHEMA,
        "compiler_sha256": entries["program:driver"],
        "target": _run([str(compiler), "-dumpmachine"]).stdout.strip(),
        "version": _run([str(compiler), "-dumpfullversion"]).stdout.strip(),
        "entries": [
            {"role": role, "sha256": digest} for role, digest in sorted(entries.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = inventory(args.compiler)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
