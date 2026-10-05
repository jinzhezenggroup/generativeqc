#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Discover provider-free CUDA imports from the real final-link diagnostics.

CMake prefixes final link commands with this launcher for Linux wheel targets.
The first strict link attempt exposes the exact ABI-level unresolved CUDA symbols
(after CUDA/cuBLAS header aliasing such as `*_v2`). The launcher generates only
the corresponding lazy Implib.so trampolines, compiles them as PIC objects, and
retries the unchanged link command with those objects added.

Unrelated undefined symbols are never synthesized and remain hard link errors.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from tools.generate_cuda_implib import generate
except ModuleNotFoundError:
    from generate_cuda_implib import generate


@dataclass(frozen=True)
class Provider:
    name: str
    base_name: str
    load_name: str
    pattern: re.Pattern[str]


PROVIDERS = (
    Provider(
        "cudart",
        "libcudart.so",
        "libcudart.so.12",
        re.compile(r"^(?:__cuda[A-Za-z0-9_]*|cuda[A-Z][A-Za-z0-9_]*)$"),
    ),
    Provider(
        "cublas",
        "libcublas.so",
        "libcublas.so.12",
        re.compile(r"^cublas(?!Lt)[A-Za-z0-9_]*$"),
    ),
    Provider(
        "cusolver",
        "libcusolver.so",
        "libcusolver.so.11",
        re.compile(r"^cusolver[A-Za-z0-9_]*$"),
    ),
)

_UNDEFINED_PATTERNS = (
    re.compile(
        r"undefined reference to\s+[`'\"‘“](?P<symbol>[A-Za-z_][A-Za-z0-9_]*)['`\"’”]"
    ),
    re.compile(r"undefined symbol:\s*(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)"),
)


def provider_for_symbol(symbol: str) -> Provider | None:
    for provider in PROVIDERS:
        if provider.pattern.fullmatch(symbol):
            return provider
    return None


def unresolved_provider_symbols(stderr: str) -> dict[str, set[str]]:
    unresolved: dict[str, set[str]] = {}
    for pattern in _UNDEFINED_PATTERNS:
        for match in pattern.finditer(stderr):
            symbol = match.group("symbol")
            provider = provider_for_symbol(symbol)
            if provider is None:
                continue
            unresolved.setdefault(provider.name, set()).add(symbol)
    return unresolved


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _compile_trampolines(
    *,
    cc: str,
    implib_root: Path,
    work_dir: Path,
    target: str,
    symbols: dict[str, set[str]],
) -> list[str]:
    objects: list[str] = []
    providers = {provider.name: provider for provider in PROVIDERS}
    for provider_name in sorted(symbols):
        provider = providers[provider_name]
        provider_dir = work_dir / provider.name
        provider_dir.mkdir(parents=True, exist_ok=True)
        generate(
            provider.base_name,
            sorted(symbols[provider_name]),
            provider.load_name,
            target,
            implib_root,
            provider_dir,
        )
        for suffix in ("init.c", "tramp.S"):
            source = provider_dir / f"{provider.base_name}.{suffix}"
            object_path = provider_dir / f"{provider.base_name}.{suffix}.o"
            result = _run([cc, "-fPIC", "-c", str(source), "-o", str(object_path)])
            if result.returncode != 0:
                sys.stdout.write(result.stdout)
                sys.stderr.write(result.stderr)
                raise RuntimeError(
                    f"failed to compile generated {provider.name} trampoline {source}"
                )
            objects.append(str(object_path))
    return objects


def _with_trampoline_objects(command: list[str], objects: list[str]) -> list[str]:
    if not objects:
        return command
    # Keep generated dlopen/dlsym users before the existing -ldl when present,
    # so --as-needed cannot discard libdl before seeing their references.
    try:
        insert_at = command.index("-ldl")
    except ValueError:
        insert_at = len(command)
    return [*command[:insert_at], *objects, *command[insert_at:]]


def _merge(destination: dict[str, set[str]], discovered: dict[str, set[str]]) -> bool:
    changed = False
    for provider, symbols in discovered.items():
        before = len(destination.setdefault(provider, set()))
        destination[provider].update(symbols)
        changed = changed or len(destination[provider]) != before
    return changed


def _format_symbols(symbols: dict[str, set[str]]) -> str:
    groups = []
    for provider in sorted(symbols):
        groups.append(f"{provider}=[{', '.join(sorted(symbols[provider]))}]")
    return " ".join(groups)


def link_with_auto_implib(
    *,
    command: list[str],
    cc: str,
    implib_root: Path,
    work_dir: Path,
    target: str,
    max_attempts: int = 4,
) -> int:
    discovered: dict[str, set[str]] = {}
    last_failure: subprocess.CompletedProcess[str] | None = None

    for _ in range(max_attempts):
        objects = (
            _compile_trampolines(
                cc=cc,
                implib_root=implib_root,
                work_dir=work_dir,
                target=target,
                symbols=discovered,
            )
            if discovered
            else []
        )
        result = _run(_with_trampoline_objects(command, objects))
        if result.returncode == 0:
            if discovered:
                print(
                    f"[generativeqc auto-implib] {_format_symbols(discovered)}",
                    file=sys.stderr,
                )
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            return 0

        last_failure = result
        new_symbols = unresolved_provider_symbols(result.stderr)
        if not _merge(discovered, new_symbols):
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            return result.returncode

    assert last_failure is not None
    sys.stdout.write(last_failure.stdout)
    sys.stderr.write(last_failure.stderr)
    print(
        "[generativeqc auto-implib] provider symbol discovery did not converge",
        file=sys.stderr,
    )
    return last_failure.returncode


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", required=True)
    parser.add_argument("--implib-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing linker command after --")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return link_with_auto_implib(
        command=args.command,
        cc=args.cc,
        implib_root=args.implib_root,
        work_dir=args.work_dir,
        target=args.target,
    )


if __name__ == "__main__":
    raise SystemExit(main())
