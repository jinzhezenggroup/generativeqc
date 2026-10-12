"""Verify graph-qualified CUDA force programs really survive the native CLI link.

This is compile/link evidence, NOT a real-GPU numerical force qualification.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def compiler_native_profiles() -> tuple[str, ...]:
    """Read the actual compiler-owned graph inventory, not a handwritten list."""
    source = Path(__file__).resolve().parents[1]
    generated = subprocess.run(
        [
            sys.executable,
            str(source / "tools/generate_native_stationary_cuda.py"),
            "--list-profiles",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    profiles = tuple(generated.split(";")) if generated else ()
    if not profiles or len(profiles) != len(set(profiles)):
        raise ValueError("invalid empty/duplicate native stationary graph inventory")
    return profiles


def verify_contract(cache: str, registry: str, symbol_listing: str) -> None:
    """Reject a binary that only advertises CUDA without its native E+F code."""
    for option, expected in (
        ("GENERATIVEQC_MONOLITHIC_CLI", "ON"),
        ("GENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA", "ON"),
        ("GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT", "OFF"),
    ):
        match = re.search(rf"^{option}:BOOL=(\S+)$", cache, flags=re.MULTILINE)
        if match is None or match.group(1) != expected:
            raise ValueError(
                f"{option} must equal {expected} in the distribution build"
            )

    linked_entries = re.findall(
        r"&native_stationary_profile_([a-z0-9_]+)\(\)", registry
    )
    if len(linked_entries) != len(set(linked_entries)):
        raise ValueError("duplicate CUDA stationary graph in linked registry")
    linked = set(linked_entries)
    expected_set = set(compiler_native_profiles())
    if linked != expected_set:
        raise ValueError(
            f"native CUDA profile registry mismatch: missing={sorted(expected_set - linked)}, "
            f"unexpected={sorted(linked - expected_set)}"
        )

    for profile in compiler_native_profiles():
        signature = f"generativeqc::dft::native_stationary_profile_{profile}()"
        if signature not in symbol_listing:
            raise ValueError(f"compiled CLI is missing native force symbol {signature}")
    if "generativeqc::dft::cuda_stationary_bindings()" not in symbol_listing:
        raise ValueError(
            "compiled CLI is missing its native stationary registry entry point"
        )
    for shared in (
        "launch_native_stationary_source_transport(",
        "launch_native_stationary_source_reduction(",
    ):
        if shared not in symbol_listing:
            raise ValueError(f"compiled CLI is missing shared CUDA kernel launcher {shared}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-tree", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    args = parser.parse_args()
    cache = (args.build_tree / "CMakeCache.txt").read_text(encoding="utf-8")
    registry = (
        args.build_tree / "generated/stationary_force/native_registry.cpp"
    ).read_text(encoding="utf-8")
    symbols = subprocess.run(
        ["nm", "--demangle", "--extern-only", "--defined-only", str(args.cli)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    verify_contract(cache, registry, symbols)
    print(
        f"Verified {len(compiler_native_profiles())} compiler-selected CUDA graphs "
        f"statically linked in {args.cli}; real-GPU E+F still requires device tests"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
