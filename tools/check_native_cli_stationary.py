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
def compiler_native_profiles(
    selection: str = "auto", automatic: str | None = None
) -> tuple[str, ...]:
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
    if selection == "auto":
        defaults = (
            subprocess.run(
                [
                    sys.executable,
                    str(source / "tools/generate_native_stationary_cuda.py"),
                    "--list-aot-profiles",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            .stdout.strip()
            .split(";")
        )
        if automatic is not None:
            arguments = [
                sys.executable,
                str(source / "tools/generate_native_stationary_cuda.py"),
                "--list-automatic-aot-profiles",
            ]
            for name in filter(None, automatic.split(";")):
                arguments.extend(("--functional", name))
            defaults.extend(
                subprocess.run(
                    arguments, check=True, capture_output=True, text=True, timeout=60
                )
                .stdout.strip()
                .split(";")
            )
        return tuple(profile for profile in profiles if profile in defaults)
    selected = tuple(sorted(set(selection.split(";")))) if selection else ()
    if any(profile not in profiles for profile in selected):
        raise ValueError("unknown explicitly selected native stationary profile")
    return selected


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
    selection = re.search(
        r"^GENERATIVEQC_STATIONARY_AOT_PROFILES:STRING=(.*)$", cache, re.MULTILINE
    )
    automatic = None
    if re.search(
        r"^GENERATIVEQC_ENABLE_AUTOMATIC_XC_CUDA_AOT:BOOL=ON$", cache, re.MULTILINE
    ):
        names = re.search(
            r"^GENERATIVEQC_AUTOMATIC_XC_CUDA_FUNCTIONALS:STRING=(.*)$",
            cache,
            re.MULTILINE,
        )
        automatic = names[1] if names else ""
    profiles = compiler_native_profiles(
        selection[1] if selection else "auto", automatic
    )
    expected_set = set(profiles)
    if linked != expected_set:
        raise ValueError(
            f"native CUDA profile registry mismatch: missing={sorted(expected_set - linked)}, "
            f"unexpected={sorted(linked - expected_set)}"
        )

    for profile in profiles:
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
            raise ValueError(
                f"compiled CLI is missing shared CUDA kernel launcher {shared}"
            )


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
    count = len(re.findall(r"&native_stationary_profile_([a-z0-9_]+)\(\)", registry))
    print(
        f"Verified {count} compiler-selected CUDA graphs "
        f"statically linked in {args.cli}; real-GPU E+F still requires device tests"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
