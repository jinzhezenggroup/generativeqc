#!/usr/bin/env python3
"""Verify that the standalone native CUDA CLI really links its ten force profiles.

This is compile/link evidence, NOT a real-GPU numerical force qualification.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

EXPECTED_PROFILES = (
    "lda_rks", "lda_uks", "pbe_rks", "pbe_uks",
    "pbe0_rks", "pbe0_uks", "r2scan_rks", "r2scan_uks",
    "b3lyp_rks", "b3lyp_uks",
)


def verify_contract(cache: str, registry: str, symbol_listing: str) -> None:
    """Reject a binary that only advertises CUDA without its native E+F code."""
    for option, expected in (
        ("GENERATIVEQC_MONOLITHIC_CLI", "ON"),
        ("GENERATIVEQC_ENABLE_NATIVE_STATIONARY_CUDA", "ON"),
        ("GENERATIVEQC_ENABLE_STATIONARY_FORCE_AOT", "OFF"),
    ):
        match = re.search(rf"^{option}:BOOL=(\S+)$", cache, flags=re.MULTILINE)
        if match is None or match.group(1) != expected:
            raise ValueError(f"{option} must equal {expected} in the distribution build")

    linked = set(re.findall(r"&native_stationary_profile_([a-z0-9_]+)\(\)", registry))
    expected_set = set(EXPECTED_PROFILES)
    if linked != expected_set:
        raise ValueError(
            f"native CUDA profile registry mismatch: missing={sorted(expected_set - linked)}, "
            f"unexpected={sorted(linked - expected_set)}"
        )

    for profile in EXPECTED_PROFILES:
        signature = f"generativeqc::dft::native_stationary_profile_{profile}()"
        if signature not in symbol_listing:
            raise ValueError(f"compiled CLI is missing native force symbol {signature}")
    if "generativeqc::dft::cuda_stationary_bindings()" not in symbol_listing:
        raise ValueError("compiled CLI is missing its native stationary registry entry point")


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
        check=True, capture_output=True, text=True,
    ).stdout
    verify_contract(cache, registry, symbols)
    print(
        f"Verified {len(EXPECTED_PROFILES)} native CUDA stationary profiles "
        f"statically linked in {args.cli}; real-GPU E+F still requires device tests"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
