"""Emit manifest-owned CPU range first-derivative AOT programs."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.integral.derivative_aot_registry import (
    AOT_ANGULAR_DOMAIN,
    component_groups,
    radial_inventory_from_payload,
)
from vibeqc_compiler.integral.range_separation import CoulombKernelFamily
from vibeqc_compiler.integral.rsh_cpu_aot import program_source


def _write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _tag(family: CoulombKernelFamily | str) -> str:
    normalized = CoulombKernelFamily(family)
    return {
        CoulombKernelFamily.SHORT_RANGE: "sr",
        CoulombKernelFamily.LONG_RANGE: "lr",
    }[normalized]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--radial-manifest", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.radial_manifest.read_text(encoding="utf-8"))
    radials = radial_inventory_from_payload(payload, backend="cpu")
    if not radials:
        raise RuntimeError("CPU derivative AOT radial inventory is empty")
    if any(
        radial.family
        not in (CoulombKernelFamily.SHORT_RANGE, CoulombKernelFamily.LONG_RANGE)
        for radial in radials
    ):
        raise RuntimeError("range derivative generator requires SR/LR radial operators")

    written = 0
    for radial in radials:
        tag = _tag(radial.family)
        for angular_value in product(AOT_ANGULAR_DOMAIN, repeat=4):
            angular = tuple(angular_value)
            shell = "".join(str(value) for value in angular)
            for group_index in range(len(component_groups(angular))):
                source, _, _ = program_source(radial, angular, group_index)
                _write_if_changed(
                    args.output_directory
                    / f"vibeqc_derivative_range_{tag}_{shell}_{group_index}.cpp",
                    source,
                )
                written += 1

    per_radial = sum(
        len(component_groups((a, b, c, d)))
        for a, b, c, d in product(AOT_ANGULAR_DOMAIN, repeat=4)
    )
    expected = per_radial * len(radials)
    if written != expected:
        raise RuntimeError(
            f"CPU range derivative AOT inventory drift: {written} != {expected}"
        )


if __name__ == "__main__":
    main()
