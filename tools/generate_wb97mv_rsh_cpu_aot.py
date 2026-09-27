#!/usr/bin/env python3
"""Emit fixed WB97M-V CPU SR/LR weighted-ERI derivative programs."""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.integral.range_separation import CoulombKernel
from vibeqc_compiler.integral.rsh_cpu_aot import (
    AOT_ANGULAR_DOMAIN,
    component_groups,
    inventory_size,
    program_source,
)
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.method.spec import RangeSeparatedExchangePrimitive


def _write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _radial(primitive: RangeSeparatedExchangePrimitive) -> CoulombKernel:
    family = {
        "short-range": "short_range",
        "long-range": "long_range",
    }[primitive.operator]
    return CoulombKernel(family, float(primitive.omega))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    method = resolve_method("WB97M-V", spin="unpolarized")
    primitives = tuple(
        primitive
        for primitive in method.primitives
        if type(primitive) is RangeSeparatedExchangePrimitive
    )
    if (
        len(primitives) != 2
        or {primitive.operator for primitive in primitives}
        != {"short-range", "long-range"}
        or {primitive.omega for primitive in primitives} != {primitives[0].omega}
    ):
        raise RuntimeError("WB97M-V range-exchange MethodIR contract drift")

    written = 0
    tags = {"short-range": "sr", "long-range": "lr"}
    for primitive in primitives:
        radial = _radial(primitive)
        tag = tags[primitive.operator]
        for angular_value in product(AOT_ANGULAR_DOMAIN, repeat=4):
            angular = tuple(angular_value)
            shell = "".join(str(value) for value in angular)
            for group_index in range(len(component_groups(angular))):
                source, _, _ = program_source(radial, angular, group_index)
                _write_if_changed(
                    args.output_directory
                    / f"vibeqc_wb97mv_rsh_{tag}_{shell}_{group_index}.cpp",
                    source,
                )
                written += 1
    if written != inventory_size():
        raise RuntimeError(f"WB97M-V CPU RSH AOT inventory drift: {written} programs")


if __name__ == "__main__":
    main()
