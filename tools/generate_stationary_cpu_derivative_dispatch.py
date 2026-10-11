"""Emit native bindings to the existing packaged CPU derivative inventory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.integral.first_derivative_schedule import (
    COMPONENT_LABELS,
    CPU_AOT_SHARDS,
    cpu_aot_symbol,
    derivative_dispatch_table,
)


def source(enabled: bool) -> str:
    """Preserve compiler-owned center/axis permutations without runtime inference."""
    lines = ["#pragma once", "#include <array>", "#include <cstddef>"]
    if enabled:
        for shard in range(CPU_AOT_SHARDS):
            lines.append(
                f'extern "C" int {cpu_aot_symbol(shard)}(unsigned, const double*, std::size_t, double*);'
            )
    lines += [
        "namespace generativeqc::integrals::generated_stationary_dispatch {",
        f"inline constexpr bool available = {'true' if enabled else 'false'};",
        f"inline constexpr std::size_t component_count = {len(COMPONENT_LABELS)};",
        "inline constexpr std::size_t eri_offset = 3 * component_count * component_count;",
    ]
    lines += [
        "using Evaluate = int (*)(unsigned, const double*, std::size_t, double*);",
        "inline constexpr Evaluate evaluators[]{",
        *(
            f"  {cpu_aot_symbol(shard) if enabled else 'nullptr'},"
            for shard in range(CPU_AOT_SHARDS)
        ),
        "};",
        "inline constexpr std::array<unsigned, 3> components[]{",
        *(
            f"  {{{label.count('x')}, {label.count('y')}, {label.count('z')}}},"
            for label in COMPONENT_LABELS
        ),
        "};",
        "inline constexpr std::array<int, 9> bindings[]{",
        *(
            "  {" + ", ".join(map(str, row)) + "},"
            for row in derivative_dispatch_table(COMPONENT_LABELS)
        ),
        "};",
    ]
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write deterministic metadata; generation never executes an integral."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enabled", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    content = source(bool(args.enabled))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text(encoding="utf-8") != content:
        args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
