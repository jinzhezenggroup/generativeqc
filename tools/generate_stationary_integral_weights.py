"""Emit portable native PBE/PBE0 weights from the shared stationary TensorIR AD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from generativeqc_compiler.method.stationary_weight_lowering import _weight_expression


def source() -> str:
    """Specialize existing pointwise AD; emit no independent density equations."""
    lines = ["#pragma once", "#include <cstddef>", "#include <cstdint>"]
    for method in ("PBE", "PBE0"):
        plan = StationaryGradientPlan(
            resolve_method(method, spin="unpolarized"),
            StationaryMeanField(SCF_POINT_MODEL),
        )
        lines.append(f"namespace generativeqc::scf::generated::{method.lower()} {{")
        for name in ("one_electron", "overlap_pulay", "coulomb", "exact_exchange"):
            if name not in plan.source_names:
                continue
            lowered, _ = _weight_expression(plan, name, dimension="nbf")
            lines += [
                f"// stationary-weight-program-{name}: {lowered.original_logical_hash}",
                f"inline double {name}([[maybe_unused]] const double* density, [[maybe_unused]] const double* weighted_density,",
                "                     std::size_t nbf, const std::int64_t* ao) {",
                "  using std::size_t;",
                f"  return {lowered.expression};",
                "}",
            ]
        lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write the deterministic generated header only when its content changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    content = source()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or output.read_text(encoding="utf-8") != content:
        output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
