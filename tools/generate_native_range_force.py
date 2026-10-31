"""Package the existing bounded host range derivative for the native KS owner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.integral.range_exchange_cpu import emit_range_exchange_cpu
from generativeqc_compiler.integral.range_separation import CoulombKernel
from generativeqc_compiler.method import compile_ks_execution_plan, resolve_method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--enabled", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    plan = compile_ks_execution_plan(resolve_method("WB97M-V"))
    radial = next(term for term in plan.exchange if term.operator == "long-range")
    symbol = "generativeqc_native_range_force_cpu"
    signature = (
        f"int {symbol}(const double*, std::size_t, std::size_t, std::size_t, "
        "const std::int64_t*, const double*, std::size_t, bool, double*, std::uint64_t*)"
    )
    header = "#pragma once\n#include <cstddef>\n#include <cstdint>\n"
    header += (
        f'extern "C" {signature};\n'
        if args.enabled
        else f"inline {signature} {{ return 1; }}\n"
    )
    header += "namespace generativeqc::scf {\n"
    header += f"inline constexpr bool native_range_force_available = {str(bool(args.enabled)).lower()};\n"
    header += f"inline constexpr double native_range_force_omega = {float(radial.omega).hex()};\n}}\n"
    outputs = {"generated_native_range_force.hpp": header}
    if args.enabled:
        outputs["generated_native_range_force.cpp"] = emit_range_exchange_cpu(
            CoulombKernel("long_range", float(radial.omega)), entry_name=symbol
        )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for name, source in outputs.items():
        path = args.output_directory / name
        if not path.exists() or path.read_text() != source:
            path.write_text(source)


if __name__ == "__main__":
    main()
