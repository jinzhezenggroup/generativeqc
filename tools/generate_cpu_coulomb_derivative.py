"""Package the shared generated full-range s/p/d/f primitive derivative owner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.integral.range_exchange_cpu import (
    emit_full_range_primitive_derivative_cpu,
)


def main() -> None:
    """Emit a build-time capability and implementation, never runtime JIT."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--enabled", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    signature = (
        "int generativeqc_full_range_primitive_derivative_cpu("
        "const double*, const double*, const unsigned*, double*)"
    )
    header = "#pragma once\n"
    header += (
        f'extern "C" {signature};\n'
        if args.enabled
        else f"inline {signature} {{ return 1; }}\n"
    )
    header += "namespace generativeqc::integrals {\n"
    header += (
        "inline constexpr bool generated_full_range_derivative_available = "
        f"{str(bool(args.enabled)).lower()};\n}}\n"
    )
    outputs = {"generated_cpu_coulomb_derivative.hpp": header}
    if args.enabled:
        outputs["generated_cpu_coulomb_derivative.cpp"] = (
            emit_full_range_primitive_derivative_cpu()
        )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for name, source in outputs.items():
        path = args.output_directory / name
        if not path.exists() or path.read_text() != source:
            path.write_text(source)


if __name__ == "__main__":
    main()
