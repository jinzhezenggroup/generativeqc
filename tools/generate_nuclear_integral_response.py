"""Emit the native fixed-projection integral source contraction with scalar AD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from generativeqc_compiler.integral.cuda import CudaEmitter
from generativeqc_compiler.method.nuclear_integral_response import (
    projected_integral_program,
)

from tools.generate_df_kernels import write_if_changed


def generate() -> str:
    """No public runtime, numerical discovery or reference engines are imported."""
    graph, roots = projected_integral_program()
    names = (
        "value",
        "first",
        "mixed",
        "weight",
        "delta_weight",
        "projection",
        "direction",
    )
    emitter = CudaEmitter(graph, dict(zip(names, names, strict=True)))
    emitter.emit(roots)
    return "\n".join(
        [
            "#pragma once",
            "#include <cmath>",
            "namespace generativeqc_nuclear_integral_ad {",
            "__device__ __forceinline__ bool contract(double value, double first, double mixed,",
            " double weight, double delta_weight, double projection, double direction, double* output) {",
            *emitter.lines,
            *(
                f" output[{index}] = {emitter.reference(root)};"
                for index, root in enumerate(roots)
            ),
            " for (unsigned field = 0; field < 4; ++field) if (!isfinite(output[field])) return false;",
            " return true;",
            "}",
            "}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_if_changed(args.output, generate())


if __name__ == "__main__":
    main()
