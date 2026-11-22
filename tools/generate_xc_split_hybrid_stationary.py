"""Generate split-hybrid point values and analytic directional potentials."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.xc.split_hybrid_stationary import (
    emit_split_hybrid_stationary_point,
)


def main() -> None:
    """Generate one header, optionally omitting the action for first-order owners."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("M06-2X", "MN15"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--value-only", action="store_true")
    args = parser.parse_args()
    modes = (False,) if args.value_only else (False, True)
    text = "\n".join(
        (
            "#pragma once",
            "#include <cmath>",
            "namespace generativeqc::dft::generated {",
            *(
                emit_split_hybrid_stationary_point(
                    args.method, with_action=action, qualifier="__device__ inline"
                )
                for action in modes
            ),
            "}",
            "",
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text() != text:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
