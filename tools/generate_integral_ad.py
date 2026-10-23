"""Emit the explicit-integral HF device contraction from canonical TensorIR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT / "python"))
from generativeqc_compiler.method.rhf_integral_ad import emit_dense_hf_terms


def main() -> None:
    """Write the deterministic generated helper consumed by the native provider."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "#pragma once\n#include <cmath>\n#include <cstddef>\n" + emit_dense_hf_terms()
    )


if __name__ == "__main__":
    main()
