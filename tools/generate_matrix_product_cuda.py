"""Generate canonical host-side matrix-product binding metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from generativeqc_compiler.tensor.matrix_product import emit_matrix_product_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(emit_matrix_product_cuda(), encoding="utf-8")


if __name__ == "__main__":
    main()
