"""Emit public/direct basis-transform requests for the shared CUDA providers."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from generativeqc_compiler.tensor.basis_transform import emit_basis_transform_cuda


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(emit_basis_transform_cuda(), encoding="utf-8")


if __name__ == "__main__":
    main()
