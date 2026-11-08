"""Emit shared C++/CUDA dispersion scalar energies and compiler-AD responses."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.geometry.dispersion_scalar import (
    emit_dispersion_scalar_header,
)


def main() -> None:
    """Generate without importing a native runtime or an independent oracle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = emit_dispersion_scalar_header()
    if not args.output.exists() or args.output.read_text() != source:
        args.output.write_text(source)


if __name__ == "__main__":
    main()
