"""Emit shared-provider COSX matrix contraction descriptors from TensorIR."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from generativeqc_compiler.dft.cosx_contraction import emit_cosx_contractions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(emit_cosx_contractions())


if __name__ == "__main__":
    main()
