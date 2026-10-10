"""Generate the shared paired weighted-Gram loop for GFN2 kernel envelopes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.method.gfn2_density_lowering import (
    emit_gfn2_density_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instrumented-output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(emit_gfn2_density_contract(), encoding="utf-8")
    if args.instrumented_output is not None:
        args.instrumented_output.parent.mkdir(parents=True, exist_ok=True)
        args.instrumented_output.write_text(
            emit_gfn2_density_contract(instrumented=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
