"""Generate the compiler-owned reciprocal MD Coulomb consumer."""

import argparse
import sys as _compiler_sys
from pathlib import Path

_compiler_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from generativeqc_compiler.integral.md_j_reciprocal_cuda import (
    emit_md_j_reciprocal_header,
)


def main() -> None:
    """Write only changed bytes so repeated generation preserves build caching."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = emit_md_j_reciprocal_header()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text(encoding="utf-8") != source:
        args.output.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
