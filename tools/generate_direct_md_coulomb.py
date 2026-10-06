"""Generate pair-precontracted Direct-J support without a runtime/GPU import."""

import argparse
import sys as _compiler_sys
from pathlib import Path

_compiler_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from generativeqc_compiler.integral.direct_md_coulomb_cuda import (
    emit_direct_md_coulomb_header,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = emit_direct_md_coulomb_header()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text() != source:
        args.output.write_text(source)


if __name__ == "__main__":
    main()
