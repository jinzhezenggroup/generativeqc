"""Generate the compiler-owned bounded Direct force page schedule."""

import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
from pathlib import Path

from generativeqc_compiler.integral.direct_force_pages import (
    emit_direct_force_page_header,
)


def main() -> None:
    """Write deterministic scheduling metadata without loading the runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = emit_direct_force_page_header()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text() != source:
        args.output.write_text(source)


if __name__ == "__main__":
    main()
