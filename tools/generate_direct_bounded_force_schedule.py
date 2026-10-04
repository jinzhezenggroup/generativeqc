"""Generate the opt-in bounded homogeneous force execution policy."""

import argparse
import sys as _compiler_sys
from pathlib import Path

_compiler_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from generativeqc_compiler.integral.direct_bounded_force_schedule import (
    emit_direct_bounded_force_schedule_header,
)


def main() -> None:
    """Write deterministic compiler output without importing the runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = emit_direct_bounded_force_schedule_header()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text(encoding="utf-8") != source:
        args.output.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()
