"""Regenerate Graph-derived arithmetic for native integral interpreters."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.integral.scalar_ad import emit_scalar_ad_header


def main() -> None:
    """Emit identical checkout and installed scalar schedules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "src/integrals/generated_scalar_ad.hpp"
    )
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    source = emit_scalar_ad_header()
    if arguments.check:
        if not arguments.output.exists() or arguments.output.read_text() != source:
            raise SystemExit(
                f"outdated generated integral arithmetic: {arguments.output}"
            )
    elif not arguments.output.exists() or arguments.output.read_text() != source:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(source)


if __name__ == "__main__":
    main()
