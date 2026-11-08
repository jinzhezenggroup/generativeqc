"""Regenerate the packaged Graph-owned scaled XC arithmetic."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.xc.point_ad import emit_point_ad_header


def main() -> None:
    """Keep checkout, installed assets and explicit regeneration identical."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "src/dft/generated_point_ad.hpp"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = emit_point_ad_header()
    if args.check:
        if not args.output.exists() or args.output.read_text() != source:
            raise SystemExit(f"outdated generated point arithmetic: {args.output}")
    elif not args.output.exists() or args.output.read_text() != source:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(source)


if __name__ == "__main__":
    main()
