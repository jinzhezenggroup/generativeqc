"""Regenerate CPU AO Gaussian polynomial jets from the shared primal Graph."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.dft.ao_axis import emit_axis_header


def main() -> None:
    """Check deterministic compiler-owned Gaussian axis coefficients."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "src/dft/generated_ao_axis.hpp"
    )
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    source = emit_axis_header()
    if arguments.check:
        if not arguments.output.exists() or arguments.output.read_text() != source:
            raise SystemExit(f"outdated generated AO arithmetic: {arguments.output}")
    elif not arguments.output.exists() or arguments.output.read_text() != source:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(source)


if __name__ == "__main__":
    main()
