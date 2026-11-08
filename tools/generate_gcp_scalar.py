"""Emit shared gCP overlap/pair primals and compiler-AD radial responses."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.geometry.gcp_scalar import emit_gcp_scalar_header


def main() -> None:
    """Generate without loading runtime libraries or independent references."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    output.parent.mkdir(parents=True, exist_ok=True)
    source = emit_gcp_scalar_header()
    if not output.exists() or output.read_text() != source:
        output.write_text(source)


if __name__ == "__main__":
    main()
