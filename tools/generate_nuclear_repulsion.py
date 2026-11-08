"""Generate nuclear-repulsion value and gradient from the shared Coulomb DAG."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.method.nuclear_repulsion import emit_nuclear_repulsion_header


def main() -> None:
    """Generate from an uninstalled checkout without loading the runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = emit_nuclear_repulsion_header()
    if not args.output.exists() or args.output.read_text() != text:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
